"""
SFT Experiment on MATH reasoning dataset.

Experiments:
1. Vary training dataset size: {128, 256, 512, 1024, full}
2. Run on filtered dataset (correct answers only)

Evaluation uses vLLM for inference on a separate GPU.
"""

import gc
import json
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch
import argparse
import torch
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from sft_helper.tokenize_prompt_and_output import tokenize_prompt_and_output
from sft_helper.utils import sft_microbatch_train_step

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "/root/assignment5-alignment-main/model"
R1_ZERO_PROMPT_PATH = "/root/assignment5-alignment-main/cs336_alignment/prompts/r1_zero.prompt"

TRAIN_PATH = "/root/assignment5-alignment-main/data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b.jsonl"
TRAIN_FILTERED_PATH = "/root/assignment5-alignment-main/data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl"
VAL_PATH = "/root/assignment5-alignment-main/data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"

OUTPUT_DIR = "/root/assignment5-alignment-main/sft_outputs"

with open(R1_ZERO_PROMPT_PATH, "r") as f:
    R1_ZERO_PROMPT_TEMPLATE = f.read()

WANDB_KEY = "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"


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.95):
    """
    启动 vLLM 推理进程，将推理模型放在与训练策略模型不同的 GPU 上。
    
    优化说明：
    - gpu_memory_utilization 默认设为 0.95（而非 0.85），在 A100 40G 上为 KV Cache 预留更多显存，
      支持更大的推理 batch，提升 vLLM 吞吐。
    - 开启 enable_prefix_caching 复用公共 prompt 前缀（如 system prompt），减少重复计算。
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json_or_jsonl(path: str):
    with open(path, "r") as f:
        raw = f.read().strip()
    if raw.startswith("["):
        return json.loads(raw)
    else:
        data = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))
        return data


def format_prompt(problem: str) -> str:
    return R1_ZERO_PROMPT_TEMPLATE.replace("{question}", problem)


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.examples = []
        for ex in data:
            problem = ex["problem"]
            reasoning = ex.get("reasoning_trace", "")
            prompt = format_prompt(problem)
            # The reasoning trace should be generated after the <think> tag
            # which is already part of the prompt template ending with "Assistant: <think>"
            output = reasoning
            self.examples.append({"prompt": prompt, "output": output, "ground_truth": ex.get("expected_answer", "")})
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer):
    prompts = [ex["prompt"] for ex in batch]
    outputs = [ex["output"] for ex in batch]
    ground_truths = [ex["ground_truth"] for ex in batch]
    tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)

    # tokenize_prompt_and_output right-pads; for decoder-only training we need left-pad
    # and an attention mask. Convert right-pad to left-pad.
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    batch_size, seq_len = input_ids.shape
    # Find actual lengths (non-pad tokens)
    # Since tokenize_prompt_and_output slices off the last token, the pad positions
    # in input_ids correspond to pad tokens from the original full sequence.
    lengths = (input_ids != pad_token_id).sum(dim=1)

    max_len = seq_len
    new_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    new_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    new_response_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i in range(batch_size):
        actual_len = lengths[i].item()
        # Copy actual tokens to the right side (left padding)
        new_input_ids[i, -actual_len:] = input_ids[i, :actual_len]
        new_labels[i, -actual_len:] = labels[i, :actual_len]
        new_response_mask[i, -actual_len:] = response_mask[i, :actual_len]
        attention_mask[i, -actual_len:] = 1

    return {
        "input_ids": new_input_ids,
        "labels": new_labels,
        "response_mask": new_response_mask,
        "attention_mask": attention_mask,
        "ground_truths": ground_truths,
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_with_vllm(llm: LLM, val_data: list, max_eval_samples: int = 500) -> dict:
    """Evaluate model with vLLM on validation set."""
    # Sample a subset for faster evaluation during training
    eval_data = random.sample(val_data, min(max_eval_samples, len(val_data)))

    prompts = []
    ground_truths = []
    for ex in eval_data:
        prompts.append(format_prompt(ex["problem"]))
        ground_truths.append(ex.get("expected_answer", ex.get("answer", "")))

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    vllm_outputs = llm.generate(prompts, sampling_params)

    total_format = 0.0
    total_answer = 0.0
    total_reward = 0.0
    n = len(vllm_outputs)

    for i, output in enumerate(vllm_outputs):
        text = output.outputs[0].text
        metrics = r1_zero_reward_fn(text, ground_truths[i])
        total_format += metrics["format_reward"]
        total_answer += metrics["answer_reward"]
        total_reward += metrics["reward"]

    return {
        "num_examples": n,
        "format_reward": total_format / n,
        "answer_reward": total_answer / n,
        "reward": total_reward / n,
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_sft(
    model,
    tokenizer,
    train_loader,
    val_data,
    num_epochs: int,
    lr: float,
    gradient_accumulation_steps: int,
    eval_steps: int,
    device: str = "cuda:0",
    run_name: str = "sft",
    llm: LLM | None = None,
) -> tuple[dict, int, int]:
    """Run SFT training."""
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()
    
    # 打印初始训练显存占用，便于监控 A100 显存使用效率
    print(f"[Memory] Training started on {device}. "
          f"Allocated: {torch.cuda.memory_allocated(device)/1e9:.2f}GB, "
          f"Reserved: {torch.cuda.memory_reserved(device)/1e9:.2f}GB")

    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    global_step = 0
    eval_step = 0
    history = {"train_loss": [], "eval_vllm": []}

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            # Forward pass (don't pass labels to avoid internal loss computation)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids)
            logits = outputs.logits

            # Compute log probs
            log_probs_dist = F.log_softmax(logits, dim=-1)
            gather_labels = labels.clone()
            gather_labels[gather_labels == -100] = 0
            policy_log_probs = log_probs_dist.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)

            # SFT loss
            loss, logging_info = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=1.0,
            )

            # microbatch_loss 已在 sft_microbatch_train_step 中除以 gradient_accumulation_steps，
            # 累加前需乘回以得到真实的未缩放 loss，否则 avg_epoch_loss 会偏小 G 倍
            epoch_loss += logging_info["microbatch_loss"].item() * gradient_accumulation_steps
            num_batches += 1

            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                wandb.log({
                    "train/loss": logging_info["microbatch_loss"].item(),
                    "train/avg_log_prob": logging_info["average_log_prob"].item(),
                    "train/epoch": epoch + (batch_idx + 1) / len(train_loader),
                    "train_step": global_step,
                })

                # In-the-loop evaluation with vLLM
                if global_step % eval_steps == 0 and llm is not None:
                    model.eval()
                    load_policy_into_vllm_instance(model, llm)
                    eval_result = evaluate_with_vllm(llm, val_data, max_eval_samples=500)
                    eval_step += 1
                    history["eval_vllm"].append((global_step, eval_result))

                    wandb.log({
                        "eval/format_reward": eval_result["format_reward"],
                        "eval/answer_reward": eval_result["answer_reward"],
                        "eval/reward": eval_result["reward"],
                        "eval_step": eval_step,
                    })

                    model.train()
                    # eval 结束后释放 eval 阶段可能残留的临时显存，保持训练显存稳定
                    torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs}, avg loss: {avg_epoch_loss:.4f}")
        wandb.log({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch": epoch + 1,
            "train_step": global_step,
        })

    return history, global_step, eval_step

def generate_reasoning_traces(
    llm: LLM,
    train_data: list,
    rollout: int = 1,
    max_tokens: int = 1024,
    max_gen_batch_size: int = 512,
) -> tuple[list, dict]:
    """
    使用当前模型为训练数据生成新的推理轨迹，并计算奖励。
    
    参数:
        llm: vLLM 推理实例
        train_data: 训练数据列表
        rollout: 每个问题生成的采样次数
        max_tokens: 生成的最大 token 数
        max_gen_batch_size: vLLM 单次推理的最大 batch 数，防止 rollout 数量大时 KV cache 溢出
    
    返回:
        updated_train_data: 过滤后的训练数据（仅保留回答正确且格式正确的样本）
        gen_metrics: 生成过程的统计指标
    """
    prompts = []
    ground_truths = []
    data_indices = []
    
    # 构建 prompt 列表：如果 rollout > 1，每个问题复制 rollout 次
    for idx, ex in enumerate(train_data):
        prompt = format_prompt(ex["problem"])
        gt = ex.get("expected_answer", "")
        for _ in range(rollout):
            prompts.append(prompt)
            ground_truths.append(gt)
            data_indices.append(idx)

    sampling_params = SamplingParams(
        temperature=0.7,  # 使用较高温度以增加生成多样性
        top_p=0.95,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    print(f"Generating {len(prompts)} reasoning traces ({rollout} rollouts x {len(train_data)} problems)...")
    
    # 分批生成：避免一次性将所有 prompt 送入 vLLM 导致 cuda:1 的 KV cache 显存溢出
    # 利用 max_gen_batch_size 控制每批大小，vLLM 内部仍会做 continuous batching
    all_vllm_outputs = []
    for start in range(0, len(prompts), max_gen_batch_size):
        end = min(start + max_gen_batch_size, len(prompts))
        batch_outputs = llm.generate(prompts[start:end], sampling_params)
        all_vllm_outputs.extend(batch_outputs)
        print(f"  Generated batch {start//max_gen_batch_size + 1}/{(len(prompts)-1)//max_gen_batch_size + 1} "
              f"({start}-{end} / {len(prompts)})")

    vllm_outputs = all_vllm_outputs

    # 统计生成指标
    total_generated = len(vllm_outputs)
    correct_count = 0
    format_correct_count = 0

    # 收集每个原始样本的最佳生成结果
    # 使用字典记录每个原始样本索引对应的正确生成
    index_to_best_trace = {}

    for i, output in enumerate(vllm_outputs):
        text = output.outputs[0].text
        gt = ground_truths[i]
        orig_idx = data_indices[i]

        metrics = r1_zero_reward_fn(text, gt)
        if metrics["format_reward"] > 0:
            format_correct_count += 1
        if metrics["reward"] > 0:
            correct_count += 1
            # 如果该原始问题还没有找到正确生成，记录此轨迹
            if orig_idx not in index_to_best_trace:
                index_to_best_trace[orig_idx] = text

    print(f"Generation complete: {format_correct_count}/{total_generated} format-correct, "
          f"{correct_count}/{total_generated} answer-correct")

    # 构建更新后的训练数据：只保留生成正确的样本
    updated_train_data = []
    for orig_idx, ex in enumerate(train_data):
        if orig_idx in index_to_best_trace:
            new_ex = {
                "problem": ex["problem"],
                "reasoning_trace": index_to_best_trace[orig_idx],
                "expected_answer": ex.get("expected_answer", ""),
            }
            updated_train_data.append(new_ex)

    gen_metrics = {
        "total_generated": total_generated,
        "format_correct": format_correct_count,
        "answer_correct": correct_count,
        "filtered_dataset_size": len(updated_train_data),
        "accuracy": correct_count / total_generated if total_generated > 0 else 0.0,
    }

    return updated_train_data, gen_metrics


def one_expert_iteration_step(
    llm: LLM | None,
    tokenizer,
    train_data,
    val_data,
    device: str = "cuda:0",
    run_name: str = "expert_iteration_step",
    seed: int = 42,
    dataset_size: int | None = None,
    batch_size: int = 8,
    num_epochs: int = 3,
    lr: float = 1e-5,
    gradient_accumulation_steps: int = 8,
    eval_steps: int = 1,
    rollout: int = 4,
    model_path: str = MODEL_PATH,
    use_wandb: bool = True,
    max_gen_batch_size: int = 512,
) -> dict:
    """
    执行一步专家迭代（Expert Iteration）。
    
    流程：
    1. 使用当前模型为训练数据生成新的推理轨迹（rollout 次）
    2. 用奖励函数筛选出格式正确且答案正确的样本
    3. 在这些过滤后的数据上进行 SFT 训练
    4. 保存模型并返回更新后的训练数据及评估指标
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # 准备数据集
    if dataset_size is not None and dataset_size < len(train_data):
        train_subset = random.sample(train_data, dataset_size)
    else:
        train_subset = train_data
        dataset_size = len(train_data)

    print(f"\n{'='*60}")
    print(f"Expert Iteration Step: {run_name}")
    print(f"Dataset size: {dataset_size}")
    print(f"Batch size: {batch_size}, Rollout: {rollout}")
    print(f"{'='*60}\n")

    # 加载 tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # 在 cuda:0 上加载策略模型用于训练
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    ).to(device)
    model.gradient_checkpointing_enable()

    # 初始化 vLLM：如果外部未传入，则在 cuda:1 上新建
    if llm is None:
        print("Initializing vLLM on cuda:1...")
        llm = init_vllm(model_path, device="cuda:1", seed=seed)

    # 将当前模型权重加载到 vLLM，以便用最新策略生成
    print("Loading policy into vLLM for generation...")
    load_policy_into_vllm_instance(model, llm)

    # 1. 生成新的推理轨迹并过滤正确样本
    updated_train_data, gen_metrics = generate_reasoning_traces(
        llm=llm,
        train_data=train_subset,
        rollout=rollout,
        max_tokens=1024,
        max_gen_batch_size=max_gen_batch_size,
    )

    print(f"Filtered train data size: {len(updated_train_data)} / {len(train_subset)} "
          f"(accuracy: {gen_metrics['accuracy']:.4f})")

    # 如果过滤后数据为空，则回退到原始数据，避免训练失败
    if len(updated_train_data) == 0:
        print("Warning: No correct generations found, falling back to original training data.")
        updated_train_data = train_subset

    # 2. 在过滤后的数据上执行 SFT 训练
    train_dataset = SFTDataset(updated_train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    # 初始化 wandb：如果外部已经统一管理了 wandb run，则不再重复 init
    if use_wandb:
        os.environ.setdefault("WANDB_API_KEY", WANDB_KEY)
        wandb.init(
            project="cs336-sft-math",
            name=run_name,
            config={
                "dataset_size": dataset_size,
                "filtered_size": len(updated_train_data),
                "lr": lr,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "rollout": rollout,
                "model": model_path,
            },
        )

    # 记录生成指标到 wandb（复用外部已打开的 run）
    wandb.log({
        "gen/total_generated": gen_metrics["total_generated"],
        "gen/format_correct": gen_metrics["format_correct"],
        "gen/answer_correct": gen_metrics["answer_correct"],
        "gen/filtered_size": gen_metrics["filtered_dataset_size"],
        "gen/accuracy": gen_metrics["accuracy"],
    })

    history, final_train_step, final_eval_step = train_sft(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_data=val_data,
        num_epochs=num_epochs,
        lr=lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_steps=eval_steps,
        device=device,
        run_name=run_name,
        llm=llm,
    )

    # 保存模型
    save_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    # 最终评估
    print("Running final vLLM evaluation...")
    load_policy_into_vllm_instance(model, llm)
    vllm_metrics = evaluate_with_vllm(llm, val_data, max_eval_samples=500)
    print(f"vLLM evaluation results: {vllm_metrics}")

    wandb.log({
        "eval/format_reward": vllm_metrics["format_reward"],
        "eval/answer_reward": vllm_metrics["answer_reward"],
        "eval/reward": vllm_metrics["reward"],
        "eval/accuracy": vllm_metrics["answer_reward"],
        "eval_step": final_eval_step + 1,
    })

    # 仅当本函数自行管理 wandb run 时才 finish
    if use_wandb:
        wandb.finish()

    # 清理 GPU 缓存
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "run_name": run_name,
        "dataset_size": dataset_size,
        "filtered_size": len(updated_train_data),
        "history": history,
        "vllm_metrics": vllm_metrics,
        "gen_metrics": gen_metrics,
        "updated_train_data": updated_train_data,
    }
    

def run_experiment(
    train_data: list,
    val_data: list,
    dataset_size: int | None,
    run_name: str,
    lr: float = 1e-5,
    batch_size: int = 16,
    num_epochs: int = 3,
    gradient_accumulation_steps: int = 8,
    eval_steps: int = 1,
    seed: int = 42,
) -> dict:
    """Run a single SFT experiment."""
    torch.manual_seed(seed)
    random.seed(seed)

    # Prepare dataset
    if dataset_size is not None and dataset_size < len(train_data):
        train_subset = random.sample(train_data, dataset_size)
    else:
        train_subset = train_data
        dataset_size = len(train_data)

    print(f"\n{'='*60}")
    print(f"Running experiment: {run_name}")
    print(f"Dataset size: {dataset_size}")
    print(f"Learning rate: {lr}, Epochs: {num_epochs}, Batch size: {batch_size}")
    print(f"{'='*60}\n")

    # Init wandb
    os.environ.setdefault("WANDB_API_KEY", WANDB_KEY)
    wandb.init(
        project="cs336-sft-math",
        name=run_name,
        config={
            "dataset_size": dataset_size,
            "lr": lr,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "model": MODEL_PATH,
        },
    )

    # Load model and tokenizer on cuda:0
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    ).to("cuda:0")
    model.gradient_checkpointing_enable()

    # Create dataset and dataloader
    train_dataset = SFTDataset(train_subset, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    # Init vLLM on cuda:1
    print("Initializing vLLM on cuda:1...")
    llm = init_vllm(MODEL_PATH, device="cuda:1", seed=seed)

    # Train
    history, final_train_step, final_eval_step = train_sft(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_data=val_data,
        num_epochs=num_epochs,
        lr=lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_steps=eval_steps,
        device="cuda:0",
        run_name=run_name,
        llm=llm,
    )

    # Save model
    save_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    # Final evaluation with vLLM
    print("Running final vLLM evaluation...")
    load_policy_into_vllm_instance(model, llm)
    vllm_metrics = evaluate_with_vllm(llm, val_data, max_eval_samples=500)
    print(f"vLLM evaluation results: {vllm_metrics}")

    wandb.log({
        "eval/format_reward": vllm_metrics["format_reward"],
        "eval/answer_reward": vllm_metrics["answer_reward"],
        "eval/reward": vllm_metrics["reward"],
        "eval/accuracy": vllm_metrics["answer_reward"],  # answer_reward is 0 or 1, so it equals accuracy
        "eval_step": final_eval_step + 1,
    })

    wandb.finish()

    # Clean up
    del model
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "run_name": run_name,
        "dataset_size": dataset_size,
        "history": history,
        "vllm_metrics": vllm_metrics,
    }


def main():
    args=argparse.ArgumentParser()
    args.add_argument("--exp_type", type=str, default="sft", help="Type of experiment to run.")
    args.add_argument("--epochs", type=int, default=3, help="Number of epochs for the sft.")
    args.add_argument("--n_ei_steps", type=int, default=5, help="Number of expert iteration steps.")
    args.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.")
    args.add_argument("--rollout", type=int, default=4, help="Number of rollouts for expert iteration experiment.")
    args.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    args.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training.")
    args.add_argument("--vllm_gpu_util", type=float, default=0.95, help="GPU memory utilization for vLLM (0.0~1.0).")
    args.add_argument("--max_gen_batch_size", type=int, default=512, help="Max batch size for vLLM generation to avoid OOM.")
    args.add_argument("--eval_steps", type=int, default=1, help="Evaluate every N training steps.")
    parsed_args = args.parse_args()
    exp_type=parsed_args.exp_type
    epochs=parsed_args.epochs
    gradient_accumulation_steps=parsed_args.gradient_accumulation_steps
    batch_size=parsed_args.batch_size
    lr=parsed_args.lr
    vllm_gpu_util=parsed_args.vllm_gpu_util
    max_gen_batch_size=parsed_args.max_gen_batch_size
    eval_steps=parsed_args.eval_steps
    
    # 全局性能优化：在 A100 上启用 TF32，充分利用 Tensor Core 加速 matmul，
    # 几乎不损失精度（相对 fp32）且比 fp32 快得多
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Login to wandb
    os.environ.setdefault("WANDB_API_KEY", WANDB_KEY)
    
    if exp_type == "sft":
        print("exp_type: SFT experiment")
        # Load data
        print("Loading datasets...")
        train_data = load_json_or_jsonl(TRAIN_PATH)
        train_filtered_data = load_json_or_jsonl(TRAIN_FILTERED_PATH)
        val_data = load_json_or_jsonl(VAL_PATH)
        print(f"Train: {len(train_data)}, Filtered: {len(train_filtered_data)}, Val: {len(val_data)}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        all_results = []

        # Experiment 1: Vary dataset sizes
        # SFT 实验保持较小 batch、较大 grad accum，确保小数据集也有充足的优化步数
        dataset_sizes = [128, 256, 512, 1024, None]  # None means full
        for size in dataset_sizes:
            run_name = f"sft_size_{size if size else 'full'}"
            result = run_experiment(
                train_data=train_data,
                val_data=val_data,
                dataset_size=size,
                run_name=run_name,
                lr=1e-5,
                batch_size=4,
                num_epochs=3,
                gradient_accumulation_steps=16,
                eval_steps=1,
            )
            all_results.append(result)

        # Experiment 2: Filtered dataset
        result_filtered = run_experiment(
            train_data=train_filtered_data,
            val_data=val_data,
            dataset_size=None,
            run_name="sft_filtered_full",
            lr=1e-5,
            batch_size=4,
            num_epochs=3,
            gradient_accumulation_steps=16,
            eval_steps=1,
        )
        all_results.append(result_filtered)

        # Save summary
        summary_path = os.path.join(OUTPUT_DIR, "sft_experiment_summary.json")
        summary = []
        for r in all_results:
            summary.append({
                "run_name": r["run_name"],
                "dataset_size": r["dataset_size"],
                "vllm_format_reward": r["vllm_metrics"]["format_reward"],
                "vllm_answer_reward": r["vllm_metrics"]["answer_reward"],
                "vllm_reward": r["vllm_metrics"]["reward"],
            })

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n=== All experiments complete ===")
        for s in summary:
            print(f"{s['run_name']}: size={s['dataset_size']}, accuracy={s['vllm_answer_reward']:.4f}")
    elif exp_type == "expert_iteration":
        print("exp_type: Expert Iteration experiment")
        # 加载数据
        print("Loading datasets...")
        train_data = load_json_or_jsonl(TRAIN_PATH)
        val_data = load_json_or_jsonl(VAL_PATH)
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        all_results = []

        for size in [512, 1024, 2048]:  # 使用不同规模的训练数据进行专家迭代实验
            run_name = f"sft_expert_size_{size}"
            current_model_path = MODEL_PATH  # 第一步从预训练模型开始
            current_train_data = train_data
            llm = None  # 延迟初始化 vLLM，直到知道模型路径

            # 为该 size 开启一个统一的 wandb run，所有 EI step 共享
            os.environ.setdefault("WANDB_API_KEY", WANDB_KEY)
            wandb.init(
                project="cs336-sft-math",
                name=run_name,
                config={
                    "dataset_size": size,
                    "lr": lr,
                    "batch_size": batch_size,
                    "num_epochs": epochs,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "rollout": parsed_args.rollout,
                    "n_ei_steps": parsed_args.n_ei_steps,
                    "model": MODEL_PATH,
                },
            )

            for step in range(parsed_args.n_ei_steps):
                print(f"\n=== Expert Iteration Step {step+1}/{parsed_args.n_ei_steps} ===")
                
                # 第一步时初始化 vLLM，后续复用同一实例（通过加载新权重更新）
                # 传入 vllm_gpu_util 参数，在 A100 上为 KV cache 预留更多显存
                if step == 0:
                    llm = init_vllm(
                        current_model_path,
                        device="cuda:1",
                        seed=42,
                        gpu_memory_utilization=vllm_gpu_util,
                    )

                # 执行一步专家迭代：生成 → 过滤 → SFT 训练
                # use_wandb=False 保证不重复创建 wandb run，所有指标写入当前已打开的 run
                ei_result = one_expert_iteration_step(
                    llm=llm,
                    tokenizer=None,
                    train_data=current_train_data,
                    val_data=val_data,
                    device="cuda:0",
                    run_name=f"{run_name}_step_{step+1}",
                    seed=42 + step,  # 每步使用不同种子以增加多样性
                    dataset_size=size,
                    batch_size=batch_size,
                    num_epochs=epochs,
                    lr=lr,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    eval_steps=eval_steps,
                    rollout=parsed_args.rollout,
                    model_path=current_model_path,
                    use_wandb=False,
                    max_gen_batch_size=max_gen_batch_size,
                )
                
                # 收集该步结果
                all_results.append(ei_result)
                
                # 更新训练数据：使用本步生成的正确轨迹作为下一步的初始数据
                current_train_data = ei_result.get("updated_train_data", current_train_data)
                
                # 更新模型路径：下一步从本步保存的模型继续迭代
                current_model_path = os.path.join(OUTPUT_DIR, f"{run_name}_step_{step+1}")
                
                print(f"Step {step+1} complete. Next model path: {current_model_path}")
                print(f"Next train data size: {len(current_train_data)}")

            # 该 size 的所有 EI step 完成后，结束当前 wandb run
            wandb.finish()
            
            # 显式释放 vLLM 实例及其显存池，避免进入下一个 size 循环时 cuda:1 显存未完全回收导致 OOM
            if llm is not None:
                del llm
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[Memory] vLLM on cuda:1 released. "
                      f"Allocated: {torch.cuda.memory_allocated('cuda:1')/1e9:.2f}GB")

        # 保存实验摘要
        summary_path = os.path.join(OUTPUT_DIR, "expert_experiment_summary.json")
        summary = []
        for r in all_results:
            summary.append({
                "run_name": r["run_name"],
                "dataset_size": r["dataset_size"],
                "filtered_size": r.get("filtered_size", r["dataset_size"]),
                "vllm_format_reward": r["vllm_metrics"]["format_reward"],
                "vllm_answer_reward": r["vllm_metrics"]["answer_reward"],
                "vllm_reward": r["vllm_metrics"]["reward"],
                "gen_accuracy": r.get("gen_metrics", {}).get("accuracy", 0.0),
            })

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n=== All experiments complete ===")
        for s in summary:
            print(f"{s['run_name']}: size={s['dataset_size']}, filtered={s['filtered_size']}, "
                  f"accuracy={s['vllm_answer_reward']:.4f}, gen_acc={s['gen_accuracy']:.4f}")

    else:
        print(f"Unknown exp_type: {exp_type}. Please choose 'sft' or 'expert_iteration'.")


if __name__ == "__main__":

    main()
