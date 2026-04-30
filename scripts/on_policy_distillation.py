#!/usr/bin/env python3
"""
On-policy Token-level Distillation.

Distills a large teacher model (Qwen2.5-14B) into a small student model
(Qwen2.5-1.5B SFT) using on-policy generated data and token-level KL divergence.

GPU layout (2x A100 40G):
    - cuda:0 : student model training + student logits computation
    - cuda:1 : VLLM (student on-policy generation) + teacher model inference

The teacher and VLLM coexist on cuda:1. To avoid OOM:
    - VLLM is initialized with a modest gpu_memory_utilization (default 0.20)
    - Teacher forward passes are chunked into micro-batches.
"""

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from scripts.sft_experiment import (
    R1_ZERO_PROMPT_TEMPLATE,
    evaluate_with_vllm,
    format_prompt,
    init_vllm,
    load_json_or_jsonl,
    load_policy_into_vllm_instance,
)
from sft_helper.tokenize_prompt_and_output import tokenize_prompt_and_output

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STUDENT_PATH = "/root/assignment5-alignment-main/sft_outputs/sft_size_full"
TEACHER_PATH = "/root/assignment5-alignment-main/qwen2.5_14B"
TRAIN_PATH = "/root/assignment5-alignment-main/data/sft-cs336-assign5-datasets/sft-reason/train.jsonl"
VAL_PATH = "/root/assignment5-alignment-main/data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"
OUTPUT_DIR = "/root/assignment5-alignment-main/sft_outputs/on_policy_distill"

WANDB_KEY = "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"


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def load_tokenizer_safe(path: str):
    """Load tokenizer, falling back to fast-only if slow tokenizer files are missing."""
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

class OnPolicyDataset(Dataset):
    """Dataset of (prompt, response) pairs generated on-policy."""

    def __init__(self, examples: list[dict], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer):
    """
    Collate a batch of {"prompt": str, "response": str} examples.
    Uses tokenize_prompt_and_output then converts right-pad to left-pad.
    """
    prompts = [ex["prompt"] for ex in batch]
    responses = [ex["response"] for ex in batch]
    ground_truths = [ex.get("ground_truth", "") for ex in batch]

    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)

    input_ids = tokenized["input_ids"]          # [B, L]  (right-padded)
    labels = tokenized["labels"]                # [B, L]
    response_mask = tokenized["response_mask"]  # [B, L]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    batch_size, seq_len = input_ids.shape
    lengths = (input_ids != pad_token_id).sum(dim=1)
    max_len = seq_len

    new_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    new_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    new_response_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i in range(batch_size):
        actual_len = lengths[i].item()
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
# On-policy generation with VLLM
# ---------------------------------------------------------------------------

def generate_on_policy_data(
    llm: LLM,
    train_data: list[dict],
    tokenizer,
    num_prompts: int | None,
    rollout: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_gen_batch_size: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """
    Generate on-policy rollouts using the student model via VLLM.

    Returns
    -------
    examples : list[dict]
        List of {"prompt": str, "response": str, "ground_truth": str}.
    metrics : dict
        Generation statistics.
    """
    if num_prompts is not None and num_prompts < len(train_data):
        sampled = random.sample(train_data, num_prompts)
    else:
        sampled = train_data

    prompts = []
    ground_truths = []
    for ex in sampled:
        prompt = format_prompt(ex["problem"])
        gt = ex.get("expected_answer", "")
        for _ in range(rollout):
            prompts.append(prompt)
            ground_truths.append(gt)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    print(f"Generating {len(prompts)} on-policy responses ({rollout} rollouts x {len(sampled)} prompts)...")
    all_outputs = []
    for start in range(0, len(prompts), max_gen_batch_size):
        end = min(start + max_gen_batch_size, len(prompts))
        batch_outputs = llm.generate(prompts[start:end], sampling_params)
        all_outputs.extend(batch_outputs)
        print(f"  Batch {start//max_gen_batch_size + 1}/{(len(prompts)-1)//max_gen_batch_size + 1} ({start}-{end})")

    examples = []
    format_correct = 0
    answer_correct = 0
    answer_reward=0.0
    for i, output in enumerate(all_outputs):
        text = output.outputs[0].text
        gt = ground_truths[i]
        reward = r1_zero_reward_fn(text, gt)
        if reward["format_reward"] > 0:
            format_correct += 1
        if reward["answer_reward"] > 0:
            answer_correct += 1
        if reward["format_reward"] > 0 and reward["answer_reward"] > 0:
            answer_reward += 1
        examples.append({
            "prompt": prompts[i],
            "response": text,
            "ground_truth": gt,
        })

    metrics = {
        "total": len(examples),
        "format_correct": format_correct,
        "answer_correct": answer_correct,
        "format_accuracy": format_correct / len(examples) if examples else 0.0,
        "answer_accuracy": answer_correct / len(examples) if examples else 0.0,
        "reward": answer_reward / len(examples) if examples else 0.0,
    }
    print(f"Generation done: {format_correct}/{len(examples)} format-correct, {answer_correct}/{len(examples)} answer-correct,reward: {answer_reward}/{len(examples)}={float(answer_reward/len(examples)):.4f}")
    return examples, metrics


# ---------------------------------------------------------------------------
# Teacher logits (micro-batched to fit cuda:1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_teacher_logits(
    teacher_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    microbatch_size: int,
    device: str = "cuda:1",
) -> torch.Tensor:
    """
    Compute teacher logits in micro-batches to avoid OOM on cuda:1.

    Returns
    -------
    teacher_logits : torch.Tensor
        Tensor of shape [B, L, vocab_size] on the *same device as input_ids*.
    """
    teacher_model.eval()
    B, L = input_ids.shape
    all_logits = []

    for start in range(0, B, microbatch_size):
        end = min(start + microbatch_size, B)
        mb_input_ids = input_ids[start:end].to(device)
        mb_attention_mask = attention_mask[start:end].to(device)
        outputs = teacher_model(input_ids=mb_input_ids, attention_mask=mb_attention_mask)
        all_logits.append(outputs.logits.cpu())  # move back to CPU to free cuda:1 mem
        del outputs
        torch.cuda.empty_cache()

    teacher_logits = torch.cat(all_logits, dim=0).to(input_ids.device)
    return teacher_logits


# ---------------------------------------------------------------------------
# Vocabulary compatibility check
# ---------------------------------------------------------------------------

def check_vocabulary_compatibility(student_tok, teacher_tok) -> dict:
    """
    Compare the vocabularies of student and teacher tokenizers.
    Returns a dict with statistics and any detected mismatches.
    """
    student_vocab = student_tok.get_vocab()
    teacher_vocab = teacher_tok.get_vocab()

    result = {
        "student_vocab_size": len(student_vocab),
        "teacher_vocab_size": len(teacher_vocab),
        "identical": student_vocab == teacher_vocab,
        "student_only_tokens": [],
        "teacher_only_tokens": [],
        "mismatched_ids": [],
    }

    if not result["identical"]:
        all_tokens = set(student_vocab.keys()) | set(teacher_vocab.keys())
        for token in all_tokens:
            s_id = student_vocab.get(token)
            t_id = teacher_vocab.get(token)
            if s_id is None:
                result["teacher_only_tokens"].append((token, t_id))
            elif t_id is None:
                result["student_only_tokens"].append((token, s_id))
            elif s_id != t_id:
                result["mismatched_ids"].append((token, s_id, t_id))

    return result


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def token_level_KL(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float = 2.0,
    mode: str = "forward",
) -> tuple[torch.Tensor, dict]:
    """
    Token-level KL distillation loss.

    Parameters
    ----------
    mode : str
        "forward"  -> KL(teacher || student)  (mode-seeking, standard KD)
        "reverse"  -> KL(student || teacher)  (mean-seeking, covers all modes)
    """
    # Align vocab sizes (student may have smaller vocab than teacher config)
    vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
    student_logits = student_logits[:, :, :vocab_size]
    teacher_logits = teacher_logits[:, :, :vocab_size]

    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    mask = response_mask.float()

    if mode == "forward":
        # Forward KL = sum_t p_t * (log p_t - log p_s)
        # Gradient equivalent to: -sum_t p_t * log p_s
        t_probs = t_log_probs.exp()
        token_loss = -(t_probs * s_log_probs).sum(dim=-1)  # [B, L]
    else:  # reverse
        # Reverse KL = sum_t p_s * (log p_s - log p_t)
        # Gradient equivalent to: -sum_t p_s * log p_t
        s_probs = s_log_probs.exp()
        token_loss = -(s_probs * t_log_probs).sum(dim=-1)  # [B, L]

    loss = (token_loss * mask).sum() / mask.sum().clamp(min=1)

    # Logging: raw KL (without temperature) for monitoring
    with torch.no_grad():
        s_log_probs_raw = F.log_softmax(student_logits, dim=-1)
        t_log_probs_raw = F.log_softmax(teacher_logits, dim=-1)
        s_probs_raw = s_log_probs_raw.exp()
        t_probs_raw = t_log_probs_raw.exp()

        if mode == "forward":
            kl_raw = (t_probs_raw * (t_log_probs_raw - s_log_probs_raw)).sum(-1)
        else:
            kl_raw = (s_probs_raw * (s_log_probs_raw - t_log_probs_raw)).sum(-1)

        avg_kl = (kl_raw * mask).sum() / mask.sum().clamp(min=1)

    return loss, {"avg_kl": avg_kl.item(), "kl_mode": mode}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_distill(
    student_model,
    teacher_model,
    tokenizer,
    train_data: list[dict],
    val_data: list[dict],
    num_epochs: int,
    lr: float,
    gradient_accumulation_steps: int,
    eval_steps: int,
    gen_every_steps: int,
    distill_temperature: float,
    teacher_microbatch: int,
    kl_mode: str,
    device: str,
    teacher_device: str,
    run_name: str,
    llm: LLM,
    num_prompts_per_gen: int | None,
    rollout: int,
    max_tokens: int,
    gen_temperature: float,
    gen_top_p: float,
    max_gen_batch_size: int,
    batch_size: int,
    seed: int,
    global_step: int = 0,
):
    """Run on-policy distillation training."""
    optimizer = AdamW(student_model.parameters(), lr=lr)
    student_model.train()

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    eval_step = 0
    history = {"train_loss": [], "eval_vllm": []}

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_loss = 0.0
        num_batches = 0

        # Outer while-loop allows mid-epoch regeneration of on-policy data
        while True:
            print(f"Generating on-policy data at step {global_step}...")
            student_model.eval()
            load_policy_into_vllm_instance(student_model, llm)
            on_policy_examples, gen_metrics = generate_on_policy_data(
                llm=llm,
                train_data=train_data,
                tokenizer=tokenizer,
                num_prompts=num_prompts_per_gen,
                rollout=rollout,
                max_tokens=max_tokens,
                temperature=gen_temperature,
                top_p=gen_top_p,
                max_gen_batch_size=max_gen_batch_size,
                seed=seed + epoch + global_step,
            )
            student_model.train()
            torch.cuda.empty_cache()

            wandb.log({
                "gen/format_accuracy": gen_metrics["format_accuracy"],
                "gen/answer_accuracy": gen_metrics["answer_accuracy"],
                "gen/total": gen_metrics["total"],
                "train_step": global_step,
            })

            dataset = OnPolicyDataset(on_policy_examples, tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=lambda batch: collate_fn(batch, tokenizer),
            )

            should_regenerate = False
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                response_mask = batch["response_mask"].to(device)

                # Student forward
                outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = outputs.logits

                # Teacher forward (micro-batched on teacher_device)
                teacher_logits = compute_teacher_logits(
                    teacher_model=teacher_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    microbatch_size=teacher_microbatch,
                    device=teacher_device,
                )

                # Distillation loss
                loss, logging_info = token_level_KL(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    response_mask=response_mask,
                    temperature=distill_temperature,
                    mode=kl_mode,
                )

                # Scale for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    wandb.log({
                        "train/distill_loss": loss.item() * gradient_accumulation_steps,
                        "train/avg_kl_div": logging_info["avg_kl"],
                        "train/epoch": epoch + (batch_idx + 1) / len(dataloader),
                        "train_step": global_step,
                    })

                    # In-the-loop evaluation
                    if global_step % eval_steps == 0:
                        student_model.eval()
                        load_policy_into_vllm_instance(student_model, llm)
                        eval_result = evaluate_with_vllm(llm, val_data, max_eval_samples=500)
                        eval_step += 1
                        history["eval_vllm"].append((global_step, eval_result))

                        wandb.log({
                            "eval/format_reward": eval_result["format_reward"],
                            "eval/answer_reward": eval_result["answer_reward"],
                            "eval/reward": eval_result["reward"],
                            "eval_step": eval_step,
                        })

                        student_model.train()
                        torch.cuda.empty_cache()

                    # Trigger mid-epoch regeneration if scheduled
                    if global_step % gen_every_steps == 0 and global_step > 0:
                        should_regenerate = True
                        break

            if not should_regenerate:
                # Natural end of epoch
                break

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}, avg distill loss: {avg_epoch_loss:.4f}")
        wandb.log({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch": epoch + 1,
            "train_step": global_step,
        })

    return history, global_step, eval_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="On-policy token-level distillation")
    parser.add_argument("--student_path", type=str, default=STUDENT_PATH)
    parser.add_argument("--teacher_path", type=str, default=TEACHER_PATH)
    parser.add_argument("--train_path", type=str, default=TRAIN_PATH)
    parser.add_argument("--val_path", type=str, default=VAL_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--run_name", type=str, default="on_policy_distill")

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--distill_temperature", type=float, default=2.0,
                        help="Temperature for softening teacher/student distributions.")
    parser.add_argument("--kl_mode", type=str, default="forward", choices=["forward", "reverse"],
                        help="KL divergence mode: forward (KL(teacher||student)) or reverse (KL(student||teacher)).")
    parser.add_argument("--teacher_microbatch", type=int, default=2,
                        help="Micro-batch size for teacher forward passes on cuda:1.")

    # On-policy generation hyperparameters
    parser.add_argument("--gen_every_steps", type=int, default=200,
                        help="Regenerate on-policy data every N training steps.")
    parser.add_argument("--num_prompts_per_gen", type=int, default=1024,
                        help="Number of prompts to sample per generation round. None = all.")
    parser.add_argument("--rollout", type=int, default=1,
                        help="Number of responses to generate per prompt.")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--gen_temperature", type=float, default=0.7)
    parser.add_argument("--gen_top_p", type=float, default=0.95)
    parser.add_argument("--max_gen_batch_size", type=int, default=128,
                        help="Max prompts per VLLM inference batch.")

    # VLLM / device
    parser.add_argument("--vllm_gpu_util", type=float, default=0.20,
                        help="GPU memory utilization for VLLM on cuda:1 (0.0~1.0).")
    parser.add_argument("--student_device", type=str, default="cuda:0")
    parser.add_argument("--teacher_device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Performance optimizations for A100
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load data
    print("Loading datasets...")
    train_data = load_json_or_jsonl(args.train_path)
    val_data = load_json_or_jsonl(args.val_path)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Wandb
    os.environ.setdefault("WANDB_API_KEY", WANDB_KEY)
    wandb.init(
        project="cs336-distill-math",
        name=args.run_name,
        config={
            "student": args.student_path,
            "teacher": args.teacher_path,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "distill_temperature": args.distill_temperature,
            "gen_every_steps": args.gen_every_steps,
            "rollout": args.rollout,
            "max_tokens": args.max_tokens,
        },
    )

    # Tokenizer (use student's tokenizer for both)
    print("Loading tokenizers...")
    student_tokenizer = load_tokenizer_safe(args.student_path)
    teacher_tokenizer = load_tokenizer_safe(args.teacher_path)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # Vocabulary compatibility check
    print("Checking vocabulary compatibility...")
    vocab_check = check_vocabulary_compatibility(student_tokenizer, teacher_tokenizer)
    print(f"  Student vocab size: {vocab_check['student_vocab_size']}")
    print(f"  Teacher vocab size: {vocab_check['teacher_vocab_size']}")
    print(f"  Vocab identical: {vocab_check['identical']}")
    if not vocab_check["identical"]:
        print(f"  !! Warning: vocabulary mismatch detected !!")
        print(f"     Student-only tokens: {len(vocab_check['student_only_tokens'])}")
        print(f"     Teacher-only tokens: {len(vocab_check['teacher_only_tokens'])}")
        print(f"     Mismatched IDs: {len(vocab_check['mismatched_ids'])}")
    else:
        print("  -> Tokenizers are fully compatible; using student tokenizer for distillation.")

    tokenizer = student_tokenizer  # unify on student tokenizer

    # Load student model on cuda:0
    print(f"Loading student model on {args.student_device}...")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(args.student_device)
    student_model.gradient_checkpointing_enable()
    print(f"[Memory] Student loaded. Allocated on {args.student_device}: "
          f"{torch.cuda.memory_allocated(args.student_device)/1e9:.2f}GB")

    # Load teacher model on cuda:1
    print(f"Loading teacher model on {args.teacher_device}...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map={"": args.teacher_device},
    )
    print(f"[Memory] Teacher loaded. Allocated on {args.teacher_device}: "
          f"{torch.cuda.memory_allocated(args.teacher_device)/1e9:.2f}GB")

    # Init VLLM on cuda:1 (student model for on-policy generation)
    print(f"Initializing VLLM on {args.teacher_device} (gpu_util={args.vllm_gpu_util})...")
    llm = init_vllm(
        model_id=args.student_path,
        device=args.teacher_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_util,
    )
    print(f"[Memory] VLLM initialized. Allocated on {args.teacher_device}: "
          f"{torch.cuda.memory_allocated(args.teacher_device)/1e9:.2f}GB")

    # Train
    print("\nStarting on-policy distillation...")
    history, final_train_step, final_eval_step = train_distill(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        num_epochs=args.num_epochs,
        lr=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        gen_every_steps=args.gen_every_steps,
        distill_temperature=args.distill_temperature,
        teacher_microbatch=args.teacher_microbatch,
        kl_mode=args.kl_mode,
        device=args.student_device,
        teacher_device=args.teacher_device,
        run_name=args.run_name,
        llm=llm,
        num_prompts_per_gen=args.num_prompts_per_gen,
        rollout=args.rollout,
        max_tokens=args.max_tokens,
        gen_temperature=args.gen_temperature,
        gen_top_p=args.gen_top_p,
        max_gen_batch_size=args.max_gen_batch_size,
        batch_size=args.batch_size,
        seed=args.seed,
        global_step=0,
    )

    # Save final model
    save_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    student_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModel saved to {save_dir}")

    # Final evaluation
    print("Running final vLLM evaluation...")
    student_model.eval()
    load_policy_into_vllm_instance(student_model, llm)
    final_eval = evaluate_with_vllm(llm, val_data, max_eval_samples=500)
    print(f"Final evaluation: {final_eval}")

    wandb.log({
        "eval/format_reward": final_eval["format_reward"],
        "eval/answer_reward": final_eval["answer_reward"],
        "eval/reward": final_eval["reward"],
        "eval_step": final_eval_step + 1,
    })
    wandb.finish()

    # Clean up
    del student_model
    del teacher_model
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("\n=== On-policy distillation complete ===")


if __name__ == "__main__":
    main()
