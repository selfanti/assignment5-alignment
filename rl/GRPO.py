
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import statistics
import torch
import torch.nn.functional as F
from typing import Literal
import os
import sys
import json
import random
import gc
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from sft_helper.tokenize_prompt_and_output import tokenize_prompt_and_output
from sft_helper.utils import compute_entropy

import wandb
import typer

MODEL_PATH = "/root/assignment5-alignment-main/sft_outputs/sft_size_full"
TRAIN_DATA_PATH = "/root/assignment5-alignment-main/data/sft-cs336-assign5-datasets/sft-reason/train.jsonl"
VAL_DATA_PATH = "/root/assignment5-alignment-main/data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"
R1_ZERO_PROMPT_PATH = "/root/assignment5-alignment-main/cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_DIR = "/root/assignment5-alignment-main/grpo_outputs"
WANDB_KEY = "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"

with open(R1_ZERO_PROMPT_PATH, "r") as f:
    R1_ZERO_PROMPT_TEMPLATE = f.read()


def format_prompt(problem: str) -> str:
    return R1_ZERO_PROMPT_TEMPLATE.replace("{question}", problem)


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


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.95):
    vllm_set_random_seed(seed)
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


def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def tokenize_rollouts(prompts, responses, tokenizer):
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]
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
    }


def compute_response_log_probs(model, input_ids, labels, attention_mask, return_token_entropy=False):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs_dist = F.log_softmax(logits, dim=-1)
    gather_labels = labels.clone()
    gather_labels[gather_labels == -100] = 0
    log_probs = log_probs_dist.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy
    return result


def evaluate_with_vllm(llm, val_data, max_eval_samples=1024):
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


def compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, normalize_by_std, ):
    """
    Compute rewards for each group of rollout responses, normalized by the group size. 
    Args:  reward_fn: Callable[[str, str], dict[str, float]] 
        Scores the rollout responses against the ground truths, 
        producing a dict with keys "reward", "format_reward", and "answer_reward".  
    rollout_responses: list[str] Rollouts from the policy. 
        The length of this list is rollout_batch_size = n_prompts_per_rollout_batch * group_size.  
    repeated_ground_truths: list[str] 
        The ground truths for the examples. The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.  
    group_size: int 
        Number of responses per question (group).  
    advantage_eps: float 
        Small constant to avoid division by zero in normalization.  
    normalize_by_std: bool 
        If True, divide by the per-group standard deviation; otherwise subtract only the group mean.  
    Returns:  tuple[torch.Tensor, torch.Tensor, dict[str, float]].  
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout response. 
        raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response. 
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    num_prompts = len(rollout_responses) // group_size
    pairs = zip(rollout_responses, repeated_ground_truths)
    rewards = [reward_fn(response, ground_truth)
               for response, ground_truth in pairs]
    advantages = []
    raw_rewards = []
    mean_format_reward = []
    mean_answer_reward = []
    for i in range(num_prompts):
        start = group_size * i
        end = group_size * (i + 1)
        prompt_rewards = [rewards[j]["reward"] for j in range(start, end)]
        mean_reward = statistics.mean(prompt_rewards)
        mean_format_reward.append(statistics.mean(
            [rewards[j]["format_reward"] for j in range(start, end)]))
        mean_answer_reward.append(statistics.mean(
            [rewards[j]["answer_reward"] for j in range(start, end)]))
        if len(prompt_rewards) > 1:
            std_reward = statistics.stdev(prompt_rewards)
        else:
            std_reward = 0.0
        for j in range(start, end):
            reward_rollout = rewards[j]["reward"]
            raw_rewards.append(reward_rollout)
            if normalize_by_std:
                denom = std_reward+advantage_eps
                advantage = (reward_rollout - mean_reward) / denom
                advantages.append(advantage)
            else:
                advantage = reward_rollout - mean_reward
                advantages.append(advantage)

    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
    metadata = {
        "batch_format_reward": statistics.mean(mean_format_reward) if mean_format_reward else 0.0,
        "batch_answer_reward": statistics.mean(mean_answer_reward) if mean_answer_reward else 0.0,
        "batch_reward_mean": statistics.mean(raw_rewards) if raw_rewards else 0.0,
        "batch_reward_std": statistics.stdev(raw_rewards) if len(raw_rewards) > 1 else 0.0,
        "batch_reward_max": max(raw_rewards) if raw_rewards else 0.0,
        "batch_reward_min": min(raw_rewards) if raw_rewards else 0.0,
    }
    return (advantages_tensor, raw_rewards_tensor, metadata)


def compute_naive_policy_gradient_loss(raw_rewards_or_advantages: torch.Tensor, policy_log_probs: torch.Tensor, ) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either the raw reward or an already-normalized advantage.  
    Args:  
    raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1)
        scalar reward/advantage for each rollout response.  
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length)
        logprobs for each token.  
    Returns:  torch.Tensor Shape (batch_size, sequence_length)
        the per-token policy-gradient loss (to be aggregated across the batch and sequence dimensions in the training loop).
    Implementation tips: 
        Broadcast the raw_rewards_or_advantages over the sequence_length dimension.
    """
    return -raw_rewards_or_advantages*policy_log_probs


def compute_grpo_clip_loss(advantages: torch.Tensor, policy_log_probs: torch.Tensor, old_log_probs: torch.Tensor, cliprange: float, ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:  
    advantages: torch.Tensor Shape (batch_size, 1)
        per-example advantages A.  
    policy_log_probs: torch.Tensor Shape (batch_size, sequence_length)
        per-token log probs from the policy being trained.  
    old_log_probs: torch.Tensor Shape (batch_size, sequence_length)
        per-token log probs from the old policy.  
    cliprange: float 
        Clip parameter ε (e.g. 0.2).  
    Returns:  tuple[torch.Tensor, dict[str, torch.Tensor]]
        loss torch.Tensor of shape (batch_size, sequence_length)
            the per-token clipped loss. 
        metadata 
            dict containing whatever you want to log. We suggest logging whether each token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of the min was lower than the LHS.  
    Implementation tips:  • Broadcast advantages over sequence_length.
    """
    policy_ratio = torch.exp(policy_log_probs - old_log_probs)
    first = advantages * policy_ratio
    clipped_ratio = torch.clip(policy_ratio, 1 - cliprange, 1 + cliprange)
    second = clipped_ratio * advantages
    loss = -torch.min(first, second)
    metadata = {"clipped": second < first}
    return (loss, metadata)


def compute_policy_gradient_loss(policy_log_probs: torch.Tensor, loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], raw_rewards: torch.Tensor | None = None, advantages: torch.Tensor | None = None, old_log_probs: torch.Tensor | None = None, cliprange: float | None = None, ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.  
    Args:  
    policy_log_probs (batch_size, sequence_length) per-token log-probabilities from the policy being trained.  
    loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".  
    raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).  
    advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).  
    old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).  
    cliprange Required for "grpo_clip"; scalar ε used for clipping.  
    Returns:  tuple[torch.Tensor, dict[str, torch.Tensor]].  
        loss (batch_size, sequence_length), per-token loss.  
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).  
    Implementation tips:  
    • Delegate to compute_naive_policy_gradient_loss or compute_grpo_clip_loss. 
    • Perform argument checks (see assertion pattern above). 
    • Aggregate any returned metadata into a single dict.
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(
            raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange)
    return (loss, metadata)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None, ) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.  
    Args:  
    tensor: torch.Tensor The data to be averaged.  
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.  
    dim: int | None Dimension over which to average. If None, compute the mean over all masked elements.  
    Returns:  torch.Tensor 
    The masked mean; shape matches tensor.mean(dim) semantics.
    """
    if dim is not None:
        return (tensor * mask).sum(dim) / mask.sum(dim)
    else:
        return (tensor * mask).sum() / mask.sum()


def grpo_microbatch_train_step( policy_log_probs: torch.Tensor, response_mask: torch.Tensor, gradient_accumulation_steps: int, loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], raw_rewards: torch.Tensor | None = None, advantages: torch.Tensor | None = None, old_log_probs: torch.Tensor | None = None, cliprange: float | None = None, ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  
    """
    Execute a forward-and-backward pass on a microbatch.  
    Args:  
    policy_log_probs (batch_size, sequence_length), 
        per-token log-probabilities from the policy being trained.  
    response_mask (batch_size, sequence_length), 
        1 for response tokens, 0 for prompt/padding.  
    gradient_accumulation_steps 
        Number of microbatches per optimizer step.  
    loss_type 
        One of "no_baseline", "reinforce_with_baseline", "grpo_clip".  
    raw_rewards 
        Needed when loss_type == "no_baseline"; shape (batch_size, 1).  
    advantages 
        Needed when loss_type != "no_baseline"; shape (batch_size, 1).  
    old_log_probs 
        Required for GRPO-Clip; shape (batch_size, sequence_length).  
    cliprange 
        Clip parameter ε for GRPO-Clip.  
    Returns:  tuple[torch.Tensor, dict[str, torch.Tensor]].  
    loss scalar tensor. 
        The microbatch loss, adjusted for gradient accumulation. We return this so we can log it. 
    metadata 
        Dict with metadata from the underlying loss call, and any other statistics you might want to log.  
    Implementation tips:  
    • You should call loss.backward() in this function. Make sure to adjust for gradient accumulation.
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError("incorrect loss type")
    loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    return loss, metadata


def main(
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type: str = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    cliprange: float = 0.2,
    eval_steps: int = 10,
    num_eval_samples: int = 1024,
    seed: int = 42,
    model_path: str = MODEL_PATH,
    train_data_path: str = TRAIN_DATA_PATH,
    val_data_path: str = VAL_DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    run_name: str = "grpo",
):
    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
    assert rollout_batch_size % micro_train_batch_size == 0, "rollout_batch_size must be divisible by micro_train_batch_size"
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    valid_loss_types = {"no_baseline", "reinforce_with_baseline", "grpo_clip"}
    assert loss_type in valid_loss_types, f"loss_type must be one of {valid_loss_types}"
    if loss_type == "grpo_clip":
        assert epochs_per_rollout_batch > 1, "GRPO-Clip should only be used when off-policy (epochs_per_rollout_batch > 1)"

    torch.manual_seed(seed)
    random.seed(seed)

    device = "cuda:0"
    vllm_device = "cuda:1"

    os.makedirs(output_dir, exist_ok=True)

    # Init wandb
    os.environ.setdefault("WANDB_API_KEY", WANDB_KEY)
    wandb.init(
        project="cs336-grpo-math",
        name=run_name,
        config={
            "n_grpo_steps": n_grpo_steps,
            "learning_rate": learning_rate,
            "rollout_batch_size": rollout_batch_size,
            "group_size": group_size,
            "epochs_per_rollout_batch": epochs_per_rollout_batch,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "loss_type": loss_type,
            "use_std_normalization": use_std_normalization,
            "cliprange": cliprange,
            "model_path": model_path,
        },
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    ).to(device)
    model.gradient_checkpointing_enable()

    # Init vLLM
    print(f"Initializing vLLM on {vllm_device}...")
    llm = init_vllm(model_path, device=vllm_device, seed=seed, gpu_memory_utilization=gpu_memory_utilization)

    # Load data
    print("Loading datasets...")
    train_data = load_json_or_jsonl(train_data_path)
    val_data = load_json_or_jsonl(val_data_path)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    global_step = 0
    best_val_reward = -float("inf")

    for step in range(n_grpo_steps):
        model.eval()

        # Sample prompts
        prompt_indices = random.sample(range(len(train_data)), n_prompts_per_rollout_batch)
        batch_prompts = []
        batch_ground_truths = []
        for idx in prompt_indices:
            ex = train_data[idx]
            batch_prompts.append(format_prompt(ex["problem"]))
            batch_ground_truths.append(ex.get("expected_answer", ex.get("answer", "")))

        # Repeat prompts and ground truths group_size times
        repeated_prompts = []
        repeated_ground_truths = []
        for i in range(n_prompts_per_rollout_batch):
            for _ in range(group_size):
                repeated_prompts.append(batch_prompts[i])
                repeated_ground_truths.append(batch_ground_truths[i])

        # Generate rollouts
        print(f"\n[Step {step+1}/{n_grpo_steps}] Generating {len(repeated_prompts)} rollouts...")
        vllm_outputs = llm.generate(repeated_prompts, sampling_params)
        rollout_responses = [output.outputs[0].text for output in vllm_outputs]

        # Compute rewards and advantages
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        # Tokenize rollouts
        print(f"Tokenizing rollouts...")
        tokenized = tokenize_rollouts(repeated_prompts, rollout_responses, tokenizer)

        # Compute old log probs if off-policy
        old_log_probs = None
        if epochs_per_rollout_batch > 1 and loss_type == "grpo_clip":
            print("Computing old log-probs...")
            with torch.no_grad():
                all_log_probs = []
                for start in range(0, rollout_batch_size, micro_train_batch_size):
                    end = min(start + micro_train_batch_size, rollout_batch_size)
                    mb_input_ids = tokenized["input_ids"][start:end].to(device)
                    mb_labels = tokenized["labels"][start:end].to(device)
                    mb_attention_mask = tokenized["attention_mask"][start:end].to(device)
                    result = compute_response_log_probs(model, mb_input_ids, mb_labels, mb_attention_mask)
                    all_log_probs.append(result["log_probs"].cpu())
                old_log_probs = torch.cat(all_log_probs, dim=0)
            old_log_probs = old_log_probs.detach()
            print(f"Old log-probs computed, shape: {old_log_probs.shape}")

        # Training epochs on this rollout batch
        for epoch in range(epochs_per_rollout_batch):
            indices = torch.randperm(rollout_batch_size)

            epoch_loss = 0.0
            epoch_clip_fraction = 0.0
            epoch_entropy = 0.0
            num_microbatches_in_step = 0

            model.train()

            for mb_idx in range(n_microbatches_per_rollout_batch):
                start = mb_idx * micro_train_batch_size
                end = min(start + micro_train_batch_size, rollout_batch_size)
                mb_indices = indices[start:end]

                mb_input_ids = tokenized["input_ids"][mb_indices].to(device)
                mb_labels = tokenized["labels"][mb_indices].to(device)
                mb_response_mask = tokenized["response_mask"][mb_indices].to(device)
                mb_attention_mask = tokenized["attention_mask"][mb_indices].to(device)

                if loss_type == "no_baseline":
                    mb_raw_rewards = raw_rewards[mb_indices].to(device).unsqueeze(1)
                    mb_advantages = None
                    mb_old_log_probs = None
                else:
                    mb_advantages = advantages[mb_indices].to(device).unsqueeze(1)
                    mb_raw_rewards = None
                    if loss_type == "grpo_clip":
                        mb_old_log_probs = old_log_probs[mb_indices].to(device)
                    else:
                        mb_old_log_probs = None

                # Forward pass
                result = compute_response_log_probs(
                    model, mb_input_ids, mb_labels, mb_attention_mask, return_token_entropy=True
                )
                policy_log_probs = result["log_probs"]
                token_entropy = result["token_entropy"]

                # Compute loss and backprop
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=cliprange if loss_type == "grpo_clip" else None,
                )

                epoch_loss += loss.item() * gradient_accumulation_steps
                num_microbatches_in_step += 1

                valid_entropy = masked_mean(token_entropy, mb_response_mask)
                epoch_entropy += valid_entropy.item()

                if loss_type == "grpo_clip" and "clipped" in metadata:
                    clip_frac = masked_mean(metadata["clipped"].float(), mb_response_mask).item()
                    epoch_clip_fraction += clip_frac

                # Optimizer step after accumulation
                if (mb_idx + 1) % gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    log_dict = {
                        "train/loss": epoch_loss / num_microbatches_in_step,
                        "train/grad_norm": grad_norm.item(),
                        "train/token_entropy": epoch_entropy / num_microbatches_in_step,
                        "train/reward_mean": reward_metadata["batch_reward_mean"],
                        "train/format_reward": reward_metadata["batch_format_reward"],
                        "train/answer_reward": reward_metadata["batch_answer_reward"],
                        "train/reward_std": reward_metadata["batch_reward_std"],
                        "train_step": global_step,
                        "grpo_step": step + 1,
                        "epoch": epoch,
                    }

                    if loss_type == "grpo_clip":
                        log_dict["train/clip_fraction"] = epoch_clip_fraction / num_microbatches_in_step

                    wandb.log(log_dict)

                    epoch_loss = 0.0
                    epoch_clip_fraction = 0.0
                    epoch_entropy = 0.0
                    num_microbatches_in_step = 0

                    # Validation
                    if global_step % eval_steps == 0:
                        model.eval()
                        load_policy_into_vllm_instance(model, llm)
                        val_metrics = evaluate_with_vllm(llm, val_data, max_eval_samples=num_eval_samples)

                        wandb.log({
                            "eval/format_reward": val_metrics["format_reward"],
                            "eval/answer_reward": val_metrics["answer_reward"],
                            "eval/reward": val_metrics["reward"],
                            "eval_step": global_step,
                        })

                        print(f"\n[Step {global_step}] Validation: "
                              f"reward={val_metrics['reward']:.4f}, "
                              f"format={val_metrics['format_reward']:.4f}, "
                              f"answer={val_metrics['answer_reward']:.4f}")

                        if val_metrics["reward"] > best_val_reward:
                            best_val_reward = val_metrics["reward"]
                            best_save_dir = os.path.join(output_dir, f"{run_name}_best")
                            os.makedirs(best_save_dir, exist_ok=True)
                            model.save_pretrained(best_save_dir)
                            tokenizer.save_pretrained(best_save_dir)
                            print(f"Saved best model to {best_save_dir}")

                        print("\nExample rollouts:")
                        for i in range(min(3, len(rollout_responses))):
                            print(f"--- Example {i+1} ---")
                            print(f"Prompt: {repeated_prompts[i][:200]}...")
                            print(f"Response: {rollout_responses[i][:500]}...")
                            print(f"Reward: {raw_rewards[i].item():.2f}, Advantage: {advantages[i].item():.4f}")

                        model.train()
                        torch.cuda.empty_cache()

        # Load updated policy into vLLM for next rollout
        if step < n_grpo_steps - 1:
            load_policy_into_vllm_instance(model, llm)

    # Final evaluation
    model.eval()
    load_policy_into_vllm_instance(model, llm)
    final_metrics = evaluate_with_vllm(llm, val_data, max_eval_samples=num_eval_samples)
    print(f"\nFinal validation: {final_metrics}")
    wandb.log({
        "eval/format_reward": final_metrics["format_reward"],
        "eval/answer_reward": final_metrics["answer_reward"],
        "eval/reward": final_metrics["reward"],
        "eval_step": global_step + 1,
    })

    final_save_dir = os.path.join(output_dir, f"{run_name}_final")
    os.makedirs(final_save_dir, exist_ok=True)
    model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"Saved final model to {final_save_dir}")

    wandb.finish()
    print("GRPO training complete!")


if __name__ == "__main__":
    typer.run(main)
