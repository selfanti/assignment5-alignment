
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import statistics
import torch
from typing import Literal
import os
MODEL_PATH=""
TRAIN_DATA_PATH=""
VAL_DATA_PATH=""
R1_ZERO_PROMPT_PATH = "/root/assignment5-alignment-main/cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_DIR = "/root/assignment5-alignment-main/grpo_outputs"
WANDB_KEY = "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"


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
    if loss_type=="no_baseline":
        assert raw_rewards is not None
        loss=compute_naive_policy_gradient_loss(raw_rewards,policy_log_probs)
    elif loss_type=="reinforce_with_baseline":
        assert advantages is not None
        loss,metatada=compute_policy_gradient_loss(policy_log_probs,"reinforce_with_baseline",raw_rewards,advantages,old_log_probs,cliprange)
    elif loss_type=="grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        loss,metatada=compute_grpo_clip_loss(advantages,policy_log_probs,old_log_probs,cliprange)
    else:
        raise ValueError("incrroct loss type")
    loss=masked_mean(loss,response_mask)/gradient_accumulation_steps
    loss.backward()
    return loss,metatada

    
def main(n_grpo_steps: int = 200,
         learning_rate: float = 1e-5,
         advantage_eps: float = 1e-6,
         rollout_batch_size: int = 256 ,
         group_size: int = 8,
         sampling_temperature: float = 1.0,
         sampling_min_tokens: int = 4, # As in Expiter, disallow empty string responses 
         sampling_max_tokens: int = 1024,
         epochs_per_rollout_batch: int = 1, # On-policy 
         train_batch_size: int = 256, # On-policy 
         gradient_accumulation_steps: int = 128, # microbatch size is 2, will fit on H100 
         gpu_memory_utilization: float = 0.85, 
         loss_type: Literal[ "no_baseline", "reinforce_with_baseline", "grpo_clip", ] = "reinforce_with_baseline", 
         use_std_normalization: bool = True, 
         ):
         
    assert train_batch_size % gradient_accumulation_steps == 0, ( "train_batch_size must be divisible by gradient_accumulation_steps" )  
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps 
    assert rollout_batch_size % group_size == 0, ( "rollout_batch_size must be divisible by group_size" )  
    n_prompts_per_rollout_batch = rollout_batch_size // group_size 
    assert train_batch_size >= group_size, ( "train_batch_size must be greater than or equal to group_size" )  
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    
    model=torch.load()
        

if __name__ == "__main__":
    main() 
    print("finished GRPO.py")
