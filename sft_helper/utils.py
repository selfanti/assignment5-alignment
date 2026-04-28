import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of a probability distribution given its logits.

    Parameters
    ----------
    logits : torch.Tensor
        A tensor of shape (batch_size, sequence_length, vocab_size) containing the logits
        for each token in the vocabulary at each position in the sequence.

    Returns
    -------
    torch.Tensor
        A tensor of shape (batch_size, sequence_length) containing the entropy of the
        probability distribution at each position in the sequence.
    """
    # Compute probabilities from logits using softmax
    probabilities = F.softmax(logits, dim=-1)

    # Compute log probabilities
    log_probabilities = F.log_softmax(logits, dim=-1)

    # Compute entropy using the formula: H(p) = -sum(p * log(p))
    # Handle 0 * log(0) = nan by treating it as 0 (standard limit convention)
    plogp = probabilities * log_probabilities
    plogp = torch.where(torch.isnan(plogp), torch.zeros_like(plogp), plogp)
    entropy = -torch.sum(plogp, dim=-1)

    return entropy
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get the per-token conditional log-probabilities from a causal language model,
    and optionally the entropy of the model's next-token distribution.

    Parameters
    ----------
    model : PreTrainedModel
        The language model to use for computing logits.
    input_ids : torch.Tensor
        A tensor of shape (batch_size, sequence_length) containing the token IDs for the input sequences.
    labels : torch.Tensor
        A tensor of shape (batch_size, sequence_length) containing the token IDs for the output sequences (labels).
    return_token_entropy : bool, optional
        Whether to return the entropy of the token distributions, by default False.

    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing:
        - "log_probs": A tensor of shape (batch_size, sequence_length) containing the conditional log-probabilities.
        - "token_entropy" (optional): A tensor of shape (batch_size, sequence_length) containing the entropy of the token distributions at each position.
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    # Compute log probabilities from logits
    log_probs_dist = F.log_softmax(logits, dim=-1)

    # Gather log probabilities for the label tokens at each position.
    # Replace -100 with a valid index (0) to avoid out-of-bounds during gather.
    gather_labels = labels.clone()
    gather_labels[gather_labels == -100] = 0
    response_log_probs = log_probs_dist.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": response_log_probs}

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy

    return result
def masked_normalize( tensor: torch.Tensor, mask: torch.Tensor, normalize_constant: float, dim: int | None = None, ) -> torch.Tensor:
    """
    Normalize a tensor along a specified dimension, while ignoring masked values.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be normalized.
    mask : torch.Tensor
        A boolean tensor of the same shape as `tensor` indicating which values to ignore (True for values to ignore).
    normalize_constant : float
        A constant value to use for normalization instead of the actual count of unmasked elements.
    dim : int | None, optional
        The dimension along which to normalize, by default None (flatten the tensor first).
    """
    # Set masked values to zero
    masked_tensor = tensor.masked_fill(mask==0, 0.0)

    # Compute the sum of unmasked values along the specified dimension
    sum_unmasked = masked_tensor.sum(dim=dim, keepdim=False)

    # Normalize by the provided constant instead of the actual count of unmasked elements
    normalized_tensor = sum_unmasked / normalize_constant

    return normalized_tensor
def sft_microbatch_train_step( policy_log_probs: torch.Tensor, response_mask: torch.Tensor, gradient_accumulation_steps: int, normalize_constant: float = 1.0, ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the loss for a microbatch during supervised fine-tuning, and return the loss along with logging information.

    Parameters
    ----------
    policy_log_probs : torch.Tensor
        A tensor of shape (batch_size, sequence_length) containing the log-probabilities of the model's predictions for the response tokens.
    response_mask : torch.Tensor
        A boolean tensor of the same shape as `policy_log_probs` indicating which tokens are part of the response (True for response tokens).
    gradient_accumulation_steps : int
        The number of microbatches to accumulate gradients over before performing an optimization step.
    normalize_constant : float, optional
        A constant value to use for normalization instead of the actual count of unmasked elements, by default 1.0.
    """
    # Compute the sum of log-probabilities for the response tokens, ignoring masked values
    sum_log_probs = masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=1)

    # Compute the mean loss over the microbatch
    loss = -sum_log_probs.mean() / gradient_accumulation_steps
    loss.backward()

    # Prepare logging information
    logging_info = {
        "microbatch_loss": loss,
        "average_log_prob": policy_log_probs.masked_select(response_mask).mean(),
    }

    return loss, logging_info

def log_generations(
    model,
    tokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: callable,
    max_new_tokens: int = 1024,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, any]:
    """
    Prompt the model to generate responses for the given prompts and log
    detailed information about each generation along with aggregate statistics.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The language model to generate from.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer.
    prompts : list[str]
        Input prompts.
    ground_truths : list[str]
        Ground-truth answers corresponding to each prompt.
    reward_fn : callable
        Function with signature ``(response: str, ground_truth: str) -> dict``.
        Must return keys ``format_reward``, ``answer_reward``, and ``reward``.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate, by default 1024.
    batch_size : int, optional
        Batch size for generation, by default 8.
    device : str, optional
        Device to run generation on, by default "cuda".

    Returns
    -------
    dict[str, any]
        A dictionary containing:
        - "examples": list of dicts, each with:
            - "prompt": str
            - "response": str
            - "ground_truth": str
            - "format_reward": float
            - "answer_reward": float
            - "reward": float
            - "avg_token_entropy": float
            - "response_length": int
        - "avg_response_length": float
        - "avg_response_length_correct": float
        - "avg_response_length_incorrect": float
        - "avg_token_entropy": float
    """
    model.eval()
    all_examples = []

    # Prepare prompt tokenization (no padding for generation)
    for start_idx in range(0, len(prompts), batch_size):
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_ground_truths = ground_truths[start_idx:end_idx]

        # Tokenize prompts (left pad for decoder-only generation)
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        tokenizer.padding_side = original_padding_side

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode generated sequences
        generated_sequences = outputs.sequences
        # scores is a tuple of logits (one per generation step)
        scores = outputs.scores  # tuple of (batch_size, vocab_size)

        # Extract only the generated tokens (excluding prompt)
        prompt_lengths = inputs["input_ids"].shape[1]
        generated_tokens = generated_sequences[:, prompt_lengths:]

        for i in range(len(batch_prompts)):
            response = tokenizer.decode(
                generated_tokens[i],
                skip_special_tokens=True,
            )
            # Truncate at </answer> if present
            if "</answer>" in response:
                response = response[:response.index("</answer>") + len("</answer>")]
            gt = batch_ground_truths[i]
            reward_info = reward_fn(response, gt)

            # Compute per-token entropy from generation scores
            # scores[j][i] is the logits for the j-th generated token of example i
            token_entropies = []
            for j in range(len(scores)):
                if j < generated_tokens.shape[1] and generated_tokens[i, j] != tokenizer.pad_token_id:
                    logits = scores[j][i:i+1]  # (1, vocab_size)
                    entropy = compute_entropy(logits)  # (1,)
                    token_entropies.append(entropy.item())

            avg_token_entropy = sum(token_entropies) / len(token_entropies) if token_entropies else 0.0
            response_length = len([t for t in generated_tokens[i] if t != tokenizer.pad_token_id])

            all_examples.append({
                "prompt": batch_prompts[i],
                "response": response,
                "ground_truth": gt,
                "format_reward": reward_info.get("format_reward", 0.0),
                "answer_reward": reward_info.get("answer_reward", 0.0),
                "reward": reward_info.get("reward", 0.0),
                "avg_token_entropy": avg_token_entropy,
                "response_length": response_length,
            })

    # Compute aggregate statistics
    avg_response_length = sum(ex["response_length"] for ex in all_examples) / len(all_examples) if all_examples else 0.0
    correct_lengths = [ex["response_length"] for ex in all_examples if ex["answer_reward"] > 0.5]
    incorrect_lengths = [ex["response_length"] for ex in all_examples if ex["answer_reward"] <= 0.5]
    avg_response_length_correct = sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0.0
    avg_response_length_incorrect = sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0.0
    avg_token_entropy = sum(ex["avg_token_entropy"] for ex in all_examples) / len(all_examples) if all_examples else 0.0

    return {
        "examples": all_examples,
        "avg_response_length": avg_response_length,
        "avg_response_length_correct": avg_response_length_correct,
        "avg_response_length_incorrect": avg_response_length_incorrect,
        "avg_token_entropy": avg_token_entropy,
    }
