from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import torch


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    prompt_token_ids = [
        tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs
    ]
    output_token_ids = [
        tokenizer.encode(o, add_special_tokens=False) for o in output_strs
    ]

    # Concatenate prompt and output tokens.
    full_token_ids = [p + o for p, o in zip(prompt_token_ids, output_token_ids)]
    prompt_and_output_lens = [len(tokens) for tokens in full_token_ids]
    max_len = max(prompt_and_output_lens)

    batch_size = len(prompt_strs)
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )

    input_ids = torch.full((batch_size, max_len - 1), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len - 1), pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool)

    for i, tokens in enumerate(full_token_ids):
        p_len = len(prompt_token_ids[i])
        o_len = len(output_token_ids[i])

        # Right-pad to max_len.
        padded = tokens + [pad_token_id] * (max_len - len(tokens))
        
        # input_ids: all tokens except the last one.
        input_ids[i] = torch.tensor(padded[:-1], dtype=torch.long)
        # labels: shifted input_ids (all tokens except the first one).
        labels[i] = torch.tensor(padded[1:], dtype=torch.long)

        # In `labels`, the first (p_len - 1) positions correspond to prompt tokens
        # (excluding the very first prompt token). The next o_len positions
        # correspond to output tokens.
        prompt_end_in_labels = max(0, p_len - 1)
        output_end_in_labels = prompt_end_in_labels + o_len
        response_mask[i, prompt_end_in_labels:output_end_in_labels] = True
    print(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


if __name__ == "__main__":
    # 加载模型：自动分配到 GPU，使用 BF16 + Flash Attention 2
    model = AutoModelForCausalLM.from_pretrained(
        "/root/assignment5-alignment/model",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",                    # ← 关键：自动分配到 GPU
        trust_remote_code=True,               # ← 关键：Qwen 等国产模型必需
    )

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/assignment5-alignment/model",
        trust_remote_code=True,
    )

    # ← 关键：设置 pad_token（Qwen 通常用 eos_token 作为 pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id