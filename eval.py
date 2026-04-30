"""
Script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH validation set.

This script:
1. Loads MATH validation examples from /data/a5-alignment/MATH/validation.jsonl
2. Formats them as string prompts using the r1_zero prompt template
3. Generates outputs for each example using vLLM
4. Calculates evaluation metrics (format reward, answer reward, overall accuracy)
5. Serializes examples, model generations, and evaluation scores to disk
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MATH_VALIDATION_PATH = "/root/assignment5-alignment-main/data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"
R1_ZERO_PROMPT_PATH = "/root/assignment5-alignment-main/cs336_alignment/prompts/r1_zero.prompt"
MODEL_PATH = "/root/assignment5-alignment-main/sft_outputs/sft_size_full"
OUTPUT_PATH = "math_evaluation_results.jsonl"

# Load the r1_zero prompt template once at module import.
with open(R1_ZERO_PROMPT_PATH, "r") as f:
    r1_zero_prompt_template = f.read()


# ---------------------------------------------------------------------------
# Data loading & prompt formatting
# ---------------------------------------------------------------------------

def load_math_validation_data(path: str) -> List[Dict[str, Any]]:
    """
    Load MATH validation examples.

    Supports both:
    - JSONL (one JSON object per line)
    - JSON array (a single JSON list of objects)
    """
    data: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        raw = f.read().strip()

    if raw.startswith("["):
        # Standard JSON array.
        data = json.loads(raw)
    else:
        # JSONL format.
        for line in raw.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def format_r1_zero_prompt(problem: str) -> str:
    """Format a math problem using the r1_zero prompt template."""
    return r1_zero_prompt_template.replace("{question}", problem)


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    examples: List[Dict[str, Any]],
    eval_sampling_params: SamplingParams,
    output_path: str = OUTPUT_PATH,
) -> Dict[str, float]:
    """
    Evaluate a language model on a list of examples, compute aggregate metrics,
    and serialize per-example results to disk.

    Parameters
    ----------
    vllm_model :
        An initialized vLLM model.
    reward_fn :
        Callable with signature ``(response: str, ground_truth: str) -> dict``.
        Must return keys such as ``format_reward``, ``answer_reward``, ``reward``.
    examples :
        List of dicts, each containing at least ``"prompt"`` and ``"ground_truth"``.
    eval_sampling_params :
        vLLM ``SamplingParams`` controlling generation behaviour.
    output_path :
        File path where the JSONL results will be written.

    Returns
    -------
    dict
        Aggregate metrics averaged over all examples.
    """
    prompts = [ex["prompt"] for ex in examples]
    ground_truths = [ex["ground_truth"] for ex in examples]

    # Generate
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # Per-example scoring
    results: List[Dict[str, Any]] = []
    total_format_reward = 0.0
    total_answer_reward = 0.0
    total_reward = 0.0
    n = len(outputs)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gt = ground_truths[i]

        metrics = reward_fn(generated_text, gt)

        results.append(
            {
                "prompt": output.prompt,
                "ground_truth": gt,
                "generated_text": generated_text,
                "reward_metrics": metrics,
            }
        )

        total_format_reward += metrics.get("format_reward", 0.0)
        total_answer_reward += metrics.get("answer_reward", 0.0)
        total_reward += metrics.get("reward", 0.0)

    # Aggregate metrics
    aggregate = {
        "num_examples": n,
        "format_reward": total_format_reward / n if n > 0 else 0.0,
        "answer_reward": total_answer_reward / n if n > 0 else 0.0,
        "reward": total_reward / n if n > 0 else 0.0,
    }

    # Serialize detailed results
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Serialize aggregate metrics alongside
    metrics_file = out_file.with_suffix(".metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"Saved {n} detailed results -> {out_file}")
    print(f"Saved aggregate metrics  -> {metrics_file}")
    print(f"Aggregate metrics: {aggregate}")

    return aggregate


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load MATH validation examples
    print(f"Loading MATH validation data from {MATH_VALIDATION_PATH} ...")
    math_examples = load_math_validation_data(MATH_VALIDATION_PATH)
    print(f"Loaded {len(math_examples)} examples.")

    # 2. Format prompts
    eval_examples = []
    for ex in math_examples:
        problem = ex["problem"]
        # The processed MATH dataset stores the gold answer under
        # ``expected_answer`` (or ``answer`` in some variants).
        ground_truth = ex.get("expected_answer", ex.get("answer", ""))
        prompt = format_r1_zero_prompt(problem)
        eval_examples.append(
            {
                "prompt": prompt,
                "ground_truth": ground_truth,
            }
        )

    # 3. Sampling parameters — greedy decoding for zero-shot evaluation
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # 4. Load model
    print(f"Loading model from {MODEL_PATH} ...")
    llm = LLM(model=MODEL_PATH)

    # 5. Evaluate
    print("Starting evaluation ...")
    metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        examples=eval_examples,
        eval_sampling_params=sampling_params,
        output_path=OUTPUT_PATH,
    )

    print("\n=== Evaluation Complete ===")
    print(f"Format reward : {metrics['format_reward']:.4f}")
    print(f"Answer reward : {metrics['answer_reward']:.4f}")
    print(f"Overall reward: {metrics['reward']:.4f}")


if __name__ == "__main__":
    main()
