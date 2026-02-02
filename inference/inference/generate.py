from typing import Dict, Any, Optional, List, Tuple
import json
import argparse
import time

import vllm
from transformers import PreTrainedTokenizer, AutoTokenizer

from reasoning_cache.utils.inference import run_inference, create_messages
from reasoning_cache.utils.fill_prompt import fill_prompt


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for model inference.
    """
    parser = argparse.ArgumentParser(description="Generate model rollouts for various tasks.")

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input dataset file")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt file")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model to use for inference with vLLM",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index of samples to process (default: 0)",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Ending index of samples to process (exclusive). If None, process to end of dataset (default: None)",
    )
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size (default: 1)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling parameter. Default is to consider all tokens (-1).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate (default: 8192)",
    )
    parser.add_argument(
        "--n_rollouts_per_problem",
        type=int,
        default=1,
        help="Number of rollouts to generate per problem (default: 1)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated rollouts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Whether to use thinking mode. Default is False.",
    )
    parser.add_argument("--enforce_eager", action="store_true", help="If set, use eager mode for inference")
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.7, help="GPU memory utilization for inference (default: 0.7)"
    )
    parser.add_argument("--model_class", type=str, default="qwen", help="Model class (default: qwen)")
    return parser.parse_args()


def generate_rollouts(
    llm: vllm.LLM,
    sampling_params: vllm.SamplingParams,
    dataset: Dict[str, Any],
    prompt: str,
    start_index: int = 0,
    end_index: Optional[int] = None,
    apply_chat_template: bool = False,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    enable_thinking: bool = False,
    model_class: str = "qwen",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generates model rollouts with support for nested fields.

    Args:
        llm: vLLM language model.
        sampling_params: Sampling parameters for the generation.
        dataset: Dataset containing input problems.
        prompt: Prompt template for the task.
        start_index: Starting index of samples to process.
        end_index: Ending index of samples to process (exclusive). If None, process to end.
        apply_chat_template: Whether to apply the chat template to the prompt.
        tokenizer: The tokenizer to be used to apply the chat template.
        enable_thinking: Whether to use thinking mode.
        model_class: Model class (default: qwen).

    Returns:
        A tuple containing:
            - A list of generated model outputs with nested structure preserved.
            - A dictionary containing timing information.
    """
    if apply_chat_template and tokenizer is None:
        raise ValueError("tokenizer is required if apply_chat_template is True.")

    if end_index is None:
        end_index = len(dataset)

    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index > len(dataset):
        raise ValueError(f"end_index ({end_index}) cannot exceed dataset size ({len(dataset)})")
    if start_index >= end_index:
        raise ValueError("start_index must be less than end_index")

    print(f"Processing samples from index {start_index} to {end_index-1} (inclusive)")

    all_prompts = []
    prompt_origins = []
    sample_indices = list(range(start_index, end_index))

    for sample_idx in sample_indices:
        filled_result = fill_prompt(dataset[sample_idx], prompt)

        if isinstance(filled_result, str):
            all_prompts.append(filled_result)
            prompt_origins.append((sample_idx, None))
        elif isinstance(filled_result, list):
            all_prompts.extend(filled_result)
            for i in range(len(filled_result)):
                prompt_origins.append((sample_idx, i))
        else:
            raise ValueError(f"prompt_filler must return str or List[str], got {type(filled_result)}")

    messages = [create_messages(prompt) for prompt in all_prompts]

    if apply_chat_template:
        try:
            if model_class == "gptoss":
                print(f"Model class: gptoss")
                processed_inputs = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, reasoning_effort="high"
                )
            else:
                print(f"Model class: standard")
                processed_inputs = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
                )
                print(f"Thinking mode: {enable_thinking}")
        except Exception as e:
            processed_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"Thinking mode not available.")
    else:
        processed_inputs = messages

    print("Generating rollouts...")
    print(f"### Example filled prompt ###\n\n{processed_inputs[0]}\n")
    print(f"Total prompts to process: {len(processed_inputs)}")

    start_time = time.time()
    outputs = run_inference(llm, processed_inputs, sampling_params)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    sample_outputs = {}
    for i, output in enumerate(outputs):
        sample_idx, nested_idx = prompt_origins[i]

        if sample_idx not in sample_outputs:
            sample_outputs[sample_idx] = dict(dataset[sample_idx])
            sample_outputs[sample_idx]["rollouts"] = {}

        output_dict = {str(j): rollout for j, rollout in enumerate(output)}
        if nested_idx is None:
            sample_outputs[sample_idx]["rollouts"] = output_dict
        else:
            sample_outputs[sample_idx]["rollouts"][str(nested_idx)] = output_dict

    final_outputs = []
    for sample_idx in sample_indices:
        if sample_idx in sample_outputs:
            final_outputs.append(sample_outputs[sample_idx])

    num_samples = len(final_outputs)
    num_prompts = len(processed_inputs)
    timing_info = {
        "total_time_seconds": elapsed_time,
        "start_time": start_time,
        "end_time": end_time,
        "num_samples_processed": num_samples,
        "num_prompts_processed": num_prompts,
        "time_per_sample_seconds": elapsed_time / num_samples if num_samples > 0 else 0,
        "time_per_prompt_seconds": elapsed_time / num_prompts if num_prompts > 0 else 0,
        "prompts_per_second": num_prompts / elapsed_time if elapsed_time > 0 else 0,
    }

    return final_outputs, timing_info


def run_generation(args: argparse.Namespace):
    """
    Run generation for a given dataset and prompt.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    # Load the dataset
    if not args.dataset_path.endswith(".json"):
        args.dataset_path = args.dataset_path + ".json"
    with open(args.dataset_path, "r") as f:
        inference_dataset = json.load(f)

    # Load the model
    model_id = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Using enforce eager: {args.enforce_eager}")
    print(f"Using GPU memory utilization: {args.gpu_memory_utilization}")
    llm = vllm.LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=args.tp_size,
        tokenizer=model_id,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n_rollouts_per_problem,
        seed=args.seed,
    )

    # Load the prompt
    with open(args.prompt_path, "r") as f:
        prompt = f.read()

    # Generate rollouts
    rollouts, timing_info = generate_rollouts(
        llm,
        sampling_params,
        inference_dataset,
        prompt,
        start_index=args.start_index,
        end_index=args.end_index,
        apply_chat_template=True,
        tokenizer=tokenizer,
        enable_thinking=args.enable_thinking,
        model_class=args.model_class,
    )

    # Save the rollouts
    if not args.output_path.endswith(".json"):
        args.output_path = args.output_path + ".json"
    with open(args.output_path, "w") as f:
        json.dump(rollouts, f, indent=4)
    print(f"Rollouts saved to {args.output_path}")

    # Save timing information with metadata
    timing_path = args.output_path.replace(".json", "_timing.json")
    timing_data = {
        **timing_info,
        "metadata": {
            "model_path": args.model_path,
            "n_rollouts_per_problem": args.n_rollouts_per_problem,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "enable_thinking": args.enable_thinking,
        },
    }
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=4)
    print(f"Timing information saved to {timing_path}")
    print(
        f"Total time: {timing_info['total_time_seconds']:.2f}s, "
        f"Time per sample: {timing_info['time_per_sample_seconds']:.2f}s, "
        f"Throughput: {timing_info['prompts_per_second']:.2f} prompts/s"
    )


if __name__ == "__main__":
    args = parse_args()
    run_generation(args)
