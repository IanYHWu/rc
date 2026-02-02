import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import vllm
from transformers import AutoTokenizer, PreTrainedTokenizer


class InferenceProblemStateS1:
    """
    A helper class to track the inference progress for a single problem using the S1 'Wait' approach.

    Attributes:
        problem (str): The original problem text.
        templated_problem (str): The problem text formatted with a prompt template.
        problem_id (str): Unique identifier for the problem.
        sample_id (str): Identifier for the specific sample/completion.
        label (str): The ground truth answer or label.
        max_steps (int): Maximum number of reasoning-summarization steps.
        reasoning_segments (List[str]): All reasoning segments (one per turn).
        current_step (int): The current step number.
        is_complete (bool): Whether the inference process is finished.
        reasoning_store (list): History of reasoning outputs.
        prompt_store (list): History of final prompts used.
    """

    def __init__(
        self,
        problem: str,
        templated_problem: str,
        problem_id: str,
        sample_id: str,
        label: str,
        max_steps: int,
    ):
        """Initializes the InferenceProblemStateS1."""
        self.problem = problem
        self.templated_problem = templated_problem
        self.problem_id = problem_id
        self.sample_id = sample_id
        self.label = label
        self.max_steps = max_steps

        self.reasoning_segments: List[str] = []
        self.current_step = 0
        self.is_complete = False
        self.completion_step = None
        self.completion_reason = None

        self.reasoning_store = []
        self.prompt_store = []

    def update_reasoning(self, response_string: str):
        """Updates the state with new reasoning output."""
        self.reasoning_store.append(response_string)
        self.reasoning_segments.append(response_string.strip())
        self.current_step += 1

    def mark_as_complete(self):
        """Marks the inference process as complete."""
        self.is_complete = True

    def get_filled_reasoning_prompt(self) -> str:
        """Generates the prompt for the next reasoning step using S1 'Wait' approach."""
        parts = [self.templated_problem]

        for segment in self.reasoning_segments:
            parts.append(segment)
            parts.append("Wait, let me continue thinking.")

        joined_prompt = "\n\n".join(parts)
        self.prompt_store.append(joined_prompt)
        return joined_prompt

    def get_full_reasoning(self) -> str:
        """Get the complete reasoning chain with Wait tokens."""
        if not self.reasoning_segments:
            return ""

        parts = []
        for segment in self.reasoning_segments[:-1]:
            parts.append(segment)
            parts.append("Wait, let me continue thinking.")
        parts.append(self.reasoning_segments[-1])

        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return f"InferenceProblemStateS1(problem_id={self.problem_id}, sample_id={self.sample_id}, step={self.current_step})"


class S1RolloutGenerator:
    """
    Generator for S1-style rollouts using the 'Wait' continuation approach.

    Attributes:
        llm_client (vllm.LLM): vLLM client for inference.
        tokenizer (PreTrainedTokenizer): Tokenizer for the model.
        reasoning_prompt_template (str): Template for reasoning prompts.
        config (Dict): Configuration parameters.
        max_steps (int): Maximum steps per rollout.
        max_thinking_tokens (int): Max tokens for initial reasoning.
        max_thinking_tokens_cont (int): Max tokens for continuation steps.
        base_sampling_params (vllm.SamplingParams): Base sampling parameters.
        n_samples_per_problem (int): Number of samples per problem.
    """

    def __init__(
        self,
        llm_client: vllm.LLM,
        tokenizer: PreTrainedTokenizer,
        reasoning_prompt_template: str,
        config: Dict[str, Any],
        sampling_params: vllm.SamplingParams,
    ) -> None:
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.reasoning_prompt_template = reasoning_prompt_template
        self.config = config
        self.max_steps = config.get("max_steps", 2)
        self.max_thinking_tokens = config.get("max_thinking_tokens", 8192)
        self.max_thinking_tokens_cont = config.get("max_thinking_tokens_cont", 2048)
        self.base_sampling_params = sampling_params
        self.n_samples_per_problem = config.get("n", 4)

    def generate_rollouts(
        self,
        prompts_batch: List[Dict[str, Any]],
    ) -> List[InferenceProblemStateS1]:
        """Generate rollouts using the S1 'Wait' approach."""
        print(f"Running rollout step 1/{self.max_steps}.")
        active_states, completed_states = self.initial_rollout_step(prompts_batch)

        for step in range(1, self.max_steps):
            print(f"Running rollout step {step + 1}/{self.max_steps}.")
            if not active_states:
                break
            active_states, completed_states = self.rollout_step(active_states, completed_states)

        for state in active_states:
            state.mark_as_complete()
            completed_states.append(state)
        return completed_states

    def run_inference(
        self,
        prompts: List[str],
        n: int,
        max_length: int,
    ) -> List[str]:
        """Generate sequences by calling the vLLM engine."""
        sampling_params = self.base_sampling_params.clone()
        sampling_params.n = n
        sampling_params.max_tokens = max_length
        vllm_output = self.llm_client.generate(prompts, sampling_params)
        return [output.text for request_output in vllm_output for output in request_output.outputs]

    def extract_and_prepare_prompts(self, prompts_batch: List[Dict[str, Any]]):
        """Extracts problems, IDs, and labels from a batch of prompts."""
        raw_prompts = [d["problem"] for d in prompts_batch]
        problem_ids = [d["problem_id"] for d in prompts_batch]
        labels = [d["label"] for d in prompts_batch]
        return raw_prompts, problem_ids, labels

    def prepare_active_states(
        self,
        problems: List[str],
        problem_ids: List[str],
        labels: List[str],
    ) -> List[InferenceProblemStateS1]:
        """Initializes InferenceProblemStateS1 objects for a batch of problems."""
        templated_problems = [self.reasoning_prompt_template.format(problem=problem) for problem in problems]
        problem_messages = [[{"role": "user", "content": problem}] for problem in templated_problems]
        templated_problems = self.tokenizer.apply_chat_template(
            problem_messages, add_generation_prompt=True, tokenize=False
        )

        raw_prompts_with_ids = [
            {
                "problem_id": f"{problem_ids[i]}",
                "sample_id": f"{n}",
                "problem": problems[i],
                "templated_problem": templated_problems[i],
                "label": labels[i],
            }
            for i in range(len(problems))
            for n in range(self.n_samples_per_problem)
        ]

        active_states = [
            InferenceProblemStateS1(
                **raw_prompt_with_id,
                max_steps=self.max_steps,
            )
            for raw_prompt_with_id in raw_prompts_with_ids
        ]
        return active_states

    def reasoning_rollout_postprocess(
        self,
        rollouts: List[str],
        active_states: List[InferenceProblemStateS1],
    ):
        """Processes the output of reasoning inference and updates active states."""
        if len(rollouts) != len(active_states):
            raise ValueError(
                f"Mismatched number of rollouts ({len(rollouts)}) and active states ({len(active_states)})."
            )
        for i, state in enumerate(active_states):
            state.update_reasoning(rollouts[i])
        return active_states

    def initial_rollout_step(self, prompts_batch: List[Dict[str, Any]]):
        """Performs the first step of the S1 rollout."""
        problems, problem_ids, labels = self.extract_and_prepare_prompts(prompts_batch)
        active_states = self.prepare_active_states(problems, problem_ids, labels)
        completed_states = []

        inference_prompts = [state.templated_problem for state in active_states]

        rollouts = self.run_inference(inference_prompts, n=1, max_length=self.max_thinking_tokens)
        active_states = self.reasoning_rollout_postprocess(rollouts, active_states)

        return active_states, completed_states

    def rollout_step(
        self,
        active_states: List[InferenceProblemStateS1],
        completed_states: List[InferenceProblemStateS1],
    ) -> Tuple[List[InferenceProblemStateS1], List[InferenceProblemStateS1]]:
        """Performs a single intermediate step of the S1 rollout."""
        filled_prompts = [state.get_filled_reasoning_prompt() for state in active_states]

        rollouts = self.run_inference(filled_prompts, n=1, max_length=self.max_thinking_tokens_cont)
        active_states = self.reasoning_rollout_postprocess(rollouts, active_states)

        return active_states, completed_states


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model inference."""
    parser = argparse.ArgumentParser(description="Generate S1-style rollouts using the 'Wait' continuation approach.")

    # --- File Paths ---
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to the input dataset file (.json)")
    parser.add_argument(
        "--reasoning_prompt_path",
        type=Path,
        required=True,
        help="Path to the reasoning prompt template",
    )
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the generated rollouts")

    # --- Model Configuration ---
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path or name of the HuggingFace model to use with vLLM"
    )
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    # --- Slicing and Batching ---
    parser.add_argument("--start_index", type=int, default=0, help="Starting index of samples to process (default: 0)")
    parser.add_argument(
        "--end_index", type=int, default=None, help="Ending index of samples to process (exclusive, default: all)"
    )

    # --- Generation Parameters ---
    parser.add_argument("--n", type=int, default=4, help="Number of completions to generate per prompt (default: 4)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (default: 0.6)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter (default: 0.95)")
    parser.add_argument(
        "--max_thinking_tokens", type=int, default=8192, help="Max tokens for reasoning steps (default: 8192)"
    )
    parser.add_argument("--max_steps", type=int, default=2, help="Number of reasoning steps to perform (default: 2)")

    return parser.parse_args()


def run_generation(args: argparse.Namespace):
    """Run generation for a given dataset and prompt."""
    # --- Load Dataset ---
    with args.dataset_path.open("r") as f:
        inference_dataset = json.load(f)

    inference_dataset = [
        {"problem": s["problem"], "problem_id": s["id"], "label": s["answer"]} for s in inference_dataset
    ]
    # --- Handle Dataset Slicing ---
    start, end = args.start_index, args.end_index
    if end is None:
        end = len(inference_dataset)
    if not (0 <= start < end <= len(inference_dataset)):
        raise ValueError(f"Invalid slice indices: start={start}, end={end}, dataset_size={len(inference_dataset)}")

    print(f"Processing samples from index {start} to {end-1} (inclusive).")
    inference_dataset = inference_dataset[start:end]

    # --- Initialize Model and Tokenizer ---
    print(f"Loading vLLM model: {args.model_path} with seed {args.seed}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"Loading tokenizer: {args.model_path}")
    llm = vllm.LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tp_size,
        seed=args.seed,
    )
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    # --- Load Prompts ---
    with args.reasoning_prompt_path.open("r") as f:
        reasoning_prompt = f.read()

    # --- Generate Rollouts ---
    config = vars(args)
    rollout_generator = S1RolloutGenerator(
        llm,
        tokenizer,
        reasoning_prompt,
        config,
        sampling_params,
    )
    all_rollouts = rollout_generator.generate_rollouts(inference_dataset)

    fields_to_save = [
        "problem",
        "label",
        "reasoning_store",
        "problem_id",
        "sample_id",
    ]
    output_data = [{key: getattr(state, key) for key in fields_to_save} for state in all_rollouts]

    with args.output_path.open("w") as f:
        json.dump(output_data, f, indent=4)
    print(f"All {len(output_data)} rollouts saved to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    run_generation(args)
