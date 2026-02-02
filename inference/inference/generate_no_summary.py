import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import vllm
from transformers import AutoTokenizer, PreTrainedTokenizer


class InferenceProblemStateNoSummary:
    """
    A helper class to track the inference progress for a single problem without summarization.

    Attributes:
        problem (str): The original problem text.
        templated_problem (str): The problem text formatted with a prompt template.
        reasoning_prompt_template (str): Template for generating reasoning prompts.
        problem_id (str): Unique identifier for the problem.
        sample_id (str): Identifier for the specific sample/completion.
        label (str): The ground truth answer or label.
        max_steps (int): Maximum number of reasoning steps.
        use_think_tags (bool): Whether to use <think> tags in reasoning.
        prev_attempt (str): The reasoning from the previous step.
        curr_reasoning (str): The reasoning from the most recent step.
        current_step (int): The current step number.
        is_complete (bool): Whether the inference process is finished.
        reasoning_store (list): History of reasoning outputs.
        reasoning_prompt_store (list): History of prompts sent for reasoning.
        prompt_store (list): History of final prompts used.
    """

    def __init__(
        self,
        problem: str,
        templated_problem: str,
        reasoning_prompt_template: str,
        problem_id: str,
        sample_id: str,
        label: str,
        max_steps: int,
        use_think_tags: bool = True,
    ):
        """Initializes the InferenceProblemStateNoSummary."""
        self.problem = problem
        self.templated_problem = templated_problem
        self.reasoning_prompt_template = reasoning_prompt_template
        self.problem_id = problem_id
        self.sample_id = sample_id
        self.label = label
        self.max_steps = max_steps
        self.use_think_tags = use_think_tags

        # Use default style: structured for reasoning
        self.reasoning_prompt_style = "structured"

        self.prev_attempt = ""
        self.curr_reasoning = ""
        self.current_step = 0
        self.is_complete = False
        self.completion_step = None
        self.completion_reason = None

        self.reasoning_store = []
        self.reasoning_prompt_store = []
        self.prompt_store = []
        self.contains_answer = []

    def update_reasoning(self, response_string: str):
        """Updates the state with new reasoning output."""
        self.reasoning_store.append(response_string)
        processed_response_string = response_string.replace("<think>", "")
        if "</think>" in processed_response_string:
            processed_response_string = processed_response_string.split("</think>")[0]
        self.curr_reasoning = processed_response_string.strip()

        # Update prev_attempt to be the current reasoning for next iteration
        self.prev_attempt = self.curr_reasoning
        self.current_step += 1

    def _check_for_answer(self, response: str) -> bool:
        """Checks if the response contains a boxed answer."""
        return "boxed{" in response

    def mark_as_complete(self):
        """Marks the inference process as complete."""
        self.is_complete = True

    def get_filled_reasoning_prompt(self, tokenizer: PreTrainedTokenizer) -> str:
        """Generates the prompt for the next reasoning step."""
        # Always use "structured" style
        filled_prompt = self.reasoning_prompt_template.format(
            **dict(
                problem=self.problem,
                prev=self.prev_attempt,
            )
        )
        templated_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": filled_prompt}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.use_think_tags,
        )

        if self.use_think_tags:
            parts = [f"{templated_prompt}<think>"]
        else:
            parts = [f"{templated_prompt}"]

        joined_parts = "\n\n".join(parts)
        self.reasoning_prompt_store.append(joined_parts)
        return joined_parts

    def __repr__(self) -> str:
        return f"InferenceProblemStateNoSummary(problem_id={self.problem_id}, sample_id={self.sample_id}, label={self.label}"


class ReasoningNoSummaryRolloutGenerator:
    """
    Manages the generation of rollouts without summarization.

    Attributes:
        llm_client (vllm.LLM): vLLM client for inference.
        tokenizer (PreTrainedTokenizer): Tokenizer for the model.
        reasoning_prompt_template (str): Template for reasoning prompts.
        config (Dict): Configuration parameters.
        max_steps (int): Maximum steps per rollout.
        max_thinking_tokens (int): Max tokens for reasoning.
        reasoning_prompt_style (str): Style of reasoning prompt.
        use_think_tags (bool): Whether to use <think> tags.
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
        """
        Initialize the ReasoningNoSummaryRolloutGenerator.
        """
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.reasoning_prompt_template = reasoning_prompt_template
        self.config = config
        self.max_steps = config.get("max_steps", 2)
        self.max_thinking_tokens = config.get("max_thinking_tokens", 8192)
        # Use default style: structured for reasoning
        self.use_think_tags = config.get("use_think_tags", False)
        self.base_sampling_params = sampling_params
        self.n_samples_per_problem = config.get("n", 4)

    def generate_rollouts(
        self,
        prompts_batch: List[Dict[str, Any]],
    ) -> List[InferenceProblemStateNoSummary]:
        """
        Generate rollouts for reasoning without summarization.
        """
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
        """
        Generate sequences by calling the vLLM engine.
        """
        sampling_params = self.base_sampling_params.clone()
        sampling_params.n = n
        sampling_params.max_tokens = max_length

        vllm_output = self.llm_client.generate(prompts, sampling_params)

        # Flatten the outputs since vLLM returns a list of completions for each prompt
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
    ) -> List[InferenceProblemStateNoSummary]:
        """Initializes InferenceProblemStateNoSummary objects for a batch of problems."""
        templated_problems = [self.reasoning_prompt_template.format(problem=problem, prev="") for problem in problems]
        problem_messages = [[{"role": "user", "content": problem}] for problem in templated_problems]
        templated_problems = self.tokenizer.apply_chat_template(
            problem_messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.use_think_tags
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
            InferenceProblemStateNoSummary(
                **raw_prompt_with_id,
                reasoning_prompt_template=self.reasoning_prompt_template,
                max_steps=self.max_steps,
                use_think_tags=self.use_think_tags,
            )
            for raw_prompt_with_id in raw_prompts_with_ids
        ]
        return active_states

    def prepare_for_inference(self, filled_prompts: List[str], apply_template: bool, enable_thinking: bool):
        """Prepares prompts for inference, optionally applying a chat template."""
        if apply_template:
            messages = [[{"role": "user", "content": filled_prompt}] for filled_prompt in filled_prompts]
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
            )
        else:
            return filled_prompts

    def reasoning_rollout_postprocess(
        self,
        rollouts: List[str],
        active_states: List[InferenceProblemStateNoSummary],
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
        """Performs the first step of the reasoning rollout."""
        problems, problem_ids, labels = self.extract_and_prepare_prompts(prompts_batch)
        active_states = self.prepare_active_states(problems, problem_ids, labels)
        completed_states = []

        filled_prompts = [self.reasoning_prompt_template.format(problem=problem, prev="") for problem in problems]
        inference_data_proto = self.prepare_for_inference(filled_prompts, apply_template=True, enable_thinking=True)

        rollouts = self.run_inference(
            inference_data_proto, n=self.n_samples_per_problem, max_length=self.max_thinking_tokens
        )
        active_states = self.reasoning_rollout_postprocess(rollouts, active_states)

        return active_states, completed_states

    def rollout_step(
        self,
        active_states: List[InferenceProblemStateNoSummary],
        completed_states: List[InferenceProblemStateNoSummary],
    ) -> Tuple[List[InferenceProblemStateNoSummary], List[InferenceProblemStateNoSummary]]:
        """Performs a single intermediate step of the reasoning rollout."""
        filled_prompts = [state.get_filled_reasoning_prompt(self.tokenizer) for state in active_states]
        final_prompts = self.prepare_for_inference(filled_prompts, apply_template=False, enable_thinking=True)

        for i, state in enumerate(active_states):
            state.prompt_store.append(final_prompts[i])

        rollouts = self.run_inference(final_prompts, n=1, max_length=self.max_thinking_tokens)
        active_states = self.reasoning_rollout_postprocess(rollouts, active_states)

        return active_states, completed_states


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model inference."""
    parser = argparse.ArgumentParser(description="Generate model rollouts without summarization using vLLM.")

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

    # --- Prompting Strategy ---
    # Note: Uses default style - 'structured' for reasoning
    parser.add_argument("--use_think_tags", action="store_true", help="If set, enclose reasoning steps in <think> tags")

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
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    print(f"Loading tokenizer: {args.model_path}")
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
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
    rollout_generator = ReasoningNoSummaryRolloutGenerator(
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
    print(f"All {len(output_data)} filtered rollouts saved to {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    run_generation(args)
