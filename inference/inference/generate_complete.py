import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time

import vllm
from transformers import AutoTokenizer, PreTrainedTokenizer


class InferenceProblemState:
    """
    A helper class to track the inference progress for a single problem.

    Attributes:
        problem (str): The original problem text.
        templated_problem (str): The problem text formatted with a prompt template.
        reasoning_prompt_template (str): Template for generating reasoning prompts.
        summarization_prompt_template (str): Template for generating summarization prompts.
        problem_id (str): Unique identifier for the problem.
        sample_id (str): Identifier for the specific sample/completion.
        label (str): The ground truth answer or label.
        max_steps (int): Maximum number of reasoning-summarization steps.
        use_think_tags (bool): Whether to use <think> tags in reasoning.
        model_class (str): The class of the model being used (e.g., 'gptoss', 'qwen').
        curr_summary (str): The current accumulated summary.
        curr_reasoning (str): The reasoning from the most recent step.
        current_step (int): The current step number.
        is_complete (bool): Whether the inference process is finished.
        reasoning_store (list): History of reasoning outputs.
        summarization_store (list): History of summarization outputs.
        reasoning_prompt_store (list): History of prompts sent for reasoning.
        summarization_prompt_store (list): History of prompts sent for summarization.
        prompt_store (list): History of final prompts used.
    """

    def __init__(
        self,
        problem: str,
        templated_problem: str,
        reasoning_prompt_template: str,
        summarization_prompt_template: str,
        problem_id: str,
        sample_id: str,
        label: str,
        max_steps: int,
        use_think_tags: bool = True,
        model_class: str = "gptoss",
    ):
        """Initializes the InferenceProblemState."""
        self.problem = problem
        self.templated_problem = templated_problem
        self.reasoning_prompt_template = reasoning_prompt_template
        self.summarization_prompt_template = summarization_prompt_template
        self.problem_id = problem_id
        self.sample_id = sample_id
        self.label = label
        self.max_steps = max_steps
        self.use_think_tags = use_think_tags
        self.model_class = model_class

        # Use default styles: structured for reasoning, summ for summarization
        self.reasoning_prompt_style = "structured"
        self.summarization_style = "summ"

        self.curr_summary = ""
        self.curr_reasoning = ""
        self.current_step = 0
        self.is_complete = False
        self.completion_step = None
        self.completion_reason = None

        self.reasoning_store = []
        self.summarization_store = []
        self.reasoning_prompt_store = []
        self.summarization_prompt_store = []
        self.prompt_store = []
        self.contains_answer = []

    def update_reasoning(self, response_string: str):
        """Updates the state with new reasoning output."""
        self.reasoning_store.append(response_string)
        if "</think>" in response_string:
            response_string = response_string.split("</think>")[0]
        processed_response_string = response_string.replace("<think>", "")
        self.curr_reasoning = processed_response_string.strip()

    def update_summarization(self, response_string: str):
        """Updates the state with new summarization output."""
        self.summarization_store.append(response_string)
        if "</think>" in response_string:
            response_string = response_string.split("</think>")[1].strip()
        processed_response_string = response_string.replace("<think>", "").strip()
        # Always use "summ" style
        self.curr_summary = processed_response_string
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
                curr_summary=self.curr_summary,
            )
        )
        if self.model_class == "gptoss":
            templated_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": filled_prompt}],
                add_generation_prompt=True,
                tokenize=False,
                reasoning_effort="high",
            )
        else:
            templated_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": filled_prompt}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.use_think_tags,
            )

        if self.use_think_tags and self.model_class != "gptoss" and "<think>" not in templated_prompt:
            parts = [f"{templated_prompt}<think>"]
        else:
            parts = [f"{templated_prompt}"]

        joined_parts = "\n\n".join(parts)
        self.reasoning_prompt_store.append(joined_parts)
        return joined_parts

    def get_filled_summarization_prompt(self) -> str:
        """Generates the prompt for the next summarization step."""
        if "<think>" in self.curr_reasoning:
            curr_chunk = self.curr_reasoning.split("<think>")[1]
        else:
            curr_chunk = self.curr_reasoning
        filled_prompt = self.summarization_prompt_template.format(
            problem=self.problem, existing_summary=self.curr_summary, reasoning=curr_chunk.strip()
        )
        self.summarization_prompt_store.append(filled_prompt)
        return filled_prompt

    def __repr__(self) -> str:
        return f"InferenceProblemState(problem_id={self.problem_id}, sample_id={self.sample_id}, label={self.label}"


class ReasoningCacheRolloutGenerator:
    """
    Manages the generation of rollouts using a single model for both reasoning and summarization.

    Attributes:
        llm_client (vllm.LLM): vLLM client for inference.
        tokenizer (PreTrainedTokenizer): Tokenizer for the model.
        reasoning_prompt_template (str): Template for reasoning prompts.
        summarization_prompt_template (str): Template for summarization prompts.
        config (Dict): Configuration parameters.
        max_steps (int): Maximum steps per rollout.
        max_thinking_tokens (int): Max tokens for reasoning.
        max_summary_tokens (int): Max tokens for summarization.
        reasoning_prompt_style (str): Style of reasoning prompt.
        use_think_tags (bool): Whether to use <think> tags.
        summarization_style (str): Style of summarization.
        base_sampling_params (vllm.SamplingParams): Base sampling parameters.
        n_samples_per_problem (int): Number of samples per problem.
        model_class (str): Model class name.
        step_timing_info (list): List of dictionaries containing timing info for each step.
    """

    def __init__(
        self,
        llm_client: vllm.LLM,
        tokenizer: PreTrainedTokenizer,
        reasoning_prompt_template: str,
        summarization_prompt_template: str,
        config: Dict[str, Any],
        sampling_params: vllm.SamplingParams,
        model_class: str,
    ) -> None:
        """
        Initialize the ReasoningCacheRolloutGenerator.
        """
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.reasoning_prompt_template = reasoning_prompt_template
        self.summarization_prompt_template = summarization_prompt_template
        self.config = config
        self.max_steps = config.get("max_steps", 2)
        self.max_thinking_tokens = config.get("max_thinking_tokens", 8192)
        self.max_summary_tokens = config.get("max_summarization_tokens", 2048)
        # Use default styles: structured for reasoning, summ for summarization
        self.use_think_tags = config.get("use_think_tags", False)
        self.base_sampling_params = sampling_params
        self.n_samples_per_problem = config.get("n", 4)
        self.model_class = model_class
        self.step_timing_info = []

    def generate_rollouts(
        self,
        prompts_batch: List[Dict[str, Any]],
    ) -> List[InferenceProblemState]:
        """
        Generate rollouts for reasoning cache.
        """
        print(f"Running rollout step 1/{self.max_steps}.")
        start_time = time.time()

        active_states, completed_states = self.initial_rollout_step(prompts_batch)

        for step in range(1, self.max_steps):
            print(f"Running rollout step {step + 1}/{self.max_steps}.")
            if not active_states:
                break
            active_states, completed_states = self.rollout_step(active_states, completed_states, step + 1)
        end_time = time.time()
        print(f"Total time taken for all steps: {end_time - start_time} seconds")

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
        for prompt in prompts:
            if prompt.count("<think>") >= 2:
                raise ValueError(f"Prompt contains multiple <think> tags: {prompt}")

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
    ) -> List[InferenceProblemState]:
        """Initializes InferenceProblemState objects for a batch of problems."""
        templated_problems = [
            self.reasoning_prompt_template.format(problem=problem, curr_summary="") for problem in problems
        ]
        problem_messages = [[{"role": "user", "content": problem}] for problem in templated_problems]
        if self.model_class == "gptoss":
            templated_problems = self.tokenizer.apply_chat_template(
                problem_messages, add_generation_prompt=True, tokenize=False, reasoning_effort="high"
            )
        else:
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
            InferenceProblemState(
                **raw_prompt_with_id,
                reasoning_prompt_template=self.reasoning_prompt_template,
                summarization_prompt_template=self.summarization_prompt_template,
                max_steps=self.max_steps,
                use_think_tags=self.use_think_tags,
                model_class=self.model_class,
            )
            for raw_prompt_with_id in raw_prompts_with_ids
        ]
        return active_states

    def prepare_for_inference(self, filled_prompts: List[str], apply_template: bool, enable_thinking: bool):
        """Prepares prompts for inference, optionally applying a chat template."""
        if apply_template:
            messages = [[{"role": "user", "content": filled_prompt}] for filled_prompt in filled_prompts]
            if self.model_class == "gptoss":
                if enable_thinking:
                    reasoning_effort = "high"
                else:
                    reasoning_effort = "medium"
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, reasoning_effort=reasoning_effort
                )
            else:
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
                )
        else:
            return filled_prompts

    def reasoning_rollout_postprocess(
        self,
        rollouts: List[str],
        active_states: List[InferenceProblemState],
    ):
        """Processes the output of reasoning inference and updates active states."""
        if len(rollouts) != len(active_states):
            raise ValueError(
                f"Mismatched number of rollouts ({len(rollouts)}) and active states ({len(active_states)})."
            )
        for i, state in enumerate(active_states):
            state.update_reasoning(rollouts[i])
        return active_states

    def summarization_rollout_postprocess(
        self,
        rollouts: List[str],
        active_states: List[InferenceProblemState],
        completed_states: List[InferenceProblemState],
    ):
        """Processes the output of summarization inference and updates active/completed states."""
        if len(rollouts) != len(active_states):
            raise ValueError(
                f"Mismatched number of rollouts ({len(rollouts)}) and active states ({len(active_states)})."
            )
        for i, state in enumerate(active_states):
            state.update_summarization(rollouts[i])

        next_active_states = []
        for state in active_states:
            if state.is_complete:
                completed_states.append(state)
            else:
                next_active_states.append(state)
        return next_active_states, completed_states

    def initial_rollout_step(self, prompts_batch: List[Dict[str, Any]]):
        """Performs the first step of the reasoning-summarization rollout."""
        problems, problem_ids, labels = self.extract_and_prepare_prompts(prompts_batch)
        active_states = self.prepare_active_states(problems, problem_ids, labels)
        completed_states = []

        filled_prompts = [
            self.reasoning_prompt_template.format(problem=problem, curr_summary="") for problem in problems
        ]
        inference_data_proto = self.prepare_for_inference(
            filled_prompts, apply_template=True, enable_thinking=self.use_think_tags
        )

        reasoning_start = time.time()
        rollouts = self.run_inference(
            inference_data_proto,
            n=self.n_samples_per_problem,
            max_length=self.max_thinking_tokens,
        )
        reasoning_end = time.time()
        reasoning_time = reasoning_end - reasoning_start

        active_states = self.reasoning_rollout_postprocess(rollouts, active_states)

        filled_prompts = [state.get_filled_summarization_prompt() for state in active_states]
        inference_data_proto = self.prepare_for_inference(
            filled_prompts, apply_template=True, enable_thinking=self.use_think_tags
        )

        summarization_start = time.time()
        rollouts = self.run_inference(inference_data_proto, n=1, max_length=self.max_summary_tokens)
        summarization_end = time.time()
        summarization_time = summarization_end - summarization_start

        active_states, completed_states = self.summarization_rollout_postprocess(
            rollouts, active_states, completed_states
        )

        self.step_timing_info.append(
            {
                "step": 1,
                "reasoning_time": reasoning_time,
                "summarization_time": summarization_time,
                "num_problems": len(problems),
                "num_samples": len(active_states) + len(completed_states),
            }
        )

        return active_states, completed_states

    def rollout_step(
        self,
        active_states: List[InferenceProblemState],
        completed_states: List[InferenceProblemState],
        step_number: int,
    ) -> Tuple[List[InferenceProblemState], List[InferenceProblemState]]:
        """Performs a single intermediate step of the reasoning-summarization rollout."""
        filled_prompts = [state.get_filled_reasoning_prompt(self.tokenizer) for state in active_states]
        final_prompts = self.prepare_for_inference(
            filled_prompts, apply_template=False, enable_thinking=self.use_think_tags
        )

        for i, state in enumerate(active_states):
            state.prompt_store.append(final_prompts[i])

        reasoning_start = time.time()
        rollouts = self.run_inference(final_prompts, n=1, max_length=self.max_thinking_tokens)
        reasoning_end = time.time()
        reasoning_time = reasoning_end - reasoning_start

        active_states = self.reasoning_rollout_postprocess(rollouts, active_states)

        filled_prompts = [state.get_filled_summarization_prompt() for state in active_states]
        final_prompts = self.prepare_for_inference(
            filled_prompts, apply_template=True, enable_thinking=self.use_think_tags
        )

        summarization_start = time.time()
        rollouts = self.run_inference(final_prompts, n=1, max_length=self.max_summary_tokens)
        summarization_end = time.time()
        summarization_time = summarization_end - summarization_start

        num_active_before = len(active_states)
        active_states, completed_states = self.summarization_rollout_postprocess(
            rollouts, active_states, completed_states
        )

        self.step_timing_info.append(
            {
                "step": step_number,
                "reasoning_time": reasoning_time,
                "summarization_time": summarization_time,
                "num_active_samples": num_active_before,
            }
        )

        return active_states, completed_states


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model inference."""
    parser = argparse.ArgumentParser(description="Generate model rollouts for various tasks using vLLM.")

    # --- File Paths ---
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to the input dataset file (.json)")
    parser.add_argument(
        "--reasoning_prompt_path",
        type=Path,
        required=True,
        help="Path to the reasoning prompt template",
    )
    parser.add_argument(
        "--summarization_prompt_path", type=Path, required=True, help="Path to the summarization prompt template"
    )
    parser.add_argument("--output_path", type=Path, required=True, help="Path to save the generated rollouts")

    # --- Model Configuration ---
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path or name of the HuggingFace model to use with vLLM"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path or name of the HuggingFace tokenizer to use with vLLM. If None, uses the same as the model path.",
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
    parser.add_argument(
        "--max_summarization_tokens", type=int, default=2048, help="Max tokens for summarization steps (default: 2048)"
    )
    parser.add_argument(
        "--max_steps", type=int, default=2, help="Number of reasoning/summarization steps to perform (default: 2)"
    )

    # --- Prompting Strategy ---
    # Note: Uses default styles - 'structured' for reasoning and 'summ' for summarization
    parser.add_argument("--use_think_tags", action="store_true", help="If set, enclose reasoning steps in <think> tags")
    parser.add_argument("--enforce_eager", action="store_true", help="If set, use eager mode for inference")
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.7, help="GPU memory utilization for inference (default: 0.7)"
    )
    parser.add_argument("--model_class", type=str, default="qwen", help="Model class (default: qwen)")

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
    if args.tokenizer_path is None:
        tok_path = args.model_path
    else:
        tok_path = args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    print(f"Loading tokenizer: {tok_path}")
    print(f"Using enforce eager: {args.enforce_eager}")
    print(f"Using GPU memory utilization: {args.gpu_memory_utilization}")
    llm = vllm.LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tp_size,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    # --- Load Prompts ---
    with args.reasoning_prompt_path.open("r") as f:
        reasoning_prompt = f.read()
    with args.summarization_prompt_path.open("r") as f:
        summarization_prompt = f.read()

    # --- Generate Rollouts ---
    config = vars(args)
    rollout_generator = ReasoningCacheRolloutGenerator(
        llm,
        tokenizer,
        reasoning_prompt,
        summarization_prompt,
        config,
        sampling_params,
        args.model_class,
    )
    all_rollouts = rollout_generator.generate_rollouts(inference_dataset)

    fields_to_save = [
        "problem",
        "label",
        "reasoning_store",
        "summarization_store",
        "problem_id",
        "sample_id",
    ]
    output_data = [{key: getattr(state, key) for key in fields_to_save} for state in all_rollouts]

    with args.output_path.open("w") as f:
        json.dump(output_data, f, indent=4)
    print(f"All {len(output_data)} filtered rollouts saved to {args.output_path}")

    # Save timing information to a separate file
    timing_output_path = args.output_path.parent / f"{args.output_path.stem}_timing.json"
    with timing_output_path.open("w") as f:
        json.dump(rollout_generator.step_timing_info, f, indent=4)
    print(f"Timing information saved to {timing_output_path}")


if __name__ == "__main__":
    args = parse_args()
    run_generation(args)
