import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        processed_response_string = response_string.replace("<think>", "")
        if "</think>" in processed_response_string:
            processed_response_string = processed_response_string.split("</think>")[0]
        self.curr_reasoning = processed_response_string.strip()

    def update_summarization(self, response_string: str):
        """Updates the state with new summarization output."""
        self.summarization_store.append(response_string)
        processed_response_string = response_string.replace("<think>", "").replace("</think>", "").strip()
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
    Manages the generation of rollouts using separate reasoning and summarization models.

    Attributes:
        reasoning_llm_client (vllm.LLM): vLLM client for reasoning.
        reasoning_tokenizer (PreTrainedTokenizer): Tokenizer for the reasoning model.
        summarization_llm_client (vllm.LLM): vLLM client for summarization.
        summarization_tokenizer (PreTrainedTokenizer): Tokenizer for the summarization model.
        reasoning_prompt_template (str): Template for reasoning prompts.
        summarization_prompt_template (str): Template for summarization prompts.
        config (Dict): Configuration parameters.
        max_steps (int): Maximum steps per rollout.
        max_thinking_tokens (int): Max tokens for reasoning.
        max_summary_tokens (int): Max tokens for summarization.
        reasoning_prompt_style (str): Style of reasoning prompt.
        use_think_tags (bool): Whether to use <think> tags.
        summarization_style (str): Style of summarization.
        reasoning_sampling_params (vllm.SamplingParams): Sampling params for reasoning.
        summarization_sampling_params (vllm.SamplingParams): Sampling params for summarization.
        n_samples_per_problem (int): Number of samples per problem.
        using_separate_models (bool): Whether different models are used for reasoning and summarization.
    """

    def __init__(
        self,
        reasoning_llm_client: vllm.LLM,
        reasoning_tokenizer: PreTrainedTokenizer,
        summarization_llm_client: vllm.LLM,
        summarization_tokenizer: PreTrainedTokenizer,
        reasoning_prompt_template: str,
        summarization_prompt_template: str,
        config: Dict[str, Any],
        reasoning_sampling_params: vllm.SamplingParams,
        summarization_sampling_params: vllm.SamplingParams,
    ) -> None:
        """
        Initialize the ReasoningCacheRolloutGenerator.

        Args:
            reasoning_llm_client: vLLM client for reasoning generation
            reasoning_tokenizer: Tokenizer for reasoning model
            summarization_llm_client: vLLM client for summarization (can be same as reasoning)
            summarization_tokenizer: Tokenizer for summarization model (can be same as reasoning)
            reasoning_prompt_template: Template for reasoning prompts
            summarization_prompt_template: Template for summarization prompts
            config: Configuration dictionary
            reasoning_sampling_params: Sampling parameters for reasoning generation
            summarization_sampling_params: Sampling parameters for summarization generation
        """
        self.reasoning_llm_client = reasoning_llm_client
        self.reasoning_tokenizer = reasoning_tokenizer
        self.summarization_llm_client = summarization_llm_client
        self.summarization_tokenizer = summarization_tokenizer
        self.tokenizer = reasoning_tokenizer

        self.reasoning_prompt_template = reasoning_prompt_template
        self.summarization_prompt_template = summarization_prompt_template
        self.config = config
        self.max_steps = config.get("max_steps", 2)
        self.max_thinking_tokens = config.get("max_thinking_tokens", 8192)
        self.max_summary_tokens = config.get("max_summarization_tokens", 2048)
        # Use default styles: structured for reasoning, summ for summarization
        self.use_think_tags = config.get("use_think_tags", False)
        self.reasoning_sampling_params = reasoning_sampling_params
        self.summarization_sampling_params = summarization_sampling_params
        self.n_samples_per_problem = config.get("n", 4)

        self.using_separate_models = reasoning_llm_client != summarization_llm_client

    def generate_rollouts(
        self,
        prompts_batch: List[Dict[str, Any]],
    ) -> List[InferenceProblemState]:
        """
        Generate rollouts for reasoning cache.
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

    def run_reasoning_inference(
        self,
        prompts: List[str],
        n: int,
        max_length: int,
    ) -> List[str]:
        """
        Generate reasoning sequences by calling the reasoning vLLM engine.
        """
        sampling_params = self.reasoning_sampling_params.clone()
        sampling_params.n = n
        sampling_params.max_tokens = max_length
        vllm_output = self.reasoning_llm_client.generate(prompts, sampling_params)
        return [output.text for request_output in vllm_output for output in request_output.outputs]

    def run_summarization_inference(
        self,
        prompts: List[str],
        n: int,
        max_length: int,
    ) -> List[str]:
        """
        Generate summarization sequences by calling the summarization vLLM engine.
        """
        sampling_params = self.summarization_sampling_params.clone()
        sampling_params.n = n
        sampling_params.max_tokens = max_length
        vllm_output = self.summarization_llm_client.generate(prompts, sampling_params)
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
        templated_problems = self.reasoning_tokenizer.apply_chat_template(
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
            )
            for raw_prompt_with_id in raw_prompts_with_ids
        ]
        return active_states

    def prepare_for_reasoning_inference(self, filled_prompts: List[str], apply_template: bool, enable_thinking: bool):
        """Prepare prompts for reasoning inference using the reasoning tokenizer."""
        if apply_template:
            messages = [[{"role": "user", "content": filled_prompt}] for filled_prompt in filled_prompts]
            return self.reasoning_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
            )
        else:
            return filled_prompts

    def prepare_for_summarization_inference(
        self, filled_prompts: List[str], apply_template: bool, enable_thinking: bool
    ):
        """Prepare prompts for summarization inference using the summarization tokenizer."""
        if apply_template:
            messages = [[{"role": "user", "content": filled_prompt}] for filled_prompt in filled_prompts]
            return self.summarization_tokenizer.apply_chat_template(
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

        # Reasoning phase
        filled_prompts = [
            self.reasoning_prompt_template.format(problem=problem, curr_summary="") for problem in problems
        ]
        inference_data_proto = self.prepare_for_reasoning_inference(
            filled_prompts, apply_template=True, enable_thinking=True
        )

        rollouts = self.run_reasoning_inference(
            inference_data_proto, n=self.n_samples_per_problem, max_length=self.max_thinking_tokens
        )
        active_states = self.reasoning_rollout_postprocess(rollouts, active_states)

        # Summarization phase
        filled_prompts = [state.get_filled_summarization_prompt() for state in active_states]
        inference_data_proto = self.prepare_for_summarization_inference(
            filled_prompts, apply_template=True, enable_thinking=False
        )
        rollouts = self.run_summarization_inference(inference_data_proto, n=1, max_length=self.max_summary_tokens)
        active_states, completed_states = self.summarization_rollout_postprocess(
            rollouts, active_states, completed_states
        )
        return active_states, completed_states

    def rollout_step(
        self, active_states: List[InferenceProblemState], completed_states: List[InferenceProblemState]
    ) -> Tuple[List[InferenceProblemState], List[InferenceProblemState]]:
        """Performs a single intermediate step of the reasoning-summarization rollout."""
        # Reasoning phase
        filled_prompts = [state.get_filled_reasoning_prompt(self.reasoning_tokenizer) for state in active_states]
        final_prompts = self.prepare_for_reasoning_inference(filled_prompts, apply_template=False, enable_thinking=True)

        for i, state in enumerate(active_states):
            state.prompt_store.append(final_prompts[i])

        rollouts = self.run_reasoning_inference(final_prompts, n=1, max_length=self.max_thinking_tokens)
        active_states = self.reasoning_rollout_postprocess(rollouts, active_states)

        # Summarization phase
        filled_prompts = [state.get_filled_summarization_prompt() for state in active_states]
        final_prompts = self.prepare_for_summarization_inference(
            filled_prompts, apply_template=True, enable_thinking=False
        )
        rollouts = self.run_summarization_inference(final_prompts, n=1, max_length=self.max_summary_tokens)
        active_states, completed_states = self.summarization_rollout_postprocess(
            rollouts, active_states, completed_states
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
        "--model_path",
        type=str,
        required=True,
        help="Path or name of the HuggingFace model to use for reasoning with vLLM",
    )
    parser.add_argument(
        "--summarization_model_path",
        type=str,
        default=None,
        help="Path or name of the HuggingFace model to use for summarization. If not provided, uses the same model as reasoning.",
    )
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for reasoning model (default: 1)")
    parser.add_argument(
        "--summarization_tp_size",
        type=int,
        default=None,
        help="Tensor parallel size for summarization model. If not provided, uses the same as --tp_size.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    # --- Slicing and Batching ---
    parser.add_argument("--start_index", type=int, default=0, help="Starting index of samples to process (default: 0)")
    parser.add_argument(
        "--end_index", type=int, default=None, help="Ending index of samples to process (exclusive, default: all)"
    )

    # --- Generation Parameters ---
    parser.add_argument("--n", type=int, default=4, help="Number of completions to generate per prompt (default: 4)")
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature for reasoning (default: 0.6)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p sampling parameter for reasoning (default: 0.95)"
    )
    parser.add_argument(
        "--summarization_temperature",
        type=float,
        default=None,
        help="Sampling temperature for summarization. If not provided, uses --temperature (default: None)",
    )
    parser.add_argument(
        "--summarization_top_p",
        type=float,
        default=None,
        help="Top-p sampling parameter for summarization. If not provided, uses --top_p (default: None)",
    )
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
    parser.add_argument(
        "--max_model_len_thinking",
        type=int,
        default=65536,
        help="Max model length for reasoning steps (default: 65536)",
    )
    parser.add_argument(
        "--max_model_len_summarization",
        type=int,
        default=65536,
        help="Max model length for summarization steps (default: 65536)",
    )
    parser.add_argument(
        "--gpu_memory_utilization_thinking",
        type=float,
        default=0.4,
        help="GPU memory utilization for reasoning steps (default: 0.4)",
    )
    parser.add_argument(
        "--gpu_memory_utilization_summarization",
        type=float,
        default=0.6,
        help="GPU memory utilization for summarization steps (default: 0.6)",
    )
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

    # --- Determine model configurations ---
    use_separate_models = args.summarization_model_path is not None
    summarization_model_path = args.summarization_model_path if use_separate_models else args.model_path
    summarization_tp_size = args.summarization_tp_size if args.summarization_tp_size is not None else args.tp_size

    # --- Initialize Models and Tokenizers ---
    print(f"Loading reasoning model: {args.model_path} with tensor_parallel_size={args.tp_size}")
    reasoning_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if use_separate_models:
        reasoning_llm = vllm.LLM(
            model=args.model_path,
            dtype="bfloat16",
            tensor_parallel_size=args.tp_size,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization_thinking,
            max_model_len=args.max_model_len_thinking,
        )
    else:
        reasoning_llm = vllm.LLM(
            model=args.model_path,
            dtype="bfloat16",
            tensor_parallel_size=args.tp_size,
            seed=args.seed,
        )

    print("Waiting for 30 seconds to ensure the model is loaded...")
    import time

    time.sleep(30)

    if use_separate_models:
        print(
            f"Loading summarization model: {summarization_model_path} with tensor_parallel_size={summarization_tp_size}"
        )
        summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_path)
        summarization_llm = vllm.LLM(
            model=summarization_model_path,
            dtype="bfloat16",
            tensor_parallel_size=args.tp_size,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization_summarization,
            max_model_len=args.max_model_len_summarization,
        )
    else:
        print(f"Using the same model for both reasoning and summarization")
        summarization_tokenizer = reasoning_tokenizer
        summarization_llm = reasoning_llm

    # Set default summarization parameters if not provided
    summarization_temperature = (
        args.summarization_temperature if args.summarization_temperature is not None else args.temperature
    )
    summarization_top_p = args.summarization_top_p if args.summarization_top_p is not None else args.top_p

    reasoning_sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )

    summarization_sampling_params = vllm.SamplingParams(
        temperature=summarization_temperature,
        top_p=summarization_top_p,
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
        reasoning_llm,
        reasoning_tokenizer,
        summarization_llm,
        summarization_tokenizer,
        reasoning_prompt,
        summarization_prompt,
        config,
        reasoning_sampling_params,
        summarization_sampling_params,
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


if __name__ == "__main__":
    args = parse_args()
    run_generation(args)
