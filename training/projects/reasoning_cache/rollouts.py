import torch
from typing import List, Dict, Any, Tuple, Callable, Optional
import copy
from collections import defaultdict
import os
import json
import uuid

from transformers import PreTrainedTokenizer
import numpy as np

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup

from projects.reasoning_cache.utils.tensor_utils import (
    concatenate_and_pad, apply_padding_to_list, apply_padding, create_position_ids, trim_right_pad_column, drop_last_columns)


class InferenceProblemState:
    """A helper class to track the inference progress for a single problem."""
    
    def __init__(
        self,
        problem: str,
        reasoning_prompt_template: str,
        summarization_prompt_template: str,
        problem_id: str,
        prompt_id: str,
        sample_id: str,
        label: str,
        starting_step: int,
        use_think_tags: bool = False,
    ):
        self.problem = problem
        self.reasoning_prompt_template = reasoning_prompt_template
        self.summarization_prompt_template = summarization_prompt_template
        self.problem_id = problem_id
        self.prompt_id = prompt_id
        self.sample_id = sample_id
        self.label = label
        self.starting_step = starting_step

        self.curr_summary = ""
        self.curr_reasoning = ""
        self.final_reward = None

        self.reasoning_rollout_store = []
        self.summarization_rollout_store = []
        self.reasoning_string_store = []
        self.summarization_string_store = []
        self.reasoning_string_complete_store = []
        self.summarization_string_complete_store = []

    def update_reasoning(self, rollouts: DataProto, response_string: str):
        self.reasoning_rollout_store.append(rollouts)
        self.reasoning_string_complete_store.append(response_string)
        processed_response_string = response_string.replace("<think>", "")
        if "</think>" in processed_response_string:
            processed_response_string = processed_response_string.split("</think>")[0]
        self.curr_reasoning = processed_response_string.strip()
        self.reasoning_string_store.append(self.curr_reasoning)

    def update_summarization(self, rollouts: DataProto, response_string: str):
        self.summarization_rollout_store.append(rollouts)
        self.summarization_string_complete_store.append(response_string)
        processed_response_string = response_string.replace("<think>", "").replace("</think>", "").strip()
        self.curr_summary = processed_response_string
        self.summarization_string_store.append(self.curr_summary)

    def get_filled_reasoning_prompt(self) -> str:
        return self.reasoning_prompt_template.format(
            problem=self.problem,
            curr_summary=self.curr_summary,
        )

    def get_filled_summarization_prompt(self) -> str:
        curr_chunk = self.curr_reasoning
        return self.summarization_prompt_template.format(
            problem=self.problem,
            existing_summary=self.curr_summary, 
            reasoning=curr_chunk.strip()
        )

    def reset_stores(self):
        self.reasoning_rollout_store = []
        self.summarization_rollout_store = []
        self.reasoning_string_store = []
        self.summarization_string_store = []
    
    def __repr__(self) -> str:
        return f"InferenceProblemState(problem_id={self.problem_id}, prompt_id={self.prompt_id}, \
            sample_id={self.sample_id}, label={self.label}, starting_step={self.starting_step})"


class ReplayBuffer:
    def __init__(self, config: Dict[str, Any]):
        self.buffer = None
        self.config = config

        buffer_path = config.reasoning_cache.get("buffer_initial_path")
        if buffer_path is not None:
            self.load_from_file(buffer_path)
        else:
            self.buffer = {}

    def save_to_file(self, filepath: str):
        """Save the replay buffer to disk in JSON format."""
        if self.buffer is None:
            raise ValueError("Buffer is not initialized")
        
        buffer_list = []
        for problem, data in self.buffer.items():
            buffer_list.append({
                "problem": problem,
                "problem_id": data["problem_id"],
                "prompt_id": data["prompt_id"],
                "sample_id": data["sample_id"],
                "label": data["label"],
                "starting_steps": data["starting_steps"],
                "summaries": data["summaries"]
            })
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(buffer_list, f, indent=4)
        print(f"Saved replay buffer with {len(buffer_list)} problems to {filepath}")

    def load_from_file(self, data_path: str):
        """Load replay buffer from a JSON file saved by save_to_file."""
        if not data_path.endswith(".json"):
            raise ValueError(f"Only JSON format is supported. Got: {data_path}")
        
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        
        if not isinstance(raw_data, list):
            raise ValueError(f"Expected list format in JSON file {data_path}")
        
        self.buffer = {}
        for item in raw_data:
            problem = item["problem"]
            summaries = item["summaries"]
            starting_steps = item["starting_steps"]
            
            if len(starting_steps) != len(summaries):
                raise ValueError(
                    f"Mismatch between number of summaries ({len(summaries)}) and "
                    f"starting_steps ({len(starting_steps)}) for problem: {problem[:50]}..."
                )
            
            self.buffer[problem] = {
                "problem_id": item["problem_id"],
                "prompt_id": item["prompt_id"],
                "sample_id": item["sample_id"],
                "label": item["label"],
                "starting_steps": starting_steps,
                "summaries": summaries
            }
        print(f"Loaded replay buffer with {len(self.buffer)} problems from {data_path}")

    def __contains__(self, problem: str) -> bool:
        return problem in self.buffer

    def __getitem__(self, problem: str) -> Dict[str, Any]:
        return self.buffer[problem]

    def sample_summaries_by_problem(self, problem: str, num_samples: int):
        if self.buffer is None:
            raise ValueError("Buffer is not loaded")
        if problem not in self.buffer:
            return []
        summaries = self.buffer[problem]["summaries"]
        starting_steps = self.buffer[problem]["starting_steps"]
        num_samples = min(num_samples, len(summaries))
        sampled_indices = np.random.choice(len(summaries), size=num_samples, replace=False)
        sampled_data = [(summaries[idx], starting_steps[idx]) for idx in sampled_indices]
        return sampled_data
    
    def purge_summaries_for_problems(self, problems: List[str]):
        if self.buffer is None:
            raise ValueError("Buffer is not loaded")
        for problem in problems:
            if problem in self.buffer:
                self.buffer[problem]["summaries"] = []
                self.buffer[problem]["starting_steps"] = []
    
    def add_states_to_buffer(self, states: List[InferenceProblemState], purge_existing: bool = False):
        if self.buffer is None:
            raise ValueError("Buffer is not loaded")
        if purge_existing:
            problem_to_states = defaultdict(list)
            for state in states:
                problem_to_states[state.problem].append(state)
            self.purge_summaries_for_problems(list(problem_to_states.keys()))
            for state in states:
                self.add_to_buffer(state)
        else:
            for state in states:
                self.add_to_buffer(state)

    def add_to_buffer(self, state: InferenceProblemState):
        if self.buffer is None:
            raise ValueError("Buffer is not loaded")
        problem = state.problem
        
        summaries_to_add = []
        starting_steps_to_add = []
        for i, summary in enumerate(state.summarization_string_store):
            summary_starting_step = state.starting_step + i + 1
            summaries_to_add.append(summary)
            starting_steps_to_add.append(summary_starting_step)
        
        if problem in self.buffer:
            self.buffer[problem]["summaries"].extend(summaries_to_add)
            self.buffer[problem]["starting_steps"].extend(starting_steps_to_add)
        else:
            self.buffer[problem] = {
                "problem_id": state.problem_id,
                "prompt_id": state.problem_id,
                "sample_id": "0",
                "label": state.label,
                "starting_steps": starting_steps_to_add,
                "summaries": summaries_to_add
            }


class ReasoningCacheRolloutGenerator:

    def __init__(
        self,
        actor_rollout_wg: RayWorkerGroup,
        tokenizer: PreTrainedTokenizer,
        reasoning_prompt_template: str,
        summarization_prompt_template: str,
        reward_function: Callable,
        config: Dict[str, Any],
        sampling_params: Dict[str, Any],
    ) -> None:
        """
        Initialize the ReasoningCacheRolloutGenerator.
        """
        self.actor_rollout_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self.reasoning_prompt_template = reasoning_prompt_template
        self.summarization_prompt_template = summarization_prompt_template
        self.config = config
        self.reward_function = reward_function

        self.max_thinking_tokens = config.reasoning_cache.get("max_thinking_tokens", 2048)
        self.max_summary_tokens = config.reasoning_cache.get("max_summary_tokens", 2048)
        self.pad_value = self.tokenizer.pad_token_id
        self.temperature = sampling_params.get("temperature", 1.0)
        self.top_p = sampling_params.get("top_p", 1.0)
        self.use_think_tags = config.reasoning_cache.get("use_think_tags", False)
        self.save_rollouts = config.reasoning_cache.get("save_rollouts", False)

        self.online_rollout_steps = config.reasoning_cache.get("online_rollout_steps", 4)
        self.online_rollout_n_samples = config.reasoning_cache.get("online_rollout_n_samples", 1)

        self.online_rollout_steps_val = config.reasoning_cache.get("online_rollout_steps_val", 4)
        self.online_rollout_n_samples_val = config.reasoning_cache.get("online_rollout_n_samples_val", 4)
        
        self.offline_rollout_steps = config.reasoning_cache.get("offline_rollout_steps", 0)
        self.offline_rollout_n_samples = config.reasoning_cache.get("offline_rollout_n_samples", 1)
        
        self.thinking_train_samples_per_online_rollout = config.reasoning_cache.get("thinking_train_samples_per_online_rollout", 1)
        self.summary_train_samples_per_online_rollout = config.reasoning_cache.get("summary_train_samples_per_online_rollout", 1)

        if self.thinking_train_samples_per_online_rollout == 0 and self.summary_train_samples_per_online_rollout == 0:
            raise ValueError("thinking_train_samples_per_online_rollout and summary_train_samples_per_online_rollout cannot both be 0")

        if self.thinking_train_samples_per_online_rollout > self.online_rollout_steps:
            raise ValueError("thinking_train_samples_per_online_rollout must be less than or equal to online_rollout_steps")
        if self.summary_train_samples_per_online_rollout > self.online_rollout_steps:
            raise ValueError("summary_train_samples_per_online_rollout must be less than or equal to online_rollout_steps")

        self.thinking_train_n_samples = config.reasoning_cache.get("thinking_train_n_samples", 1)
        self.summary_train_n_samples = config.reasoning_cache.get("summary_train_n_samples", 1)
        self.omit_online_rollout_initial_thinking_step = config.reasoning_cache.get("omit_online_rollout_initial_thinking_step", False)
        self.use_train_state_counter = config.reasoning_cache.get("use_train_state_counter", True)

        self.thinking_reward_rollout_steps = config.reasoning_cache.get("thinking_reward_rollout_steps", 0)
        self.summary_reward_rollout_steps = config.reasoning_cache.get("summary_reward_rollout_steps", 1)
        self.thinking_reward_n_samples = config.reasoning_cache.get("thinking_reward_n_samples", 1)
        self.summary_reward_n_samples = config.reasoning_cache.get("summary_reward_n_samples", 4)

        if config.reasoning_cache.get("use_replay_buffer", False):
            if self.summary_train_samples_per_online_rollout > 0:
                raise ValueError("Training on summaries is not currently supported with replay buffer")
            print(f"Using replay buffer from {config.reasoning_cache.get('replay_buffer_path', None)}")
            self.use_replay_buffer = True
            self.buffer_sample_summaries_per_problem = config.reasoning_cache.get("buffer_sample_summaries_per_problem", 1)
            self.buffer_purge_existing_on_add = config.reasoning_cache.get("buffer_purge_existing_on_add", False)
            
            self.replay_buffer = ReplayBuffer(config)
            print("Replay buffer loaded successfully.")
            print("Will use buffer to continue rollouts from existing summaries when available.")
            if self.buffer_purge_existing_on_add:
                print("Note: Existing summaries will be purged when adding new summaries for the same problem.")
        else:
            self.replay_buffer = None
            self.use_replay_buffer = False
            self.buffer_purge_existing_on_add = False
    
    def save_replay_buffer(self, filepath: str):
        if not self.use_replay_buffer or self.replay_buffer is None:
            print("Replay buffer is not enabled or not initialized. Skipping save.")
            return
        if not filepath.endswith(".json"):
            filepath = filepath + ".json"
        self.replay_buffer.save_to_file(filepath)
        
    def generate_rollouts(
        self,
        prompts_batch: DataProto,
        eval_mode: bool = False,
    ) -> Tuple[DataProto, Dict[str, Any]]:
        """
        Generate rollouts for reasoning cache.
        """
        # Prepare online rollout states and prompts
        problems, problem_ids, labels = self.extract_problems_and_labels(prompts_batch)
        online_rollout_states = self.prepare_online_rollout_states(problems, problem_ids, labels)
        
        # Generate online rollouts for all problems in the batch
        print("Generating online rollouts...")
        if eval_mode:
            online_rollout_states = self.rollout_step_thinking(online_rollout_states, n=self.online_rollout_n_samples_val)
            num_online_rollout_steps = self.online_rollout_steps_val
        else:
            online_rollout_states = self.rollout_step_thinking(online_rollout_states, n=self.online_rollout_n_samples)
            num_online_rollout_steps = self.online_rollout_steps
        
        online_rollout_states = self.rollout_step_summarization(online_rollout_states)
        for step in range(1, num_online_rollout_steps):
            online_rollout_states = self.rollout_step_thinking(online_rollout_states)
            online_rollout_states = self.rollout_step_summarization(online_rollout_states)
        
        online_rollout_metrics = self.compute_online_rollout_metrics(online_rollout_states)
        if eval_mode:
            if self.save_rollouts:
                self.save_online_rollouts(online_rollout_states)
            return None, online_rollout_metrics
        
        # Add all rollouts to buffer if enabled
        if self.use_replay_buffer:
            print("Adding rollouts to buffer...")
            self.replay_buffer.add_states_to_buffer(online_rollout_states, purge_existing=self.buffer_purge_existing_on_add)
            num_problems_in_buffer = len(self.replay_buffer.buffer)
            print(f"Buffer now contains {num_problems_in_buffer} unique problems.")
        
        # Prepare training states and prompts
        thinking_train_states, summary_train_states = self.postprocess_online_rollout_states(online_rollout_states)
        print(f"Num thinking train states: {len(thinking_train_states)}")
        print(f"Num summary train states: {len(summary_train_states)}")

        # Generate training rollouts
        print("Generating train rollouts...")
        thinking_train_states = self.rollout_step_thinking(thinking_train_states, n=self.thinking_train_n_samples)
        summary_train_states = self.rollout_step_summarization(summary_train_states, n=self.summary_train_n_samples)

        # Prepare reward states and prompts
        thinking_reward_states, summary_reward_states = self.postprocess_train_states(thinking_train_states, summary_train_states)

        # Generate reward rollouts
        print("Generating thinking reward rollouts...")
        if self.thinking_reward_rollout_steps > 0 and len(thinking_train_states) > 0:
            thinking_reward_states = self.rollout_step_summarization(
                thinking_reward_states, n=self.thinking_reward_n_samples)
            thinking_reward_states = self.rollout_step_thinking(thinking_reward_states)
            for step in range(1, self.thinking_reward_rollout_steps):
                thinking_reward_states = self.rollout_step_summarization(thinking_reward_states)
                thinking_reward_states = self.rollout_step_thinking(thinking_reward_states)
            thinking_train_states = self.postprocess_train_reward_states(thinking_train_states, thinking_reward_states)
        else:
            thinking_train_states = self.postprocess_train_reward_states(thinking_train_states, None)

        print("Generating summary reward rollouts...")
        if len(summary_train_states) > 0:
            summary_reward_states = self.rollout_step_thinking(summary_reward_states, n=self.summary_reward_n_samples)
            if self.summary_reward_rollout_steps > 0:
                for step in range(1, self.summary_reward_rollout_steps):
                    summary_reward_states = self.rollout_step_summarization(summary_reward_states)
                    summary_reward_states = self.rollout_step_thinking(summary_reward_states)
            summary_train_states = self.postprocess_train_reward_states(summary_train_states, summary_reward_states)

        # Postprocess train states and obtain DataProto for training
        merged_data_proto = self.postprocess_completed_states(thinking_train_states, summary_train_states)
        print("Done generating rollouts.")
        return merged_data_proto, online_rollout_metrics

    def save_online_rollouts(self, online_rollout_states: List[InferenceProblemState]):
        fields = [
            "problem",
            "label",
            "problem_id",
            "prompt_id",
            "sample_id",
            "reasoning_string_store",
            "summarization_string_store",
            "reasoning_string_complete_store",
            "summarization_string_complete_store"
            ]
        serialized = [{field: getattr(state, field) for field in fields} for state in online_rollout_states]
        save_dir = self.config.trainer.get("rollout_data_dir", None)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "online_val_rollouts.json"), "w") as f:
            json.dump(serialized, f, indent=4)

    def run_inference(
        self, 
        prompts_batch: DataProto, 
        n: int,
        max_length: int,
    ) -> Tuple[DataProto, Dict[str, Any]]:
        """
        Generate rollouts for reasoning cache.
        """
        prompts_batch_padded, pad_size = pad_dataproto_to_divisor(
            prompts_batch, self.actor_rollout_wg.world_size
        )
        prompts_batch_padded.meta_info["sampling_params"] = {
                "n": n,
                "max_tokens": max_length,
        }
        rollouts = self.actor_rollout_wg.generate_sequences(
            prompts_batch_padded
        )
        rollouts = unpad_dataproto(rollouts, pad_size * n)
        return rollouts

    def prepare_for_inference(self, prompts: List[str], enable_thinking: bool):
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        filled_prompts_with_chat_template = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
            )
        if enable_thinking:
            filled_prompts_with_chat_template = [p + "<think>" if "<think>" not in p else p for p in filled_prompts_with_chat_template]
        tokenized_prompts = self.tokenizer(
            filled_prompts_with_chat_template,
            add_special_tokens=False,
        )
        tokenized_prompts = [torch.tensor(t) for t in tokenized_prompts["input_ids"]]
        tokenized_prompts, attention_mask = apply_padding(
            tokenized_prompts, self.pad_value, direction="left"
        )
        position_ids = create_position_ids(attention_mask)

        inference_data_proto = DataProto.from_dict(
            tensors={
                "input_ids": tokenized_prompts,
                "attention_mask": attention_mask,
                "position_ids": position_ids
            }
        )
        return inference_data_proto
    
    def extract_problems_and_labels(self, prompts_batch: DataProto):
        raw_prompts = prompts_batch.non_tensor_batch["raw_prompt"]
        problems = [
            s[0]["content"] for s in raw_prompts
        ]        
        problem_ids = prompts_batch.non_tensor_batch["problem_ids"]
        label_dicts = prompts_batch.non_tensor_batch["reward_model"]
        labels = np.array([label_dict["ground_truth"] for label_dict in label_dicts])

        return problems, problem_ids, labels

    def create_snapshot_state(self, existing_state, thinking_ind, summary_ind):
        thinking_strings = existing_state.reasoning_string_store
        summarization_strings = existing_state.summarization_string_store

        snapshot = {
            "problem": existing_state.problem,
            "problem_id": existing_state.problem_id,
            "prompt_id": f"{existing_state.prompt_id}_{existing_state.sample_id}",
            "sample_id": "0",
            "label": existing_state.label,
            "starting_step": existing_state.starting_step,
        }

        snapshot_state = InferenceProblemState(
            **snapshot,
            reasoning_prompt_template=self.reasoning_prompt_template,
            summarization_prompt_template=self.summarization_prompt_template,
            use_think_tags=self.use_think_tags
        )

        if len(thinking_strings) > 0:
            snapshot_state.curr_reasoning = thinking_strings[thinking_ind - 1] if thinking_ind > 0 else ""
        if len(summarization_strings) > 0:
            snapshot_state.curr_summary = summarization_strings[summary_ind - 1] if summary_ind > 0 else ""
        return snapshot_state
    
    def prepare_online_rollout_states(
        self,
        problems: List[str],
        problem_ids: List[str],
        labels: List[str],
    ) -> List[InferenceProblemState]:
        """
        Prepare active states for reasoning cache.
        If replay buffer is enabled, checks each problem and initializes with sampled summary if available.
        """
        states = []
        num_from_buffer = 0
        num_from_scratch = 0
        
        for i in range(len(problems)):
            problem = problems[i]
            problem_id = problem_ids[i]
            label = labels[i]
            
            state_data = {
                "problem_id": f"{problem_id}",
                "prompt_id": f"{problem_id}",
                "sample_id": "0",
                "problem": problem,
                "label": label,
            }
            
            if self.use_replay_buffer and problem in self.replay_buffer:
                sampled_data = self.replay_buffer.sample_summaries_by_problem(
                    problem, self.buffer_sample_summaries_per_problem
                )
                
                for summary, summary_starting_step in sampled_data:
                    starting_step = summary_starting_step
                    state = InferenceProblemState(
                        **state_data,
                        reasoning_prompt_template=self.reasoning_prompt_template,
                        summarization_prompt_template=self.summarization_prompt_template,
                        use_think_tags=self.use_think_tags,
                        starting_step=starting_step
                    )
                    state.curr_summary = summary
                    states.append(state)
                    num_from_buffer += 1
            else:
                state = InferenceProblemState(
                    **state_data,
                    reasoning_prompt_template=self.reasoning_prompt_template,
                    summarization_prompt_template=self.summarization_prompt_template,
                    use_think_tags=self.use_think_tags,
                    starting_step=0
                )
                states.append(state)
                num_from_scratch += 1
        
        if self.use_replay_buffer:
            print(f"Prepared {len(states)} rollout states: {num_from_buffer} from buffer, {num_from_scratch} from scratch")
        return states

    def postprocess_online_rollout_states(self, online_rollout_states: List[InferenceProblemState]):
        """
        Sample intermediate states from online rollouts to create training inputs.
        """
        thinking_train_states = []
        summary_train_states = []
        train_state_counter = 0
        
        for state in online_rollout_states:
            num_thinking_steps = min(len(state.reasoning_string_store), self.thinking_train_samples_per_online_rollout)
            if self.omit_online_rollout_initial_thinking_step:
                assert num_thinking_steps > 1, "num_thinking_steps must be > 1 if omit_online_rollout_initial_thinking_step is True"
                all_thinking_indices = list(range(1, len(state.reasoning_string_store)))
                num_thinking_steps = min(num_thinking_steps, len(all_thinking_indices))
            else:
                all_thinking_indices = list(range(len(state.reasoning_string_store)))
            if num_thinking_steps > 0:
                sampled_thinking_indices = np.random.choice(
                    all_thinking_indices,
                    size=num_thinking_steps, 
                    replace=False
                )
                for ind in sampled_thinking_indices:
                    snapshot_state = self.create_snapshot_state(state, thinking_ind=ind, summary_ind=ind)
                    if self.use_train_state_counter:
                        snapshot_state.prompt_id = f"{snapshot_state.prompt_id}_{train_state_counter}"
                        train_state_counter += 1
                    thinking_train_states.append(snapshot_state)

            num_summary_steps = min(len(state.summarization_string_store), self.summary_train_samples_per_online_rollout)
            if num_summary_steps > 0:
                sampled_summary_indices = np.random.choice(
                    len(state.summarization_string_store),
                    size=num_summary_steps,
                    replace=False
                )
                for ind in sampled_summary_indices:
                    snapshot_state = self.create_snapshot_state(state, thinking_ind=ind + 1, summary_ind=ind)
                    snapshot_state.prompt_id = f"{snapshot_state.prompt_id}_{train_state_counter}"
                    train_state_counter += 1
                    summary_train_states.append(snapshot_state)
                
        return thinking_train_states, summary_train_states
        
    def postprocess_train_states(
        self, 
        thinking_train_states: List[InferenceProblemState], 
        summary_train_states: List[InferenceProblemState]
    ):
        """
        Create reward rollout inputs from training states.
        """
        thinking_reward_states = []
        summary_reward_states = []
        for state in thinking_train_states:
            thinking_reward_states.append(self.create_snapshot_state(state, thinking_ind=1, summary_ind=0))
        for state in summary_train_states:
            summary_reward_states.append(self.create_snapshot_state(state, thinking_ind=1, summary_ind=1))
        return thinking_reward_states, summary_reward_states

    def postprocess_train_reward_states(
        self, 
        train_states: List[InferenceProblemState], 
        reward_states: Optional[List[InferenceProblemState]] = None
    ):
        """
        Compute and then assign train state rewards using reward states.
        """
        reward_dict = defaultdict(list)
        if reward_states is None:
            for state in train_states:
                final_reasoning_string = state.reasoning_string_complete_store[-1]
                score = self.reward_function(final_reasoning_string, state.label)
                state.final_reward = score
        else:
            for state in reward_states:
                final_reasoning_string = state.reasoning_string_complete_store[-1]
                score = self.reward_function(final_reasoning_string, state.label)
                reward_dict[state.prompt_id].append(score)
            for key, val in reward_dict.items():
                reward_dict[key] = np.mean(val)
            for state in train_states:
                state_reward = reward_dict[f"{state.prompt_id}_{state.sample_id}"]
                state.final_reward = state_reward
        return train_states

    def broadcast_states(self, active_states, n):
        broadcasted_states = []
        for i, state in enumerate(active_states):
            for j in range(n):
                curr_state = copy.deepcopy(state)
                curr_state.sample_id = f"{j}"
                broadcasted_states.append(curr_state) 
        return broadcasted_states

    def reasoning_rollout_postprocess(
        self, 
        rollouts: DataProto, 
        active_states: List[InferenceProblemState], 
        n: int
    ):
        decoded_responses = self.tokenizer.batch_decode(rollouts.batch["responses"], skip_special_tokens=True)
        assert len(decoded_responses) == len(active_states) * n
        if n > 1:
            active_states = self.broadcast_states(active_states, n)

        for i, state in enumerate(active_states):
            state.update_reasoning(rollouts[i], decoded_responses[i])
        return active_states

    def summarization_rollout_postprocess(
        self, 
        rollouts: DataProto, 
        active_states: List[InferenceProblemState],
        n: int
    ):
        decoded_responses = self.tokenizer.batch_decode(rollouts.batch["responses"], skip_special_tokens=True)
        assert len(decoded_responses) == len(active_states) * n
        if n > 1:
            active_states = self.broadcast_states(active_states, n)

        for i, state in enumerate(active_states):
            state.update_summarization(rollouts[i], decoded_responses[i])
        return active_states
    
    def rollout_step_thinking(
        self, 
        states: List[InferenceProblemState],
        n: int = 1
    ):
        if len(states) == 0:
            return states
        filled_prompts = [state.get_filled_reasoning_prompt() for state in states]
        inference_data_proto = self.prepare_for_inference(filled_prompts, enable_thinking=self.use_think_tags)
        rollouts = self.run_inference(inference_data_proto, n=n, max_length=self.max_thinking_tokens)
        states = self.reasoning_rollout_postprocess(rollouts, states, n=n)
        return states

    def rollout_step_summarization(
        self, 
        states: List[InferenceProblemState],
        n: int = 1
    ):
        if len(states) == 0:
            return states
        filled_prompts = [state.get_filled_summarization_prompt() for state in states]
        inference_data_proto = self.prepare_for_inference(filled_prompts, enable_thinking=False)
        rollouts = self.run_inference(inference_data_proto, n=n, max_length=self.max_summary_tokens)
        states = self.summarization_rollout_postprocess(rollouts, states, n=n)
        return states

    def compute_online_rollout_metrics(
        self,
        online_rollout_states: List[InferenceProblemState],
    ):
        """
        Compute metrics for online rollouts.
        """
        initial_scores = []
        final_scores = []
        problem_ids = [state.problem_id for state in online_rollout_states]
        
        for state in online_rollout_states:
            initial_reasoning_string = state.reasoning_string_complete_store[0]
            final_reasoning_string = state.reasoning_string_complete_store[-1]
            initial_reasoning_score = self.reward_function(initial_reasoning_string, state.label)
            final_reasoning_score = self.reward_function(final_reasoning_string, state.label)
            initial_scores.append(initial_reasoning_score)
            final_scores.append(final_reasoning_score)

        problem_score_dict = defaultdict(list)
        for p_id, score in zip(problem_ids, final_scores):
            problem_score_dict[p_id].append(score)

        scores_by_problem = list(problem_score_dict.values())
        zero_adv_zero_by_problem = [1 if any(x == 1 for x in score_list) else 0 for score_list in scores_by_problem]

        metrics = {
            "online_rollout_initial_score_mean": np.mean(initial_scores), 
            "online_rollout_final_score_mean": np.mean(final_scores),
            "online_rollout_final_score_bon_mean": np.mean(zero_adv_zero_by_problem),
        }
        return metrics

    def merge_data(
        self,
        data_protos: List[DataProto],
    ) -> Dict[str, torch.Tensor]:
        """
        Merge data protos.
        """
        prompt_tensors = [dp.batch["prompts"] for dp in data_protos]
        padded_prompt_tensor, left_pad_amounts = concatenate_and_pad(
            prompt_tensors, self.pad_value, padding_side="left")
        response_tensors = [dp.batch["responses"] for dp in data_protos]
        padded_response_tensor, right_pad_amounts = concatenate_and_pad(
            response_tensors, self.pad_value, padding_side="right")
        
        input_ids_tensors = [dp.batch["input_ids"] for dp in data_protos]
        attention_mask_tensors = [dp.batch["attention_mask"] for dp in data_protos]

        def pad_both_sides(tensors, pad_amounts_left, pad_amounts_right, pad_value):
            padded_tensors = apply_padding_to_list(
                tensors, pad_amounts_left, pad_value, padding_side="left")
            padded_tensors = apply_padding_to_list(
                padded_tensors, pad_amounts_right, pad_value, padding_side="right")
            return torch.stack(padded_tensors)
        
        padded_input_ids_tensors = pad_both_sides(
            input_ids_tensors, left_pad_amounts, right_pad_amounts, self.pad_value)
        padded_attention_mask_tensors = pad_both_sides(
            attention_mask_tensors, left_pad_amounts, right_pad_amounts, 0)

        padded_position_ids_tensors = create_position_ids(padded_attention_mask_tensors)

        padded_response_tensor, num_dropped = trim_right_pad_column(padded_response_tensor, self.pad_value)
        padded_position_ids_tensors = drop_last_columns(padded_position_ids_tensors, num_dropped)
        padded_attention_mask_tensors = drop_last_columns(padded_attention_mask_tensors, num_dropped)
        padded_input_ids_tensors = drop_last_columns(padded_input_ids_tensors, num_dropped)

        assert padded_response_tensor.shape[-1] + padded_prompt_tensor.shape[-1] == padded_input_ids_tensors.shape[-1]
        assert padded_input_ids_tensors.shape[-1] == padded_position_ids_tensors.shape[-1]
        assert padded_input_ids_tensors.shape[-1] == padded_attention_mask_tensors.shape[-1]

        output_dict = {
            "input_ids": padded_input_ids_tensors,
            "position_ids": padded_position_ids_tensors,
            "attention_mask": padded_attention_mask_tensors,
            "prompts": padded_prompt_tensor,
            "responses": padded_response_tensor
        }
        return output_dict

    def postprocess_completed_states(
        self, 
        thinking_train_states: List[InferenceProblemState], 
        summary_train_states: List[InferenceProblemState]
    ):
        reasoning_data_protos = [state.reasoning_rollout_store[-1] for state in thinking_train_states]
        summarization_data_protos = [state.summarization_rollout_store[-1] for state in summary_train_states]
        combined_data_protos = reasoning_data_protos + summarization_data_protos
        combined_states = thinking_train_states + summary_train_states

        prompt_ids_thinking = [f"R-{state.prompt_id}" for state in thinking_train_states]
        prompt_ids_summary = [f"S-{state.prompt_id}" for state in summary_train_states]
        prompt_ids = prompt_ids_thinking + prompt_ids_summary
        sample_ids = [state.sample_id for state in combined_states]
        problem_ids = [state.problem_id for state in combined_states]
        labels = [state.label for state in combined_states]
        rewards = [state.final_reward for state in combined_states]
        starting_step = [state.starting_step for state in combined_states]

        summary_indicator = [0] * len(thinking_train_states) + [1] * len(summary_train_states)
        summary_indicator = torch.as_tensor(summary_indicator)

        merged_rollout_data_dict = self.merge_data(combined_data_protos)
        merged_rollout_data_dict["summary_indicator"] = summary_indicator
        meta_dict = {"prompt_ids": prompt_ids,
                    "problem_ids": problem_ids,
                    "sample_ids": sample_ids,
                    "labels": labels,
                    "rewards": rewards,
                    "starting_step": starting_step,
                    }
        
        merged_data_proto = DataProto.from_dict(tensors=merged_rollout_data_dict, non_tensors=meta_dict)
        print(f"Merged data proto: {merged_data_proto.batch}")
        if self.save_rollouts:
            save_dir = self.config.trainer.get("rollout_data_dir", None)
            merged_data_proto.save_to_disk(os.path.join(save_dir, "latest_rollout_data_proto.pkl"))

        return merged_data_proto
        