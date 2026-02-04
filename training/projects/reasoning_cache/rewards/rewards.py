from typing import Dict, Union, Tuple, Any, Callable
from collections import defaultdict

from transformers import PreTrainedTokenizer

from verl.protocol import DataProto



class RewardProcessor:
    """
    Handles non task-specific rewards e.g. format rewards.

    This class processes and combines task-specific rewards with format-based rewards.
    It enforces formatting requirements like think tags and verification phrases,
    applying penalties or zeroing rewards based on configuration settings.

    Attributes:
        task_reward_func: Function that computes task-specific rewards
        config: Configuration object containing reward and verification settings
        end_think_token_id: Token that marks the end of thinking phase
        start_verification_tag: Tag that marks the start of verification
        end_verification_tag: Tag that marks the end of verification
    """

    def __init__(self, task_reward_func: Callable, tokenizer: PreTrainedTokenizer, config: Any) -> None:
        """
        Initialize the RewardProcessor.
        """
        self.task_reward_func = task_reward_func
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, data: DataProto) -> Tuple[Dict[str, Union[str, bool, float]], Dict[str, Union[str, bool, float]]]:
        reward_dict = {}
        score_dict = {}
        bonus_dict = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]
            if not data_item.non_tensor_batch['is_completion']:
                continue

            response_tensor = data_item.batch['responses']
            response_str = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
            ground_truth = str(data_item.non_tensor_batch['labels'])
            score = self.task_reward_func(
                response_str,
                ground_truth,
            )
            boxed_present = "oxed{" in response_str

            # Auxilliary bonus/penalties go here
            # reward is the final reward for the sample
            # score is the correctness score for the sample
            reward = score
            completion_steps = data_item.non_tensor_batch['completion_steps']
            exploration_bonus = self.apply_exploration_bonus(completion_steps, score, boxed_present)
            bonus_dict["exploration_bonus"].append(exploration_bonus)
            reward = reward + exploration_bonus
            # End of auxilliary bonus/penalties

            problem_id = data_item.non_tensor_batch['problem_ids']
            sample_id = data_item.non_tensor_batch['sample_ids']
            if problem_id not in score_dict:
                score_dict[problem_id] = {}
                reward_dict[problem_id] = {}
            score_dict[problem_id][sample_id] = score
            reward_dict[problem_id][sample_id] = reward

        return score_dict, reward_dict, bonus_dict

    def apply_exploration_bonus(self, completion_steps, score, boxed_present):
        exploration_bonus = self.config.reasoning_cache.get("exploration_bonus", 0.0)
        min_steps_for_bonus = self.config.reasoning_cache.get("min_steps_for_bonus", 0)
        bonus_correct_only = self.config.reasoning_cache.get("bonus_correct_only", True)
        bonus_boxed_only = self.config.reasoning_cache.get("bonus_boxed_only", False)
        max_steps = self.config.reasoning_cache.get("max_steps", 8)

        if exploration_bonus > 0:
            if bonus_correct_only and score != 1:
                return 0
            if bonus_boxed_only and not boxed_present:
                return 0
            if completion_steps < min_steps_for_bonus:
                bonus_reward = 0 
            else:
                bonus_reward = exploration_bonus * (completion_steps / (max_steps - 1))
            return bonus_reward
        return 0