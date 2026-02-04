from typing import Dict

import torch
import numpy as np

from verl.protocol import DataProto
from projects.reasoning_cache.utils.parsing_utils import build_reward_dict
        

def compute_advantage(
    data: DataProto
):
    with torch.no_grad():
        reward_result = build_reward_dict(data, "prompt_ids", "sample_ids", "rewards")
        advantage_dict = compute_grpo_advantage_dict(reward_result)
        print("Advantage dict length:", len(advantage_dict))
        print("Advantage dict num elements:", sum(len(v) for v in advantage_dict.values()))
        advantage_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        response_mask = data.batch["response_mask"]

        for i in range(len(data)):
            data_item = data[i]
            prompt_id = data_item.non_tensor_batch['prompt_ids']
            sample_id = data_item.non_tensor_batch['sample_ids']
            advantage = advantage_dict[prompt_id][sample_id]
            advantage_tensor[i, :] = advantage
        advantage_tensor = advantage_tensor * response_mask

        data.batch['advantages'] = advantage_tensor
        data.batch['returns'] = advantage_tensor

    return data

    
def compute_grpo_advantage_dict(
    reward_result: Dict[str, float], 
    epsilon: float = 1e-6
):
    advantage_dict = {}
    for prompt_id, sample_id_to_score in reward_result.items():
        advantage_dict[prompt_id] = {}
        sample_scores = list(sample_id_to_score.values())
        if len(sample_scores) == 1:
            mean = 0
            std = 1
        else:
            mean = np.mean(sample_scores)
            std = np.std(sample_scores)
        for sample_id, score in sample_id_to_score.items():
            advantage_dict[prompt_id][sample_id] = (score - mean) / (std + epsilon)

    return advantage_dict
