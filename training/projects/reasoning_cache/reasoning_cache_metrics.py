import torch
from typing import Any, Dict
import numpy as np
from verl import DataProto
from collections import defaultdict


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def split_thinking_and_summary(batch: DataProto) -> tuple:
    """
    Split batch into thinking and summary samples.

    Returns:
        tuple: (batch_thinking, batch_summary) where batch_summary may be None
               if summary_indicator is not present (summary-free mode).
    """
    if "summary_indicator" not in batch.batch:
        # Summary-free mode: all samples are thinking samples
        return batch, None

    is_summary = batch.batch["summary_indicator"]
    summary_indices = [i for i, is_summary in enumerate(is_summary) if is_summary]
    reasoning_indices = [i for i, is_summary in enumerate(is_summary) if not is_summary]

    batch_thinking = batch[reasoning_indices] if reasoning_indices else None
    batch_summary = batch[summary_indices] if summary_indices else None

    return batch_thinking, batch_summary


def split_offline_and_online(batch: DataProto) -> DataProto:
    is_offline = batch.non_tensor_batch["state_type"] == "offline"
    offline_indices = [i for i, is_offline in enumerate(is_offline) if is_offline]
    online_indices = [i for i, is_offline in enumerate(is_offline) if not is_offline]
    if len(offline_indices) == 0:
        return None, batch
    return batch[offline_indices], batch[online_indices]


def build_problem_score_dict(data: DataProto) -> Dict[str, Any]:
    problem_ids = data.non_tensor_batch["problem_ids"]
    rewards = data.non_tensor_batch["rewards"]
    problem_score_dict = defaultdict(list)
    for p, r in zip(problem_ids, rewards):
        problem_score_dict[p].append(r)
    return problem_score_dict


def safe_mean_list(l):
    return np.mean(l) if len(l) > 0 else 0.0


def safe_mean_tensor(t):
    return torch.mean(t).detach().item() if t.numel() > 0 else 0.0


def safe_max_tensor(t):
    return torch.max(t).detach().item() if t.numel() > 0 else 0.0


def safe_min_tensor(t):
    return torch.min(t).detach().item() if t.numel() > 0 else 0.0


def compute_zero_adv_metrics(batch: DataProto) -> Dict[str, Any]:
    problem_score_dict = build_problem_score_dict(batch)
    scores_by_problem = list(problem_score_dict.values())
    zero_adv_zero_by_problem = [1 if all(x == 0 for x in score_list) else 0 for score_list in scores_by_problem]
    zero_adv_one_by_problem = [1 if all(x == 1 for x in score_list) else 0 for score_list in scores_by_problem]
    return zero_adv_zero_by_problem, zero_adv_one_by_problem


def compute_data_metrics(batch: DataProto, config: Dict[str, Any]) -> Dict[str, Any]:
    batch_thinking, batch_summary = split_thinking_and_summary(batch)
    response_mask = batch.batch["response_mask"].bool()

    # Score metrics
    scores = batch.non_tensor_batch["rewards"]

    # Thinking metrics (always present)
    if batch_thinking is not None:
        thinking_scores = batch_thinking.non_tensor_batch["rewards"]
        response_info_thinking = _compute_response_info(batch_thinking)
        prompt_length_thinking = response_info_thinking["prompt_length"]
        response_length_thinking = response_info_thinking["response_length"]
        thinking_max_length = (batch_thinking.batch["attention_mask"][:, -1] == 1).sum().item()
    else:
        thinking_scores = []
        prompt_length_thinking = torch.tensor([])
        response_length_thinking = torch.tensor([])
        thinking_max_length = 0

    # Summary metrics (may be None in summary-free mode)
    if batch_summary is not None:
        summary_scores = batch_summary.non_tensor_batch["rewards"]
        response_info_summary = _compute_response_info(batch_summary)
        prompt_length_summary = response_info_summary["prompt_length"]
        response_length_summary = response_info_summary["response_length"]
    else:
        summary_scores = []
        prompt_length_summary = torch.tensor([])
        response_length_summary = torch.tensor([])

    # Advantages and returns
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]
    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    # Zero-advantage metrics
    zero_adv_zero_by_problem, zero_adv_one_by_problem = compute_zero_adv_metrics(batch)

    metrics = {
        # scores
        "scores/final_score_mean": safe_mean_list(scores),
        "scores/final_score_mean_thinking": safe_mean_list(thinking_scores),
        "scores/final_score_mean_summary": safe_mean_list(summary_scores),
        # adv
        "advantages/mean": safe_mean_tensor(valid_adv),
        "advantages/max": safe_max_tensor(valid_adv),
        "advantages/min": safe_min_tensor(valid_adv),
        "advantages/zero_adv_zero": safe_mean_list(zero_adv_zero_by_problem),
        "advantages/zero_adv_one": safe_mean_list(zero_adv_one_by_problem),
        # returns
        "returns/mean": safe_mean_tensor(valid_returns),
        "returns/max": safe_max_tensor(valid_returns),
        "returns/min": safe_min_tensor(valid_returns),
        # response length
        "rollouts/mean_thinking_response_length": safe_mean_tensor(response_length_thinking),
        "rollouts/max_thinking_response_length": safe_max_tensor(response_length_thinking),
        "rollouts/min_thinking_response_length": safe_min_tensor(response_length_thinking),
        "rollouts/mean_summary_response_length": safe_mean_tensor(response_length_summary),
        "rollouts/max_summary_response_length": safe_max_tensor(response_length_summary),
        "rollouts/min_summary_response_length": safe_min_tensor(response_length_summary),
        "rollouts/num_thinking_seq_max_length": thinking_max_length,
        # prompt length
        "rollouts/mean_thinking_prompt_length": safe_mean_tensor(prompt_length_thinking),
        "rollouts/max_thinking_prompt_length": safe_max_tensor(prompt_length_thinking),
        "rollouts/min_thinking_prompt_length": safe_min_tensor(prompt_length_thinking),
        "rollouts/mean_summary_prompt_length": safe_mean_tensor(prompt_length_summary),
        "rollouts/max_summary_prompt_length": safe_max_tensor(prompt_length_summary),
        "rollouts/min_summary_prompt_length": safe_min_tensor(prompt_length_summary),
    }
    return metrics
