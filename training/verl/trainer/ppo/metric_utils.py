# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List, Callable
import numpy as np
from verl import DataProto
from collections import Counter, defaultdict
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
import os


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


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


def compute_data_metrics(
    batch: DataProto,
    use_critic: bool = True,
    experiment_name: str = "",
    global_steps: int = 0,
    test_freq: int = -1,
    tokenizer: any = None,
) -> Dict[str, Any]:

    initial_score_tensor = batch.batch["token_level_scores"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        "scores/final_score_mean": torch.mean(initial_score_tensor).detach().item(),
        # adv
        "advantages/mean": torch.mean(valid_adv).detach().item(),
        "advantages/max": torch.max(valid_adv).detach().item(),
        "advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "returns/mean": torch.mean(valid_returns).detach().item(),
        "returns/max": torch.max(valid_returns).detach().item(),
        "returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "values/mean": torch.mean(valid_values).detach().item(),
                "values/max": torch.max(valid_values).detach().item(),
                "values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "values/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "rollouts/mean_response_length": torch.mean(response_length).detach().item(),
        "rollouts/max_response_length": torch.max(response_length).detach().item(),
        "rollouts/min_response_length": torch.min(response_length).detach().item(),
        # prompt length
        "rollouts/mean_prompt_length": torch.mean(prompt_length).detach().item(),
        "rollouts/max_prompt_length": torch.max(prompt_length).detach().item(),
        "rollouts/min_prompt_length": torch.min(prompt_length).detach().item(),
    }

    if (global_steps - 1) % test_freq == 0:
        # advantage histograms
        advantages = valid_adv.detach().cpu().tolist()
        sns.histplot(advantages)
        os.makedirs(f"outputs/{experiment_name}/{global_steps}", exist_ok=True)
        plt.savefig(f"outputs/{experiment_name}/{global_steps}/advantage.png")
        plt.clf()
        plt.close()

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughput_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str],
    sample_inputs: list[str],
    infos_dict: dict[str, list[Any]],
    seed: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.

    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample

    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                for n in ns:
                    # Best/Worst-of-N
                    [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                        data=var_vals,
                        subset_size=n,
                        reduce_fns=[np.max, np.min],
                        seed=seed,
                    )
                    metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = (
                        bon_mean,
                        bon_std,
                    )
                    metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = (
                        won_mean,
                        won_std,
                    )
                    # TODO: enable majority vote for Countdown properly.
                    # We probably need to use Sympy to reduce all candidates to a common form.

                    # # Majority voting
                    # if var2vals.get("pred", None) is not None:
                    #     vote_data = [
                    #         {"val": val, "pred": pred}
                    #         for val, pred in zip(var_vals, var2vals["pred"])
                    #     ]
                    #     [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                    #         data=vote_data,
                    #         subset_size=n,
                    #         reduce_fns=[
                    #             partial(calc_maj_val, vote_key="pred", val_key="val")
                    #         ],
                    #         seed=seed,
                    #     )
                    #     metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = (
                    #         maj_n_mean,
                    #         maj_n_std,
                    #     )

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
