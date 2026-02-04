import torch
import verl.utils.torch_functional as verl_F


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str, grouping: torch.Tensor = None):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "step-mean-token-mean":
        assert grouping is not None, "grouping is required for step-mean-token-mean"
        masked_loss = loss_mat * loss_mask
        ids_1d = grouping.view(-1)
        id_counts = torch.bincount(ids_1d)
        normalization_factors = id_counts[ids_1d].unsqueeze(1)
        step_normalized_loss = masked_loss / (normalization_factors.float() + 1e-8)
        num_active_tokens = loss_mask.sum()
        loss = step_normalized_loss.sum() / (num_active_tokens + 1e-8)
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
        
    return loss


def compute_policy_loss(old_log_prob,
                        log_prob,
                        advantages,
                        response_mask,
                        cliprange=None,
                        cliprange_low=None,
                        cliprange_high=None,
                        clip_ratio_c=3.0,
                        loss_agg_mode="token-mean",
                        grouping=None):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior        

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low,
                                           1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1,
                                    pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(
        loss_mat=pg_losses, 
        loss_mask=response_mask, 
        loss_agg_mode=loss_agg_mode, 
        grouping=grouping
    )

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower