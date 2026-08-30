from __future__ import annotations

import copy

import torch

from bdse.experiments.train import _boundary_focused_pair_subsample


def _reference(batch: dict[str, torch.Tensor], cfg: dict) -> dict[str, torch.Tensor]:
    """V64.3.3 row-wise BFAR sampler, retained only as an equivalence oracle."""
    sampler_cfg = cfg['training']['boundary_pair_sampler']
    pairs = batch['pair_indices'].long()
    pair_valid = batch['pair_valid'].bool()
    B, P, _ = pairs.shape
    max_pairs = int(sampler_cfg['max_pairs'])
    a, b = pairs[..., 0], pairs[..., 1]
    target = batch['teacher_a_star'].long().reshape(B, 1)
    winner_pair = (a == target) | (b == target)
    hard = batch['teacher_hard_violation'].bool()
    K = hard.shape[1]
    hard_a = torch.gather(hard, 1, a.clamp(0, K - 1))
    hard_b = torch.gather(hard, 1, b.clamp(0, K - 1))
    hard_cross = hard_a ^ hard_b
    abs_margin = batch['pair_margins'].float().abs()
    inf = torch.full_like(abs_margin, float('inf'))
    valid_abs = torch.where(pair_valid, abs_margin, inf)
    sorted_abs = torch.sort(valid_abs, dim=1).values
    valid_count_int = pair_valid.sum(dim=1).clamp_min(1)
    median_index = ((valid_count_int - 1) // 2).reshape(B, 1)
    margin_scale = torch.gather(sorted_abs, 1, median_index).clamp_min(float(sampler_cfg['min_margin_scale']))
    normalized_abs = abs_margin / margin_scale
    near_score = 1.0 / (1.0 + normalized_abs / float(sampler_cfg['near_tie_tau']))
    w = batch['pair_weights'].float().clamp_min(0.0)
    wmax = torch.where(pair_valid, w, torch.zeros_like(w)).max(dim=1, keepdim=True).values.clamp_min(1e-6)
    weight_score = w / wmax
    score = (
        float(sampler_cfg['winner_bonus']) * winner_pair.float()
        + float(sampler_cfg['hard_cross_bonus']) * hard_cross.float()
        + float(sampler_cfg['near_tie_bonus']) * near_score
        + float(sampler_cfg['pair_weight_bonus']) * weight_score
    )
    index_preference = (P - torch.arange(P, dtype=score.dtype)) * 1e-7
    chosen_rows = []
    category_specs = (
        (winner_pair, int(sampler_cfg['winner_quota'])),
        (hard_cross, int(sampler_cfg['hard_cross_quota'])),
        (near_score, int(sampler_cfg['near_tie_quota'])),
    )
    for row in range(B):
        selected = torch.zeros(P, dtype=torch.bool)
        valid_row = pair_valid[row]
        for category, raw_quota in category_specs:
            quota = min(max(0, raw_quota), max_pairs - int(selected.sum().item()))
            if quota <= 0:
                continue
            if category.dtype == torch.bool:
                category_mask = category[row] & valid_row & ~selected
                category_rank = score[row]
            else:
                category_mask = valid_row & ~selected
                category_rank = category[row] + 1e-3 * score[row]
            candidate_ids = torch.nonzero(category_mask, as_tuple=False).flatten()
            if candidate_ids.numel() == 0:
                continue
            rank_values = category_rank[candidate_ids] + index_preference[candidate_ids]
            take = min(quota, int(candidate_ids.numel()))
            ids = candidate_ids[torch.topk(rank_values, k=take, largest=True, sorted=False).indices]
            selected[ids] = True
        remaining = max_pairs - int(selected.sum().item())
        if remaining > 0:
            candidate_ids = torch.nonzero(valid_row & ~selected, as_tuple=False).flatten()
            if candidate_ids.numel() > 0:
                rank_values = score[row, candidate_ids] + index_preference[candidate_ids]
                take = min(remaining, int(candidate_ids.numel()))
                ids = candidate_ids[torch.topk(rank_values, k=take, largest=True, sorted=False).indices]
                selected[ids] = True
                remaining -= take
        if remaining > 0:
            pad_ids = torch.nonzero(~selected, as_tuple=False).flatten()[:remaining]
            selected[pad_ids] = True
        chosen_rows.append(torch.nonzero(selected, as_tuple=False).flatten()[:max_pairs])
    chosen = torch.sort(torch.stack(chosen_rows, dim=0), dim=1).values
    out = copy.deepcopy(batch)
    for key in ('pair_indices', 'pair_valid', 'pair_margins', 'pair_weights', 'pair_residuals'):
        value = out.get(key)
        if value is None:
            continue
        gather_shape = [B, max_pairs] + [1] * (value.ndim - 2)
        gather_index = chosen.reshape(gather_shape).expand(B, max_pairs, *value.shape[2:])
        out[key] = torch.gather(value, 1, gather_index)
    return out


def test_vectorized_pair_sampler_matches_v64_3_3_selection() -> None:
    torch.manual_seed(73)
    B, P, K = 5, 23, 8
    # Fixed pseudo-random pairs, with enough duplicate semantics to exercise all quotas.
    a = torch.randint(0, K, (B, P))
    b = (a + torch.randint(1, K, (B, P))) % K
    pairs = torch.stack([a, b], dim=-1)
    valid = torch.rand(B, P) > 0.15
    valid[-1, 7:] = False  # exercise fixed-shape invalid padding path
    batch = {
        'pair_indices': pairs,
        'pair_valid': valid,
        'pair_margins': torch.randn(B, P) + torch.arange(P).float()[None] * 1e-3,
        'pair_weights': torch.rand(B, P) + torch.arange(P).float()[None] * 1e-4,
        'pair_residuals': torch.randn(B, P),
        'teacher_a_star': torch.randint(0, K, (B,)),
        'teacher_hard_violation': torch.rand(B, K) > 0.65,
    }
    cfg = {
        'training': {
            'global_step': 1,
            'current_epoch': 0,
            'epochs': 4,
            'steps_per_epoch': 10,
            'boundary_pair_sampler': {
                'enabled': True,
                'full_every_n_steps': 4,
                'full_last_n_steps': 0,
                'max_pairs': 11,
                'winner_quota': 3,
                'hard_cross_quota': 3,
                'near_tie_quota': 4,
                'winner_bonus': 16.0,
                'hard_cross_bonus': 10.0,
                'near_tie_bonus': 6.0,
                'pair_weight_bonus': 3.0,
                'near_tie_tau': 0.5,
                'min_margin_scale': 1.0,
            },
        }
    }
    expected = _reference(copy.deepcopy(batch), cfg)
    actual = _boundary_focused_pair_subsample(copy.deepcopy(batch), cfg)
    for key in ('pair_indices', 'pair_valid', 'pair_margins', 'pair_weights', 'pair_residuals'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)
