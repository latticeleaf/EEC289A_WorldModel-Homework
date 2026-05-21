""""Student one-step plus rollout loss.

Key changes from original:
1. Exponential step-weighting (gamma): step 0 is weighted 1.0 and step h is weighted gamma^h to help with early errors in the rollout which compound into later ones which kills VPT.
2. Huber (L1-smooth) term alongside MSE: MSE drives precision on easy cases and Huber prevents large outlier predictions from dominating gradients and destabilising GRU hidden state.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .rollout import open_loop_rollout

def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)

def rollout_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer, warmup_steps: int, horizon: int, gamma: float = 0.97) -> torch.Tensor:
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    # exponential step weighting
    weights = torch.tensor(
        [gamma ** h for h in range(int(horizon))],
        dtype=pred_norm.dtype,
        device=pred_norm.device,
    ).view(1, -1, 1)
    diff = pred_norm - target_norm
    mse_loss = (weights * diff ** 2).mean()
    huber_loss = (weights * diff.abs()).mean()
    return mse_loss + 0.1 * huber_loss

def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    one = one_step_delta_loss(model, states, actions, normalizer)
    horizon = int(loss_cfg.get("rollout_train_horizon", 15))
    warmup = int(cfg["eval"].get("warmup_steps", 10))
    gamma = float(loss_cfg.get("rollout_gamma", 0.97))
    roll = rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon, gamma=gamma)
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 1.0)) * roll
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
