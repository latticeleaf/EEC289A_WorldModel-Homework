"""Student one-step plus rollout loss.

Key changes:
1. exponential step-weighting (gamma): step 0 is weighted 1.0 and step h is weighted gamma^h to help with early errors in the rollout which
compound into later ones which kills VPT.

2. Huber (L1-smooth) term along MSE: MSE drives precision on easy cases and Huber prevents large outlier predictions from dominating gradients
and destabilising GRU hidden state. 

3. Multi-window rollout: sample 'num_rollout_windows' independent random start positions per batch and average their losses. More rollout signal
per gradient step without increasing sequence length. 

4. Psuedo-curriculum via global step counter: rollout horizon starts at 'horizon_start,' and ramps up linearly to 'rollout_train_horizon' 
over 'curriculum_steps' updates. The model masters short rollouts before handling long ones. 
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout

_GLOBAL_STEP: int = 0
                      
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
    huber_loss = (weights * diff.abs()).mean() # huber term
    return mse_loss + 0.1 * huber_loss

def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict,) -> tuple[torch.Tensor, dict]: 
    global _GLOBAL_STEP
    _GLOBAL_STEP += 1
    
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]

    # one-step loss
    one = one_step_delta_loss(model, states, actions, normalizer)

    # curriculum rollout horizon
    max_horizon = int(loss_cfg.get("rollout_train_horizon", 35))
    start_horizon = int(loss_cfg.get("horizon_start", 10))
    curriculum_steps = int(loss_cfg.get("curriculum_steps", 1000))
    if _GLOBAL_STEP >= curriculum_steps:
        horizon = max_horizon
    else:
        frac = _GLOBAL_STEP / max(curriculum_steps, 1)
        horizon = int(start_horizon + frac * (max_horizon - start_horizon))

    warmup = int(cfg["eval"].get("warmup_steps", 10))
    gamma = float(loss_cfg.get("rollout_gamma", 0.97))
    
    num_windows = int(loss_cfg.get("num_rollout_windows", 2))
    roll = torch.tensor(0.0, device=states.device)
    for _ in range(num_windows):
        roll = roll + rollout_loss(
        model, states, actions, normalizer, 
        warmup_steps=warmup, 
        horizon=horizon, 
        gamma=gamma
        )
    roll = roll / num_windows
    
    total = (
        float(loss_cfg.get("one_step_weight", 1.0)) * one 
        + float(loss_cfg.get("rollout_weight", 1.0)) * roll
    )
    
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
