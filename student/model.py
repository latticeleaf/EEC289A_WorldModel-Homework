"""Student world model.

- GRU hidden state to carry temporal context and preserve history of the rollout for each predicted step
- 2-layer head lets model learn more complex relationship between internal memory and predicted state change 
(mapping hidden -> delta)
- delta_limit=2.0 soft-clamps via tanh, tighter than the baseline to prevent runaway predictions from feeding back and exploding

"""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_gru: bool = True,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self.obs_dim = obs_dim

        #Encoder: maps (obs, act) -> feature vector
        in_dim = obs_dim + act_dim
        layers: list[nn.Module] = []
        for _ in range(int(num_layers)):
            layers += [
                nn.Linear(in_dim, hidden_dim), 
                nn.SiLU()
            ]
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # Recurrent core
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None

        # Prediction head uses two-layers
        self.head = nn.Linear(hidden_dim, hidden_dim)
            
    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        feat = self.encoder(torch.cat([obs_norm, act_norm], dim=-1))
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        raw_delta = self.head(feat)
        # Soft clamp: tanh gradients stay alive near boundary due to tanh
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
