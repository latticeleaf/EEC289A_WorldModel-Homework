"""Student world model.

- GRU hidden state to carry temporal context and preserve history of the rollout for each predicted step
- removed soft clamp
- added physics features

""" 

from __future__ import annotations

import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
    def forward(self, x):
        return x + self.block(x)
        
class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 4,
        use_gru: bool = False,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self.obs_dim = obs_dim

        #Encoder: maps (obs, act) -> feature vector
        in_dim = obs_dim + act_dim + 4
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(int(num_layers) - 1):
            layers.append(ResidualBlock(hidden_dim))
        self.encoder = nn.Sequential(*layers)

        # Recurrent core
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None

        self.head = nn.Linear(hidden_dim, obs_dim)
            
    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        # Add physics features: sin and cos of pol angle (dim 1)
        angle = obs_norm[:, 1:2]
        angular_vel = obs_norm[:, 3:4]
        cart_vel = obs_norm[:, 2:3]
        physics = torch.cat([
            torch.sin(angle),
            torch.cos(angle),
            angle * angular_vel,
            cart_vel * angular_vel,
        ], dim=-1)
        x = torch.cat([obs_norm, act_norm, physics], dim=-1)
        feat = self.encoder(x)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        raw_delta = self.head(feat)
        return raw_delta, hidden

