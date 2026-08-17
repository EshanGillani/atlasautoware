"""
Neural architectures for the driving policy.
============================================

The observation is two very different things concatenated: a 1-D range image
from the lidar, and a short vector of physical state.  Feeding both into one
flat MLP wastes the structure — so the lidar goes through a small 1-D
convolutional encoder first.

Why a 1-D CNN over the beams
----------------------------
Adjacent lidar beams are spatially adjacent in the world, and the features that
matter (a wall receding, a gap opening, a car-sized blob) are *translation
equivariant*: a gap two metres to the left means the same thing as a gap two
metres to the right, mirrored.  A convolution learns one gap detector and
applies it at every bearing; a fully-connected layer has to learn the same
detector separately for all 108 inputs, from far more data.  Strided
convolutions then build a small summary that the policy head can combine with
the physical state.

Everything here is deliberately small.  The policy has to run inside a 50 Hz
control loop on a Jetson while SLAM, the lidar driver and a YOLO detector are
also running; a network that needs 15 ms is useless no matter how well it
drives.  The actor — the only part that ships on the car — is ~113k parameters
and measures ~0.9 ms per tick on a laptop CPU, against a 20 ms budget.  The
twin critics (~357k) exist only during training and never leave the desktop.

Requires torch.  Nothing else in the stack imports this module unless you are
actually training or deploying a policy, so the rest keeps working without it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class LidarEncoder(nn.Module):
    """1-D conv stack over the range image -> a fixed-width embedding."""

    def __init__(self, n_beams, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=3, padding=3),   # ~1/3
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # ~1/6
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),  # ~1/12
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():                        # infer the flattened width
            n_flat = self.net(torch.zeros(1, 1, n_beams)).numel()
        self.head = nn.Linear(n_flat, out_dim)
        self.out_dim = out_dim

    def forward(self, lidar):                        # (B, n_beams)
        h = self.net(lidar.unsqueeze(1))
        return F.relu(self.head(h.flatten(1)))


class Torso(nn.Module):
    """Lidar embedding + physical state -> shared hidden representation.

    `n_beams=0` means the observation has no range image at all — the
    decision-layer policy (rl/duel.py) reasons over gaps and closing speeds,
    not raw lidar.  The convolutional encoder is then skipped entirely rather
    than fed a zero-length input, and this is a plain MLP.
    """

    def __init__(self, n_beams, n_state, hidden=256, embed=64):
        super().__init__()
        self.n_beams = int(n_beams)
        self.encoder = LidarEncoder(self.n_beams, embed) if self.n_beams else None
        in_dim = (embed if self.encoder is not None else 0) + n_state
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.out_dim = hidden

    def forward(self, obs):                          # (B, n_beams + n_state)
        if self.encoder is None:
            return self.mlp(obs)
        lidar, state = obs[:, :self.n_beams], obs[:, self.n_beams:]
        return self.mlp(torch.cat([self.encoder(lidar), state], dim=1))


class SquashedGaussianPolicy(nn.Module):
    """SAC actor: a Gaussian in pre-squash space, tanh-squashed into [-1, 1].

    The tanh is what bounds the action, and it is also why the log-probability
    needs the change-of-variables correction below — without it the entropy
    term is wrong and the temperature tuning silently misbehaves.
    """

    def __init__(self, n_beams, n_state, n_actions=2, hidden=256):
        super().__init__()
        self.torso = Torso(n_beams, n_state, hidden)
        self.mu = nn.Linear(hidden, n_actions)
        self.log_std = nn.Linear(hidden, n_actions)

    def forward(self, obs):
        h = self.torso(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs, deterministic=False, with_logprob=True):
        """-> (action in [-1,1], log_prob or None)."""
        mu, log_std = self(obs)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        noise = torch.randn_like(mu)
        pre = mu + std * noise
        action = torch.tanh(pre)
        if not with_logprob:
            return action, None
        # log N(pre; mu, std) minus the tanh Jacobian, in the numerically
        # stable form: log(1 - tanh(u)^2) = 2*(log2 - u - softplus(-2u))
        logp = (-0.5 * noise.pow(2) - log_std
                - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))).sum(1)
        logp -= (2.0 * (torch.log(torch.tensor(2.0)) - pre
                        - F.softplus(-2.0 * pre))).sum(1)
        return action, logp


class QNetwork(nn.Module):
    """Twin critics in one module — SAC needs two, and sharing the call site
    keeps the update readable.  They do NOT share weights; the whole point of
    the pair is that their independent errors bound the overestimation bias."""

    def __init__(self, n_beams, n_state, n_actions=2, hidden=256):
        super().__init__()
        self.q1_torso = Torso(n_beams, n_state, hidden)
        self.q2_torso = Torso(n_beams, n_state, hidden)
        self.q1_head = nn.Sequential(
            nn.Linear(hidden + n_actions, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1))
        self.q2_head = nn.Sequential(
            nn.Linear(hidden + n_actions, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1))

    def forward(self, obs, action):
        q1 = self.q1_head(torch.cat([self.q1_torso(obs), action], 1)).squeeze(-1)
        q2 = self.q2_head(torch.cat([self.q2_torso(obs), action], 1)).squeeze(-1)
        return q1, q2

    def q1_only(self, obs, action):
        return self.q1_head(torch.cat([self.q1_torso(obs), action], 1)).squeeze(-1)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
