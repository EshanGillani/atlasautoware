"""
Soft Actor-Critic — the learning algorithm, plus checkpoint I/O.
================================================================

SAC rather than PPO or DQN, for reasons specific to this problem:

  - **Continuous actions.**  Steering and speed are continuous; discretising
    them throws away exactly the fine control that wins lap time.
  - **Off-policy.**  Every step ever collected stays in the replay buffer and
    gets reused. That matters enormously here because collecting a step means
    simulating the car — and, in a practice session, because you can train on
    logged real driving instead of only on fresh rollouts.
  - **Maximum-entropy objective.**  SAC maximises reward *plus* policy entropy,
    with the trade-off (`alpha`) tuned automatically against a target entropy.
    In practice that means it keeps exploring alternative lines instead of
    locking onto the first one that completes a lap, and it is far less
    sensitive to hyperparameters than PPO — which matters when you have one
    week before a competition, not one month of tuning runs.

Implementation is the standard modern SAC: twin critics with a target network,
a tanh-squashed Gaussian actor, and learned temperature.

Requires torch.
"""

import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .features import ObsSpec
from .networks import QNetwork, SquashedGaussianPolicy, count_parameters


class ReplayBuffer:
    """Flat, preallocated circular buffer.  Preallocated because growing a
    Python list to a million transitions mid-training is how you discover the
    Jetson's memory limit at 3am."""

    def __init__(self, obs_dim, action_dim, capacity=400_000):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        i = self.idx
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, rng):
        idx = rng.integers(0, len(self), size=int(batch_size))
        return (self.obs[idx], self.action[idx], self.reward[idx],
                self.next_obs[idx], self.done[idx])


class SAC:
    """The agent: networks, optimizers, one update step, and checkpointing."""

    def __init__(self, spec, action_dim=2, hidden=256, gamma=0.99, tau=0.005,
                 lr=3e-4, alpha_lr=3e-4, target_entropy=None, device=None,
                 seed=0):
        torch.manual_seed(seed)
        self.spec = spec
        self.device = torch.device(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.action_dim = int(action_dim)

        n_b, n_s = spec.n_beams, spec.n_state
        self.actor = SquashedGaussianPolicy(n_b, n_s, action_dim, hidden).to(self.device)
        self.critic = QNetwork(n_b, n_s, action_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Learned temperature. The convention -dim(A) as the entropy target is
        # from the SAC paper and works without tuning across a wide range of
        # tasks; parameterising log(alpha) keeps alpha positive by construction.
        self.target_entropy = float(target_entropy if target_entropy is not None
                                    else -action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def n_parameters(self):
        return count_parameters(self.actor) + count_parameters(self.critic)

    # ── acting ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        t = torch.as_tensor(obs, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(t, deterministic=deterministic,
                                      with_logprob=False)
        return action.squeeze(0).cpu().numpy()

    # ── learning ────────────────────────────────────────────────────────────
    def update(self, batch):
        obs, action, reward, next_obs, done = (
            torch.as_tensor(x, device=self.device) for x in batch)

        # --- critics: regress toward the entropy-augmented Bellman target ---
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            # the min of the twin targets is what bounds overestimation
            target_v = torch.min(q1_t, q2_t) - self.alpha * next_logp
            target_q = reward + (1.0 - done) * self.gamma * target_v

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        # --- actor: maximise Q - alpha*logp (critics frozen for this step) ---
        for p in self.critic.parameters():
            p.requires_grad_(False)
        new_action, logp = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        actor_loss = (self.alpha * logp - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        # --- temperature: drive entropy toward the target ---
        alpha_loss = -(self.log_alpha
                       * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(),
                             self.critic_target.parameters()):
                pt.mul_(1.0 - self.tau).add_(self.tau * p)

        return dict(critic_loss=float(critic_loss.detach()),
                    actor_loss=float(actor_loss.detach()),
                    alpha=float(self.alpha),
                    entropy=float(-logp.detach().mean()))

    def behavior_clone(self, batch_obs, batch_action):
        """One supervised step toward a demonstrated action.

        Used for warm-starting: with a residual action space the expert action
        is zero, so this teaches the policy to defer to the MPC before it ever
        explores.  Regressing the *mean* (pre-squash) rather than a sampled
        action keeps the gradient clean and leaves the entropy term alone.
        """
        obs = torch.as_tensor(batch_obs, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(batch_action, dtype=torch.float32,
                                 device=self.device)
        mu, _ = self.actor(obs)
        loss = F.mse_loss(torch.tanh(mu), target)
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        return float(loss.detach())

    # ── checkpoints ─────────────────────────────────────────────────────────
    def save(self, path, meta=None):
        """Write a checkpoint that carries its own observation contract.

        The spec goes in the file so the ROS node can refuse a policy that was
        trained against a different observation layout, instead of running it
        on silently wrong inputs.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        torch.save(dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            log_alpha=self.log_alpha.detach().cpu(),
            spec=self.spec.to_dict(),
            fingerprint=self.spec.fingerprint(),
            action_dim=self.action_dim,
            meta=meta or {},
        ), path)
        side = os.path.splitext(path)[0] + '.json'
        with open(side, 'w') as f:
            json.dump(dict(fingerprint=self.spec.fingerprint(),
                           spec=self.spec.to_dict(), meta=meta or {}), f, indent=2)

    @staticmethod
    def load(path, device=None, hidden=256):
        """-> (agent, meta).  Raises if the checkpoint is unreadable."""
        dev = torch.device(device or ('cuda' if torch.cuda.is_available()
                                      else 'cpu'))
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        stored = ckpt['spec']
        # A decision-layer policy has no lidar block, so it must be rebuilt as
        # a flat spec — reconstructing it as an ObsSpec would silently give the
        # network a CNN encoder the checkpoint has no weights for.
        if stored.get('flat') or stored.get('n_beams') == 0:
            from .duel import FlatObsSpec
            spec = FlatObsSpec.from_dict(stored)
        else:
            spec = ObsSpec.from_dict(stored)
        agent = SAC(spec, action_dim=int(ckpt.get('action_dim', 2)),
                    hidden=hidden, device=str(dev))
        agent.actor.load_state_dict(ckpt['actor'])
        if 'critic' in ckpt:
            agent.critic.load_state_dict(ckpt['critic'])
            agent.critic_target = copy.deepcopy(agent.critic)
        agent.actor.eval()
        return agent, ckpt.get('meta', {})
