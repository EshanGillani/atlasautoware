"""
Learned driving policy for the AtlasAutoware stack.
===================================================

A neural policy that takes the lidar scan plus the car's state and outputs a
bounded *correction* to the MPC's command — never a raw command of its own.
That structure is what makes learning practical on a real race car:

    features.py   what the policy sees, and the residual action envelope.
                  Pure numpy — shared by training and deployment so the two
                  cannot drift apart.
    networks.py   1-D CNN lidar encoder + MLP; SAC actor and twin critics.
    env.py        the f110_gym dynamics wrapped for RL, with the reward.
    sac.py        Soft Actor-Critic, replay buffer, checkpoint I/O.

Train with `atlas run train-rl`, evaluate against the MPC with
`atlas run eval-rl`, deploy with `atlas run rl-drive`.  See docs/rl.md.

Only `features` imports cleanly without torch; the rest need it, which is why
nothing else in the stack imports this package at startup.
"""

from .features import ObsSpec, ResidualAction, build_observation   # noqa: F401

__all__ = ['ObsSpec', 'ResidualAction', 'build_observation']
