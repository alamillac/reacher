import copy
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from noise import OUNoise
from replay_buffer import PrioritizedReplayBuffer

from .model import Actor, Critic

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 5e-4  # learning rate of the actor
LR_CRITIC = 5e-4  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay
UPDATE_EVERY = 4  # how often to update the network
MIN_BUFFER_SIZE = 1e4  # minimum buffer size before learning

ACTOR_HIDDEN_LAYER_1 = 256
ACTOR_HIDDEN_LAYER_2 = 128

CRITIC_HIDDEN_LAYER_1 = 256
CRITIC_HIDDEN_LAYER_2 = 128

# Gradient clipping
VALUE_MAX_GRAD_NORM = 1.0

# Prioritized Experience Replay (PER)
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_INCREMENT = 0.001
PER_EPSILON = 1e-5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    """Interacts with and learns from the environment."""

    def __init__(
        self, state_size, action_size, seed=0, batch_size=BATCH_SIZE, add_noise=True
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.add_noise = add_noise

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size,
            action_size,
            seed,
            fc1_units=ACTOR_HIDDEN_LAYER_1,
            fc2_units=ACTOR_HIDDEN_LAYER_2,
        ).to(device)
        self.actor_target = Actor(
            state_size,
            action_size,
            seed,
            fc1_units=ACTOR_HIDDEN_LAYER_1,
            fc2_units=ACTOR_HIDDEN_LAYER_2,
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            state_size,
            action_size,
            seed,
            fc1_units=CRITIC_HIDDEN_LAYER_1,
            fc2_units=CRITIC_HIDDEN_LAYER_2,
        ).to(device)
        self.critic_target = Critic(
            state_size,
            action_size,
            seed,
            fc1_units=CRITIC_HIDDEN_LAYER_1,
            fc2_units=CRITIC_HIDDEN_LAYER_2,
        ).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Gradient clipping
        self.value_max_grad_norm = VALUE_MAX_GRAD_NORM

        # Replay memory
        self.memory = PrioritizedReplayBuffer(
            action_size, BUFFER_SIZE, batch_size, PER_ALPHA
        )
        self.beta = PER_BETA_START

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # For metrics
        self.metrics = []
        self.num_steps = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps and if enough samples are available in memory
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > MIN_BUFFER_SIZE:
            # Get random subset and learn
            experiences = self.memory.sample()
            actor_loss, critic_loss = self.learn(experiences, GAMMA)
            self.num_steps += 1
            self.metrics.append(("critic_loss", critic_loss, self.num_steps))
            self.metrics.append(("actor_loss", actor_loss, self.num_steps))

    def act_train(self, states):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if self.add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def act(self, states):
        """Returns actions for given state as per current policy."""
        actions = self.act_train(states)
        return actions

    def reset(self):
        self.noise.reset()

    def pop_metrics(self):
        metrics = self.metrics
        self.metrics = []
        return metrics

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, sampling_weights, exp_idx = (
            experiences
        )

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        # As we are using Prioritized Experience Replay (PER), we need to multiply the loss by the importance sampling weights
        buffer_size = len(self.memory)
        weights = (buffer_size * sampling_weights) ** (
            -self.beta
        )  # Importance sampling weights
        weights = weights / weights.max()  # Normalize the weights

        critic_weighted_loss = (
            weights * F.mse_loss(Q_expected, Q_targets, reduction="none")
        ).mean()

        # Update the priorities
        td_errors = torch.abs(Q_expected - Q_targets).detach().squeeze().cpu().numpy()
        self.memory.update_priority(exp_idx, td_errors + PER_EPSILON)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_local.parameters(), self.value_max_grad_norm
        )  # Gradient clipping
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # ------------------- update beta ------------------- #
        self.beta = min(1.0, self.beta + PER_BETA_INCREMENT)

        return actor_loss.item(), critic_weighted_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, path):
        torch.save(self.actor_local.state_dict(), path)

    def load(self, path):
        self.actor_local.load_state_dict(torch.load(path))
        self.actor_target.load_state_dict(torch.load(path))

    def get_state(self):
        return {
            "actor_local_state_dict": self.actor_local.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_local_state_dict": self.critic_local.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "beta": self.beta,
        }

    def load_state(self, state):
        self.actor_local.load_state_dict(state["actor_local_state_dict"])
        self.actor_target.load_state_dict(state["actor_target_state_dict"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])

        self.critic_local.load_state_dict(state["critic_local_state_dict"])
        self.critic_target.load_state_dict(state["critic_target_state_dict"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
        self.beta = state["beta"]
