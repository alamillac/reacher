import copy
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

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

# Prioritized Experience Replay (PER)
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_INCREMENT = 0.001
PER_EPSILON = 1e-5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=0, batch_size=BATCH_SIZE, add_noise=True):
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

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, batch_size, seed)
        self.beta = PER_BETA_START

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Initialize losses
        self.num_steps = 0
        self.actor_total_loss = 0
        self.critic_total_loss = 0

    def step(self, states, actions, rewards, next_states, dones, act_info=None):
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
            self.actor_total_loss += actor_loss
            self.critic_total_loss += critic_loss

    def act_train(self, states):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if self.add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1), None

    def act(self, states):
        """Returns actions for given state as per current policy."""
        actions, _ = self.act_train(states)
        return actions

    def reset(self):
        self.noise.reset()

        # Reset losses
        self.num_steps = 0
        self.actor_total_loss = 0
        self.critic_total_loss = 0

    def get_losses(self):
        if self.num_steps == 0:
            return [(0, "actor_loss"), (0, "critic_loss")]

        return [
            (self.actor_total_loss / self.num_steps, "actor_loss"),
            (self.critic_total_loss / self.num_steps, "critic_loss"),
        ]

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
            self.critic_local.parameters(), 1
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


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(
            size=self.mu.shape
        )
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = deque(maxlen=self.buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

        # Prioritized Experience Replay
        self.priorities = np.zeros(self.buffer_size)
        self.max_priority = 1.0
        self.next_idx = 0
        self.current_size = 0

        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        if self.current_size < self.buffer_size:
            self.current_size += 1

        # Add priority
        self.priorities[self.next_idx] = self.max_priority
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    def update_priority(self, idx, priorities):
        self.priorities[idx] = priorities
        self.max_priority = max(np.max(priorities), self.max_priority)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        priorities = (
            self.priorities[: self.current_size] ** PER_ALPHA
        )  # To make the sampling more uniform and reduce overfitting
        sampling_weights = priorities / np.sum(priorities)
        idx_experiences = np.random.choice(
            range(self.current_size),
            size=self.batch_size,
            replace=False,
            p=sampling_weights,
        )  # Sample based on the priority

        idx_adjusted = self.next_idx - self.current_size + idx_experiences
        experiences = [self.memory[idx] for idx in idx_adjusted]

        sampling_weights = (
            torch.from_numpy(sampling_weights[idx_experiences]).float().to(device)
        )
        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            sampling_weights,
            idx_experiences,
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
