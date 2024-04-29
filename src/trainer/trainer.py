import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch

LINE_CLEAR = "\x1b[2K"


class Trainer:
    def __init__(
        self,
        max_episodes=2000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        save_every=100,
        save_checkpoint_path="checkpoint.pth",
        save_model_path="model.pth",
        override_checkpoint=False,
    ):
        """Deep Q-Learning.

        Params
        ======
            max_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        self.max_t = max_t
        self.max_episodes = max_episodes
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.save_checkpoint_path = save_checkpoint_path
        self.save_model_path = save_model_path
        self.save_every = save_every
        self.override_checkpoint = override_checkpoint

    def plot_scores(self, scores):
        # plot the scores
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel("Score")
        plt.xlabel("Episode #")
        plt.show()

    def test(self, env, agent, num_episodes=100):
        scores = []
        max_score = -np.Inf
        min_score = np.Inf
        for i_episode in range(num_episodes):
            states = env.reset()
            score = 0
            for step in range(self.max_t):
                actions = agent.act(states)
                states, rewards, dones = env.step(actions)
                score += np.mean(rewards)
                print(f"\rStep {step} Score: {score:.2f}", end="")
                done = np.any(dones)  # if any agent is done, then the episode is done
                if done:
                    break

            max_score = max(score, max_score)
            min_score = min(score, min_score)
            scores.append(score)
            avg_score = np.mean(scores)
            print(
                f"\rEpisode {i_episode + 1} Score: {score:.2f} Min Score: {min_score:.2f} Max Score: {max_score:.2f} Average Score: {avg_score:.2f}"
            )
        env.close()
        return scores

    def _train(self, env, agent, print_step=False):
        init_episode, eps = self.load_checkpoint(self.save_checkpoint_path, agent)
        for i_episode in range(init_episode, self.max_episodes + 1):
            states = env.reset()
            score = 0
            for step in range(self.max_t):
                actions = agent.act(states, eps)
                next_states, rewards, dones = env.step(actions)
                agent.step(states, actions, rewards, next_states, dones)
                states = next_states
                score += np.mean(rewards)
                if print_step:
                    print(
                        f"\rEpisode {i_episode} Step {step} Score: {score:.2f}", end=""
                    )
                done = np.any(dones)  # if any agent is done, then the episode is done
                if done:
                    break
            eps = max(self.eps_end, self.eps_decay * eps)  # decrease epsilon

            # Save the checkpoint
            if (i_episode + 1) % self.save_every == 0:
                self.save_checkpoint(self.save_checkpoint_path, agent, i_episode)

            yield i_episode, score

    def save_checkpoint(self, path, agent, i_episode):
        agent_state = agent.get_state()
        checkpoint = {
            "i_episode": i_episode,
            "agent_state": agent_state,
        }
        torch.save(checkpoint, path)

    def _init_epsilon(self, i_episode):
        return max(self.eps_end, self.eps_decay**i_episode)

    def load_checkpoint(self, path, agent):
        # Check if the file exists
        if not os.path.isfile(path) or self.override_checkpoint:
            return 0, self.eps_start

        checkpoint = torch.load(path)
        agent.load_state(checkpoint["agent_state"])

        eps = self._init_epsilon(checkpoint["i_episode"])
        return checkpoint["i_episode"], eps

    def train_until(self, env, agent, desired_score, consecutive_episodes=100):
        scores_window = deque(maxlen=consecutive_episodes)  # last scores
        scores = []  # list containing scores from each episode
        for i_episode, score in self._train(env, agent, print_step=False):
            scores.append(score)

            scores_window.append(score)  # save most recent score
            avg_score = np.mean(scores_window)

            if avg_score >= desired_score and i_episode > consecutive_episodes:
                print(
                    f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {avg_score:.2f}"
                )
                break

            print(f"\rEpisode {i_episode} Average Score: {avg_score:.2f}", end="")
            if i_episode % 100 == 0:
                print(f"\rEpisode {i_episode} Average Score: {avg_score:.2f}")

        agent.save(self.save_model_path)
        env.close()
        return scores

    def train(self, env, agent):
        scores_window = deque(maxlen=100)  # last 100 scores
        scores = []  # list containing scores from each episode
        max_score = -np.Inf
        for i_episode, score in self._train(env, agent, print_step=True):
            scores.append(score)

            scores_window.append(score)  # save most recent score
            avg_score = np.mean(scores_window)
            max_score = max(avg_score, max_score)

            print(
                f"\rEpisode {i_episode} Average Score: {avg_score:.2f} Max avg Score: {max_score:.2f}"
            )

            if i_episode % 10 == 0:
                print(f"Memory size: {len(agent.memory)}")

        agent.save(self.save_model_path)
        env.close()
        return scores
