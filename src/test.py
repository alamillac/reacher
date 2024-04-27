#from agent import Agent, Trainer
from agent import Trainer
from environment import Env
import numpy as np

env_path = {
    "reacher_one": ("Reacher_One_Linux/Reacher.x86_64", "reacher_one.pth"),
    "reacher_many": ("Reacher_Linux/Reacher.x86_64", "reacher_many.pth"),
    "crawler": ("Crawler_Linux/Crawler.x86_64", "crawler.pth"),
}

env_filename, save_path = env_path["reacher_one"]

class Agent:
    def __init__(self, state_size, action_size, num_agents, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.num_agents = num_agents

    def act(self, state):
        actions = np.random.randn(self.num_agents, self.action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        return actions

env = Env(env_filename, train_mode=False, seed=0)

trainer = Trainer()
agent = Agent(state_size=env.state_size, action_size=env.action_size, num_agents=env.num_agents, seed=0)
#agent.load(save_path)

trainer.test(env, agent, num_episodes=2)
