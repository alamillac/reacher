import numpy as np

from ddpg import DDPGAgent
from environment import Env
from td3 import TD3Agent
from trainer import Trainer

env_path = {
    "reacher_one": ("Reacher_One_Linux/Reacher.x86_64", "reacher.pth"),
    "reacher_many": ("Reacher_Linux/Reacher.x86_64", "reacher.pth"),
    "crawler": ("Crawler_Linux/Crawler.x86_64", "crawler.pth"),
}

# env_filename, save_path = env_path["reacher_many"]
# env_filename, save_path = env_path["reacher_one"]
env_filename, save_path = env_path["crawler"]

env = Env(env_filename, train_mode=False, seed=0)

trainer = Trainer(max_t=300)
# agent = DDPGAgent(state_size=env.state_size, action_size=env.action_size, seed=0)

low = np.array([-1] * env.action_size)
high = np.array([1] * env.action_size)
action_bounds = (low, high)
agent = TD3Agent(
    state_size=env.state_size, action_bounds=action_bounds, n_envs=env.num_agents
)
agent.load(save_path)

trainer.test(env, agent, num_episodes=20)
