from ddpg import Agent
from environment import Env
from trainer import Trainer

env_path = {
    "reacher_one": ("Reacher_One_Linux/Reacher.x86_64", "reacher.pth"),
    "reacher_many": ("Reacher_Linux/Reacher.x86_64", "reacher.pth"),
    "crawler": ("Crawler_Linux/Crawler.x86_64", "crawler.pth"),
}

# env_filename, save_path = env_path["reacher_many"]
env_filename, save_path = env_path["reacher_one"]

env = Env(env_filename, train_mode=False, seed=0)

trainer = Trainer(max_t=300)
agent = Agent(state_size=env.state_size, action_size=env.action_size, seed=0)
agent.load(save_path)

trainer.test(env, agent, num_episodes=2)
