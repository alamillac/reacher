from ddpg import Agent
from environment import Env
from trainer import Trainer

env_path = {
    "reacher_one": (
        "Reacher_One_Linux/Reacher.x86_64",
        "reacher.pth",
        "reacher_one_checkpoint.pth",
    ),
    "reacher_many": (
        "Reacher_Linux/Reacher.x86_64",
        "reacher.pth",
        "reacher_many_checkpoint.pth",
    ),
    "crawler": (
        "Crawler_Linux/Crawler.x86_64",
        "crawler.pth",
        "crawler_checkpoint.pth",
    ),
}

env_filename, save_path, save_checkpoint_path = env_path["reacher_many"]

env = Env(env_filename, train_mode=True)

print("Number of agents:", env.num_agents)
print("Number of actions:", env.action_size)
print("States have length:", env.state_size)

trainer = Trainer(
    max_episodes=2000,
    max_t=300,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    save_model_path=save_path,
    save_checkpoint_path=save_checkpoint_path,
    override_checkpoint=True,
)

agent = Agent(state_size=env.state_size, action_size=env.action_size, seed=0)

# scores = trainer.train_until(env, agent, desired_score=13, consecutive_episodes=100)
scores = trainer.train(env, agent)
trainer.plot_scores(scores)