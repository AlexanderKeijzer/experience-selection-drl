import gym
import numpy as np

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise

from hyperparams import PARAMS_ZOO

env = gym.make('Pendulum-v0')

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=PARAMS_ZOO["noise_std"] * np.ones(n_actions))

model = TD3(
    policy=PARAMS_ZOO["policy"],
    env=env,
    learning_rate=PARAMS_ZOO["learning_rate"],
    gamma=PARAMS_ZOO["gamma"],
    buffer_size=PARAMS_ZOO["buffer_size"],
    learning_starts=PARAMS_ZOO["learning_starts"],
    gradient_steps=PARAMS_ZOO["gradient_steps"],
    train_freq=PARAMS_ZOO["train_freq"],
    policy_kwargs=PARAMS_ZOO["policy_kwargs"],
    action_noise=action_noise,
    tensorboard_log="./ddpg_pendulum_log/",
    verbose=1,
)
model.learn(total_timesteps=PARAMS_ZOO["n_timesteps"], log_interval=10)
model.save("td3_pendulum")
