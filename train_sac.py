import gym

from stable_baselines3 import SAC

from hyperparams import PARAMS_ZOO

from callback_fft import FFTCallback

env = gym.make('Pendulum-v0')

callback = FFTCallback()

model = SAC(
    policy=PARAMS_ZOO["policy"],
    env=env,
    learning_rate=PARAMS_ZOO["learning_rate"],
    tensorboard_log="./ddpg_pendulum_log/",
    verbose=1,
)
model.learn(total_timesteps=PARAMS_ZOO["n_timesteps"], log_interval=10, callback=callback)
model.save("sac_pendulum")
