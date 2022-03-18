import gym

from stable_baselines3 import DDPG, TD3, SAC

env = gym.make('Pendulum-v0')
#model = DDPG.load("ddpg_pendulum")
#model = TD3.load("td3_pendulum")
model = SAC.load("sac_pendulum")

obs = env.reset()
dones = False
while not dones:
    action, _states = model.predict(obs)
    print(action)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()