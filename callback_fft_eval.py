from msilib.schema import Error
from gym import Env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from scipy.fft import fft, fftfreq, fftn
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

class FFTEvalCallback(BaseCallback):

    def __init__(self, eval_env: Env, verbose: int = 0, ):
        super().__init__(verbose)
        self.env = eval_env
        self.data = []
        self.n_dones = -1
        self.EVAL_EVERY = 10
        self.EVAL_ROLLOUTS = 10

    def _on_step(self) -> None:
        if sum(self.locals["dones"]) > 0:
            self.n_dones = self.n_dones + 1
            if self.n_dones % self.EVAL_EVERY == 0:
                yf_list = []
                for _ in range(self.EVAL_ROLLOUTS):
                    actions, rewards = self._do_rollout()
                    
                    data = np.squeeze(rewards)
                    N = len(data)
                    yf = fft(data)
                    yf = 2.0/N * np.abs(yf[0:N//2])
                    yf_list.append(yf)
                yf_m, yf_std = np.mean(yf_list, axis=0), np.std(yf_list, axis=0)
                T = self.env.dt
                xf = fftfreq(N, T)[:N//2]

                self.data.append((xf, yf_m, yf_std))

                for i, (xf, yf_m, yf_std) in enumerate(self.data):
                    plt.plot(xf, yf_m, label='Episode ' + str(i*self.EVAL_EVERY))
                    plt.fill_between(xf, yf_m-yf_std, yf_m+yf_std, alpha=0.4)
                plt.grid()
                plt.legend()
                plt.xlabel("Frequency [Hz]")
                plt.ylabel("Rewards [-]")
                plt.show()

        return super()._on_step()

    def _do_rollout(self):
        obs = self.env.reset()
        done, state = False, None
        actions = []
        rewards = []
        while not done:
            action, state = self.locals['self'].predict(obs, state=state, deterministic=False)
            actions.append(action)
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
        return actions, rewards