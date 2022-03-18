from msilib.schema import Error
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from scipy.fft import fft, fftfreq, fftn
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

class FFTCallback(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.data = []
        self.n_dones = -1

    def _on_step(self) -> None:
        if sum(self.locals["dones"]) > 0:
            self.n_dones = self.n_dones + 1
            if self.n_dones % 10 == 0:
                rb = self.locals['replay_buffer']
                data = np.squeeze(_get_episode_actions_from_buffer(rb))
                plt.plot(data)
                plt.show()
                yf = fft(data)
                N = len(data)
                T = self.locals['env'].unwrapped.get_attr("dt")[0]
                xf = fftfreq(N, T)[:N//2]

                self.data.append((xf, yf, N))

                for xf, yf, N in self.data:
                    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
                plt.grid()
                plt.show()

        return super()._on_step()

def _get_episode_start(rb: ReplayBuffer) -> Tuple[int, bool]:
    wrapped = False
    p = rb.pos - 1
    while sum(rb.dones[p]) == 0:
        p = p - 1
        if p < 0:
            if wrapped:
                p = rb.pos
                break
            if rb.full:
                p = rb.size-1
                wrapped = True
            else:
                p = 0
                break
    return p+1, wrapped

def _get_episode_reward_from_buffer(rb: ReplayBuffer) -> np.array:
    p, wrapped = _get_episode_start(rb)
    if not wrapped:
        return rb.rewards[p:rb.pos-1]
    else:
        return np.concatenate((rb.rewards[0:rb.pos-1], rb.rewards[p:]))
        
def _get_episode_observations_from_buffer(rb: ReplayBuffer) -> np.array:
    p, wrapped = _get_episode_start(rb)
    if not wrapped:
        return rb.observations[p:rb.pos-1]
    else:
        return np.concatenate((rb.observations[0:rb.pos-1], rb.observations[p:]))

def _get_episode_actions_from_buffer(rb: ReplayBuffer) -> np.array:
    p, wrapped = _get_episode_start(rb)
    if not wrapped:
        return rb.actions[p:rb.pos-1]
    else:
        return np.concatenate((rb.actions[0:rb.pos-1], rb.actions[p:]))