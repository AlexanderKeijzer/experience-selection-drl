PARAMS_ZOO = {
    "n_timesteps": 20000,
    "policy": 'MlpPolicy',
    "learning_rate": 1e-3,
    "gamma": 0.98,
    "buffer_size": 200000,
    "learning_starts": 10000,
    "noise_type": 'normal',
    "noise_std": 0.1,
    "gradient_steps": -1,
    "train_freq": (1, "episode"),
    "policy_kwargs": {"net_arch": [400, 300]},
}