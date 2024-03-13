import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from space_env import SpaceScalEnv
from utils.config_reader import read_config
import numpy as np

ray.init(local_mode=True)

env_name = "3DBall"
file_name = "E:\wspace\\rl_tutorial\\builds\\3dball_single\\UnityEnvironment.exe"
from_checkpoint = None
num_workers = 0
framework = "torch"

tune.register_env(
    "unity3d",
    lambda c: Unity3DEnv(
        file_name=c["file_name"],
        no_graphics=True,
        episode_horizon=c["episode_horizon"],
        port=5104
    ),
)


checkpoint_path = "E:\wspace\\rl_tutorial\\rllib_results\\PPO\\PPO_unity3d_98cf5_00000_0_2024-03-03_14-26-09\checkpoint_000040"

exp_config = read_config(checkpoint_path)['config']
exp_config['num_workers'] = 0
exp_config['env_config']["no_graphics"] = True
if file_name:
    exp_config['env_config']['file_name'] = file_name
agent = PPO(exp_config)
agent.load_checkpoint(checkpoint_path)
policy = agent.get_policy(env_name)

# policy.export_model(checkpoint_path, 9)


env = Unity3DEnv(file_name=file_name, no_graphics=False, port=5105)
policy_name = env_name
for _ in range(1000):
    score = np.zeros(12)
    state = env.reset()
    for t in range(5000):
        actions = agent.compute_actions(state, policy_id=policy_name, unsquash_actions=False)
        s, r, d, i = env.step(actions)
        # print("Time: " + str(t) + "; Dones: " + str(list(d.values())))
        state = s
        score += np.array(list(r.values()))
        if d["__all__"]:
            break
    print("Score: " + str(score))
ray.shutdown()