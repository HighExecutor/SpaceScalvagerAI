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

env = "SpaceScalEnv"
file_name = "E:\\Projects\\SpaceScalvagerAI\\SpaceScalvager\\Build\\Train\\SpaceScalvager.exe"

tune.register_env(
    "SpaceScalEnv",
    lambda c: SpaceScalEnv(file_name=c["file_name"], no_graphics=c["no_graphics"]),
)

checkpoint_path = "C:\\Users\\mihai\\ray_results\\PPO\\PPO_SpaceScalEnv_8b617_00000_0_2024-02-21_15-21-46\\checkpoint_002850"
# checkpoint_path = "C:\\Users\\mihai\\ray_results\\PPO\\PPO_SpaceScalEnv_e3b3f_00000_0_2024-02-18_21-40-02\\checkpoint_001550"
# checkpoint_path = "C:\\Users\\mihai\\ray_results\\PPO\\PPO_SpaceScalEnv_ceb5b_00000_0_2024-02-16_18-28-32\checkpoint_000650"

exp_config = read_config(checkpoint_path)['config']
exp_config['num_workers'] = 0
exp_config['env_config']["no_graphics"] = True
if file_name:
    exp_config['env_config']['file_name'] = file_name
agent = PPO(exp_config)
agent.load_checkpoint(checkpoint_path)
policy_name = SpaceScalEnv.get_policy_name()
policy = agent.get_policy(policy_name)


env = SpaceScalEnv(file_name=file_name, no_graphics=False, time_scale=4, port=7001)
for _ in range(1000):
    score = np.zeros(4)
    state = env.reset()
    for t in range(5000):
        actions = agent.compute_actions(state, policy_id=policy_name, unsquash_actions=False)
        s, r, d, i = env.step(actions)
        # print("Time: " + str(t) + "; Dones: " + str(list(d.values())))
        state = s
        score += np.array(list(r.values()))
        if d[policy_name + "?team=0_0"]:
            break
    print("Score: " + str(score))
ray.shutdown()