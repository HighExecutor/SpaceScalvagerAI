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
file_name = "E:\wspace\\rl_tutorial\\builds\\SpaceScalvager\\SpaceScalvager.exe"

tune.register_env(
    "SpaceScalEnv",
    lambda c: SpaceScalEnv(file_name=c["file_name"], no_graphics=c["no_graphics"]),
)

base_dir = "E:\wspace\\rl_tutorial\\rllib_results\\"
# checkpoint_path = "<exp_series>\\<PPO>\\<run_name>\<checkpoint_xxxxxx>"
checkpoint_path = "PPO_2024-03-22_14-19-58\PPO_SpaceScalEnv_e1a3f_00000_0_2024-03-22_14-19-58\checkpoint_000023"
checkpoint_path = os.path.join(base_dir, checkpoint_path)

exp_config = read_config(checkpoint_path)['config']
exp_config['num_rollout_workers'] = 0
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
    state, info = env.reset()
    for t in range(5000):
        state = {k: np.concatenate(v) for k, v in state.items()}
        actions = agent.compute_actions(state, policy_id=policy_name, unsquash_actions=False)
        s, r, d, _, i = env.step(actions)
        state = s
        score += np.array(list(r.values()))
        if d[policy_name + "?team=0_0"]:
            break
    print("Score: " + str(score))
ray.shutdown()