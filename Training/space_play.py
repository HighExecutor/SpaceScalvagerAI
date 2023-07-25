import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from space_env import SpaceScalEnv

ray.init(local_mode=True)

args = argparse.Namespace
args.env = "SpaceScalEnv"
args.file_name = "E:\\Projects\\SpaceScalvagerAI\\SpaceScalvager\\Build\\SpaceScalvager.exe"
args.from_checkpoint = None
args.framework = "torch"
args.num_workers = 0
args.no_graphics = False

tune.register_env(
    "SpaceScalEnv",
    lambda c: SpaceScalEnv(file_name=c["file_name"], no_graphics=c["no_graphics"]),
)

agent = PPO.from_checkpoint("C:\\Users\\mihai\\ray_results\\PPO\\PPO_SpaceScalEnv_70c52_00000_0_2023-07-13_13-07-22\\checkpoint_000550")
policy = agent.get_policy("SpaceScalvager")

env = SpaceScalEnv(file_name=args.file_name, no_graphics=False)
for _ in range(1000):
    score = 0.0
    state = env.reset()
    for _ in range(500):
        flat_obs = state["SpaceScalvager?team=0_0"]
        act = agent.compute_single_action(flat_obs, policy_id="SpaceScalvager")
        s, r, d, i = env.step({"SpaceScalvager?team=0_0": act})
        state = s
        score += r["SpaceScalvager?team=0_0"]
        if d["SpaceScalvager?team=0_0"]:
            break
    print("Score: " + str(score))
ray.shutdown()