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
args.file_name = "C:\\wspace\\projects\\space_main\\SpaceScalvagerAI\\SpaceScalvager\\Build\\SpaceScalvager.exe"
args.from_checkpoint = None
args.stop_iters = 999999
args.stop_timesteps = 999999999
args.stop_reward = 9999.0
args.framework = "torch"
args.num_workers = 1
args.no_graphics = False

tune.register_env(
    "SpaceScalEnv",
    lambda c: SpaceScalEnv(file_name=c["file_name"], no_graphics=c["no_graphics"]),
)

policies, policy_mapping_fn = SpaceScalEnv.get_policy_configs_for_game("SpaceShip")

config = (
    PPOConfig()
        .environment(
        env="SpaceScalEnv",
        disable_env_checking=True,
        env_config={"file_name": args.file_name,
                    "no_graphics": args.no_graphics},
    )
        .framework(args.framework)
        .rollouts(
        num_rollout_workers=args.num_workers if args.file_name else 0,
        rollout_fragment_length=200,
        batch_mode="complete_episodes"
    )
        .training(
        lr=0.0003,
        lambda_=0.95,
        gamma=0.99,
        sgd_minibatch_size=256,
        train_batch_size=4096,
        num_sgd_iter=8,
        vf_loss_coeff=1.0,
        clip_param=0.2,
        entropy_coeff=0.002,
        model={"fcnet_hiddens": [64, 64],
               "vf_share_layers": False},
    )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus=1)
        .debugging(log_level="INFO")
)
stop = {
    "training_iteration": args.stop_iters,
    "timesteps_total": args.stop_timesteps,
    "episode_reward_mean": args.stop_reward,
}


tune.run(
        'PPO',
        config=config.to_dict(),
        stop=stop,
        verbose=3,
        checkpoint_freq=50,
        checkpoint_at_end=False,
    )

ray.shutdown()
