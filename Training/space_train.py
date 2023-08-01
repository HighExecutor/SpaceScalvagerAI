import argparse

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from space_env import SpaceScalEnv
from curriculum import curriculum_config, curriculum_fn, CurriculumCallback, single_task

ray.init(local_mode=False)

args = argparse.Namespace
args.env = "SpaceScalEnv"
args.file_name = "E:\\Projects\\SpaceScalvagerAI\\SpaceScalvager\\Build\\SpaceScalvager.exe"
# args.file_name = None
args.from_checkpoint = None
args.stop_iters = 999999
args.stop_timesteps = 999999999
args.stop_reward = 9999.0
args.framework = "torch"
args.num_workers = 4
args.no_graphics = False

policies, policy_mapping_fn = SpaceScalEnv.get_policy_configs_for_game("SpaceScalvager")

tune.register_env(
    "SpaceScalEnv",
    lambda c: SpaceScalEnv(file_name=c["file_name"], no_graphics=c["no_graphics"],
                           curriculum_config=curriculum_config),
)

config = (
    PPOConfig()
    .environment(
        env="SpaceScalEnv",
        disable_env_checking=True,
        env_config={"file_name": args.file_name,
                    "no_graphics": args.no_graphics},
        env_task_fn=curriculum_fn
    )
    .framework(args.framework)
    .rollouts(
        num_rollout_workers=args.num_workers if args.file_name else 0,
        rollout_fragment_length=1000,
        batch_mode="complete_episodes"
    )
    .training(
        lr=0.0003,
        lambda_=0.95,
        gamma=0.99,
        sgd_minibatch_size=512,
        train_batch_size=16384,
        num_sgd_iter=8,
        vf_loss_coeff=1.0,
        clip_param=0.2,
        entropy_coeff=0.02,
        model={"fcnet_hiddens": [64, 64],
               "vf_share_layers": False},
    )
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    .resources(num_gpus=1)
    .debugging(log_level="INFO")
    .callbacks(CurriculumCallback)
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
