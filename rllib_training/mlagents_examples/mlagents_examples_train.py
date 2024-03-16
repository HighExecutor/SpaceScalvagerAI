import argparse
import os

import ray
from ray import air, tune
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.algorithms.ppo import PPOConfig

# "3DBall"
# "3DBallHard"
# "GridFoodCollector"
# "Pyramids"
# "SoccerStrikersVsGoalie"
# "SoccerTwos"
# "Sorter"
# "Tennis"
# "VisualHallway"
# "Walker"

local_mode = True
env = "3DBall"
file_name = "E:\wspace\\rl_tutorial\\builds\\3dball\\UnityEnvironment.exe"
# file_name = None
from_checkpoint = None
num_workers = 0 if local_mode else 4
stop_iters = 9999
stop_timesteps = 100000000
stop_reward = 9999.0
horizon = 1000
framework = "torch"

if __name__ == "__main__":
    ray.init(local_mode=local_mode)


    tune.register_env(
        "unity3d",
        lambda c: Unity3DEnv(
            file_name=c["file_name"],
            no_graphics=False,
            episode_horizon=c["episode_horizon"],
        ),
    )

    # Get policies (different agent types; "behaviors" in MLAgents) and
    # the mappings from individual agents to Policies.
    policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game(env)
    config = {
        "env": "unity3d",
        "disable_env_checking": True,
        "env_config": {
            "file_name": file_name,
            "episode_horizon": horizon,
        },
        # For running in editor, force to use just one Worker (we only have
        # one Unity running)!
        "num_workers": num_workers,
        # Other settings.
        "lr": 0.0003,
        "lambda": 0.95,
        "gamma": 0.99,
        "sgd_minibatch_size": 64,
        "train_batch_size": 12000,
        "num_gpus": 1,
        # "num_sgd_iter": 20,
        "num_sgd_iter": 3,
        "rollout_fragment_length": 200,
        # "batch_mode": "complete_episodes",
        "clip_param": 0.2,
        # Multi-agent setup for the particular env.
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "model": {
            # "fcnet_hiddens": [512, 512],
            "fcnet_hiddens": [128, 128],
            "fcnet_activation": "swish"
        },
        "normalize_actions": True,
        "framework": framework,
        "no_done_at_end": True,
    }
    # Switch on Curiosity based exploration for Pyramids env
    # (not solvable otherwise).
    if env == "Pyramids":
        config["exploration_config"] = {
            "type": "Curiosity",
            "eta": 0.1,
            "lr": 0.001,
            # No actual feature net: map directly from observations to feature
            # vector (linearly).
            "feature_net_config": {
                "fcnet_hiddens": [],
                "fcnet_activation": "relu",
            },
            "sub_exploration": {
                "type": "StochasticSampling",
            },
            "forward_net_activation": "relu",
            "inverse_net_activation": "relu",
        }
    elif env == "GridFoodCollector":
        config["model"] = {
            "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [256, [10, 10], 1]],
        }
    elif env == "Sorter":
        config["model"]["use_attention"] = True

    stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        "episode_reward_mean": stop_reward,
    }

    # Run the experiment.
    results = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=air.RunConfig(
            stop=stop,
            verbose=3,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True,
            ),
            local_dir="E:\wspace\\rl_tutorial\\rllib_results\\"
        ),
    ).fit()

    ray.shutdown()
