from typing import Dict, Callable
from space_env import SpaceScalEnv
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv


def register_envs(experiment_config):
    from ray import tune
    if experiment_config["config"]["env"] == "SpaceScalvager":
        env_class = SpaceScalEnv
    else:
        env_class = Unity3DEnv
    tune.register_env(
        experiment_config["config"]["env"],
        lambda c: env_class(**experiment_config['config']['env_config'])
    )