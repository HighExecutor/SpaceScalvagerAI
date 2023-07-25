from typing import Dict

from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy

from space_env import SpaceScalEnv


def curriculum_fn(
        train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    cur_task_data = task_settable_env.get_task()
    cur_task = cur_task_data["task_id"]
    if cur_task_data["task_data"]["mean_reward"] < train_results["episode_reward_mean"]:
        new_task = cur_task + 1
        return new_task
    return cur_task


class CurriculumCallback(DefaultCallbacks):
    def on_episode_start(
            self,
            *,
            worker: "RolloutWorker",
            base_env: SpaceScalEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            **kwargs) -> None:
        cl_ml_env = base_env.envs[0]
        current_task_id = cl_ml_env.cur_task
        episode.custom_metrics['curriculum_task_id'] = current_task_id


curriculum_config = {
    0: {
        "name": "initial",
        "mean_reward": 0.3,
        "env_args": {
            "max_steps": 800,
            "sell_price": 0
        }
    },
    1: {
        "name": "hit_more",
        "mean_reward": 0.6,
        "env_args": {
            "max_steps": 2000,
            "sell_price": 0
        }
    },
    2: {
        "name": "hit_more_and_try_sell",
        "mean_reward": 1.0,
        "env_args": {
            "max_steps": 3000,
            "sell_price": 1
        }
    },
    3: {
        "name": "hit and sell",
        "mean_reward": 1.5,
        "env_args": {
            "max_steps": 5000,
            "sell_price": 2
        }
    }
}

single_task = {
    0: {
        "name": "initial",
        "mean_reward": 0.3,
        "env_args": {
            "max_steps": 0
        }
    }
}
