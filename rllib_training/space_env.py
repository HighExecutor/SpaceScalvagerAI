import numpy as np
import time
import random
from gymnasium.spaces import Box, MultiDiscrete, Tuple as TupleSpace
from typing import Callable, Optional, Tuple, List, Dict

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


class SpaceScalEnv(MultiAgentEnv, TaskSettableEnv):
    # Default base port when connecting directly to the Editor
    _BASE_PORT_EDITOR = 5004
    # Default base port when connecting to a compiled environment
    _BASE_PORT_ENVIRONMENT = 5106
    # The worker_id for each environment instance
    _WORKER_ID = 0

    def __init__(self, env_name: str = "SpaceScalEnv",
                 file_name: str = None,
                 time_scale: int = 1,
                 port: Optional[int] = None,
                 seed: int = 0,
                 no_graphics: bool = False,
                 timeout_wait: int = 300,
                 curriculum_config=None,
                 additional_args: List[str] = None,
                 ):
        super().__init__()
        if curriculum_config is None:
            curriculum_config = {}
        self.env_name = env_name

        if file_name is None:
            print(
                "No game binary provided, will use a running Unity editor "
                "instead.\nMake sure you are pressing the Play (|>) button in "
                "your editor to start."
            )

        import mlagents_envs
        from mlagents_envs.environment import UnityEnvironment

        # Try connecting to the Unity3D game instance. If a port is blocked
        port_ = None
        while True:
            if port_ is not None:
                time.sleep(random.randint(1, 10))
            port_ = port or (
                self._BASE_PORT_ENVIRONMENT if file_name else self._BASE_PORT_EDITOR
            )
            # cache the worker_id and
            # increase it for the next environment
            worker_id_ = SpaceScalEnv._WORKER_ID if file_name else 0
            SpaceScalEnv._WORKER_ID += 1
            self.environment_parameters_side_channel = EnvironmentParametersChannel()
            self.environment_parameters_side_channel.set_float_parameter("time_scale", time_scale)
            try:
                self.unity_env = UnityEnvironment(
                    file_name=file_name,
                    worker_id=worker_id_,
                    base_port=port_,
                    seed=seed,
                    side_channels=[self.environment_parameters_side_channel],
                    no_graphics=no_graphics,
                    timeout_wait=timeout_wait,
                    additional_args=additional_args,
                )
                print("Created UnityEnvironment for port {}".format(port_ + worker_id_))
            except mlagents_envs.exception.UnityWorkerInUseException:
                pass
            else:
                break

        # ML-Agents API version.
        self.api_version = self.unity_env.API_VERSION.split(".")
        self.api_version = [int(s) for s in self.api_version]

        # Keep track of how many times we have called `step` so far.
        self.episode_timesteps = 0
        # Curriculum
        self.curriculum_config = curriculum_config
        self.changes_to_task = None
        self.cur_task = None
        self.setup_curriculum_config()

    def step(
            self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        all_agents = []
        for behavior_name in self.unity_env.behavior_specs:
            if self.api_version[0] > 1 or (
                    self.api_version[0] == 1 and self.api_version[1] >= 4
            ):
                actions = []
                for agent_id in self.unity_env.get_steps(behavior_name)[0].agent_id:
                    key = behavior_name + "_{}".format(agent_id)
                    all_agents.append(key)
                    if key not in action_dict:
                        action_dict[key] = np.zeros(6, dtype=np.int64)
                    actions.append(action_dict[key])
                if actions:
                    if isinstance(actions[0], Tuple):
                        action_tuple = ActionTuple(continuous=np.array([actions[0][0]]),
                                                   discrete=np.array([actions[0][1]]))
                    elif actions[0].dtype == np.float32:
                        action_tuple = ActionTuple(continuous=np.array(actions))
                    else:
                        action_tuple = ActionTuple(discrete=np.array(actions))
                    self.unity_env.set_actions(behavior_name, action_tuple)
            else:
                raise ValueError("Check mlagents|rllib versions")
        self.unity_env.step()

        obs, rewards, dones, truncated, infos = self._get_step_results()

        # Global horizon reached? -> Return __all__ done=True, so user
        # can reset. Set all agents' individual `done` to True as well.
        self.episode_timesteps += 1
        return obs, rewards, dones, truncated, infos

    def reset(self, **kwargs) -> Tuple[MultiAgentDict, MultiAgentDict]:
        # Curriculum
        if self.changes_to_task:
            self.update_and_send_task_to_unity(self.changes_to_task)
        """Resets the entire Unity3D scene (a single multi-agent episode)."""
        self.episode_timesteps = 0
        self.unity_env.reset()
        obs, _, _, _, infos = self._get_step_results()
        return obs, infos

    def _get_step_results(self):
        """Collects those agents' obs/rewards that have to act in next `step`.

        Returns:
            Tuple:
                obs: Multi-agent observation dict.
                    Only those observations for which to get new actions are
                    returned.
                rewards: Rewards dict matching `obs`.
                dones: Done dict with only an __all__ multi-agent entry in it.
                    __all__=True, if episode is done for all agents.
                infos: An (empty) info dict.
        """
        obs = {}
        rewards = {}
        infos = {}
        dones = {}
        all_done = False
        for behavior_name in self.unity_env.behavior_specs:
            decision_steps, terminal_steps = self.unity_env.get_steps(
                behavior_name)

            for agent_id, idx in decision_steps.agent_id_to_index.items():
                key = behavior_name + "_{}".format(agent_id)
                dones[key] = False

                os = tuple(o[idx] for o in decision_steps.obs)
                os = os[0] if len(os) == 1 else os
                obs[key] = os
                rewards[key] = decision_steps.reward[idx]  # rewards vector

            for agent_id, idx in terminal_steps.agent_id_to_index.items():
                key = behavior_name + "_{}".format(agent_id)

                dones[key] = True
                # Only overwrite rewards (last reward in episode), b/c obs
                # here is the last obs.
                # Unless key does not exist in obs.
                if key not in obs:
                    os = tuple(o[idx] for o in terminal_steps.obs)
                    obs[key] = os = os[0] if len(os) == 1 else os
                rewards[key] = terminal_steps.reward[idx]  # rewards vector

        # Only use dones if all agents are done, then we should do a reset.
        all_done = not (False in list(dones.values()))
        dones["__all__"] = all_done
        return obs, rewards, dones, {"__all__": False}, infos

    @staticmethod
    def get_policy_configs_for_game(
            game_name: str,
    ) -> Tuple[dict, Callable[[AgentID], PolicyID]]:
        obs_spaces = {
            "SpaceScalvager": TupleSpace([
                Box(float("-inf"), float("inf"), (4,)),
                Box(float("-inf"), float("inf"), (35,)),
            ])
        }

        action_spaces = {
            "SpaceScalvager": MultiDiscrete([3, 3, 3, 3, 3, 2])
        }

        policies = {
            game_name: PolicySpec(
                observation_space=obs_spaces[game_name],
                action_space=action_spaces[game_name],
            ),
        }

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            return game_name

        return policies, policy_mapping_fn

    @staticmethod
    def get_policy_name():
        return "SpaceScalvager"

    # Curriculum
    def setup_curriculum_config(self):
        if self.curriculum_config:
            assert self.curriculum_config[0]
            self.update_and_send_task_to_unity(0)

    def get_task(self) -> dict:
        return {"task_id": self.cur_task, "task_data": self.curriculum_config[self.cur_task]}

    def set_task(self, changes_to_task):
        if changes_to_task != self.cur_task and changes_to_task in self.curriculum_config:
            self.changes_to_task = changes_to_task

    def update_and_send_task_to_unity(self, change_to_task):
        self.cur_task = change_to_task
        for k, v in self.curriculum_config[self.cur_task]["env_args"].items():
            self.environment_parameters_side_channel.set_float_parameter(k, v)
        self.changes_to_task = None
