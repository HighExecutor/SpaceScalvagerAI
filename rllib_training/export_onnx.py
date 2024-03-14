from __future__ import annotations
import os

from typing import Callable
import ray
import gym.spaces
import torch
from torch import nn
from ray.rllib.utils.typing import List
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.registry import get_algorithm_class
import numpy as np
from utils.export_dist_funcs import sample_multiaction, sample_diagonal_gaussian_action_distribution_deterministic, sample_diagonal_gaussian_action_distribution
from utils.config_reader import read_config, read_yaml
from utils.env_helper import register_envs
from ray.rllib.algorithms.ppo import PPO

# base_dir = ".\\rllib_results\\"
# checkpoint_path = "<exp_series>\\<PPO>\\<run_name>\<checkpoint_xxxxxx>"
# checkpoint_path = "C:\\Users\\mihai\\ray_results\\PPO\\PPO_SpaceScalEnv_8b617_00000_0_2024-02-21_15-21-46\\checkpoint_002850"
checkpoint_path = "E:\wspace\\rl_tutorial\\rllib_results\\PPO\\PPO_unity3d_2ae30_00000_0_2024-03-02_18-34-48\checkpoint_000075"
# file_name = None  # specify path if checkpoint is trained on another build
# file_name = "E:\\Projects\\SpaceScalvagerAI\\SpaceScalvager\\Build\\Train\\SpaceScalvager.exe"
file_name = "E:\wspace\\rl_tutorial\\builds\\3dball\\UnityEnvironment.exe"

export_config = {
    "algorithm": "PPO",
    "policy_id": "3DBall",
    # "checkpoint_path": os.path.join(base_dir, checkpoint_path),
    "checkpoint_path": checkpoint_path,
    "onnx_model_suffix": ".onnx",
    "model_config": {},
    "env_config": {
        "file_name": file_name,
        "no_graphics": True,
    },
    "export_params": True,
    "opset_version": 9,
}

def model_forward(original_model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
        Forward pass through the FCNet model.
    """
    # Get features by passing inputs through hidden layers.
    original_model._features = original_model._hidden_layers(inputs)
    # Get logits from features.
    logits = original_model._logits(original_model._features) if original_model._logits else original_model._features
    # NOTE: Not quite sure what this is but included since it's in original model forward method.
    if original_model.free_log_std:
        logits = original_model._append_free_log_std(logits)
    return logits

def get_actions_nums(original_model):
    n_cont_actions = 0
    n_discr_actions = 0
    discr_lengths = []
    act_space = original_model.action_space
    if isinstance(act_space, gym.spaces.Box):
        n_cont_actions = act_space.shape[0]
    if isinstance(act_space, gym.spaces.MultiDiscrete):
        n_discr_actions = act_space.shape[0]
        discr_lengths = act_space.nvec
    if isinstance(act_space, gym.spaces.Tuple):
        for space in act_space:
            if isinstance(space, gym.spaces.Box):
                n_cont_actions = space.shape[0]
            if isinstance(space, gym.spaces.MultiDiscrete):
                n_discr_actions = space.shape[0]
                discr_lengths = space.nvec
    return n_cont_actions, n_discr_actions, discr_lengths

def get_input_names(original_model, discr_actions):
    input_names = []
    if isinstance(original_model.obs_space, gym.spaces.Tuple):
        input_names = ["obs_" + str(i) for i in range(len(original_model.obs_space.original_space))]
    else:
        input_names = ["obs_0"]
    if discr_actions > 0:
        input_names.append("action_masks")
    return input_names

def get_output_names(n_cont_actions, n_discr_actions):
    output_names = ["continuous_actions", "discrete_actions", "version_number", "memory_size",
     "continuous_action_output_shape",
     "discrete_action_output_shape"]
    if n_cont_actions == 0:
        output_names.remove("continuous_actions")
        output_names.remove("continuous_action_output_shape")
    if n_discr_actions == 0:
        output_names.remove("discrete_actions")
        output_names.remove("discrete_action_output_shape")
    return output_names


class FCNetWrapperForMLAgents(nn.Module):
    """
       A wrapper class to convert RLLib FCNet torch models into a nn.Module that is compatible with MLAgents.
       Adds extra buffers, requires a simplified model.forward(), and requires action distribution sampling to be added to the forward method.
    """

    def __init__(self, config: dict, original_model: nn.Module, model_forward_method: Callable,
                 model_action_distribution_sampler: Callable) -> None:
        """
            Initialise the wrapper model.
        """
        super(FCNetWrapperForMLAgents, self).__init__()
        # Store config and original RLLib models
        self.config = config
        self.original_model = original_model

        # Register extra info as buffers in model.
        model_export_version = self.config.get("version_number", 3)  # 3 corresponds to ModelApiVersion.MLAgents2_0
        self.register_buffer("version_number", torch.Tensor([model_export_version]).cuda())
        self.version_number: torch.Tensor

        memory_size = self.config.get("memory_size", 0)  # 0 corresponds to no LSTM/recurrent outputs.
        self.register_buffer("memory_size_vector", torch.Tensor([int(memory_size)]).cuda())
        self.memory_size_vector: torch.Tensor

        # Get extra info that MLAgents needs from config
        n_cont_actions, n_discr_actions, discr_lengths = get_actions_nums(self.original_model)
        config['cont_actions'] = n_cont_actions
        config['discr_actions'] = n_discr_actions
        config['discr_lengths'] = discr_lengths
        self.input_names = get_input_names(self.original_model, n_discr_actions)
        # Get input and output names from config.
        self.output_names = get_output_names(n_cont_actions, n_discr_actions) # Corresponds to continuous actions and discrete actions with stochastic inference.
        if n_cont_actions:
            self.register_buffer("continuous_act_size_vector", torch.Tensor([int(n_cont_actions)]).cuda())
            self.continuous_act_size_vector: torch.Tensor
        if n_discr_actions:
            discrete_action_lengths = [int(i) for i in discr_lengths]
            self.register_buffer("discrete_act_size_vector", torch.Tensor([discrete_action_lengths]).cuda())
            self.discrete_act_size_vector: torch.Tensor

        # Set modified forward model and action distribution sampler from methods passed in.
        self.model_forward_method = model_forward_method
        self.action_distribution_sampler = model_action_distribution_sampler

    def forward(self, *inputs: List[torch.Tensor]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Run a forward pass through the wrapped model.
        """
        # Get action prediction distributions from model.
        logits = self.model_forward_method(self.original_model, inputs[0])
        # Sample actions from distributions.
        mask = inputs[1] if len(inputs) > 1 else None
        sampled_cont, sampled_disc = self.action_distribution_sampler(logits, self.config, mask)

        results = []
        if sampled_cont is not None:
            results.append(sampled_cont)
        if sampled_disc is not None:
            results.append(sampled_disc)
        results.extend([self.version_number, self.memory_size_vector])
        if hasattr(self, "continuous_act_size_vector"):
            results.append(self.continuous_act_size_vector)
        if hasattr(self, "discrete_act_size_vector"):
            results.append(self.discrete_act_size_vector)

        # Return sampled actions and other extra info for MLAgents.
        return tuple(results)


def get_restored_policy(checkpoint_path: str, policy_id, algorithm) -> Policy:
    """
        Returns a named policy that has been restored from a checkpoint.
    """
    # Restore checkpoint to algorithm.
    # algorithm.restore(checkpoint_path)
    algorithm.load_checkpoint(checkpoint=checkpoint_path)
    # Get policy from algorithm using policy id
    return algorithm.get_policy(policy_id)


def generate_sample_mask(export_config):
    discrete_size = export_config['model_config']['discr_lengths'].sum()
    return torch.ones([1, discrete_size]).cuda()


def get_sample_inputs_from_policy(policy: Policy) -> torch.Tensor:
    """
        Generate a batch of dummy data for use in model tracing with ONNX exporter.
    """
    # Sample from the policy's observation space.
    test_data = policy.observation_space.sample()
    # Get it into the correct shape.
    test_data = np.expand_dims(test_data, axis=0)
    # Create torch tensor with input data.
    return torch.tensor(test_data).cuda()


def get_mlagents_model_from_rllib_model(model_config: dict, model: nn.Module,
                                        model_forward_method: Callable = model_forward,
                                        action_sampling_method: Callable = sample_diagonal_gaussian_action_distribution_deterministic) -> FCNetWrapperForMLAgents:
    """
        Get a wrapped RLLib torch model compatible with MLAgents.
    """
    # Create wrapped model for exporting to ONNX.
    return FCNetWrapperForMLAgents(model_config, model, model_forward_method, action_sampling_method)


def export_onnx_model(model: nn.Module, sample_data: torch.Tensor, onnx_export_path: str, export_params: bool,
                      opset_version: int, input_names: list, output_names: list, dynamic_axes: dict) -> None:
    """
        Export an torch.nn.Module model as an ONNX model.
    """
    # Export ONNX model from torch model.
    torch.onnx.export(
        model,
        sample_data,
        onnx_export_path,
        export_params=export_params,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )


def export_onnx_model_from_rllib_checkpoint(config) -> None:
    """
        Export the RLLib model as an MLAgents-compatible ONNX model.
    """
    # Create checkpoint and ONNX model output paths.
    checkpoint_path = config["checkpoint_path"]
    onnx_out_path = checkpoint_path + config["onnx_model_suffix"]
    experiment_config = read_config(checkpoint_path)
    experiment_config['config']['num_workers'] = 0
    for env_key, env_value in config["env_config"].items():
        experiment_config['config']['env_config'][env_key] = env_value
    register_envs(experiment_config)
    # algorithm = get_algorithm_class(config["algorithm"])(experiment_config['config'])
    algorithm = PPO(experiment_config['config'])
    policy_id = config["policy_id"]
    # Get policy from algorithm and restore from checkpoint.
    policy = get_restored_policy(checkpoint_path, policy_id, algorithm)
    print("Exporting policy:", policy_id, "\n\tfrom:\n", checkpoint_path, "\n\tto:\n", onnx_out_path)

    # Get RLLib policy model from policy.
    model = policy.model
    # Choose sampling method according to action outputs for model
    sampling_method = sample_multiaction
    # Get wrapped model.
    mlagents_model = get_mlagents_model_from_rllib_model(config["model_config"], model, model_forward, sampling_method)
    # Get sample inputs for tracing model.
    sample_obs = get_sample_inputs_from_policy(policy)
    sample_inputs = (sample_obs,)
    if export_config['model_config']['discr_actions'] > 0:
        sample_mask = generate_sample_mask(export_config)
        sample_inputs = (sample_obs, sample_mask)
    # Get export params and opset version for ONNX export settings.
    export_params = config["export_params"]
    opset_version = config["opset_version"]
    # Create list of input names, output names, and make the appropriate axes dynamic.
    input_names = mlagents_model.input_names
    output_names = mlagents_model.output_names
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes.update({output_names[0]: {0: "batch"}})
    if config['model_config']['cont_actions'] > 0 and config['model_config']['discr_actions'] > 0:
        # if we have both types of actions
        dynamic_axes.update({output_names[1]: {0: "batch"}})
    # Export ONNX model from RLLib checkpoint.
    export_onnx_model(mlagents_model, sample_inputs, onnx_out_path, export_params, opset_version, input_names,
                      output_names, dynamic_axes)


if __name__ == "__main__":
    # Init ray to start algorithm
    ray.init()
    # Export onnx model from rllib checkpoint using the algorithm provided
    export_onnx_model_from_rllib_checkpoint(export_config)
    print("export done")
    ray.shutdown()
