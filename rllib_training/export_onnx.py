from __future__ import annotations
import os

from typing import Callable
import ray
import torch
from torch import nn
from ray.rllib.utils.typing import List
from ray.rllib.policy.policy import Policy
import numpy as np
from utils.export_dist_funcs import sample_multicategorical_action_distribution_deterministic
from utils.config_reader import read_config, read_yaml
from utils.env_helper import register_envs
from ray.rllib.algorithms.ppo import PPO

from space_env import SpaceScalEnv

base_dir = "E:\wspace\\rl_tutorial\\rllib_results\\"
# checkpoint_path = "<exp_series>\\<PPO>\\<run_name>\<checkpoint_xxxxxx>"
checkpoint_path = "PPO_2024-03-22_14-19-58\PPO_SpaceScalEnv_e1a3f_00000_0_2024-03-22_14-19-58\checkpoint_000023"
# file_name = None  # specify path if checkpoint is trained on another build
file_name = "E:\wspace\\rl_tutorial\\builds\\SpaceScalvager\\SpaceScalvager.exe"

export_params = True
opset_version = 9
version_number = 3
memory_size = 0

checkpoint_path = os.path.join(base_dir, checkpoint_path)
onnx_model_suffix = ".onnx"

discrete_actions = [3, 3, 3, 3, 2]

input_names = ["obs_0", "obs_1" "action_masks"]
output_names = ["discrete_actions", "version_number", "memory_size",
                "discrete_action_output_shape"]


class RLLibTorchModelWrapper(nn.Module):

    def __init__(self, original_model: nn.Module,
                 model_action_distribution_sampler: Callable) -> None:
        """
            Initialise the wrapper model.
        """
        super().__init__()
        self.original_model = original_model

        # Register extra info as buffers in model.
        self.register_buffer("version_number", torch.Tensor([version_number]).cuda())
        self.version_number: torch.Tensor


        self.register_buffer("memory_size_vector", torch.Tensor([int(memory_size)]).cuda())
        self.memory_size_vector: torch.Tensor

        self.register_buffer("discrete_act_size_vector", torch.Tensor([discrete_actions]).cuda())
        self.discrete_act_size_vector: torch.Tensor

        # Set modified forward model and action distribution sampler from methods passed in.
        self.action_distribution_sampler = model_action_distribution_sampler


    def rllib_model_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
            Forward pass through the FCNet model.
        """
        # Get features by passing inputs through hidden layers.
        self.original_model._features = self.original_model._hidden_layers(inputs)
        # Get logits from features.
        logits = self.original_model._logits(
            self.original_model._features) if self.original_model._logits else self.original_model._features
        # NOTE: Not quite sure what this is but included since it's in original model forward method.
        if self.original_model.free_log_std:
            logits = self.original_model._append_free_log_std(logits)
        return logits

    def forward(self, *inputs: List[torch.Tensor]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Run a forward pass through the wrapped model.
        """
        # Get action prediction distributions from model.
        logits = self.model_forward_method(self.original_model, inputs[0])
        # Sample actions from distributions.
        mask = inputs[1] if len(inputs) > 1 else None
        sampled_disc = self.action_distribution_sampler(logits, self.config, mask)

        results = [sampled_disc, self.version_number, self.memory_size_vector, self.discrete_act_size_vector]
        return tuple(results)


def generate_sample_mask():
    discrete_size = discrete_actions.sum()
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


def export_onnx_model_from_rllib_checkpoint() -> None:
    """
        Export the RLLib model as an MLAgents-compatible ONNX model.
    """
    # Create checkpoint and ONNX model output paths.
    onnx_out_path = checkpoint_path + "onnx_model_suffix"
    experiment_config = read_config(checkpoint_path)
    experiment_config['config']['num_rollout_workers'] = 0
    register_envs(experiment_config)

    algorithm = PPO(experiment_config['config'])
    policy_id = SpaceScalEnv.get_policy_name()
    algorithm.load_checkpoint(checkpoint=checkpoint_path)
    policy = algorithm.get_policy(policy_id)
    print("Exporting policy:", policy_id, "\n\tfrom:\n", checkpoint_path, "\n\tto:\n", onnx_out_path)

    # Get RLLib policy model from policy.
    model = policy.model
    # Choose sampling method according to action outputs for model
    sampling_method = sample_multicategorical_action_distribution_deterministic
    # Get wrapped model.
    mlagents_model = RLLibTorchModelWrapper(model, sampling_method)
    # Get sample inputs for tracing model.
    sample_obs = get_sample_inputs_from_policy(policy)
    sample_mask = generate_sample_mask()
    sample_inputs = (sample_obs, sample_mask)
    # Get export params and opset version for ONNX export settings.
    # Create list of input names, output names, and make the appropriate axes dynamic.
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes.update({name: {0: "batch"} for name in output_names})
    # Export ONNX model from RLLib checkpoint.
    export_onnx_model(mlagents_model, sample_inputs, onnx_out_path, export_params, opset_version, input_names,
                      output_names, dynamic_axes)


if __name__ == "__main__":
    # Init ray to start algorithm
    ray.init()
    export_onnx_model_from_rllib_checkpoint()
    print("export done")
    ray.shutdown()
