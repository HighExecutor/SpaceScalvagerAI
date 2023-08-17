from __future__ import annotations
import math
from numbers import Number

import argparse
from typing import Callable, Tuple, Union
import ray
from ray import tune
import torch
from torch import nn
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical
from ray.rllib.utils.typing import TensorType, List
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch.distributions.categorical import Categorical
from ray.rllib.algorithms.registry import get_algorithm_class
import numpy as np

from space_env import SpaceScalEnv


export_config = {
    "algorithm": "PPO",
    "checkpoint_path": "C:\\Users\\mihai\\ray_results\\PPO\\PPO_SpaceScalEnv_8027d_00000_0_2023-08-17_12-19-08\\checkpoint_000100",
    "onnx_model_suffix": "-model-1.onnx",
    "policy_id": "SpaceScalvager",
    "model_config": {
        "version_number": 3,
        "num_discrete_actions": 6,
        "discrete_action_lengths": (3, 3, 3, 3, 3, 2),
        "memory_size": 0,
        "input_names": [
            "obs_0",
            "obs_1"
        ],
        "output_names": [
            "discrete_actions",
            "version_number",
            "memory_size",
            "discrete_action_output_shape"
        ]
    },
    "export_params": True,
    "opset_version": 9,
}


args = argparse.Namespace
args.env = "SpaceScalEnv"
args.file_name = "E:\\Projects\\SpaceScalvagerAI\\SpaceScalvager\\Build\\Train\\SpaceScalvager.exe"
args.stop_iters = 9999
args.stop_timesteps = 99999999
args.stop_reward = 9999.0
args.framework = "torch"

policies, policy_mapping_fn = SpaceScalEnv.get_policy_configs_for_game("SpaceScalvager")

exp_config = (
    PPOConfig()
        .environment(
        env="SpaceScalEnv",
        disable_env_checking=True,
        env_config={"file_name": args.file_name,
                    "no_graphics": True},
    )
        .framework(args.framework)
        .rollouts(
        num_rollout_workers=0,
        rollout_fragment_length=2000,
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
        model={"fcnet_hiddens": [32, 32],
               "vf_share_layers": False},
    )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus=1)
        .debugging(log_level="INFO")
).to_dict()

tune.register_env(
    "SpaceScalEnv",
    lambda c: SpaceScalEnv(file_name=c["file_name"], no_graphics=True),
)

class CategoricalWithoutT(Categorical):

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """
            Numerically stable implementation of the operation value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            if isinstance(sum_exp, Number):
                return m + math.log(sum_exp)
            else:
                return m + torch.log(sum_exp)

    def logsumexp(self, x, dim=None, keepdim=False):
        if dim is None:
            x, dim = x.view(-1), 0
        xm, _ = torch.max(x, dim, keepdim=True)
        x = torch.where(
            (xm == float('inf')) | (xm == float('-inf')),
            xm,
            xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
        return x if keepdim else x.squeeze(dim)

    @override(Categorical)
    def __init__(self, probs=None, logits: torch.Tensor = None, validate_args=None, deterministic=False):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            if not deterministic:
                # self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
                self.logits = logits - self.log_sum_exp(logits, dim=-1, keepdim=True)
                # self.logits = logits - self.logsumexp(logits, dim=-1, keepdim=True)
            else:
                self.logits = logits
        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.size()[-1]
        batch_shape = self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        super(Categorical, self).__init__(batch_shape, validate_args=validate_args)

    @override(Categorical)
    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).transpose(0, 1)
        return samples_2d.reshape(self._extended_shape(sample_shape))


class TorchMultiCategoricalWithoutT(TorchMultiCategorical):
    @override(TorchMultiCategorical)
    def __init__(self, inputs: List[TensorType], model: TorchModelV2,
                 input_lens: Union[List[int], np.ndarray, Tuple[int, ...]], action_space=None, deterministic=False):
        super().__init__(inputs, model, input_lens)
        # If input_lens is np.ndarray or list, force-make it a tuple.
        inputs_split = self.inputs.split(tuple(input_lens), dim=1)
        self.cats = [CategoricalWithoutT(logits=input_, deterministic=deterministic) for input_ in inputs_split]
        # Used in case we are dealing with an Int Box.
        self.action_space = action_space


def sample_multicategorical_action_distribution_deterministic(logits: torch.Tensor,
                                                              config: dict = None,
                                                              mask: torch.Tensor = None) -> torch.Tensor:
    """
        Sample action samples from the action distribution, in this case, a multicategorical distribution.
    """
    action_lengths = config["discrete_action_lengths"]
    if mask is not None:
        logits = logits * mask
    dist = TorchMultiCategoricalWithoutT(logits, None, action_lengths, deterministic=True)
    return dist.deterministic_sample()


def model_forward(original_model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    original_model._features = original_model._hidden_layers(inputs)
    logits = original_model._logits(original_model._features) if original_model._logits else original_model._features
    if original_model.free_log_std:
        logits = original_model._append_free_log_std(logits)
    return logits


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
        # Get extra info that MLAgents needs from config
        model_export_version = self.config.get("version_number", 3)  # 3 corresponds to ModelApiVersion.MLAgents2_0
        num_discrete_actions = self.config.get("num_discrete_actions",
                                               self.original_model.num_outputs // 2)
        discrete_action_lengths = self.config.get("discrete_action_lengths", [1 for _ in range(num_discrete_actions)])
        assert num_discrete_actions == len(discrete_action_lengths)
        memory_size = self.config.get("memory_size", 0)  # 0 corresponds to no LSTM/recurrent outputs.
        # Register extra info as buffers in model.
        self.register_buffer("version_number", torch.Tensor([model_export_version]).cuda())
        self.version_number: torch.Tensor

        self.register_buffer("memory_size_vector", torch.Tensor([int(memory_size)]).cuda())
        self.memory_size_vector: torch.Tensor
        # Get input and output names from config.
        self.input_names = self.config.get("input_names", ["obs_0"])  # corresponds to single vector or observations
        self.output_names = self.config.get("output_names",
                                            ["discrete_actions", "version_number", "memory_size",
                                             "discrete_action_output_shape"])
        self.input_names.append("action_masks")
        if num_discrete_actions:
            discrete_action_lengths = [int(i) for i in discrete_action_lengths]
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
        logits = self.model_forward_method(self.original_model, torch.concatenate((inputs[0], inputs[1]), 1))
        # Sample actions from distributions.
        sampled_disc = self.action_distribution_sampler(logits, self.config, inputs[2])

        results = [sampled_disc, self.version_number, self.memory_size_vector]

        if hasattr(self, "discrete_act_size_vector"):
            results.append(self.discrete_act_size_vector)

        # Return sampled actions and other extra info for MLAgents.
        return tuple(results)

        # # Return sampled actions and other extra info for MLAgents.
        # return sampled_actions, self.version_number, self.memory_size_vector, self.continuous_act_size_vector, self.discrete_act_size_vector


def get_restored_policy(checkpoint_path: str, policy_id: str, algorithm: Algorithm) -> Policy:
    """
        Returns a named policy that has been restored from a checkpoint.
    """
    # Restore checkpoint to algorithm.
    # algorithm.restore(checkpoint_path)
    algorithm.load_checkpoint(checkpoint=checkpoint_path)

    # Get policy from algorithm using policy id
    return algorithm.get_policy(policy_id)


def get_sample_inputs_from_policy(policy: Policy) -> torch.Tensor:
    """
        Generate a batch of dummy data for use in model tracing with ONNX exporter.
    """
    # Sample from the policy's observation space.
    obs0, obs1 = [s.sample() for s in policy.observation_space.original_space]
    # Get it into the correct shape.
    obs0 = np.expand_dims(obs0, axis=0)
    obs1 = np.expand_dims(obs1, axis=0)
    # Create torch tensor with input data.
    return torch.tensor(obs0).cuda(), torch.tensor(obs1).cuda()


def get_mlagents_model_from_rllib_model(model_config: dict, model: nn.Module,
                                        model_forward_method: Callable = model_forward,
                                        action_sampling_method: Callable = sample_multicategorical_action_distribution_deterministic) -> FCNetWrapperForMLAgents:
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


def export_onnx_model_from_rllib_checkpoint(config, algorithm: Algorithm) -> None:
    """
        Export the RLLib model as an MLAgents-compatible ONNX model.
    """
    # Create checkpoint and ONNX model output paths.
    checkpoint_path = config["checkpoint_path"]
    onnx_out_path = checkpoint_path + config["onnx_model_suffix"]
    # Get policy id.
    policy_id = config["policy_id"]
    print("Exporting policy:", policy_id, "\n\tfrom:\n", checkpoint_path, "\n\tto:\n", onnx_out_path)
    # Get policy from algorithm and restore from checkpoint.
    policy = get_restored_policy(checkpoint_path, policy_id, algorithm)
    # Get RLLib policy model from policy.
    model = policy.model
    # Choose sampling method according to action outputs for model
    sampling_method = sample_multicategorical_action_distribution_deterministic
    # Get wrapped model.
    mlagents_model = get_mlagents_model_from_rllib_model(config["model_config"], model, model_forward, sampling_method)
    # Get sample inputs for tracing model.
    sample_obs0, sample_obs1 = get_sample_inputs_from_policy(policy)
    sample_mask = torch.ones([1, sum(config['model_config']['discrete_action_lengths'])]).cuda()
    sample_inputs = (sample_obs0, sample_obs1, sample_mask)
    # Get export params and opset version for ONNX export settings.
    export_params = config["export_params"]
    opset_version = config["opset_version"]
    # Create list of input names, output names, and make the appropriate axes dynamic.
    input_names = mlagents_model.input_names
    output_names = mlagents_model.output_names
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes.update({output_names[0]: {0: "batch"}})
    dynamic_axes.update({output_names[1]: {0: "batch"}})
    # Export ONNX model from RLLib checkpoint.
    export_onnx_model(mlagents_model, sample_inputs, onnx_out_path, export_params, opset_version, input_names,
                      output_names, dynamic_axes)


if __name__ == "__main__":
    # Init ray to start algorithm
    ray.init()
    # Run only one worker.
    exp_config["num_workers"] = 0
    # Export onnx model from rllib checkpoint using the algorithm provided
    export_onnx_model_from_rllib_checkpoint(export_config, get_algorithm_class(export_config["algorithm"])(
        config=exp_config))
    print("export done")
