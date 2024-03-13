from __future__ import annotations
import math
from numbers import Number

from typing import Tuple, Union
import torch
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical
from ray.rllib.utils.typing import TensorType, List
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch.distributions.categorical import Categorical
import numpy as np

def sample_diagonal_gaussian_action_distribution_deterministic(diagonal_gaussian_inputs: torch.Tensor,
                                                               config: dict = None) -> torch.Tensor:
    """
        Sample action samples from the action distribution, in this case, a diagonal guassian.
    """
    # Get means and log standard deviations of action predictions from input tensor.
    mu, _ = torch.chunk(diagonal_gaussian_inputs, 2, dim=1)
    # Return sample from diagonal gaussian distribution.
    return mu


def sample_diagonal_gaussian_action_distribution(diagonal_gaussian_inputs: torch.Tensor,
                                                 config: dict = None) -> torch.Tensor:
    """
        Sample action samples from the action distribution, in this case, a diagonal guassian.
    """
    # Get means and log standard deviations of action predictions from input tensor.
    mu, log_std = torch.chunk(diagonal_gaussian_inputs, 2, dim=1)
    # Get standard deviation.
    std = torch.exp(log_std)
    # Sample random epsilon from Normal(0, 1) distribution.
    eps = torch.randn_like(std)
    # Return sample from diagonal gaussian distribution.
    return eps * std + mu


class CategoricalWithoutT(Categorical):

    @override(Categorical)
    def __init__(self, probs=None, logits: torch.Tensor = None, validate_args=None, deterministic=True):
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
                self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            else:
                # If deterministic just outputs logits here!
                self.logits = logits
        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.size()[-1]
        batch_shape = self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        super(Categorical, self).__init__(batch_shape, validate_args=validate_args)


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

    action_lengths = config["discr_lengths"]
    if mask is not None:
        logits = logits * mask
    dist = TorchMultiCategoricalWithoutT(logits, None, action_lengths, deterministic=True)
    # dist = TorchMultiCategorical(inputs=logits, model=None, input_lens=action_lengths)
    return dist.deterministic_sample()


def sample_multiaction(logits: torch.Tensor, config: dict = None, mask: torch.Tensor = None):
    cont_acts = None
    cont_logits_size = config['cont_actions'] * 2
    if config['cont_actions'] > 0:
        # Clip to given bounds, just in case the squashed values were
        # outside [-1.0, 1.0].
        cont_acts = sample_diagonal_gaussian_action_distribution_deterministic(logits[:, :cont_logits_size], config)
        s_low = -1.0
        s_high = 1.0
        cont_acts = s_low + (cont_acts + 1.0) * (s_high - s_low) / 2.0
        cont_acts = torch.clip(cont_acts, s_low, s_high)

    discr_acts = None
    if config['discr_actions'] > 0:
        masked_logits = logits[:, cont_logits_size:]
        if mask is not None:
            masked_logits = masked_logits * mask
        discr_acts = sample_multicategorical_action_distribution_deterministic(masked_logits, config)
    return cont_acts, discr_acts
