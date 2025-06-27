"""
Augmentation Tools

This module provides a set of augmentation tools designed to enhance, randomize, and distort tensor
data.

Citation
--------
If you use this code, please cite the following, and read function docs for further info/citations
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,
CVPR 2018. https://arxiv.org/abs/1903.03148

License
-------
Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under
the License.
"""

from __future__ import annotations
from typing import Union, Tuple, List

import torch
import neurite as ne

from neurite.samplers import Sampler


__all__ = [
    "random_crop",
    "random_clip",
    "random_gamma"
]


def random_crop(
    input_tensor: torch.Tensor,
    crop_proportion: Union[Sampler, float] = 0.5,
    prob: Union[Sampler, float] = 1,
    forbidden_dims: Union[Tuple, List] = (0, 1),
    seed: Union[Sampler, int] = None,
):
    """
    Apply random crops ~iid Ber() to the input tensor along the specified dimensions.

    This function selects a subset of the allowed dimensions (excluding `forbidden_dims`)
    from successful iid Bernoulli trials of each dimension specified by `prob`. The proportion of
    the dimension's size to crop can either be fixed or sampled.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The tensor to be randomly cropped. It is assumed to have batch and channel dimensions.
    crop_proportion : Union[Sampler, float], optional
        The proportion that is randomly cropped from any allowed dimension. By default 0.5
        - If a `float` is provided, it represents the maximum proportion (0 to 1) to crop,
        sampled from independent uniform distributinos for each allowed dimension. A value of `0.5`
        means up to 50% of each dimension can be cropped.
        - If a `Sampler` is provided, cropped proportions are dynamically sampled based on the
        specified distribution
    prob : Union[Sampler, float], optional
        The probability of cropping each allowed dimension. By default 1.0
        - If a `float` is provided, it's used as a fixed probability for all eligible dimensions.
        - If a `Sampler` is provided, probabilities are dynamically generated for each dimension.
    forbidden_dims : Union[Tuple[int, ...], List[int]], optional
        Dimensions that should never be cropped. By defult `(0, 1)` (batch and channel dimensions)
    seed : Union[Sampler, int], optional
        A random seed or sampler to control the randomness of cropping operations. If provided, it
        ensures reproducibility of the cropping. Defaults to `None`.

    Returns
    -------
    torch.Tensor
        The randomly cropped tensor with the same number of dimensions as `input_tensor`.
        The size of each cropped dimension is reduced based on the sampled `crop_proportion`.

    Examples
    --------
    >>> tensor = torch.randn(2, 3, 64, 64)
    >>> # Cropping up to 80% of each dimension, with each having a 50% chance of being cropped
    >>> cropped_tensor = random_crop(tensor, crop_proportion=0.8, prob=0.5)
    >>> print(cropped_tensor.shape)
    torch.Size([2, 3, 51, 51])

    >>> # Cropping between 25% to 75% of each dimension
    >>> cropped_tensor = random_crop(tensor, crop_proportion=Uniform(0.25, 0.75))
    >>> print(cropped_tensor.shape)
    torch.Size([2, 3, 45, 45])

    >>> # Specifying forbidden dimensions (e.g., not cropping the last dimension)
    >>> cropped_tensor = random_crop(tensor, forbidden_dims=[0, 1, 3])
    >>> print(cropped_tensor.shape)
    torch.Size([2, 3, 64, 32])
    """

    # Initialize random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Calculate the list of allowable dimensions
    allowed_dims = [x for x in range(input_tensor.dim()) if x not in forbidden_dims]

    # If `crop_proportion` is float, interpret it as upper bound of uniform distribution.
    crop_sampler = ne.samplers.make_sampler(
        ne.samplers.Uniform,
        ne.utils.utils.range_endpoints(0, crop_proportion)
    )

    # If prob is a sampler, sample from it
    if isinstance(prob, Sampler):
        prob = prob()

    # Make prob into a Bernoulli distribution
    prob = ne.samplers.make_sampler(ne.samplers.Bernoulli, prob)

    # Make empty list of slices which we will modufy
    slices = [slice(None)] * input_tensor.dim()

    # I think `translation_min` will always be zero. Keep it as such.
    translation_min = 0

    # Iterate through each dimension and make croppings for them independently.
    for dim in allowed_dims:

        # Decide if we are to crop the current dimension
        if bool(prob()):

            # Determine the size of this dimension
            dim_size = input_tensor.shape[dim]

            # Sample a random proportion of the dimension to crop and convert it to a point along
            # the axis.
            crop_size = round((1 - crop_sampler()) * dim_size)

            # Calculate the maximum translation to avoid going out of bounds.
            translation_max = dim_size - crop_size

            # Prevent errors in the case of no translation
            if translation_min != translation_max:

                # Sample a valid translation (can't be out of bounds!)
                translation = ne.samplers.RandInt(translation_min, translation_max)()

            else:

                translation = 0

            # Make the slice and override the current (presumably null) slice.
            slices[dim] = slice(translation, crop_size + translation)

    return input_tensor[slices]


def random_clip(
    input_tensor: torch.Tensor,
    clip_min: Union[float, int, Sampler] = 0,
    clip_max: Union[float, int, Sampler] = 1,
    clip_prob: Union[float, int, Sampler] = 0.5,
    seed: Union[int, Sampler] = None,
) -> torch.Tensor:
    """
    Randomly clip values in a tensor to a specified range with a given probability.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor whose values may be clipped.
    clip_min : Union[float, int, Sampler], optional
        The minimum value for clipping. If a `Sampler` is provided, it is sampled dynamically.
        By default, 0.
    clip_max : Union[float, int, Sampler], optional
        The maximum value for clipping. If a `Sampler` is provided, it is sampled dynamically.
        By default, 1.
    clip_prob : Union[float, int, Sampler], optional
        The probability of applying the clipping operation. If a `Sampler` is provided, it is
        sampled dynamically. By default, 0.5.
    seed : Union[int, Sampler], optional
        A seed value or a sampler for reproducibility. Default is None.

    Returns
    -------
    torch.Tensor
        The tensor, clipped based on the specified parameters, or unchanged if the clipping
        operation is not applied.

    Examples
    --------
    ### Fixed range clipping
    >>> input_tensor = torch.tensor([1.5, -0.5, 3.0])
    >>> clipped_tensor = random_clip(input_tensor, clip_min=0, clip_max=1)
    >>> print(clipped_tensor)
    tensor([1.0, 0.0, 1.0])

    ### Dynamic range clipping
    >>> from my_samplers import UniformSampler
    >>> input_tensor = torch.tensor([1.5, -0.5, 3.0])
    >>> clipped_tensor = random_clip(
                            input_tensor,
                            clip_min=UniformSampler(0, 0.5), 
                            clip_max=UniformSampler(1.5, 2.0)
                        )
    >>> print(clipped_tensor)
    tensor([1.5000, 0.1732, 1.6281])

    ### Reproducibility with a seed
    >>> input_tensor = torch.tensor([1.5, -0.5, 3.0])
    >>> clipped_tensor1 = random_clip(input_tensor, clip_min=0, clip_max=1, seed=42)
    >>> clipped_tensor2 = random_clip(input_tensor, clip_min=0, clip_max=1, seed=42)
    >>> print(clipped_tensor1 is clipped_tensor2)
    True
    """
    # If prob is a sampler, sample from it
    if isinstance(clip_prob, Sampler):
        clip_prob = clip_prob()

    # Make prob into a Bernoulli distribution
    clip_prob = ne.samplers.make_sampler(ne.samplers.Bernoulli, clip_prob)

    # Sample Bernoulli trial to determine whether to clip
    if bool(clip_prob()):

        # Initialize random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # If `clip_min` is float, interpret it as a fixed minimum for clipping (clipping floor).
        clip_min = ne.samplers.make_sampler(ne.samplers.Fixed, clip_min)

        # If `clip_max` is float, interpret it as a fixed maximum for clipping (clipping ceiling).
        clip_max = ne.samplers.make_sampler(ne.samplers.Fixed, clip_min)

        # Sample and apply clips
        return input_tensor.clip_(clip_min(), clip_max())
    else:
        return input_tensor


def random_gamma(
    input_tensor: torch.Tensor,
    gamma: Union[Sampler, float] = 1.0,
    prob: Union[Sampler, float] = 1.0,
    seed: Union[Sampler, int] = None,
) -> torch.Tensor:
    """
    Apply randomized nonlinear gamma scaling to the input tensor with a specified probability.

    The gamma scaling operation adjusts the contrast of the input tensor by applying a non-linear
    operation. Specifically, each element in the tensor is raised to the power of `gamma`. This can
    enhance or diminish the contrast of the input data.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The tensor to which the gamma scaling operation will be applied. Assumed to have a range
        suitable for gamma correction (typically normalized between 0 and 1).
    gamma : Union[float, Sampler], optional
        The gamma value to apply for the scaling operation.
        - If a `float` is provided, it represents a fixed gamma value.
        - If a `Sampler` is provided, the gamma value is dynamically sampled based on the specified
          distribution.
        By default `1.0`, which leaves the tensor unchanged.
    prob : Union[float, Sampler], optional
        The probability of applying the gamma scaling operation.
        - If a `float` is provided, it's used as a fixed probability for the operation.
        - If a `Sampler` is provided, probabilities are dynamically generated for each invocation.
        By default `1.0` (always apply).
    seed : Union[int, Sampler], optional
        A random seed or sampler to control the randomness of the gamma operation. If provided,
        it ensures reproducibility of the operation. Defaults to `None`.

    Returns
    -------
    torch.Tensor
        The tensor after applying the gamma scaling operation. If the operation is not applied
        (based on `prob`), the original `input_tensor` is returned unchanged.

    Examples
    --------
    ### Fixed gamma scaling operation
    >>> tensor = torch.tensor([0.25, 0.5, 0.75])
    >>> gamma_tensor = random_gamma(tensor, gamma=2.0, prob=1.0)
    >>> print(gamma_tensor)
    tensor([0.0625, 0.2500, 0.5625])

    ### Randomized gamma scaling operation with a range of gamma values
    >>> from neurite.torch.random import Uniform
    >>> tensor = torch.tensor([0.25, 0.5, 0.75])
    >>> gamma_sampler = Uniform(0.5, 1.5)
    >>> gamma_tensor = random_gamma(tensor, gamma=gamma_sampler, prob=0.8)
    >>> print(gamma_tensor)
    tensor([0.1768, 0.5000, 0.8367])

    ### Applying gamma scaling operation with reproducibility
    >>> tensor = torch.tensor([0.25, 0.5, 0.75])
    >>> gamma_tensor1 = random_gamma(tensor, gamma=2.0, prob=1.0, seed=42)
    >>> gamma_tensor2 = random_gamma(tensor, gamma=2.0, prob=1.0, seed=42)
    >>> print(torch.equal(gamma_tensor1, gamma_tensor2))
    True
    """
    # Initialize random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # If prob is a sampler, sample from it
    if isinstance(prob, Sampler):
        prob = prob()

    # Make prob into a Bernoulli distribution
    prob = ne.samplers.make_sampler(ne.samplers.Bernoulli, prob)

    # Sample Bernoulli trial to determine whether to apply gamma scaling operation
    if bool(prob()):

        # Sample gamma
        gamma = ne.samplers.make_sampler(ne.samplers.Fixed, gamma)()

        # Apply nonlinear gamma scaling operation
        return input_tensor.pow(gamma)
    else:
        return input_tensor
