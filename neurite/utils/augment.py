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

# Standard library imports
from typing import Union, Tuple, List

# Third party imports
import torch


def random_crop(
    input_tensor: torch.Tensor,
    crop_proportion: Union[float, int] = 0.5,
    prob: Union[float, int] = 1,
    forbidden_dims: Union[Tuple, List] = (0, 1),
    seed: Union[int, None] = None,
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
    crop_proportion : float or int, optional
        The maximum proportion (0 to 1) to crop, sampled from independent uniform distributions
        for each allowed dimension. A value of `0.5` means up to 50% of each dimension can be
        cropped. By default 0.5.
    prob : float or int, optional
        The probability of cropping each allowed dimension. Used as a fixed probability for all
        eligible dimensions. By default 1.0.
    forbidden_dims : Union[Tuple[int, ...], List[int]], optional
        Dimensions that should never be cropped. By default `(0, 1)` (batch and channel dimensions)
    seed : int, optional
        A random seed to control the randomness of cropping operations. If provided, it
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

    >>> # Specifying forbidden dimensions (e.g., not cropping the last dimension)
    >>> cropped_tensor = random_crop(tensor, forbidden_dims=[0, 1, 3])
    >>> print(cropped_tensor.shape)
    torch.Size([2, 3, 64, 32])
    """

    raise NotImplementedError(
        "random_crop is deprecated. Use neurite.nn.functional.crop with "
        "size or scale_factor instead.",
    )


def random_clip(
    input_tensor: torch.Tensor,
    clip_min: Union[float, int] = 0,
    clip_max: Union[float, int] = 1,
    clip_prob: Union[float, int] = 0.5,
    seed: Union[int, None] = None,
) -> torch.Tensor:
    """
    Randomly clip values in a tensor to a specified range with a given probability.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor whose values may be clipped.
    clip_min : float or int, optional
        The minimum value for clipping. By default, 0.
    clip_max : float or int, optional
        The maximum value for clipping. By default, 1.
    clip_prob : float, optional
        The probability of applying the clipping operation. By default, 0.5.
    seed : int, optional
        A seed value for reproducibility. Default is None.

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

    ### Reproducibility with a seed
    >>> input_tensor = torch.tensor([1.5, -0.5, 3.0])
    >>> clipped_tensor1 = random_clip(input_tensor, clip_min=0, clip_max=1, seed=42)
    >>> clipped_tensor2 = random_clip(input_tensor, clip_min=0, clip_max=1, seed=42)
    >>> print(clipped_tensor1 is clipped_tensor2)
    True
    """
    raise NotImplementedError(
        "random_clip is deprecated. Use neurite.nn.functional.clip with "
    )
