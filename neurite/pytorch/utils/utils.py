"""
Functional utilities for the neurite project.

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
__all__ = [
    "identity",
    "soft_quantize",
    "mse",
    "create_gaussian_kernel",
    "gaussian_smoothing",
    "bernoulli",
    "apply_bernoulli_mask",
    "subsample_tensor",
    "subsample_tensor_random_dims",
    "upsample_tensor",
    "make_range",
    "random_clear_label",
    "sample_image_from_labels",
    "is_instantiated_normalization",
    "make_downsampling_conv_blocks",
    "make_upsampling_conv_blocks",
    "derive_dense_displcement_field_from_affines",
    "make_grid",
    "make_sample_checkerboard_image",
    "make_sample_flow",
    "cross_expand",
    "filter_dim",
    "crop_to_nearest_multiple",
    "logistic",
    "dice",
    "log_dice"
]

from typing import Union, List, Tuple
import inspect
import einops
import torch
import torch.nn.functional as F
from torch import nn
import neurite.pytorch as ne


def identity(input_argument):
    "Returns the `input_argument`."
    return input_argument


def soft_quantize(
    input_tensor: torch.Tensor,
    nb_bins: Union[int, ne.samplers.Sampler] = 16,
    softness: Union[float, int, ne.samplers.Sampler] = 1.0,
    min_clip: Union[float, int, ne.samplers.Sampler] = -float('inf'),
    max_clip: Union[float, int, ne.samplers.Sampler] = float('inf'),
    return_log: bool = False
) -> torch.Tensor:
    """
    This function softly maps continuous values to discrete bins while retaining smoothness,
    controlled by the `softness` parameter.

    This function is used to discretize continuous values into bins while preserving some continuity
    or smoothness in the data. It is especially useful in the context of machine learning, where it
    is desirable to have a differentiable version of a quantized quantity, allowing for backprop.
    Hard quantization is non-differentiable and creates gradients of zero, making gradient-based
    optimization impossible.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to softly quantize.
    nb_bins : float, int, or Sampler, optional
        The number of discrete bins to softly quantize the input values into. By default 16
    softness : float, int, or Sampler, optional
        The softness factor for quantization. A higher value gives smoother quantization.
        By default 1.0
    min_clip : float, int, or Sampler, optional
        Clip data lower than this value before calculating bin centers. By default `-float('inf')`
    max_clip : float, int, or Sampler, optional
        Clip data higher than this value before calculating bin centers. By default `float('inf')`
    return_log : bool, optional
        Optionally return the log of the softly quantized tensor. By default False

    Returns
    -------
    torch.Tensor
        Softly quantized tensor with the same dimensions as `input_tensor`.

    Examples
    --------
    >>> import torch
    >>> import matplotlib.pyplot as plt
    # Make a random 3D tensor with zero mean and unit variance.
    >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
    # Compute the softly quantized tensor with a low softness to approximate (and visualize) a
    # pseudo-hard quantization.
    >>> softly_quantized_tensor = soft_quantize(input_tensor, nb_bins=4, softness=0.5)
    # Visualize the softly quantized tensor.
    >>> plt.imshow(softly_quantized_tensor[0, 0, 16])
    """
    # Initialize and draw realizations from samplers from input arguments
    nb_bins = ne.samplers.make_sampler(ne.samplers.Fixed, nb_bins)()
    softness = ne.samplers.make_sampler(ne.samplers.Fixed, softness)()
    min_clip = ne.samplers.make_sampler(ne.samplers.Fixed, min_clip)()
    max_clip = ne.samplers.make_sampler(ne.samplers.Fixed, max_clip)()

    # Invert softness
    softness = 1 / softness

    # Optionally clip `input_tensor`
    input_tensor.clip_(min_clip, max_clip)

    # Get the bin centers
    bin_centers = torch.linspace(
        start=input_tensor.min(),
        end=input_tensor.max(),
        steps=nb_bins,
        device=input_tensor.device
    )

    # Compute the distance between each element in `input_tensor` and the centers of the bins.
    # The resultant has `nb_bins` channels in the last dimension, each corresponding to the distance
    # between that element's intensity (in pixel/voxel space) to the center of each of the bins.
    distances_to_bin_centers = torch.abs(input_tensor.unsqueeze(-1) - bin_centers)

    # Apply softmax along last dimension
    softly_quantized = F.softmax(-softness * distances_to_bin_centers, dim=-1)

    # Compute the softly quantized value by averaging bin centers weighted by softmax values
    softly_quantized = (softly_quantized * bin_centers).sum(dim=-1)

    # Optionally convert to log domain
    if return_log:
        softly_quantized.log_()

    return softly_quantized


def mse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean squared error (MSE) between the elements of `tensor1` and `tensor2`.

    Parameters
    ----------
    tensor1 : torch.Tensor
        An input tensor.
    tensor2 : torch.Tensor
        A tensor with the same shape as `tensor2`.

    Returns
    -------
    torch.Tensor
        The mean squared error between `tensor1` and `tensor2`.

    Examples
    --------
    >>> import torch
    # First tensor with zero mean, unit variance
    >>> tensor1 = torch.randn((1, 16, 16, 16))
    # Other tensor with zero mean, unit variance, and same shape as `tensor1`
    >>> tensor2 = torch.randn((1, 16, 16, 16))
    # Calculate loss
    >>> loss = mse(tensor1, tensor2)
    # Print loss (should be approximately 2.0)
    >>> print(loss)
    """

    return torch.mean((tensor1 - tensor2) ** 2)


def create_gaussian_kernel(
    kernel_size: Union[int, ne.samplers.Sampler] = 3,
    sigma: Union[float, int, ne.samplers.Sampler] = 1,
    ndim: int = 3,
    nchannels: int = 1
) -> torch.Tensor:
    """
    Create a {1D, 2D, 3D} Gaussian kernel.

    Parameters
    ----------
    kernel_size : Sampler or int, optional
        Size of Gaussian kernel. Default is 3.
    sigma : float, int, or Sampler, optional
        Standard deviation of the Gaussian kernel. Default is 1.
    ndim : int
        Dimensionality of the gaussian kernel. Default is 3.

    Returns
    -------
    torch.Tensor
        Tensor representing the {1D, 2D, 3D} Gaussian kernel with batch and channel dimensions.

    Examples
    --------
    >>> import torch
    # Make the kernel!
    >>> gaussian_kernel = create_gaussian_kernel(3, 1, 3)
    # Print shape (should have batch and channel dimensions)
    >>> gaussian_kernel.shape()
    torch.Size([1, 1, 3, 3, 3])
    """
    # Initialize and sample parameters
    kernel_size = ne.samplers.make_sampler(ne.samplers.Fixed, kernel_size)()
    sigma = ne.samplers.make_sampler(ne.samplers.Fixed, sigma)()

    # Create a coordinate grid centered at zero
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    grid = torch.stack(torch.meshgrid([coords] * ndim), -1)

    # Calculate the Gaussian function
    kernel = torch.exp(-((grid ** 2).sum(-1) / (2 * sigma ** 2)))

    # Normalize the kernel so that the sum of all elements is 1
    kernel = kernel / kernel.sum()

    # Reshape to 5D tensor for conv3d
    kernel = kernel.view(1, 1, *([kernel_size] * ndim))

    # Repeat the kernel for each channel (depth-wise convolution)
    if nchannels > 1:
        kernel = kernel.repeat(nchannels, nchannels, *([1] * ndim))

    return kernel


def gaussian_smoothing(
    input_tensor: torch.Tensor,
    kernel_size: Union[int, ne.samplers.Sampler] = 3,
    sigma: Union[float, int, ne.samplers.Sampler] = 1,
) -> torch.Tensor:
    """
    Applies Gaussian smoothing to the {1D, 2D, 3D} input tensor based on the given kernel size and
    sigma. Assumes tensor has a batch and channel dimension.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor, assumed to be 1D, 2D, or 3D.
    kernel_size : Sampler or int, optional
        Size of the Gaussian kernel, default is 3.
    sigma : float, int, or Sampler, optional
        Standard deviation of the Gaussian kernel, default is 1.

    Returns
    -------
    smoothed_tensor : torch.Tensor
        The smoothed tensor.

    Examples
    --------
    >>> import torch
    # Make a random input tensor
    >>> input_tensor = torch.rand(1, 1, 16, 16, 16)
    # Smooth it
    >>> smoothed_tensor = gaussian_smoothing(input_tensor)
    """
    # Sampling parameters
    kernel_size = ne.samplers.make_sampler(ne.samplers.Fixed, kernel_size)()
    sigma = ne.samplers.make_sampler(ne.samplers.Fixed, sigma)()

    # Infer dimensionality in voxel/pixel space. Squeeze to remove batch and/or channel dims.
    ndim = input_tensor.dim() - 2

    # Initialize the gaussian kernel
    gaussian_kernel = create_gaussian_kernel(
        kernel_size=kernel_size,
        sigma=sigma,
        ndim=ndim,
        nchannels=input_tensor.shape[1]
    )

    # Calculate padding size
    padding = kernel_size // 2
    # Make the padding symmetric and
    padding = torch.tensor(padding).repeat(ndim * 2)
    # Convert to tuple (F.pad takes a tuple of ints, not tensors)
    padding = tuple(padding.tolist())

    # Pad `input_tensor`
    padded_input_tensor = F.pad(input_tensor, padding, mode='reflect')

    # Make dictionary for the different convolution dimensionalities
    conv_fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[ndim]

    # Apply the smoothig operation
    smoothed_tensor = conv_fn(
        input=padded_input_tensor,
        weight=gaussian_kernel,
        padding=0,
    )

    return smoothed_tensor


def bernoulli(p: float = 0.5, shape: tuple = (1,)) -> torch.Tensor:
    """
    Sample from a Bernoulli distribution with a specified probability and shape.

    Parameters
    ----------
    p : float, optional
        Probability of realizing a success (i.e., the probability of a 1) from the Bernoulli
        distribution. By default, 0.5. Must be in the range [0, 1].
    shape : tuple, optional
        Shape of the output tensor, specifying the number of independent Bernoulli trials. Each
        entry represents the dimensions of the resulting tensor. By default, (1).

    Returns
    -------
    bernoulli_result: torch.Tensor
        A tensor of Bernoulli-distributed random samples with values of 0 or 1, representing results
        of independent Bernoulli trials.

    Examples
    --------
    >>> # Generate samples from the Bernoulli distribution
    >>> samples = bernoulli(p=0.25, shape=(1, 32, 32, 32))
    >>> # Mean (expectation) should be ~=`p`
    >>> print(samples.mean())
    tensor(0.2471)
    """
    # Make sampling domain.
    # Each element in this tensor represents the probability of realizing a 1.
    sampling_domain = torch.tensor(p).repeat(shape)

    # Sample from the bernoulli distribution
    bernoulli_result = torch.bernoulli(sampling_domain)

    return bernoulli_result


def apply_bernoulli_mask(input_tensor, p: float = 0.5, returns: str = None) -> torch.Tensor:
    """
    Apply a Bernoulli mask to a tensor.

    This function samples a Bernoulli mask with the parameter `p`, representing the probability of
    success (e.g. realizing a 1) and applies it to `input_tensor` by element-wise multiplcation. The
    The elements of `input_tensor` corresponding to successes in the mask are preserved, while
    failures (e.g. zeros) are set to zero.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor to be masked.
    p : float, optional
        Probability of realizing a success (i.e., the probability of a 1) in the mask. Successes are
        preserved in the input tensor such that higher values of this parameter correspond to more
        elements of the input tensor being preserved. By default 0.5. Must be in the range [0, 1].
    returns : str, {None, 'successes', 'failures'}
        Optionally return the subset of the input tensor corresponding to Bernoulli {'successes',
        'failures'}. By default None (returns the original tensor with failures set to zero)
        - Setting `returns = 'successes'` might be useful in sampling a subset of a large tensor to
        estimate the statistics of it. Such operations such as `torch.quantile()` are especially
        unfriendly to a large sample size.

    Returns
    -------
    torch.Tensor
        Masked tensor with approximately `p` * 100% elements preserved (or 1 - (`p` * 100%))
        elements dropped out.

    Examples
    --------
    ## Standard use case
    >>> # Define input tensor. (Filled with ones for demonstration purposes)
    >>> input_tensor = torch.ones((1, 32, 32, 32))
    >>> # Mask the tensor.
    >>> masked_tensor = apply_bernoulli_mask(input_tensor, p=0.9)
    >>> # Return the average value of the tensor of ones, approximating the expectation of the mask
    >>> # in this special case.
    >>> masked_tensor.mean()

    ## Returning successes only (as a flattened tensor representing elements from successful trials)
    >>> Define input tensor. (Filled with ones for demonstration purposes)
    >>> input_tensor = torch.ones((1, 32, 32, 32))
    >>> # Get masked tensor
    >>> masked_tensor = apply_bernoulli_mask(input_tensor, p=0.9, returns='successes')
    >>> # Compute original shape and masked shape
    >>> original_shape, masked_shape = input_tensor.flatten().shape[0], masked_tensor.shape[0]
    >>> # Compute difference in size as a percent. Should be ~= `p`
    >>> print((masked_shape/original_shape))
    """
    # Sample the Bernoulli mask with parameter `p`
    bernoulli_mask = bernoulli(p=p, shape=input_tensor.shape)

    # Clone the input tensor for future computations
    masked = torch.clone(input_tensor)

    if returns == 'successes':
        # Get all elements from `input_tensor` corresponding to Bernoulli failures.
        masked = masked[bernoulli_mask == 1]

    elif returns == 'failures':
        # Get all elements from `input_tensor` corresponding to Bernoulli failures.
        masked = masked[bernoulli_mask == 0]

    elif returns is None:
        # Drop out (zero) all bernoulli failures.
        masked[bernoulli_mask == 0] = 0

    else:
        raise ValueError(f"{returns} isn't supported!")

    return masked


def subsample_tensor(
    input_tensor: torch.Tensor,
    subsampling_dimension: int = 0,
    stride: int = 2
) -> torch.Tensor:
    """
    Subsamples `input_tensor` by a factor `stride` along the specified dimension.

    The `subsample_tensor()` function provides a convenient way to downsample a specified dimension
    of a PyTorch tensor by a given stride. This type of downsampling is achieved by interleaving
    dropouts, meaning that every `stride`-th element along the selected dimension is kept, while the
    others are discarded. This operation can be applied to tensors of any dimensionality, making it
    versatile for a variety of tensor structures.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The tensor to sample from.
    subsampling_dimension : int, optional
        The dimension (or axis) along which the subsampling will occur. By default 0.
    stride : int, optional
        Factor by which to subsample (interleave dropouts). By default 2.

    Returns
    -------
    subsampled_tensor : torch.Tensor
        Tensor that has been subsampled.

    Examples
    --------
    >>> import torch
    # Defining two dimensional input tensor of shape (5, 5)
    >>> input_tensor = torch.arange(25).view(5, 5)
    # Visualize the tensor
    >>> print(input_tensor)
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    # Lets subsample along the first dimension (the columns)
    >>> subsampled_tensor = subsample_tensor(input_tensor, subsampling_dimension=1)
    # With the default stride (of 2), every other column should have been dropped out.
    >>> print(subsampled_tensor)
    tensor([[ 0,  2,  4],
            [ 5,  7,  9],
            [10, 12, 14],
            [15, 17, 19],
            [20, 22, 24]])
    # We could, of course, keep the default `subsampling_dimension=0` and subsample the rows:
    >>> subsampled_tensor = subsample_tensor(input_tensor, subsampling_dimension=1)
    >>> print(subsampled_tensor)
    tensor([[ 0,  1,  2,  3,  4],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24]])
    """
    # Make a list of slices that we will modify individually.
    slices = [slice(None)] * input_tensor.ndim

    # Slice the `axis` dimension with a given step size. Keep everything else the same.
    slices[subsampling_dimension] = slice(None, None, stride)

    # Slice the `input_tensor` with all slices to make the subsampled tensor.
    subsampled_tensor = input_tensor[tuple(slices)]

    return subsampled_tensor


def subsample_tensor_random_dims(
    input_tensor: torch.Tensor,
    stride: int = 2,
    forbidden_dims: list = (0, 1),
    p: float = 0.5,
    max_concurrent_subsamplings: int = None
) -> torch.Tensor:
    """
    Subsamples the input tensor along randomly selected dimensions, with constraints
    on which dimensions to subsample (`forbidden_dims`), the stride, and the probability of
    subsampling.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor to be subsampled. Assumed to have batch and channel dimensions.
    stride : Sampler or int or tuple optional
        The stride value to use when subsampling a given dimension. Can be int, Sampler, or tuple
        corresponding to the range of strides to sample. By default, 2.
            - When stride is an int, the stride is considered to be fixed
            - When the stride is a tuple of two elements, the elements correspond to the upper and
            lower bounds of a uniformly distributed integer sampler.
            - When a sampler is passed, use that sampler to sample the strides at each calll
            - A stride of 1 does not result in any subsampling.
            - A stride of 2 will reduce the elements of the selected dimension by 1/2.
    forbidden_dims : list, optional
        A list of dimensions that should not be subsampled. If None, no dimensions
        are forbidden from subsampling. Default is (0, 1) to ignore batch and channel dimensions.
    p : float, optional
        The probability of selecting each dimension for subsampling. This probability 
        is applied as an independent Bernoulli trial for each dimension. By default, 0.5.
    max_concurrent_subsamplings : int, optional
        The maximum number of dimensions that can be subsampled simultaneously. If
        None, the number of concurrent subsamplings is set to the number of dimensions
        in `input_tensor`. Default is None.

    Returns
    -------
    torch.Tensor
        The subsampled tensor after applying the specified dimensional subsampling.

    Examples
    --------
    >>> import torch
    >>> # Defining input tensor with batch and channel dimensions, and spatial dims=(5, 5)
    >>> input_tensor = torch.arange(25).view(1, 1, 5, 5)
    >>> # Visualize the tensor
    >>> print(input_tensor)
    tensor([[[[ 0,  1,  2,  3,  4],
              [ 5,  6,  7,  8,  9],
              [10, 11, 12, 13, 14],
              [15, 16, 17, 18, 19],
              [20, 21, 22, 23, 24]]]])
    >>> # Subsample the tensor. This may now (randomly) subsample more than one dimension.
    >>> subsampled_tensor = subsample_tensor_random_dims(input_tensor)
    >>> print(subsampled_tensor)
    tensor([[[[ 0,  3],
              [10, 13],
              [20, 23]]]])
    >>> # Subsample by defining the stride range.
    >>> subsampled_tensor = subsample_tensor_random_dims(input_tensor, stride=(3, 4))
    >>> print(subsampled_tensor)
    tensor([[[[ 0,  4],
              [20, 24]]]])
    """
    # Determine how many dimensions should be subsampled at once
    if max_concurrent_subsamplings is None:
        # If None, we will subsample at most *all* of them (at once!)
        max_concurrent_subsamplings = input_tensor.dim()

    elif max_concurrent_subsamplings <= input_tensor.dim():
        # Great. It's already defined :)
        pass

    elif max_concurrent_subsamplings > input_tensor.dim():
        # Sometimes, you might try to define a `max_concurrent_subsamplings` that's not possible :(
        raise ValueError(
            f"Your tensor doesn't have {max_concurrent_subsamplings} dimensions!"
        )

    # Sample the dimensions (to subsample) by randomly permuting the list of allowed dimensions and
    # taking the first `max_concurrent_subsamplings`
    dimensions_to_subsample = torch.randperm(
        input_tensor.dim()
    )[:max_concurrent_subsamplings]

    # Remove all forbidden dimensions (dimensions that should not be subsampled)
    if forbidden_dims is not None:
        # Convert to tensor
        forbidden_dims = torch.Tensor(forbidden_dims)
        # Make mask to remove elements in `dimensions_to_subsample` that are in `forbidden_dims`
        mask = torch.isin(dimensions_to_subsample, forbidden_dims)
        # Invert mask and apply
        dimensions_to_subsample = dimensions_to_subsample[~mask]

    # We might not want to subsample the same number of dimensions every time as defined by
    # `max_concurrent_subsamplings`, so we'll mask some out with iid Bernoulli trials. 
    dimensions_to_subsample = apply_bernoulli_mask(
        input_tensor=dimensions_to_subsample,
        p=p,
        returns='successes'
    )

    # If the stride is an int we'll set it to be a fixed sampler.
    # This prevents us from trying to stride 0 elements (not possible), and one element (no effect).
    if isinstance(stride, int | float):
        stride_sampler = ne.samplers.make_sampler(ne.samplers.Fixed, stride)
    else:
        stride_sampler = ne.samplers.make_sampler(ne.samplers.RandInt, stride)

    # Perform the subsampling.
    for dimension in dimensions_to_subsample:
        # Sample the stride
        sampled_stride = stride_sampler()
        # Apply the subsampling operation
        input_tensor = subsample_tensor(
            input_tensor,
            subsampling_dimension=dimension,
            stride=sampled_stride
        )

    return input_tensor


def upsample_tensor(
    input_tensor: torch.Tensor,
    shape: tuple,
    mode: str = 'nearest',
) -> torch.Tensor:
    """
    Upsamples 1D, 2D, or 3D tensors to a given `shape`.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor to be upsampled. Assumed to have batch and channel dimensions.
    shape : tuple
        Spatial dimensions (without batch or channel dimensions) to upsample `input_tensor` into.
    mode : str, optional
        The interpolation mode to use. By default None. Options (WRT spatial dimensions) include:
            - 'nearest' (default)
            - 'linear' (1D-only)
            - 'bilinear' (2D-only)
            - 'bicubic' (2D-only)
            - 'trilinear' (3D-only)
            - 'area'

    Examples
    --------
    >>> # 2D Upsampling
    >>> input_tensor = torch.randn(1, 3, 32, 32)  # (B, C, H, W)
    >>> upsampled_tensor = upsample_tensor(input_tensor, shape=(64, 64), mode='bilinear')
    >>> print(upsampled_tensor.shape)
    torch.Size([1, 3, 64, 64])
    >>> # 3D Upsampling
    >>> input_tensor = torch.randn(1, 3, 32, 32, 32)  # (B, C, D, H, W)
    >>> upsampled_tensor = upsample_tensor(input_tensor, shape=(64, 64, 64), mode='bilinear')
    >>> print(upsampled_tensor.shape)
    torch.Size([1, 3, 64, 64, 64])
    """
    # Calculate the spatial dimensions (disregarding batch and channel)
    spatial_dims = input_tensor.dim() - 2
    if spatial_dims not in [1, 2, 3]:
        raise ValueError(
            f"Unsupported tensor dimensionality: {spatial_dims} spatial dimensions. "
            "Only 1D, 2D, and 3D tensors are supported."
        )

    # Perform the upsampling operation
    upsampled = F.interpolate(input_tensor, size=shape, mode=mode)

    return upsampled


def make_range(*args, **kwargs) -> tuple:
    """
    Creates a tuple specigying the bounds for a range of numbers `(min, max)`.

    This function generates a tuple containing the min and max values for a range. The range can be
    flexibly defined through positional and/or keyword arguments. If only one positional argument is
    provided, it is interpreted as `max` with `min` defaulting to 0. Keyword arguments can be used
    to explicitly set `min` and/or `max`, overriding positional arguments.

    Parameters
    ----------
    min : int or float, optional
        The minimum value of the range.
    max : int or float, optional
        The maximum value of the range.

    Returns
    -------
    tuple of (int or float, int or float)
        A tuple containing the minimum and maximum values `(min, max)`.

    Examples
    --------
    >>> # Using two positional arguments
    >>> rng = make_range(0, 19.7)
    >>> print(rng)
    (0, 19.7)
    >>> # Using one positional argument
    >>> rng = make_range(5)
    >>> print(rng)
    (0, 5)
    >>> # Using keyword arguments
    >>> rng = make_range(min=0.6, 1)
    >>> print(rng)
    (0.6, 1)
    """
    # Return arguments of type {Sampler, list, tuple} as-is
    for arg in args:
        if isinstance(arg, ne.samplers.Sampler):
            return arg
        elif isinstance(arg, (list, tuple)):
            return arg

    # Return keyword arguments of type {Sampler, list, tuple} as-is
    for arg in kwargs.values():
        if isinstance(arg, ne.samplers.Sampler):
            return arg
        elif isinstance(arg, (list, tuple)):
            return arg

    # Setting default values
    min, max = 0, 1
    # Handle positional arguments
    if len(args) == 2:
        min, max = args

    elif len(args) == 1:
        if isinstance(args[0], list | tuple):
            # if the argument is a list, unpack it:
            min, max = args[0]
        else:
            # if only one input arg is defined, interpret it as `max`
            min, max = 0, args[0]

    # Handle kwargs (if they exist)
    if 'min' in kwargs:
        min = kwargs['min']
    if 'max' in kwargs:
        max = kwargs['max']

    # Thrown an error if min is greater than max
    if max <= min:
        raise ValueError("`max` must be greater than `min`.")

    return (min, max)


def random_clear_label(
    input_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    prob: Union[float, int, ne.samplers.Sampler] = 0.5,
    exclude_zero: bool = True,
    seed: int = None
) -> torch.Tensor:
    """
    Randomly clears/erases regions from an image corresponding to randomly selected regions in a
    label map.

    This function identifies unique labels within the `label_tensor` and, based on a specified
    probability, clears (sets to zero) the corresponding regions in the `input_tensor`. This can be
    used for tasks such as data augmentation, where certain labels are randomly omitted to
    simulate occlusions or missing annotations.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Image or tensor to clear.
    label_tensor : torch.Tensor
        Label map corresponding to sampling domain from which to select regions for clearing.
    prob : Union[float, int, Sampler], optional
        Probability of any label/region being selected for erasure as determined by iid Bernoulli
        trials, by default 0.5.
    exclude_zero : bool, optional
        Optionally exclude zero (uaually background) from the list of potential regions to clear
        (never clear zero labels), by default True.
    seed : int, optional
        A random seed or sampler to control the randomness of label clearing operations. If
        provided, it ensures reproducibility of the clearing process. By default, None.

    Returns
    -------
    torch.Tensor
        The modified tensor with specified labels cleared (set to zero). If no labels are cleared,
        the original `input_tensor` is returned unchanged.

    Examples
    --------
    ### Clearing labels with a fixed probability
    >>> input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> label_tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> cleared_tensor = random_clear_label(input_tensor, label_tensor, prob=0.5)
    >>> print(cleared_tensor)
    tensor([0.0, 0.0, 0.3, 0.0, 0.5, 0.6])

    ### Excluding label `0` from being cleared
    >>> input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> label_tensor = torch.tensor([0, 0, 0, 0, 0, 0])
    >>> cleared_tensor = random_clear_label(input_tensor, label_tensor, prob=1.0, exclude_zero=True)
    >>> print(cleared_tensor)
    torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    ### Reproducibility with a seed
    >>> input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> label_tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> cleared_tensor1 = random_clear_label(input_tensor, label_tensor, prob=1.0, seed=42)
    >>> cleared_tensor2 = random_clear_label(input_tensor, label_tensor, prob=1.0, seed=42)
    >>> print(torch.equal(cleared_tensor1, cleared_tensor2))
    True
    """
    # Initialize random seed if provided
    if seed is not None:
        if isinstance(seed, ne.samplers.Sampler):
            seed = seed()
        torch.manual_seed(seed)

    # Determine all unique labels
    unique_labels = torch.unique(label_tensor)
    # Optionally exclude zero label (usually background)
    if exclude_zero:
        unique_labels = unique_labels[unique_labels != 0]

    # Apply Bernoulli mask to determine which labels to clear
    labels_to_clear = apply_bernoulli_mask(unique_labels, prob, returns='successes')

    # Clear the specified labels in the input tensor
    for label in labels_to_clear:
        input_tensor.masked_fill_(label_tensor == label, 0)

    return input_tensor


def sample_image_from_labels(
    label_tensor: torch.Tensor,
    mean_sampler: ne.samplers.Sampler = ne.samplers.Uniform(0, 1),
    noise_sampler: ne.samplers.Sampler = ne.samplers.Normal,
    noise_variance: Union[float, int, ne.samplers.Sampler] = 0.25
) -> torch.Tensor:
    """
    Sample textures/intensities from an integer label map.

    This function identifies all unique integer labels in the `label_tensor`, and assigns each a
    mean intensity to the labeled region in the corresponding output image (`sampled_image`). The
    mean intensity serves as the mean for a noise distribution modeled by `noise_sampler`. The
    variance of the noise model may be a fixed quantity or sampled from another distribution defined
    by `noise_variance`.

    Parameters
    ----------
    label_tensor : torch.Tensor
        A tensor with batch and channel dimensions containing integer labels defining distinct
        regions.
    mean_sampler : Sampler
        A `Sampler` from which to draw the mean intensity for each region defined by each label in
        the `label_tensor`. By default, `Uniform(0, 1)`
    noise_sampler : Sampler
        A `Sampler` that is used to model the noise within a particular label/region. The mean for
        the sampler is defined by the mean region intensity (sampled from `mean_sampler`).
        By default, `Normal`.
    noise_variance : float, int, or Sampler
        The variance of the noise model. It can be a fixed quantity (int or float), or a sampled
        quantity in the case a `Sampler` is passed. By default, 0.25.

    Returns
    -------
    torch.Tensor
        A tensor of sampled image intensities with the same shape as `label_tensor`.
    """
    # Make the variance
    noise_variance = ne.samplers.make_sampler(ne.samplers.Fixed, noise_variance)
    # Extract unique labels
    unique_labels = torch.unique(label_tensor)

    # Initialize the sampled image
    sampled_image = torch.zeros_like(label_tensor).float()

    # Iteratevly texturize/sample intensities for each region as specified by a label
    for label in unique_labels:
        # Determine the mean value of the region
        mean_region_intensity = mean_sampler()

        # Sample the texturized region
        texturized_redion = noise_sampler(
            mean_region_intensity, noise_variance()
        )(label_tensor[label_tensor == label].shape)

        # Assign the textures to the region of the label
        sampled_image[label_tensor == label] = texturized_redion

    return sampled_image


def is_instantiated_normalization(obj: object) -> bool:
    """
    Determine if an object is a normalization layer that has been instantiated.

    Parameters
    ----------
    obj : object
        Object to checked.

    Returns
    -------
    bool
        Whether the object is an instantiated normalization layer or not.
    """
    # Get all classes from torch.nn.modules.normalization
    normalization_classes = tuple(
        cls for _, cls in inspect.getmembers(torch.nn.modules.normalization, inspect.isclass)
        if issubclass(cls, torch.nn.Module)
    )
    return isinstance(obj, normalization_classes)


def make_downsampling_conv_blocks(
    ndim: int,
    nb_features: List[int],
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    norms: Union[str, nn.Module, None] = None,
    activations: Union[str, nn.Module, None] = "relu",
    pool_mode: str = "max",
    pool_kernel_size: int = 2,
    order: str = 'nca',
    return_residual: bool = False,
) -> nn.ModuleList:
    """
    Create an `nn.ModuleList` of downsampling conv blocks based the number of features per layer.

    Parameters
    ----------
    ndim : int
        Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
    nb_features : List[int]
        Number of features at each level of the downsampling convs.
    kernel_size : int, optional
        Size of the convolving kernel at each level. Default is 3.
    stride : int, optional
        Stride of the convolution. Default is 1.
    padding : int, optional
        Padding added to all sides of the input. Default is 1.
    norms : list, str, nn.Module, or None, optional
        Normalization layers for each downsampling conv block.
    activations : list, str, nn.Module, or None, optional
        Activation function for each downsampling conv block.
    pool_mode : str, optional
        Pooling mode ('max' or 'avg'). Default is 'max'.
    pool_kernel_size : int, optional
        Kernel size for pooling. Default is 2.
    order : str, optional
        The order of operations in each downsampling conv block. Default is 'nca' (normalization ->
        convolution -> activation). Each character in the string can be specified an arbitrary
        number of times in any order. Each character in the string represents one of the following:
        - `'c'`: Convolution
        - `'n'`: Normalization
        - `'a'`: Activation
    return_residual : bool
        Optionally return a residual (skip connection) from the output of the forward pass.

    Returns
    -------
    nn.ModuleList
        A list of downsampling convolutional blocks.

    Examples
    --------
    >>> downsampling_conv_blocks = make_downsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[3, 16, 32],
    ...     kernel_size=3,
    ...     norms=["batch", "instance", "batch"],
    ...     activations="relu"
    ... )
    >>> print(downsampling_conv_blocks)
    ModuleList(...)
    """

    # Normalization layers
    if not isinstance(norms, list):
        norms = [norms] * len(nb_features)

    # Activation layers
    if not isinstance(activations, list):
        activations = [activations] * len(nb_features)

    # Init container for downsampling convs
    downsampling_conv_blocks = nn.Sequential()

    # Make downsampling conv block and append to list of them
    for i in range(len(nb_features) - 1):

        downsampling_conv_block = ne.modules.DownsampleConvBlock(
            ndim=ndim,
            in_channels=nb_features[i],
            out_channels=nb_features[i + 1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norms[i],
            activation=activations[i],
            pool_mode=pool_mode,
            pool_kernel_size=pool_kernel_size,
            order=order,
            return_residual=return_residual,
        )

        downsampling_conv_blocks.append(downsampling_conv_block)

    return downsampling_conv_blocks


def make_upsampling_conv_blocks(
    ndim: int,
    nb_features: List[int],
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    upsample_kernel_size: int = 4,
    upsample_stride: int = 2,
    upsample_padding: int = 1,
    norms: List[Union[str, nn.Module, None]] = None,
    activations: List[Union[str, nn.Module, None]] = None,
    order: str = 'nca',
    accepts_residuals: bool = True,
) -> nn.ModuleList:
    """
    Create an `nn.ModuleList` of upsampling conv blocks based the number of features per layer/
    level.

    Parameters
    ----------
    ndim : int
        Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
    nb_features : List[int]
        Number of features at each upsampling conv block.
    kernel_size : int, optional
        Size of the convolving kernel at each level. Default is 3.
    stride : int, optional
        Stride of the convolution at each level. Default is 1.
    padding : int, optional
        Padding added to all sides of the input at each level. Default is 1.
    upsample_kernel_size : int, optional
        Kernel size for the transposed convolution at each level. Default is 4.
    upsample_stride : int, optional
        Stride for the transposed convolution at each upsampling conv block. Default is 2.
    upsample_padding : int, optional
        Padding for the transposed convolution at each level. Default is 1.
    norms : list, str, nn.Module, or None, optional
        Normalization layers for each upsampling conv block at each level. If a list, must have the
        same length as nb_features.
    activations : list, str, nn.Module, or None, optional
        Activation functions for each upsampling conv block at each level. If a list, must have the
        same length as nb_features.
    order : str, optional
        The order of operations in the block. Default is 'nca' (normalization -> convolution ->
        activation). Each character in the string can be specified an arbitrary number of times
        in any order. Each character in the string represents one of the following:
        - `'c'`: Convolution
        - `'n'`: Normalization
        - `'a'`: Activation
    accepts_residuals : bool
        If True, the blocks are configured to accept residual connections. This doubles the
        expected number of input channels, allowing the blocks to concatenate skip features with
        the main input.

    Returns
    -------
    nn.ModuleList
        A list of upsampling convolutional blocks.

    Notes
    -----
    - If `norms` or `activations` are a list, they must be the same length as the nb_features.

    Examples
    --------
    >>> upsampling_conv_blocks = make_upsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[32, 16, 4],
    ...     upsample_kernel_size=4,
    ...     norms=["batch", "instance", "batch"],
    ...     activations="relu"
    ... )
    >>> print(upsampling_conv_blocks)
    ModuleList(...)
    """

    # Normalization layers
    if not isinstance(norms, list):
        norms = [norms] * len(nb_features)

    # Activation layers
    if not isinstance(activations, list):
        activations = [activations] * len(nb_features)

    # Init upsampling conv blocks container
    upsampling_conv_blocks = nn.Sequential()

    # make the number of features for the upsampling conv blocks
    nb_features = [*nb_features, nb_features[-1]]

    # Make upsampling conv block and append to list of them
    for i in range(len(nb_features) - 1):

        upsampling_conv_block = ne.modules.UpsampleConvBlock(
            ndim=ndim,
            in_channels=nb_features[i],
            out_channels=nb_features[i + 1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            upsample_kernel_size=upsample_kernel_size,
            upsample_stride=upsample_stride,
            upsample_padding=upsample_padding,
            norm=norms[-i],
            activation=activations[-i],
            order=order,
            accepts_residuals=accepts_residuals,
        )

        upsampling_conv_blocks.append(upsampling_conv_block)

    return upsampling_conv_blocks


def derive_dense_displcement_field_from_affines(
    affine_a: torch.Tensor,
    affine_b: torch.Tensor,
    grid_size: tuple,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    normalize: bool = True
) -> torch.Tensor:
    """
    Derive a dense displacement field from affine matrices.

    Parameters
    ----------
    affine_a : torch.Tensor
        Affine matrix A of shape (batch_size, ndim, ndim + 1), where ndim is 2 or 3.
    affine_b : torch.Tensor
        Affine matrix B of shape (batch_size, ndim, ndim + 1), same shape as affine_A.
    grid_size : tuple
        Spatial size of the grid, e.g., (H, W) for 2D or (D, H, W) for 3D.
    device : torch.device, optional
        Device for computations, default is 'cpu'.
    dtype : torch.dtype, optional
        Data type for computations, default is torch.float32.
    normalize : bool, optional
        If True, grid coordinates are normalized to [-1, 1]. Default is True.

    Returns
    -------
    torch.Tensor
        Dense displacement field of shape (batch_size, ndim, *grid_size), where each
        vector represents displacement in each dimension from a to b.

    Examples
    --------
    ### Dense displacement field for 2x scaled affines
    >>> # Make first affine with ones
    >>> aff_a_2d = torch.eye(2, 2 + 1).unsqueeze(0)
    >>> # Dilate original affine by 2
    >>> aff_b_2d = aff_a_2d * 2
    >>> grid_size_2d = (128, 128)
    >>> displacement_field = derive_dense_displacement_field_from_affines(
                                aff_a_2d, aff_b_2d, grid_size_2d
                            )
    """

    # Input validation (ensuring F.affine_grid() will be happy)
    assert affine_a.dim() == 3 and affine_b.dim() == 3, "Affine matrices must be 3D tensors"
    assert affine_a.shape == affine_b.shape, "Affine matrices must have the same shape"

    # Validate rectangular shape of affine matricies
    batch_size, ndim, ndim_plus_one = affine_a.shape
    assert ndim_plus_one == ndim + 1, "Affine shape should be (batch_size, ndim, ndim+1)"
    assert ndim in [2, 3], "Only 2D and 3D transformations are supported"

    # Generate grids/flows for A and B using torch's affine_grid()
    grid_a = F.affine_grid(affine_a, size=(batch_size, 1, *grid_size), align_corners=True)
    grid_b = F.affine_grid(affine_b, size=(batch_size, 1, *grid_size), align_corners=True)

    # Order of dimensions to permute (nD)
    permuting_order = [0, ndim_plus_one] + list(range(1, ndim_plus_one))
    # Calculate the displacement
    displacement = grid_b - grid_a

    # Permute the dimensions and make contiguious. Returns shape: (B, ndim, *grid_size)
    displacement = displacement.permute(*permuting_order).contiguous()

    if not normalize:
        # Scale the displacement by the grid size
        scale = torch.tensor(grid_size, device=device, dtype=dtype).view(1, ndim, *[1] * ndim)
        displacement *= scale

    return displacement


def make_grid(
    size: Tuple[int],
    device: Union[str, torch.device] = "cpu",
    dtype: Union[str, torch.dtype] = torch.float32,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Generate a grid of spatial coordinates.

    This function defines the coordinate axes by generating vectors for each spatial dimension
    represented by the elements of `shape`. It then creates a grid representing all spatial coords.

    Parameters
    ----------
    size : Tuple[int] 
        Size of the spatial dimensions of the input tensor. e.g. (H, W) or (D, W, H)
    device : Union[str, torch.device], optional
        The device on which the grid will reside. By default "cpu"
    dtype : Union[str, torch.dtype], optional
        The data type of the tensor grid, by default None
    normalize : bool, optional
        Normalize each dimension of the grid to the range [-1, 1]. Otherwise, the grid coords span
        from 0 to `size[i] - 1` for each dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape `1, *size, len(size)` representing the grid of spatial coordinates.

    Examples
    --------
    ### Make a 2d grid of size (3, 2)
    >>> grid = make_grid(size=(3, 2))
    >>> print(grid.shape)
    torch.Size([1, 2, 3, 2])
    >>> print(grid)
    tensor([[[[-1., -1.],
            [ 0.,  0.],
            [ 1.,  1.]],
            [[-1.,  1.],
            [-1.,  1.],
            [-1.,  1.]]]])
    """

    # Define coordinate axes/vectors: for each dimension in `size`, create a 1D vector for the
    # coord system
    axes = []

    for axis_length in size:

        # Construct the axis for the ith spatial dimension
        if normalize:
            # Create the axis on [-1, 1], with the origin (ideally) at zero
            axis = torch.linspace(-1, 1, steps=axis_length, device=device, dtype=dtype)

        else:
            # Create the axis to the `axis_length`
            axis = torch.linspace(0, axis_length, steps=axis_length, device=device, dtype=dtype)
        axes.append(axis)

    # Make grid as a tuple of torch.Tensor
    grid = torch.meshgrid(*axes, indexing="ij")

    # Stack the grid tuples to make a tensor, and create new leading singleton dimension
    grid = torch.stack(grid)

    # Move the coordinate dim/axis to the back
    grid = grid.moveaxis(0, -1).contiguous().unsqueeze(0)

    return grid


def make_sample_checkerboard_image(
        image_shape: tuple = (1, 1, 16, 16),
        square_size: int = 3,
        device: str = "cpu"
):
    """
    Generate a checkerboard pattern in 2D or 3D.

    This function creates an image with a checkerboard pattern where alternating
    squares of size `square_size` are filled with ones, while the rest remain zero.

    Parameters
    ----------
    image_shape : tuple, optional
        Shape of the output image tensor. The expected format is:
        - (B, C, H, W) for 2D images
        - (B, C, D, H, W) for 3D images
        Default is (1, 1, 16, 16) for a single-channel 2D image.
    square_size : int, optional
        The size of each square in the checkerboard pattern.
        The default value is 3.

    Returns
    -------
    torch.Tensor
        A tensor of shape `image_shape` containing a checkerboard pattern.
        Alternating squares are set to 1.

    Example
    -------
    >>> img = make_sample_checkerboard_image((1, 1, 6, 6), square_size=2)
    >>> img[0, 0]
    tensor([[1., 1., 0., 0., 1., 1.],
            [1., 1., 0., 0., 1., 1.],
            [0., 0., 1., 1., 0., 0.],
            [0., 0., 1., 1., 0., 0.],
            [1., 1., 0., 0., 1., 1.],
            [1., 1., 0., 0., 1., 1.]])
    """

    # Extract spatial dimensions
    spatial_dims = image_shape[2:]

    # Init the checkerboard tensor on the device
    checkerboard_image = torch.zeros(image_shape, device=device)

    # Create starting points on the axes for the squares (either light or dark)
    checkerboard_startpoints_for_axes = []
    for dim in spatial_dims:

        # Make the starting points alternate every `square_size`
        startpoints_for_axis = torch.arange(0, dim, square_size)
        checkerboard_startpoints_for_axes.append(startpoints_for_axis)

    # Get the cartesian product of all dims to make points in (2D or 3D) space
    checkerboard_start_coords = torch.cartesian_prod(*checkerboard_startpoints_for_axes)

    for start_coord in checkerboard_start_coords:
        # Fill image with ones for all spatial dims starting at the point
        if start_coord.sum().item() % (2 * square_size) == 0:
            slices = tuple(slice(i, i + square_size) for i in start_coord)
            checkerboard_image[(..., *slices)] = 1

    return checkerboard_image


def make_sample_flow(
    shape: tuple = (1, 1, 16, 16),
    device: str = 'cpu',
    shift_size: int = 1,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Makes a simple flow field for testing registration in N-dimensional space.

    This function generates a flow field with channels that represent the transformations to each
    spatial dimension. E.g. channel 1 represents the dense transformation on the x-axis, channel 2
    represents the dense transformation on the y axis, and so on...

    Parameters
    ----------
    shape : tuple, optional
        Shape of the input tensor, expected as (B, C, *spatial_dims).
        Default is (1, 1, 4, 4) for a 2D case.
    device : str, optional
        The device to allocate tensors to ('cpu' or 'cuda').

    Returns
    -------
    flow_field : torch.Tensor
        A tensor representing the flow field, shaped as (B, n_spatial_dims, *spatial_dims).
        The first spatial dimension is shifted by +1 in a normalized manner.

    Example
    -------
    >>> flow = create_sample_flow((1, 1, 4, 4), device='cpu')
    >>> flow.shape
    torch.Size([1, 2, 4, 4])

    >>> flow_3d = create_sample_flow((1, 1, 4, 4, 4), device='cpu')
    >>> flow_3d.shape
    torch.Size([1, 3, 4, 4, 4])
    """
    spatial_dims = shape[2:]  # Extract spatial dimensions
    n_spatial_dims = len(spatial_dims)

    # Create a flow field tensor
    flow_field = torch.zeros(shape[0], n_spatial_dims, *spatial_dims, device=device)

    # Shift along the first spatial dimension
    flow_field[:, 0, ...] = shift_size

    # Optionally normalize
    if normalize:
        flow_field[:, 0, ...] /= (spatial_dims[0] - 1)

    return flow_field


def cross_expand(
    x1: torch.Tensor,
    x2: torch.Tensor,
    return_batched: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expands `x1` and `x2` along new dimensions to create pairwise combinations.

    Each slice in `x1` is expanded along a new axis to match every slice in `x2`, and vice versa.
    This is essentially just taking the cartesian product of two tensors at their second dimension.

    Parameters
    ----------
    x1 : torch.Tensor
        Input tensor of shape (B, Sx1, Cx1, ...), where Sx1 is the number of slices or subimages.
    x2 : torch.Tensor
        Input tensor of shape (B, Sx2, Cx2, ...), where Sx2 is the number of slices or subimages.
    return_batched : bool, optional
        Return paired expanded tensors patched into the batch dimension.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        - If `return_batched=True`, returns:
            - `batched_paired_tensors` paired expanded tensors patched into the batch dimension.
        - If `return_batched=False`, returns
            - `x1_expanded` of shape (B, Sx1, Sx2, Cx1, ...) where each slice in `x1` is expanded.
            - `x2_expanded` of shape (B, Sx1, Sx2, Cx2, ...) where each slice in `x2` is expanded.

    References
    ----------
    J. G. Ortiz et al., "UniverSeg: Universal Medical Image Segmentation," 
    GitHub repository, 2023. Available: https://github.com/JJGO/UniverSeg

    Examples
    --------
    ### Cross expansion of two 2D tensors
    >>> x1 = torch.randn(1, 3, 4, 5, 6)
    >>> x2 = torch.randn(1, 7, 8, 9, 10)
    >>> x1_cross_expanded, x2_cross_expanded = cross_expand(x1, x2)
    >>> print(x1_cross_expanded.shape, x2_cross_expanded.shape)
    torch.Size([1, 3, 7, 4, 5, 6]) torch.Size([1, 3, 7, 8, 9, 10])
    """

    # Unpack to get Sx1 and Sx2 slice dimensions
    Bx1, Sx1, Cx1, *x1_spatial = x1.shape  # Could've used x1.size(1), but I like it this way :)
    Bx2, Sx2, Cx2, *x2_spatial = x2.shape

    if Bx1 != Bx2:
        raise ValueError(
            f"The input tensors must have the same number of batches. Got Bx1={Bx1} and Bx2={Bx2}")

    # n-Dimensional reshaping/cartesian product of tensors
    x1_expanded = einops.repeat(x1, "Bx1 Sx1 Cx1 ... -> Bx1 Sx1 Sx2 Cx1 ...", Sx2=Sx2)
    x2_expanded = einops.repeat(x2, "Bx2 Sx2 Cx2 ... -> Bx2 Sx1 Sx2 Cx2 ...", Sx1=Sx1)

    if return_batched:

        # Raise an error if we're not going to be able to concatenate them
        if Bx1 != Bx2 or x1_spatial != x2_spatial:
            raise ValueError(
                "The tensors must match in their batch and spatial dimensions. Got:"
                f"x1.shape: {x1.shape}, x2.shape: {x2.shape}"
            )

        # Concatenate the expanded tensors along their batch dimension
        paired_tensors = torch.cat([x1_expanded, x2_expanded], dim=3)

        # Take advantage of the batch dimension collect the slices/subimages
        batched_paired_tensors = einops.rearrange(
            paired_tensors, "B Sx1 Sx2 C ... -> (B Sx1 Sx2) C ..."
        )

        return batched_paired_tensors

    else:

        return x1_expanded, x2_expanded


def filter_dim(
    tensor: torch.Tensor,
    dim: int = 0,
    verbose: bool = False
) -> torch.Tensor:
    """
    Filters dimensions of a tensor that contain NaNs, infinite values, or are entirely zero.

    Parameters:
    ----------
    tensor : torch.Tensor
        An n-dimensional tensor.
    verbose : bool, optional
        If True, prints the number of elements filtered for each condition. Default is False.

    Returns:
    -------
    torch.Tensor
        The filtered tensor containing only dim elements without NaNs, infinite values, and not
        entirely zeros.
    """

    dims_to_test = list(range(tensor.dim()))
    dims_to_test.remove(dim)

    # Create mask for batches without any NaN values.
    nan_mask = ~torch.isnan(tensor).any(dim=dims_to_test)
    nan_mask = torch.nonzero(nan_mask, as_tuple=True)[0]

    # Filter out batches that contain NaNs
    filtered_tensor = torch.index_select(tensor, dim, nan_mask)

    # Create mask for batches without any infinite values
    inf_mask = ~torch.isinf(filtered_tensor).any(dim=dims_to_test)
    inf_mask = torch.nonzero(inf_mask, as_tuple=True)[0]

    # Filter out batches that contain infinite values
    filtered_tensor = torch.index_select(filtered_tensor, dim, inf_mask)

    # Create mask for batches that are not entirely zeros
    zero_mask = ~torch.all(filtered_tensor == 0, dim=dims_to_test)
    zero_mask = torch.nonzero(zero_mask, as_tuple=True)[0]
    # Filter out batches that are entirely zeros
    filtered_tensor = torch.index_select(filtered_tensor, dim, zero_mask)

    if verbose:
        # Print number of batches removed due to NaNs
        n_nans = torch.sum(~nan_mask)
        print("N Batches with NaNs: ", n_nans)

        # Pring number of batches removed due to infinite values
        n_infs = torch.sum(~inf_mask)
        print("N Batches with Inf: ", n_infs)

        # Print number of batches removed because they were entirely zeros
        n_zeros = torch.sum(zero_mask)
        print("N Batches with Zero: ", n_zeros)

    has_zero_dim = torch.any(
        torch.tensor(filtered_tensor.shape) == 0
    )

    if has_zero_dim:
        zero_dims = []
        for d, size in enumerate(tensor.shape):
            if size == 0:
                zero_dims.append(d)

        raise ValueError(
            f"Dimension {zero_dims} of the filtered tensor has shape == 0."
        )

    return filtered_tensor


def crop_to_nearest_multiple(tensor, multiple=128):
    """
    Crops the spatial dimensions of a tensor to the nearest multiple of
    `multiple`. Supports 1D, 2D, or 3D spatial dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor with shape (B, C, *spatial_dims), where `spatial_dims`
        can represent 1D, 2D, or 3D spatial dimensions.
    multiple : int, optional
        The multiple to which spatial dimensions are cropped. Default is 128.

    Returns
    -------
    torch.Tensor
        The tensor with spatial dimensions cropped to the nearest multiple of
        `multiple`.

    Examples
    --------
    >>> import torch
    >>> tensor_1d = torch.randn(1, 3, 250)  # 1D spatial tensor
    >>> cropped_1d = crop_to_nearest_multiple(tensor_1d, multiple=64)
    >>> cropped_1d.shape
    torch.Size([1, 3, 192])

    >>> tensor_2d = torch.randn(1, 3, 250, 330)  # 2D spatial tensor
    >>> cropped_2d = crop_to_nearest_multiple(tensor_2d, multiple=128)
    >>> cropped_2d.shape
    torch.Size([1, 3, 128, 256])

    >>> tensor_3d = torch.randn(1, 3, 100, 250, 330)  # 3D spatial tensor
    >>> cropped_3d = crop_to_nearest_multiple(tensor_3d, multiple=64)
    >>> cropped_3d.shape
    torch.Size([1, 3, 64, 192, 320])
    """
    # Ensure the tensor has at least 3 dimensions (batch, channel, and spatial)
    if tensor.ndim < 3:
        raise ValueError(
            "Tensor must have at least 3 dimensions (B, C, *spatial_dims)."
        )

    # Get the spatial dimensions (ignoring batch and channel dimensions)
    spatial_dims = tensor.shape[2:]

    # Compute the new spatial shape (nearest multiple of `multiple`)
    new_spatial_shape = [
        dim - (dim % multiple) for dim in spatial_dims
    ]

    # Compute the starting indices to center the crop
    start_indices = [
        (dim - new_dim) // 2 for dim, new_dim in zip(
            spatial_dims, new_spatial_shape
        )
    ]

    # Compute the slices for cropping (batch and channel are untouched)
    slices = [slice(None), slice(None)] + [
        slice(start, start + new_dim)
        for start, new_dim in zip(start_indices, new_spatial_shape)
    ]

    # Apply the slices to crop the tensor
    cropped_tensor = tensor[tuple(slices)]

    return cropped_tensor


def logistic(
    logits: torch.Tensor,  # TODO: Should this be called `input_tensor` to make it more general?
    slope: float = 1.0,
    lower_asymptote: float = 0.0,
    upper_asymptote: float = 1.0,  # TODO: Maybe call these `min_output` and `max_output`
) -> torch.Tensor:

    """
    Apply a scaled and shifted logistic function to input logits.

    This function computes a generalized logistic (sigmoid) function that maps input logits to a
    range bounded by `lower_asymptote` and `upper_asymptote`. The `slope` parameter controls the
    steepness of the transition between these asymptotic values.

    Parameters
    ----------
    logits : torch.Tensor
        Unnormalized output (score), such as the outputs of a segmentation model.
    slope : float
        The slope of the logistic function. A higher value results in a steeper transition between
        the asymptotic bounds. Default is 1.0.
    lower_asymptote : float, optional
        The lower bound of output values (asymptote) as logits tend to infinity. Default is 0.0.
    upper_asymptote : float, optional
        The maximum bound output values (asymptote) as logits tend to negative infinity.
        Default is 1.0.

    Returns
    -------
    torch.Tensor
        Result of the logistic function which can be interpreted as probabilities/normalized scores.
    """

    # Validate upper and lower bounds of logistic
    assert upper_asymptote > lower_asymptote, (
        "`upper_asymptote` must be greater than `lower_asymptote."
    )

    # Compute the numerator of logistic. By default, 1.0
    numerator = upper_asymptote - lower_asymptote

    # Compute denominator of logistic with the modulated slope
    denominator = 1 + torch.exp(-slope * logits)

    # Shift by the lower asymptote and return
    return lower_asymptote + (numerator / denominator)


def dice(
    seg1: torch.Tensor,
    seg2: torch.Tensor,
    smooth_numerator: float = 1e-12,
    smooth_denominator: float = 1e-12,
) -> torch.Tensor:
    """
    Compute the Dice score between two segmentation tensors (e.g. ground truth, predictions, etc...)

    Parameters
    ----------
    seg1 : torch.Tensor
        First segmentation tensor of shape (B, C, *spatial_dims).
    seg2 : torch.Tensor
        Second segmentation tensor with the same shape as `seg1`
    smooth_numerator : float, optional
        Smoothing constant added to the numerator.
    smooth_denominator : float, optional
        Smoothing constant added to the denominator.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B, C) whose entries represent the dice score for each batch and class.

    Examples
    --------
    >>> # Make shape for example segmentations with shape (B, C, *spatial_dims)
    >>> shape = (1, 5, 64, 64)
    >>> # Sample `seg1` and `seg2` from neurite's uniform samplers
    >>> seg1 = ne.samplers.Uniform(0, 1)(shape)
    >>> seg2 = ne.samplers.Uniform(0, 1)(shape)
    >>> dice_score = ne.utils.dice(seg1, seg2)
    >>> dice_score
    tensor([[0.5068, 0.4974, 0.5031, 0.4982, 0.4999]])
    """

    # Ensure `seg1` can be interpreted as valid probabilities
    assert seg1.min() >= 0 and seg1.max() <= 1, (
        f"`seg1` must be between zero and one. Got seg1.min()={seg1.min()}, "
        f"seg1.max()={seg1.max()}"
    )

    # Ensure `seg2` can be interpreted as valid probabilities
    assert seg2.min() >= 0 and seg2.max() <= 1, (
        f"`seg2` must be between zero and one. Got seg2.min()={seg2.min()}, "
        f"seg2.max()={seg2.max()}"
    )

    # Flatten spatial dimensions while preserving batch and channel dims
    seg1 = seg1.view(seg1.size(0), seg1.size(1), -1).contiguous()
    seg2 = seg2.view(seg2.size(0), seg2.size(1), -1).contiguous()

    # Per-class intersection
    intersection = (seg2 * seg1).sum(dim=2)

    # Per-class union
    union = seg2.sum(dim=2) + seg1.sum(dim=2)

    # Compute the dice score with intersection, smooth, & union
    dice_score = (2 * intersection + smooth_numerator) / (union + smooth_denominator)

    return dice_score


def log_dice(
    seg1: torch.Tensor,
    seg2: torch.Tensor,
    smooth_numerator: float = 1e-12,
    smooth_denominator: float = 1e-12,
) -> torch.Tensor:
    """
    Compute the logarithm of the soft Dice coefficient between `seg1` and `seg2` in the log
    domain using the logsumexp trick.

    Parameters
    ----------
    seg1 : torch.Tensor
        Ground truth tensor with values in [0, 1]. Expected shape is (B, C, *spatial_dims)
    seg2 : torch.Tensor
        Logits/raw score. Expected shape is (B, C, *spatial_dims)
    smooth_numerator : float, optional
        Smoothing constant added to the numerator to avoid log(0), by default 1e-12.
    smooth_denominator : float, optional
        Smoothing constant added to the denominator to avoid log(0), by default 1e-12.

    Returns
    -------
    torch.Tensor
        The log Dice coefficient computed per batch and channel.

    Examples
    --------
    >>> # Computing log_dice of random tensors
    >>> seg1 = ne.samplers.RandInt(0, 1)((1, 1, 32, 32))
    >>> seg2 = ne.samplers.RandInt(0, 1)((1, 1, 32, 32))
    >>> log_dice = ne.utils.log_dice(seg1, seg2)
    >>> # Expecting log(0.5) ~= -0.69314
    >>> log_dice
    tensor([[-0.6970]])
    >>> # Converting to linear domain, should be about 0.5
    >>> torch.exp(log_dice)
    tensor([[0.4981]])
    """

    # Flatten all spatial dims into one axis
    seg1 = seg1.view(seg1.size(0), seg1.size(1), -1).contiguous()
    seg2 = seg2.view(seg2.size(0), seg2.size(1), -1).contiguous()

    # Reshape and convert numerator smoothing factor into log domain to play nicely w/ stacking
    log_smooth_numerator = torch.tensor(
        smooth_numerator,
        device=seg1.device
    ).expand(seg1.size(0), seg1.size(1)).log()

    # Reshape and convert denominator smoothing factor into log domain to play nicely w/ stacking
    log_smooth_denominator = torch.tensor(
        smooth_denominator,
        device=seg1.device
    ).expand(seg1.size(0), seg1.size(1)).log()

    # Map seg tensors into the log domain
    log_seg1 = torch.log(seg1)
    log_seg2 = torch.log(seg2)

    # Compute numerically stable intersection with logsumexp trick on the sum
    log_intersection = torch.logsumexp(log_seg1 + log_seg2, dim=2)

    # Add log of x2 factor and the log of the intersection, and stack with smoothing for logexpsum
    numerator_stack = torch.stack(
        [
            torch.tensor(2).log() + log_intersection,
            log_smooth_numerator,
        ]
    )

    # Compute log numerator using logsumexp trick
    log_numerator = torch.logsumexp(numerator_stack, dim=0)

    # Calculate summed logs safely
    log_sum_seg1 = torch.logsumexp(log_seg1, dim=2)
    log_sum_seg2 = torch.logsumexp(log_seg2, dim=2)

    # We are going add the smoothing term `smooth_denominator` by stacking it alongside log_sum_*
    # Stack them
    stacked_logsums_with_smoothing = torch.stack(
        [
            log_sum_seg1,
            log_sum_seg2,
            log_smooth_denominator,
        ]
    )

    # Compute the log union using the logsumexp trick
    log_union = torch.logsumexp(stacked_logsums_with_smoothing, dim=0)

    # Compute the dice score by negating (dividing in linear domain)
    log_dice = log_numerator - log_union

    return log_dice
