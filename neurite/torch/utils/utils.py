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
    "derive_dense_displacement_field_from_affines"
]

from typing import Union, List
import inspect
import torch
import torch.nn.functional as F
from torch import nn
from neurite.torch.random import Fixed, RandInt, Sampler, Normal, Uniform
from neurite.torch import modules


def identity(input_argument):
    "Returns the `input_argument`."
    return input_argument


def soft_quantize(
    input_tensor: torch.Tensor,
    nb_bins: Union[int, Sampler] = 16,
    softness: Union[float, int, Sampler] = 1.0,
    min_clip: Union[float, int, Sampler] = -float('inf'),
    max_clip: Union[float, int, Sampler] = float('inf'),
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
        Clip data lower than this value before calculating bin centers. By default -float('inf')
    max_clip : float, int, or Sampler, optional
        Clip data higher than this value before calculating bin centers. By default float('inf')
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
    nb_bins = Fixed.make(nb_bins)()
    softness = Fixed.make(softness)()
    min_clip = Fixed.make(min_clip)()
    max_clip = Fixed.make(max_clip)()

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


def mse(input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean squared error (MSE) between the predicted and target values.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The tensor representing the model's prediction(s).
    target_tensor : torch.Tensor
        The target or ground truth values.

    Returns
    -------
    torch.Tensor
        The mean squared error between `input_tensor` and `target_tensor`.

    Examples
    --------
    >>> import torch
    # Input tensor with zero mean, unit variance
    >>> input_tensor = torch.randn((1, 16, 16, 16))
    # Target tensor with zero mean, unit variance
    >>> target_tensor = torch.randn((1, 16, 16, 16))
    # Calculate loss
    >>> loss = mse(input_tensor, target_tensor)
    # Print loss (should be approximately 2.0)
    >>> print(loss)
    """

    return torch.mean((input_tensor - target_tensor) ** 2)


def create_gaussian_kernel(
    kernel_size: Union[int, Sampler] = 3,
    sigma: Union[float, int, Sampler] = 1,
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
    kernel_size = Fixed.make(kernel_size)()
    sigma = Fixed.make(sigma)()

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
    kernel_size: Union[int, Sampler] = 3,
    sigma: Union[float, int, Sampler] = 1,
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
    kernel_size = Fixed.make(kernel_size)()
    sigma = Fixed.make(sigma)()

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
        stride_sampler = Fixed(stride)
    else:
        stride_sampler = RandInt.make(stride)

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
        if isinstance(arg, Sampler):
            return arg
        elif isinstance(arg, (list, tuple)):
            return arg

    # Return keyword arguments of type {Sampler, list, tuple} as-is
    for arg in kwargs.values():
        if isinstance(arg, Sampler):
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
    prob: Union[float, int, Sampler] = 0.5,
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
        if isinstance(seed, Sampler):
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
    mean_sampler: Sampler = Uniform(0, 1),
    noise_sampler: Sampler = Normal,
    noise_variance: Union[float, int, Sampler] = 0.25
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
    noise_variance = Fixed.make(noise_variance)
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
    downsampling_conv_blocks = nn.ModuleList()

    # Make downsampling conv block and append to list of them
    for i in range(len(nb_features) - 1):

        downsampling_conv_block = modules.DownsampleConvBlock(
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
            order=order
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
    order: str = 'nca'
) -> nn.ModuleList:
    """
    Create an `nn.ModuleList` of upsampling conv blocks based the number of features per layer/level.

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
    upsampling_conv_blocks = nn.ModuleList()

    # make the number of features for the upsampling conv blocks
    nb_features = [*nb_features, nb_features[-1]]

    # Make upsampling conv block and append to list of them
    for i in range(len(nb_features) - 1):

        upsampling_conv_block = modules.UpsampleConvBlock(
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
        )

        upsampling_conv_blocks.append(upsampling_conv_block)

    return upsampling_conv_blocks


def derive_dense_displacement_field_from_affines(
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
