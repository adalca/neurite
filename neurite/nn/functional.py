"""
Tensor operations and functions for Neurite.

A collection of functions for manipulating and analyzing PyTorch tensors, 
with applications a focus on imaging. 

Notes
-----
- All functions assume tensors follow the (B, C, *spatial_dims) convention.
- Some utilities accept `Sampler` objects from `neurite.samplers` for
  stochastic behavior.
"""

# Standard library imports
from typing import Union, List, Tuple, Literal, Type, Optional

# Third party imports
import einops
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# Custom imports
import neurite as ne
from neurite.samplers import Sampler


__all__ = [
    "identity",
    "soft_quantize",
    "mse",
    "gaussian_smoothing",
    "gaussian_antialiasing",
    "apply_bernoulli_mask",
    "subsample",
    "subsample_tensor_random_dims",
    "upsample",
    "resample",
    "random_clear_label",
    "sample_image_from_labels",
    "affine_to_dense_shift",
    "volshape_to_ndgrid",
    "checkerboard",
    "constant_shift_field",
    "cross_expand",
    "filter_dim",
    "crop_to_nearest_multiple",
    "logistic",
    "dice",
    "log_dice",
    "reduce",
    "random_flip",
    "resize",
]


def identity(input_argument):
    "Returns the `input_argument`."
    return input_argument


def soft_quantize(
    input_tensor: torch.Tensor,
    nb_bins: int = 16,
    softness: Union[float, int] = 1.0,
    min_clip: Union[float, int] = -float('inf'),
    max_clip: Union[float, int] = float('inf'),
    return_log: bool = False
) -> torch.Tensor:
    """
    Quantize continuous values into discrete bins.

    Instead of assigning each value to a single bin, use a soft assignment based on the distance
    between each value and the bin centers. Particularly useful for taking gradients during
    quantization.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to softly quantize.
    nb_bins : float, int, optional
        The number of discrete bins to softly quantize the input values into. By default, 16
    softness : float, int, optional
        The softness factor for quantization. A higher value gives smoother quantization.
        By default 1.0
    min_clip : float, int, optional
        Clip data lower than this value before calculating bin centers. By default `-float('inf')`
    max_clip : float, int, optional
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

    # Invert softness to control sensitivity in softmax: higher input softness → sharper bins
    softness = 1 / softness
    input_tensor.clip_(min_clip, max_clip)

    bin_centers = torch.linspace(
        start=input_tensor.min(),
        end=input_tensor.max(),
        steps=nb_bins,
        device=input_tensor.device
    )

    # Compute distance to each bin center and apply soft assignment via softmax
    distances_to_bin_centers = torch.abs(input_tensor.unsqueeze(-1) - bin_centers)
    softly_quantized = F.softmax(-softness * distances_to_bin_centers, dim=-1)
    softly_quantized = (softly_quantized * bin_centers).sum(dim=-1)

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
    # Calculate mse
    >>> mse_value = mse(tensor1, tensor2)
    # Print `mse_value` (should be approximately 2.0)
    >>> print(mse_value)
    """

    return torch.mean((tensor1 - tensor2) ** 2)


def gaussian_smoothing(
    input_tensor: torch.Tensor,
    kernel_size: Union[int, List[int], Sampler] = 3,
    sigma: Union[float, int, List[float], List[int], Sampler] = 1,
) -> torch.Tensor:
    """
    Apply Gaussian smoothing to the {1D, 2D, 3D} input tensor.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor, assumed to be 1D, 2D, or 3D.
    kernel_size : int, List[int], or Sampler, optional
        Size of the Gaussian kernel. If int, same size is used for all dimensions.
        If List[int], different sizes can be specified per dimension. Default is 3.
    sigma : float, int, List[float], List[int], or Sampler, optional
        Standard deviation of the Gaussian kernel. If float/int, same sigma is used
        for all dimensions. If List, different sigmas can be specified per dimension.
        Default is 1.

    Returns
    -------
    smoothed_tensor : torch.Tensor
        The smoothed tensor.

    Examples
    --------
    >>> import torch
    >>> # Make an input tensor ~N(1, 0)
    >>> input_tensor = torch.rand(1, 1, 16, 16, 16)
    >>> # Smooth it with uniform kernel
    >>> smoothed_tensor = gaussian_smoothing(input_tensor)
    >>> # Smooth with per-dimension parameters
    >>> smoothed_tensor = gaussian_smoothing(
    ...     input_tensor,
    ...     kernel_size=[3, 5, 7],
    ...     sigma=[0.5, 1.0, 1.5]
    ... )
    """

    # Infer spatial dimensionality (subtract batch and channel dims)
    ndim = input_tensor.dim() - 2

    gaussian_kernel_ = ne.utils.utils.gaussian_kernel(
        kernel_size=kernel_size,
        sigma=sigma,
        ndim=ndim,
        nchannels=input_tensor.shape[1]
    ).float()

    if isinstance(kernel_size, list):
        padding_per_dim = [ks // 2 for ks in kernel_size]
    else:
        padding_per_dim = [kernel_size // 2] * ndim

    # F.pad expects padding in reverse order: [left, right, top, bottom, front, back]
    padding = []
    for pad in reversed(padding_per_dim):
        padding.extend([pad, pad])

    # Pad input tensor
    padding = tuple(padding)
    padded_input_tensor = F.pad(input_tensor, padding, mode='reflect')

    # Apply the smoothig operation using depthwise convolution
    # groups==nchannels ensures each channel is blurred independently
    conv_fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[ndim]
    smoothed_tensor = conv_fn(
        input=padded_input_tensor,
        weight=gaussian_kernel_,
        padding=0,
        groups=input_tensor.shape[1],
    )

    return smoothed_tensor


def gaussian_antialiasing(
    input_tensor: torch.Tensor,
    stride: Union[int, List[int]] = 2,
    kernel_size: Union[int, List[int], Sampler] = None,
    sigma: Union[float, int, List[float], List[int], Sampler] = None,
    subsampling_dimension: Union[List[int], int, None] = None
) -> torch.Tensor:
    """
    Apply Gaussian antialiasing by combining Gaussian blur with downsampling.

    This function reduces aliasing artifacts when downsampling by first applying
    a Gaussian blur filter followed by subsampling. This is particularly important
    in medical imaging to preserve structural information during downsampling operations.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor to be downsampled with antialiasing, assumed to be 1D, 2D, or 3D.
    stride : int or List[int], optional
        Downsampling stride. If int, the same stride is applied to all spatial dimensions.
        If List[int], different strides can be specified per dimension. Default is 2.
    kernel_size : int, List[int], or Sampler, optional
        Size of the Gaussian kernel for antialiasing. If int, same size is used for all
        dimensions. If List[int], different sizes can be specified per dimension.
        If None, automatically computed as 2 * stride + 1 per dimension. Default is None.
    sigma : float, int, List[float], List[int], or Sampler, optional
        Standard deviation of the Gaussian kernel. If float/int, same sigma is used for
        all dimensions. If List, different sigmas can be specified per dimension.
        If None, automatically computed as stride / 2 per dimension. Default is None.
    subsampling_dimension : List[int], int, or None, optional
        Dimensions to apply antialiasing and subsampling. If None, applies to all
        spatial dimensions. Default is None.

    Returns
    -------
    torch.Tensor
        Antialiased and downsampled tensor with reduced spatial dimensions.

    Examples
    --------
    >>> import torch
    >>> import neurite.nn.functional as nef
    >>> # Create a 3D medical image tensor
    >>> input_tensor = torch.randn(1, 1, 64, 64, 64)
    >>> # Apply Gaussian antialiasing with 2x downsampling
    >>> antialiased_tensor = nef.gaussian_antialiasing(input_tensor, stride=2)
    >>> print(antialiased_tensor.shape)
    torch.Size([1, 1, 32, 32, 32])

    >>> # Apply different strides per dimension
    >>> antialiased_tensor = nef.gaussian_antialiasing(
    ...     input_tensor, stride=[2, 2, 4], sigma=1.5
    ... )
    >>> print(antialiased_tensor.shape)
    torch.Size([1, 1, 32, 32, 16])

    >>> # Apply per-dimension antialiasing parameters
    >>> antialiased_tensor = nef.gaussian_antialiasing(
    ...     input_tensor,
    ...     stride=[2, 2, 4],
    ...     kernel_size=[5, 5, 9],
    ...     sigma=[1.0, 1.0, 2.0]
    ... )
    >>> print(antialiased_tensor.shape)
    torch.Size([1, 1, 32, 32, 16])
    """
    # Infer spatial dimensionality
    ndim = input_tensor.dim() - 2
    if ndim not in [1, 2, 3]:
        raise ValueError(
            f"Unsupported spatial dimensions: {ndim}. Only 1D, 2D, and 3D are supported.")

    # Convert stride to list if it's a single int
    if isinstance(stride, int):
        if stride <= 0:
            raise ValueError(f"Stride must be positive, got {stride}")
        strides = [stride] * ndim
    else:
        strides = list(stride)
        if len(strides) != ndim:
            raise ValueError(
                f"Stride list length {len(strides)} must match spatial dimensions {ndim}")
        if any(s <= 0 for s in strides):
            raise ValueError(f"All strides must be positive, got {strides}")

    # Auto-compute kernel size if not provided
    if kernel_size is None:
        kernel_size = [2 * s + 1 for s in strides]

    # Validate kernel_size if provided as list
    elif isinstance(kernel_size, list):
        if len(kernel_size) != ndim:
            raise ValueError(
                f"kernel_size list length {len(kernel_size)} must match spatial dimensions {ndim}")
        if any(ks <= 0 for ks in kernel_size):
            raise ValueError(f"All kernel sizes must be positive, got {kernel_size}")

    if sigma is None:
        # Compute per-dimension sigmas: stride / 2 for each dimension
        sigma = [s / 2.0 for s in strides]
    elif isinstance(sigma, list):
        if len(sigma) != ndim:
            raise ValueError(f"sigma list length {len(sigma)} must match spatial dimensions {ndim}")
        if any(s <= 0 for s in sigma):
            raise ValueError(f"All sigma values must be positive, got {sigma}")

    # Apply Gaussian smoothing first for antialiasing
    smoothed_tensor = gaussian_smoothing(
        input_tensor=input_tensor,
        kernel_size=kernel_size,
        sigma=sigma
    )

    antialiased_tensor = subsample(
        input_tensor=smoothed_tensor,
        stride=strides,
        subsampling_dimension=subsampling_dimension
    )

    return antialiased_tensor


def apply_bernoulli_mask(input_tensor, p: float = 0.5, returns: str = None) -> torch.Tensor:
    """
    Apply a Bernoulli mask to a tensor.

    Sample a Bernoulli mask with the parameter `p`, representing the probability of
    success (e.g. realizing a 1) and apply it to `input_tensor` via element-wise multiplcation. The
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
    #### Standard use case

    ```python
    # Define input tensor.
    input_tensor = torch.ones((1, 32, 32, 32))

    # Mask the tensor.
    masked_tensor = apply_bernoulli_mask(input_tensor, p=0.9)

    # Return the average value of the tensor of ones, approximating the
    # expectation of the mask in this special case.
    masked_tensor.mean()
    ```

    #### Return successes only (as a flattened tensor representing elements from successful trials)
    ```python
    # Define input tensor.
    input_tensor = torch.ones((1, 32, 32, 32))

    # Get masked tensor
    masked_tensor = apply_bernoulli_mask(input_tensor, p=0.9, returns='successes')

    # Compute original shape and masked shape
    original_shape, masked_shape = input_tensor.flatten().shape[0], masked_tensor.shape[0]

    # Compute difference in size as a percent. Should be ~= `p`
    print((masked_shape/original_shape))
    ```
    """

    # Sample the Bernoulli mask with parameter `p`
    bernoulli_mask = ne.utils.utils.bernoulli(p=p, shape=input_tensor.shape)
    masked = torch.clone(input_tensor)

    # Get successes or failures
    if returns == 'successes':
        masked = masked[bernoulli_mask == 1]
    elif returns == 'failures':
        masked = masked[bernoulli_mask == 0]

    elif returns is None:
        masked[bernoulli_mask == 0] = 0

    else:
        raise ValueError(f"{returns} isn't supported!")

    return masked


def subsample(
    input_tensor: torch.Tensor,
    stride: Union[List, Tuple, int, None] = 2,
    subsampling_dimension: Union[List, Literal[0, 1, 2], int, None] = None,
) -> torch.Tensor:
    """
    Subsamples `input_tensor` by a factor `stride` along the specified dimension.

    Downsample a specified dimension of a PyTorch tensor by a given stride. This is achieved by
    interleaving dropouts, meaning that every `stride`-th element along the selected dimension is
    kept, while the others are discarded.

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
    # Define 2D tensor of shape (5, 5)
    >>> input_tensor = torch.arange(25).view(5, 5)
    # Visualize the tensor
    >>> print(input_tensor)
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    # Subsample along the first dimension (the columns)
    >>> subsampled_tensor = subsample(input_tensor, subsampling_dimension=1)
    # With the default stride (of 2), every other column should have been dropped out.
    >>> print(subsampled_tensor)
    tensor([[ 0,  2,  4],
            [ 5,  7,  9],
            [10, 12, 14],
            [15, 17, 19],
            [20, 22, 24]])
    # We could, of course, keep the default `subsampling_dimension=0` and subsample the rows:
    >>> subsampled_tensor = subsample(input_tensor, subsampling_dimension=1)
    >>> print(subsampled_tensor)
    tensor([[ 0,  1,  2,  3,  4],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24]])
    """

    if isinstance(subsampling_dimension, torch.Tensor):
        raise TypeError("subsampling_dimension must be an int, list, tuple, or None, not a Tensor")

    # Infer the number of spatial dimensions
    n_spatial = input_tensor.dim() - 2

    # Precompute list of (empty) slices
    slices = [slice(None)] * input_tensor.ndim

    if isinstance(stride, int):
        strides = [stride] * n_spatial
    elif isinstance(stride, (tuple, list)):
        strides = list(stride)

    # If `None` is passed, subsample all dimensions
    if subsampling_dimension is None:
        subsampling_dimension = list(range(n_spatial))

    if isinstance(subsampling_dimension, int):
        strides[subsampling_dimension] = stride
        slices[subsampling_dimension + 2] = slice(None, None, strides[subsampling_dimension])
    elif isinstance(subsampling_dimension, (list, tuple)):
        for dim in subsampling_dimension:
            strides[dim] = strides[dim]
            slices[dim + 2] = slice(None, None, strides[dim])

    return input_tensor[tuple(slices)]


def subsample_tensor_random_dims(
    input_tensor: torch.Tensor,
    stride: int = 2,
    forbidden_dims: list = (0, 1),
    p: float = 0.5,
    max_concurrent_subsamplings: int = None
) -> torch.Tensor:
    """
    Subsample the input tensor along randomly selected dimensions

    This extends `neurite.utils.subsample()` by applying constraints on which dimensions to
    subsample (`forbidden_dims`), the stride, and the probability of subsampling.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor to be subsampled. Assumed to have batch and channel dimensions.
    stride : int,  optional
        The stride value to use when subsampling a given dimension. By default, 2.
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
    >>> # Define input tensor with batch and channel dimensions, and spatial dims=(5, 5)
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
    >>> subsampled_tensor = subsample_tensor_random_dims(input_tensor, stride=4)
    >>> print(subsampled_tensor)
    tensor([[[[ 0,  4],
              [20, 24]]]])
    """

    if max_concurrent_subsamplings is None:
        max_concurrent_subsamplings = input_tensor.dim()

    elif max_concurrent_subsamplings <= input_tensor.dim():
        pass

    elif max_concurrent_subsamplings > input_tensor.dim():
        raise ValueError(
            f"Your tensor doesn't have {max_concurrent_subsamplings} dimensions!"
        )

    dimensions_to_subsample = torch.randperm(
        input_tensor.dim()
    )[:max_concurrent_subsamplings]

    if forbidden_dims is not None:
        forbidden_dims = torch.Tensor(forbidden_dims)
        mask = torch.isin(dimensions_to_subsample, forbidden_dims)
        dimensions_to_subsample = dimensions_to_subsample[~mask]

    # Apply Bernoulli mask to vary the number of dimensions subsampled each time
    dimensions_to_subsample = apply_bernoulli_mask(
        input_tensor=dimensions_to_subsample,
        p=p,
        returns='successes'
    )

    for dimension in dimensions_to_subsample:
        # Adjust dimension index to account for batch and channel dims
        input_tensor = subsample(
            input_tensor=input_tensor,
            subsampling_dimension=int(dimension) - 2,
            stride=stride
        )

    return input_tensor


def upsample(
    input_tensor: torch.Tensor,
    mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
    scale_factor: float = 2,
    shape: tuple = None,
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
        Interpolation mode for upsampling. Options include 'nearest', 'linear',
        'bicubic', 'area', and 'nearest-exact'. Default is 'linear'.

    Examples
    --------
    >>> # 2D Upsampling
    >>> input_tensor = torch.randn(1, 3, 32, 32)  # (B, C, H, W)
    >>> upsampled_tensor = upsample(input_tensor, shape=(64, 64), mode='bilinear')
    >>> print(upsampled_tensor.shape)
    torch.Size([1, 3, 64, 64])
    >>> # 3D Upsampling
    >>> input_tensor = torch.randn(1, 3, 32, 32, 32)  # (B, C, D, H, W)
    >>> upsampled_tensor = upsample(input_tensor, shape=(64, 64, 64), mode='bilinear')
    >>> print(upsampled_tensor.shape)
    torch.Size([1, 3, 64, 64, 64])
    """

    # Get the correct {'linear', 'bilinear', 'trilinear'} interpolation mode
    if mode == 'linear':
        mode = ne.utils.util.infer_linear_interpolation_mode(input_tensor.dim() - 2)

    spatial_dims = input_tensor.dim() - 2
    if spatial_dims not in [1, 2, 3]:
        raise ValueError(
            f"Unsupported tensor dimensionality: {spatial_dims} spatial dimensions. "
            "Only 1D, 2D, and 3D tensors are supported."
        )

    upsampled = F.interpolate(
        input=input_tensor,
        size=shape,
        mode=mode,
        scale_factor=scale_factor,
    )

    return upsampled


def resample(
    input_tensor: torch.Tensor,
    resample_dimension: Union[int, List[int]] = None,
    downsample_stride: Union[int, List[int]] = 2,
    upsample_scale_factor: Union[Union[int, float], List[Union[int, float]]] = 2,
    mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
    shape: tuple = None,
) -> torch.Tensor:
    """
    Subsample `input_tensor` by a factor `stride`, then upsample it by `scale_factor`.

    Combines `subsample` and `upsample` by first subsampling `input_tensor` along a
    given dimension by `stride`, then upsampling back to `shape`.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The tensor to resample.
    resample_dimension : int or list of ints, optional
        The dimension(s) that should be resampled. If None, all dimensions are resampled.
        Default is None.
    downsample_stride : int or list of ints, optional
        Factor by which to subsample. Default is 2.
    upsample_scale_factor : int, float or list of ints or floats, optional
        Factor by which to upsample. Default is 2.
    mode : str, optional
        Interpolation mode for upsampling. Options include 'nearest', 'linear',
        'bicubic', 'area', and 'nearest-exact'. Default is 'linear'.
    shape : tuple
        Spatial dimensions (without batch or channel dims) to upsample the subsampled tensor into.

    Returns
    -------
    torch.Tensor
        The resampled tensor with the same batch and channel dims as `input_tensor` and spatial dims
        equal to `shape`.

    Examples
    --------
    >>> import torch
    >>> input_tensor = torch.randn(1, 3, 32, 32)
    >>> # Subsample rows/cols by 2, then upsample to (64, 64)
    >>> res = resample(
    ...     input_tensor, shape=(64, 64),
    ...     subsampling_dimension=2, stride=2,
    ...     mode='bilinear'
    ... )
    >>> print(res.shape)
    torch.Size([1, 3, 64, 64])
    """

    resampled = subsample(
        input_tensor,
        subsampling_dimension=resample_dimension,
        stride=downsample_stride
    )

    resampled = upsample(
        resampled,
        shape=shape,
        mode=mode,
        scale_factor=upsample_scale_factor,
    )

    return resampled


def random_clear_label(
    input_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    prob: Union[float, int, Sampler] = 0.5,
    exclude_zero: bool = True,
    seed: int = None
) -> torch.Tensor:
    """
    Erase regions of an image from randomly selected regions in a label map.

    Identify unique labels within the `label_tensor` and, based on a specified probability,
    designate regions of the `input_tensor` to be erased (set to zero).

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
    if seed is not None:
        if isinstance(seed, Sampler):
            seed = seed()
        torch.manual_seed(seed)

    unique_labels = torch.unique(label_tensor)
    # Optionally exclude zero label (usually background)
    if exclude_zero:
        unique_labels = unique_labels[unique_labels != 0]

    labels_to_clear = apply_bernoulli_mask(unique_labels, prob, returns='successes')

    for label in labels_to_clear:
        input_tensor.masked_fill_(label_tensor == label, 0)

    return input_tensor


def sample_image_from_labels(
    label_tensor: torch.Tensor,
    mean_sampler: Sampler = ne.samplers.Uniform(0, 1),
    noise_sampler: Sampler = ne.samplers.Normal,
    noise_variance: Union[float, int, Sampler] = 0.25
) -> torch.Tensor:
    """
    Generate an image from a label map by sampling a random intensity for each label.

    Identify all unique integer labels in `label_tensor` and assigns each a mean intensity in the
    corresponding output image (`sampled_image`). The mean intensity serves as the mean for a noise
    distribution modeled by `noise_sampler`. The variance of the noise model may be a fixed quantity
    or sampled from another distribution defined by `noise_variance`.

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
    noise_variance = ne.samplers.make_sampler(ne.samplers.Fixed, noise_variance)
    unique_labels = torch.unique(label_tensor)
    sampled_image = torch.zeros_like(label_tensor).float()

    # Iteratevly texturize/sample intensities for each region as specified by a label
    for label in unique_labels:
        mean_region_intensity = mean_sampler()
        texturized_redion = noise_sampler(
            mean_region_intensity, noise_variance()
        )(label_tensor[label_tensor == label].shape)
        sampled_image[label_tensor == label] = texturized_redion

    return sampled_image


def affine_to_dense_shift(
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
    >>> Dense displacement field for 2x scaled affines
    >>> # Make first affine with ones
    >>> aff_a_2d = torch.eye(2, 2 + 1).unsqueeze(0)
    >>> # Dilate original affine by 2
    >>> aff_b_2d = aff_a_2d * 2
    >>> grid_size_2d = (128, 128)
    >>> displacement_field = affine_to_dense_shift(
    ...     aff_a_2d, aff_b_2d, grid_size_2d
    ... )
    """

    # Input validation (ensuring F.affine_grid() will be happy)
    assert affine_a.dim() == 3 and affine_b.dim() == 3, "Affine matrices must be 3D tensors"
    assert affine_a.shape == affine_b.shape, "Affine matrices must have the same shape"

    # Validate rectangular shape of affine matricies
    batch_size, ndim, ndim_plus_one = affine_a.shape
    assert ndim_plus_one == ndim + 1, "Affine shape should be (batch_size, ndim, ndim+1)"
    assert ndim in [2, 3], "Only 2D and 3D transformations are supported"

    grid_a = F.affine_grid(affine_a, size=(batch_size, 1, *grid_size), align_corners=True)
    grid_b = F.affine_grid(affine_b, size=(batch_size, 1, *grid_size), align_corners=True)

    # Rearrange from (B, *grid_size, ndim) to (B, ndim, *grid_size)
    permuting_order = [0, ndim_plus_one] + list(range(1, ndim_plus_one))
    displacement = grid_b - grid_a
    displacement = displacement.permute(*permuting_order).contiguous()

    if not normalize:
        # Convert from normalized [-1, 1] coordinates to voxel coordinates
        scale = torch.tensor(grid_size, device=device, dtype=dtype).view(1, ndim, *[1] * ndim)
        displacement *= scale

    return displacement


def volshape_to_ndgrid(
    size: Tuple[int],
    device: Union[str, torch.device] = "cpu",
    dtype: Union[str, torch.dtype] = torch.float32,
    normalize: bool = False,
    indexing: Literal["ij", "xy"] = "ij",
    stack: bool = False,
) -> torch.Tensor:
    """
    Generate a grid of spatial coordinates.

    Define the coordinate axes by generating vectors for each spatial dimension represented by the
    elements of `shape`, then creates a grid representing all spatial coords.

    Parameters
    ----------
    size : Tuple[int]
        Size of the spatial dimensions of the input tensor. e.g. (H, W) or (D, W, H)
    device : Union[str, torch.device], optional
        The device on which the grid will reside. By default "cpu"
    dtype : Union[str, torch.dtype], optional
        The data type of the tensor grid, by default ``torch.float32``
    indexing : Literal["ij", "xy"], optional
        Indexing mode passed to ``torch.meshgrid``. Defaults to ``"ij"``.
    normalize : bool, optional
        Normalize each dimension of the grid to the range [-1, 1]. 
        Otherwise, the grid coords span from 0 to `size[i] - 1` for each dimension.
        Default is False
    stack : bool, optional
        If True, stack the grid tensors along the last dimension to return a single tensor of
        shape `(*size, len(size))`. If False, return a tuple of tensors, each of shape
        `(*size)`. Default is False.

    Returns
    -------
    torch.Tensor
        the meshgrid of spatial coordinates
        if stack=False, a tuple of len(size) tensors of shape `*size` 
        if stack=True, a tensor of shape `*size, len(size)`

    Examples
    --------
    ### Make a 2d grid of size (3, 2)
    >>> the_grid = volshape_to_ndgrid(size=(3, 2))
    >>> print(the_grid.shape)
    torch.Size([1, 2, 3, 2])
    >>> print(the_grid)
    tensor([[[[-1., -1.],
            [ 0.,  0.],
            [ 1.,  1.]],
            [[-1.,  1.],
            [-1.,  1.],
            [-1.,  1.]]]])
    """

    if normalize:
        axes = [torch.linspace(-1, 1, steps=sz, device=device, dtype=dtype) for sz in size]
    else:
        axes = [torch.arange(0, sz, device=device, dtype=dtype) for sz in size]

    grid = torch.meshgrid(*axes, indexing=indexing)

    if stack:
        grid = torch.stack(grid, dim=-1).contiguous()

    return grid


def checkerboard(
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
    >>> img = checkerboard((1, 1, 6, 6), square_size=2)
    >>> img[0, 0]
    tensor([[1., 1., 0., 0., 1., 1.],
            [1., 1., 0., 0., 1., 1.],
            [0., 0., 1., 1., 0., 0.],
            [0., 0., 1., 1., 0., 0.],
            [1., 1., 0., 0., 1., 1.],
            [1., 1., 0., 0., 1., 1.]])
    """

    spatial_dims = image_shape[2:]
    checkerboard_image = torch.zeros(image_shape, device=device)

    # Generate grid of square starting positions
    checkerboard_startpoints_for_axes = []
    for dim in spatial_dims:
        startpoints_for_axis = torch.arange(0, dim, square_size)
        checkerboard_startpoints_for_axes.append(startpoints_for_axis)

    checkerboard_start_coords = torch.cartesian_prod(*checkerboard_startpoints_for_axes)

    # Fill alternating squares based on coordinate sum parity
    for start_coord in checkerboard_start_coords:
        if start_coord.sum().item() % (2 * square_size) == 0:
            slices = tuple(slice(i, i + square_size) for i in start_coord)
            checkerboard_image[(..., *slices)] = 1

    return checkerboard_image


def constant_shift_field(
    shape: tuple = (1, 1, 16, 16),
    shift_size: int = 1,
    normalize: bool = False,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Makes a simple flow field for testing registration in N-dimensional space.

    This function generates a flow field with channels that represent the transformations to each
    spatial dimension. E.g. channel 1 represents the dense transformation on the x-axis, channel 2
    represents the dense transformation on the y axis, and so on...

    Parameters
    ----------
    shape : tuple, optional
        Shape of the input tensor, expected as (B, C, *spatial_dims). Default is (1, 1, 4, 4) for a
        2D case.
    shift_size : int, list of int, or torch.Tensor, optional
        Shift magnitude for each axis. If int, same shift on all axes. If list/tuple, length must
        equal number of spatial dims. If Tensor, must have shape (n_spatial_dims,). Default is 1.
    normalize : bool, optional
        If True, normalize the first spatial channel by (size - 1), where
        size is the extent of that axis. Default is False.

    Returns
    -------
    flow_field : torch.Tensor
        A tensor representing the flow field, shaped as (B, n_spatial_dims, *spatial_dims).
        The first spatial dimension is shifted by +1 in a normalized manner.

    Example
    -------
    >>> flow = constant_shift_field((1, 1, 4, 4), device='cpu')
    >>> flow.shape
    torch.Size([1, 2, 4, 4])

    >>> flow_3d = constant_shift_field((1, 1, 4, 4, 4), device='cpu')
    >>> flow_3d.shape
    torch.Size([1, 3, 4, 4, 4])
    """

    spatial_dims = shape[2:]
    n_spatial_dims = len(spatial_dims)

    if isinstance(shift_size, int):
        shift_size = torch.tensor([shift_size] * n_spatial_dims)
    elif isinstance(shift_size, (list, tuple)):
        shift_size = torch.tensor(shift_size)
    elif isinstance(shift_size, torch.Tensor):
        pass
    else:
        raise ValueError(
            f'shift_size must be a tensor, got {type(shift_size)}: {shift_size}'
        )

    assert shift_size.shape[0] == n_spatial_dims, (
        f'shift_size must have {n_spatial_dims} elements. Got {shift_size.shape}: {shift_size}'
    )

    flow_field = torch.zeros(shape[0], n_spatial_dims, *spatial_dims, device=device)
    # Reshape shift_size for broadcasting across spatial dimensions
    shift_size = shift_size.view(1, -1, *[1] * n_spatial_dims)
    flow_field += shift_size

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

    # Unpack shape to extract slice dimensions and spatial dims
    Bx1, Sx1, Cx1, *x1_spatial = x1.shape
    Bx2, Sx2, Cx2, *x2_spatial = x2.shape

    if Bx1 != Bx2:
        raise ValueError(
            f"The input tensors must have the same number of batches. Got Bx1={Bx1} and Bx2={Bx2}")

    # Create pairwise combinations via cartesian product
    x1_expanded = einops.repeat(x1, "Bx1 Sx1 Cx1 ... -> Bx1 Sx1 Sx2 Cx1 ...", Sx2=Sx2)
    x2_expanded = einops.repeat(x2, "Bx2 Sx2 Cx2 ... -> Bx2 Sx1 Sx2 Cx2 ...", Sx1=Sx1)

    if return_batched:

        if Bx1 != Bx2 or x1_spatial != x2_spatial:
            raise ValueError(
                "The tensors must match in their batch and spatial dimensions. Got:"
                f"x1.shape: {x1.shape}, x2.shape: {x2.shape}"
            )

        paired_tensors = torch.cat([x1_expanded, x2_expanded], dim=3)

        # Flatten pairwise combinations into batch dimension
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
    Filters slices of a tensor that contain NaNs, infinite values, or are entirely zero.

    Parameters
    ----------
    tensor : torch.Tensor
        An n-dimensional tensor.
    verbose : bool, optional
        If True, prints the number of elements filtered for each condition. Default is False.

    Returns
    -------
    torch.Tensor
        The filtered tensor containing only slice elements without NaNs, infinite values, and not
        entirely zeros.
    """

    dims_to_test = list(range(tensor.dim()))
    dims_to_test.remove(dim)

    # Remove NaNs
    nan_mask = ~torch.isnan(tensor).any(dim=dims_to_test)
    nan_mask = torch.nonzero(nan_mask, as_tuple=True)[0]
    filtered_tensor = torch.index_select(tensor, dim, nan_mask)

    # Remove infs
    inf_mask = ~torch.isinf(filtered_tensor).any(dim=dims_to_test)
    inf_mask = torch.nonzero(inf_mask, as_tuple=True)[0]
    filtered_tensor = torch.index_select(filtered_tensor, dim, inf_mask)

    # Remove all zeros
    zero_mask = ~torch.all(filtered_tensor == 0, dim=dims_to_test)
    zero_mask = torch.nonzero(zero_mask, as_tuple=True)[0]
    filtered_tensor = torch.index_select(filtered_tensor, dim, zero_mask)

    if verbose:
        n_nans = torch.sum(~nan_mask)
        print("N Batches with NaNs: ", n_nans)

        n_infs = torch.sum(~inf_mask)
        print("N Batches with Inf: ", n_infs)

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
    Crop the spatial dimensions of a tensor to the nearest multiple of
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

    assert upper_asymptote > lower_asymptote, (
        "`upper_asymptote` must be greater than `lower_asymptote."
    )

    numerator = upper_asymptote - lower_asymptote
    denominator = 1 + torch.exp(-slope * logits)
    return lower_asymptote + (numerator / denominator)


def dice(
    *segs: torch.Tensor,
    smooth_numerator: float = 1e-12,
    smooth_denominator: float = 1e-12,
    reduction: str = 'mean',
    reduction_dim: Union[int, Tuple[int, ...]] = (0, 1),
    keepdims: bool = True,
) -> torch.Tensor:
    """
    Compute Dice score over multiple segmentation maps.

    Parameters
    ----------
    *segs : torch.Tensor
        Two or more segmentation tensors of shape (B, C, *spatial_dims) with values in [0, 1].
    smooth_numerator : float, optional
        Smoothing constant added to the numerator.
    smooth_denominator : float, optional
        Smoothing constant added to the denominator.
    reduction : str, optional
        The type of reduction to apply. Supported values for multidimensional reductions are:
        'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'; for single-dimension
        reductions: 'argmin', 'argmax', and all multidimensionals. Default is 'mean'.
    reduction_dim : int or tuple of ints, optional
        Dimension(s) over which to apply the reduction. For multidimensional reductions, pass a
        tuple of dimensions; for single-dimension reductions, pass an integer. Default is (0, 1)
    keepdims : bool, optional
        Whether to retain reduced dimensions as a singleton. Default is True.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B, C) whose entries represent the dice score for each batch and class.

    Examples
    --------
    >>> # Compute dice for 2 segmentation tensors (batch=2, classes=1, H=W=32) with no reduction
    >>> seg1 = torch.rand((2, 1, 32, 32))
    >>> seg2 = torch.rand((2, 1, 32, 32))
    >>> score = dice(seg1, seg2, reduction=None)
    >>> print(score.shape)
    torch.Size([2, 1])

    >>> # Compute the dice for three classes (batch=2, classes=3)
    >>> segs = [torch.rand((2, 3, 64, 64)) for _ in range(3)]
    >>> per_class = dice(*segs, reduction='mean')
    >>> print(per_class.shape)
    tensor([[0.2487]])
    """

    nsegs = len(segs)

    if nsegs < 2:
        raise ValueError(
            'Provide at least two segmentation tensors.'
        )

    if not all(segs[0].shape == seg.shape for seg in segs):
        shapes = {seg.shape for seg in segs}
        raise ValueError(f'All segmentations must share shape; got {shapes}')

    for seg in segs:
        if seg.min() < 0 or seg.max() > 1:
            raise AssertionError(
                f'Segmentations must be in [0,1]; '
                f'got min {seg.min()}, max {seg.max()}'
            )

    # Flatten spatial dimensions while preserving batch and channel dims
    segs_flat = [seg.flatten(2) for seg in segs]

    # Compute intersection and union
    intersection = torch.stack(segs_flat, dim=0).prod(dim=0).sum(dim=2)
    union = torch.stack(segs_flat, dim=0).sum(dim=(0, 3))

    # Dice for N tensors: N * intersection / union
    dice_score = (nsegs * intersection + smooth_numerator) / (union + smooth_denominator)

    if reduction is None:
        return dice_score

    return reduce(
        tensor=dice_score,
        reduction=reduction,
        dim=reduction_dim,
        keepdims=keepdims,
    )


def log_dice(
    *segs,
    smooth_numerator: float = 1e-12,
    smooth_denominator: float = 1e-12,
    reduction: str = 'mean',
    reduction_dim: Union[int, Tuple[int, ...]] = (0, 1),
    keepdims: bool = True,
    enforce_valid_probabilities: bool = False,
) -> torch.Tensor:
    """
    Compute the Dice coefficient in the log domain given two tensors representing log probabilities.

    Parameters
    ----------
    *segs : torch.Tensor
        Two or more segmentation tensors of shape (B, C, *spatial_dims) representing
        log-probabilities.
    smooth_numerator : float, optional
        Smoothing constant added to the numerator to avoid log(0). By default, 1e-12.
    smooth_denominator : float, optional
        Smoothing constant added to the denominator to avoid log(0). By default, 1e-12.
    reduction : str, optional
        The type of reduction to apply. Supported values for multidimensional reductions are:
        'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'; for single-dimension
        reductions: 'argmin', 'argmax', and all multidimensionals. Default is 'mean'.
    reduction_dim : int or tuple of ints, optional
        Dimension(s) over which to apply the reduction. For multidimensional reductions, pass a
        tuple of dimensions; for single-dimension reductions, pass an integer. Default is (0, 1)
    keepdims : bool, optional
        Whether to retain reduced dimensions as a singleton. Default is True.
    enforce_valid_probabilities : bool, optional
        Ensure input segmentations represent valid probabilities by checking that ensuring
        exp(seg1) and exp(seg2) sum to 1.

    Returns
    -------
    torch.Tensor
        The Dice coefficient in the log domain over batch and channel dimensions.

    Notes
    -----
    The Dice coefficient for two probability maps is:
        Dice(seg1, seg2) = 2 * seg1 * seg2 / (seg1^2 + seg2^2).

    In log space, given L_1 = log(seg1) and L_2 = log(seg2):
        LogDice(L_1, L_2) = log(2) + L_1 + L_2 - log(exp(2 * L_1) + exp(2 * L_2)).

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

    if len(segs) < 2:
        raise ValueError(
            'Provide at least two segmentation tensors.'
        )

    if not all(segs[0].shape == seg.shape for seg in segs):
        shapes = {seg.shape for seg in segs}
        raise ValueError(
            f'All segmentations must share shape; got {shapes}'
        )

    if enforce_valid_probabilities:
        assert all(torch.all(seg <= 0) for seg in segs), (
            "ne.utils.log_dice expects input tensors to represent log-probabilities (be entirely "
            f"negative) but got the following maximum values: {[seg.max().item() for seg in segs]}"
        )
        assert all(torch.allclose(seg.exp().sum(), 1.0) for seg in segs), (
            "Input tensors are not valid probability distributions (probs don't sum to 1.0). Got "
            f"the following sums: {[seg.exp().sum().item() for seg in segs]}"
        )

    segs_flat = [seg.flatten(2) for seg in segs]
    n_segs = len(segs_flat)

    # Convert smoothing constants to log domain for numerical stability with logsumexp
    log_smooth_numerator = torch.tensor(
        smooth_numerator,
        device=segs_flat[0].device
    ).expand(segs_flat[0].shape).log()

    log_smooth_denominator = torch.tensor(
        smooth_denominator,
        device=segs_flat[0].device
    ).expand(segs_flat[0].shape).log()

    # Compute N * intersection in log space: log(N) + L_1 + L_2 + ...
    numerator = segs_flat[0] + torch.log(torch.tensor(n_segs, device=segs_flat[0].device))
    for seg in segs_flat[1:]:
        numerator = numerator + seg
    numerator = torch.logsumexp(torch.stack([numerator, log_smooth_numerator], dim=-1), dim=-1)

    # Compute union in log space using logsumexp: log(e^(N*L_1) + e^(N*L_2) + ...)
    scaled_segs = [n_segs * seg for seg in segs_flat]
    denominator = torch.logsumexp(
        torch.stack(
            [*scaled_segs, log_smooth_denominator], dim=-1
        ),
        dim=-1
    )

    # Dice = numerator / denominator, computed as subtraction in log space
    log_dice_vals = numerator - denominator

    if reduction is None:
        return log_dice_vals

    return reduce(
        tensor=log_dice_vals,
        reduction=reduction,
        dim=reduction_dim,
        keepdims=keepdims,
    )


def reduce(
    tensor: torch.Tensor,
    reduction: str = 'mean',
    dim: Union[Tuple[int, ...], int] = None,
    keepdims: bool = False,
) -> torch.Tensor:
    """
    Apply any torch reduction on a tensor.

    This function applies a reduction (e.g., mean, sum, median) on the input tensor across one or
    more dimensions. For reductions that operate on multiple dimensions, the `dim` can be
    a tuple of dimensions. For reductions that operate on a single dimension (e.g., argmin, argmax),
    `dim` must be an integer.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to reduce.
    reduction : str, optional
        The type of reduction to apply. Supported values for multidimensional reductions are:
        None, 'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'; for single
        dimension reductions: 'argmin', 'argmax', and all multidimensionals. Default is None; all
        dimensions are reduced to return a scalar.
    dim : int or tuple of ints, optional
        Dimension(s) over which to apply the reduction. For multidimensional reductions, pass a
        tuple of dimensions; for single-dimension reductions, pass an integer. Default is (0, 1).
    keepdims : bool, optional
        Whether to retain reduced dimensions as a singleton. Default is False.

    Returns
    -------
    torch.Tensor
        The reduced tensor.

    Raises
    ------
    AssertionError
        If a single-dimension reduction (e.g., 'argmin', 'argmax') is requested with a
        `dim` that is not an integer.

    Examples
    --------
    >>> # Make a random tensor
    >>> input_tensor = torch.randn(3, 4, 128, 128)
    >>> # Getting the means from each batch
    >>> reduce(input_tensor, reduction='mean', dim=(1, 2, 3))
    tensor([-0.0004, -0.0021, -0.0052])
    >>> # Getting the largest value from each batch
    >>> reduce(input_tensor, reduction='amax', dim=(1, 2, 3))
    tensor([4.6618, 3.9218, 4.1831])
    """

    # PyTorch multidimensional reductions (also work for single dimensions)
    torch_multidim_reductions = [
        'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean', None
    ]

    # PyTorch single-dimension-only reductions
    torch_singledim_reductions = ['argmin', 'argmax']

    if reduction in torch_multidim_reductions:
        return getattr(torch, reduction)(tensor, dim=dim, keepdims=keepdims)

    elif reduction in torch_singledim_reductions:

        assert isinstance(dim, int), (
            f"Reduction type {reduction} is only compatable with one reduction dimension. Got "
            f"{dim}"
        )

        return getattr(torch, reduction)(tensor, dim=dim, keepdims=keepdims)

    else:
        raise ValueError(
            f"ne.utils.reduce received an invaid `reduction`. Got {reduction}. Valid options"
            " are {'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean', 'argmin', "
            "'argmax'}"
        )


def infer_linear_interpolation_mode(
    num_spatial: Literal[1, 2, 3]
):
    """
    Infer the interpolation mode for `F.interpolate()` from tensor dimensions.

    Parameters
    ----------
    num_spatial : {1, 2, 3}
        Tensor with batch and channel dimensions, and with {1, 2, 3} spatial dimensions.

    Returns
    -------
    mode : str
        Interpolation mode string:
        - 'linear' for 1D
        - 'bilinear' for 2D
        - 'trilinear' for 3D

    Examples
    --------
    >>> # Look at output for different number of spatial dims
    >>> infer_linear_interpolation_mode(1)
    'linear'
    >>> infer_linear_interpolation_mode(3)
    'trilinear'
    """
    if num_spatial == 1:
        return 'linear'
    elif num_spatial == 2:
        return 'bilinear'
    elif num_spatial == 3:
        return 'trilinear'


# Map normalization types to PyTorch classes
NORMALIZATION_MAP = {
    "batch": {
        1: nn.BatchNorm1d,
        2: nn.BatchNorm2d,
        3: nn.BatchNorm3d,
    },
    "instance": {
        1: nn.InstanceNorm1d,
        2: nn.InstanceNorm2d,
        3: nn.InstanceNorm3d,
    },
    "layer": nn.LayerNorm,
    "group": nn.GroupNorm,
}


def build_normalization(
    normalization_type: Union[
        str,
        Type[nn.Module],
        nn.Module,
        None
    ],
    ndim: Optional[int] = None,
    num_features: Optional[int] = None,
    num_groups: Optional[int] = None,
    eps: float = 1e-5,
    affine: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory for various normalization layers.

    Parameters
    ----------
    normalization_type : str or nn.Module
        Type of normalization. Must be one of 'batch', 'instance', 'layer', 'group', or a custom
        `nn.Module` class.
            - `batch` performs normalization per channel. The mean and variance are calculated
            across the B, and *spatial dimensions for each channel C.
    ndim : int, optional
        Dimensionality for batch/instance normalization:
        - 1 -> *Norm1d
        - 2 -> *Norm2d
        - 3 -> *Norm3d
        Required for 'batch' or 'instance' normalizations.
    num_features : int, optional
        Number of input features or channels. Required for 'batch', 'instance', 'layer', and 'group'
        normalizations. For layer normalization, this is the size of the normalized dimension. For
        batch and instance normalizations, this is typically the number of channels/features.
    num_groups : int, optional
        Number of groups for GroupNorm. Required for 'group' normalization.
    eps : float, optional
        A value added to the denominator for numerical stability. Default is 1e-5.
    affine : bool, optional
        If True, the layer has learnable affine parameters. Default is True.
    **kwargs : dict, optional
        Additional keyword arguments are passed directly to the normalization class constructor.
        This enables further customization without modifying this class.

    Returns
    -------
    nn.Module
        Configured and initialized normalization layer.

    Examples
    --------
    >>> # Dummy input with 2 spatial dims ~N(0, 1)
    >>> x = torch.randn(1, 16, 32, 32)

    ### Normalize with a custom normalization layer
    >>> norm_a = nn.InstanceNorm2d(16)
    >>> norm_A = build_normalization(norm_a)
    >>> norm_A(x)
    ...

    ### Normalize with a custom, uninitialized normalization layer
    >>> norm_b = nn.InstanceNorm2d
    >>> norm_B = build_normalization(norm_b, num_features=16)
    >>> norm_B(x)
    ...

    ### Normalize with text-based input
    >>> norm_C = build_normalization(normalization_type='instance', ndim=2, num_features=16)
    >>> norm_C(x)
    ...
    """

    if ne.utils.is_instantiated_normalization(normalization_type):
        normalization = normalization_type
        return

    if isinstance(normalization_type, type) and issubclass(normalization_type, nn.Module):

        if num_features is None:
            raise ValueError("`num_features` must be specified for custom normalizations.")

        normalization = normalization_type(
            num_features=num_features, eps=eps, affine=affine, **kwargs
        )
        return

    if normalization_type not in NORMALIZATION_MAP:

        raise ValueError(
            f"Invalid normalization_type '{normalization_type}'. Must be one of "
            f"{list(NORMALIZATION_MAP.keys())} or a custom nn.Module subclass."
        )

    if normalization_type in ("batch", "instance"):

        if ndim not in (1, 2, 3):

            raise ValueError(
                "For 'batch' or 'instance' normalization, ndim must be 1, 2, or 3."
            )

        if num_features is None:
            raise ValueError(
                "`num_features` must be specified for 'batch' or 'instance' normalization."
            )

        normalization_class = NORMALIZATION_MAP[normalization_type][ndim]
        normalization = normalization_class(
            num_features=num_features, eps=eps, affine=affine, **kwargs
        )

    elif normalization_type == "layer":
        if num_features is None:
            raise ValueError(
                "`num_features` (normalized shape) must be specified for 'layer' normalization."
            )

        normalization = nn.LayerNorm(
            num_features, eps=eps, elementwise_affine=affine, **kwargs
        )

    elif normalization_type == "group":
        if num_groups is None:
            raise ValueError("For 'group' normalization, `num_groups` must be specified.")

        if num_features is None:
            raise ValueError("`num_features` must be specified for 'group' normalization.")

        normalization = nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine, **kwargs)

    return normalization


def random_flip(
    dim: int,
    *args,
    prob: float = 0.5
):
    """
    Randomly flips an image (or set of images) along the given dimension.

    Parameters
    ----------
    dim : int
        The dimension along which to flip. Note that the first dimension
        is the channel dimension.
    *args : torch.Tensor
        The image(s) to flip.
    prob : float
        The probability of flipping the image(s).

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor]
        The flipped image(s).
    """
    result = tuple([arg.flip([dim]) for arg in args]) if ne.utils.utils.bernoulli(prob) else args
    if len(args) == 1:
        return result[0]
    return result


def resize(
    image: torch.Tensor,
    scale_factor: List[float] = None,
    shape: List[int] = None,
    nearest: bool = False
) -> torch.Tensor:
    """
    Resize an image with the option of scaling and/or setting to a new shape.

    Parameters
    ----------
    image: torch.Tensor
        An input tensor with shape (C, H, W[, D]) to resize.
    scale_factor: float or List[float], optional
        Multiplicative factor(s) for scaling the input tensor. If a float, then the same
        scale factor is applied to all spatial dimensions. If a tuple, then the scaling
        factor for each dimension should be provided.
    shape: List[int], optional
        Target shape of the output tensor.
    nearest: bool, optional
        If True, use nearest neighbor interpolation. Otherwise, use linear interpolation.

    Returns:
    --------
    torch.Tensor
        The resized tensor with the shape specified by `shape` or scaled by `scale_factor`.

    Notes
    -----
    TODO: This function has numpy operations and other general things (e.g. antialiasing) that
    Adrian wants to refactor.
    """
    ndim = image.ndim - 1

    if scale_factor is not None and scale_factor != 1:

        target_shape = [int(s * scale_factor + 0.5) for s in image.shape[1:]]

        # Preserve original dtype for nearest interpolation (requires float for linear)
        reset_type = None
        if not torch.is_floating_point(image):
            if nearest:
                reset_type = image.dtype
            image = image.type(torch.float32)

        linear = 'trilinear' if image.ndim - 1 == 3 else 'bilinear'
        mode = 'nearest' if nearest else linear

        if nearest:
            image = torch.nn.functional.interpolate(image.unsqueeze(0), target_shape, mode=mode)
        else:
            image = torch.nn.functional.interpolate(image.unsqueeze(0), target_shape, mode=mode)
        image = image.squeeze(0)

        if reset_type is not None:
            image = image.type(reset_type)

    if shape is not None:

        # Compute center-aligned padding per dimension
        padding = []
        baseshape = image.shape[1:]
        for d in range(ndim):
            diff = shape[d] - baseshape[d]
            if diff > 0:
                half = diff / 2
                a, b = int(np.floor(half)), int(np.ceil(half))
                padding.extend([a, b])
            else:
                padding.extend([0, 0])

        # F.pad expects reversed dimension order
        padding.reverse()
        image = torch.nn.functional.pad(image, padding)

        # Compute center-aligned crop per dimension
        slicing = [slice(0, image.shape[0])]
        baseshape = image.shape[1:]
        for d in range(ndim):
            diff = baseshape[d] - shape[d]
            if diff > 0:
                half = diff / 2
                a, b = int(np.floor(half)), int(np.ceil(half))
                slicing.append(slice(a, baseshape[d] - b))
            else:
                slicing.append(slice(0, baseshape[d]))

        image = image[tuple(slicing)]

    return image
