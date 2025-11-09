"""
Tensor operations and functions for Neurite.

A collection of functions for manipulating and analyzing PyTorch tensors,
with applications a focus on imaging.

Notes
-----
- All functions assume tensors follow the (B, C, *spatial_dims) convention.
"""

# Standard library imports
from collections.abc import Sequence
from typing import Union, List, Tuple, Literal, Type, Optional

# Third party imports
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# Custom imports
import neurite as ne


def identity(input_argument):
    "Returns the `input_argument`."
    return input_argument


def gaussian_smoothing(
    input_tensor: torch.Tensor,
    kernel_size: Union[int, Sequence[int]] = 3,
    sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
) -> torch.Tensor:
    """
    Apply Gaussian smoothing to the {1D, 2D, 3D} input tensor.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor, assumed to be 1D, 2D, or 3D.
    kernel_size : int, List[int], or optional
        Size of the Gaussian kernel. If int, same size is used for all dimensions.
        If Sequence[int], different sizes can be specified per dimension.
    sigma : float, int, Sequence[float], Sequence[int], default=1
        Standard deviation of the Gaussian kernel. If float/int, same sigma is used
        for all dimensions. If Sequence, different sigmas can be specified per dimension.

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
        kernel_size=kernel_size, sigma=sigma, ndim=ndim, nchannels=input_tensor.shape[1]).float()

    if isinstance(kernel_size, Sequence):
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
        input=padded_input_tensor, weight=gaussian_kernel_, padding=0, groups=input_tensor.shape[1])

    return smoothed_tensor


def gaussian_antialiasing(
    input_tensor: torch.Tensor,
    stride: Union[int, Sequence[int]] = 2,
    kernel_size: Union[int, Sequence[int], None] = None,
    sigma: Union[float, int, Sequence[float], Sequence[int], None] = None,
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
    stride : int or Sequence[int], default=2
        Downsampling stride. If int, the same stride is applied to all spatial dimensions.
        If Sequence[int], different strides can be specified per dimension.
    kernel_size : int, Sequence[int], default=None
        Size of the Gaussian kernel for antialiasing. If int, same size is used for all
        dimensions. If Sequence[int], different sizes can be specified per dimension.
        If None, automatically computed as 2 * stride + 1 per dimension.
    sigma : float, int, Sequence[float], Sequence[int], default=None
        Standard deviation of the Gaussian kernel. If float/int, same sigma is used for
        all dimensions. If Sequence, different sigmas can be specified per dimension.
        If None, automatically computed as stride / 2 per dimension.
    subsampling_dimension : Sequence[int], int, or None, default=None
        Dimensions to apply antialiasing and subsampling. If None, applies to all
        spatial dimensions.

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
        input_tensor=input_tensor, kernel_size=kernel_size, sigma=sigma)

    antialiased_tensor = subsample(
        input_tensor=smoothed_tensor, stride=strides, subsampling_dimension=subsampling_dimension)

    return antialiased_tensor


def apply_bernoulli_mask(
    input_tensor: torch.Tensor,
    p: Union[float, int] = 0.5,
    returns: Union[str, None] = None
) -> torch.Tensor:
    """
    Apply a Bernoulli mask to a tensor in (B, C, *spatial) format.

    Sample a Bernoulli mask with the parameter `p`, representing the probability of
    success (e.g. realizing a 1) and apply it to `input_tensor` via element-wise multiplcation. The
    The elements of `input_tensor` corresponding to successes in the mask are preserved, while
    failures (e.g. zeros) are set to zero.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor to be masked.
    p : float, default=0.5
        Probability of realizing a success (i.e., the probability of a 1) in the mask. Successes are
        preserved in the input tensor such that higher values of this parameter correspond to more
        elements of the input tensor being preserved. Must be in the range [0, 1].
    returns : {None, 'successes', 'failures'}, default=None
        Optionally return the subset of the input tensor corresponding to Bernoulli {'successes',
        'failures'}. If None, returns the original tensor with failures set to zero.
        Setting `returns = 'successes'` might be useful in sampling a subset of a large tensor to
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
    return ne.apply_bernoulli_mask(input_tensor, p=p, returns=returns)


def subsample(
    input_tensor: torch.Tensor,
    stride: Union[Sequence[int], int, None] = 2,
    subsampling_dimension: Union[Sequence[int], Literal[0, 1, 2], int, None] = None,
) -> torch.Tensor:
    """
    Subsamples `input_tensor` by a factor `stride` along the specified dimension.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The tensor to sample from, with shape (B, C, *spatial).
    stride : Sequence[int], int, or None, default=2
        Factor by which to subsample (interleave dropouts).
    subsampling_dimension : Sequence[int], Literal[0, 1, 2], int, or None, default=None
        The spatial dimension(s) to subsample (0-indexed among spatial dims). If None, subsamples
        all spatial dimensions.

    Returns
    -------
    torch.Tensor
        Tensor that has been subsampled.

    Examples
    --------
    >>> import torch
    # Define tensor of shape (1, 1, 5, 5)
    >>> input_tensor = torch.arange(25).view(1, 1, 5, 5)
    # Subsample along spatial dimension 1 (width)
    >>> subsampled = subsample(input_tensor, stride=2, subsampling_dimension=1)
    >>> print(subsampled.shape)
    torch.Size([1, 1, 5, 3])

    # Subsample all spatial dimensions
    >>> input_tensor = torch.randn(2, 3, 32, 32)
    >>> subsampled = subsample(input_tensor, stride=2)
    >>> print(subsampled.shape)
    torch.Size([2, 3, 16, 16])
    """
    return ne.subsample(
        input_tensor=input_tensor,
        stride=stride,
        subsampling_dimension=subsampling_dimension,
        non_spatial_dims=(0, 1)
    )


def subsample_random_dims(
    input_tensor: torch.Tensor,
    stride: int = 2,
    forbidden_dims: Sequence[int] = (0, 1),
    p: float = 0.5,
    max_concurrent_subsamplings: Union[int, None] = None
) -> torch.Tensor:
    """
    Subsample the input tensor along randomly selected dimensions

    This extends `neurite.utils.subsample()` by applying constraints on which dimensions to
    subsample (`forbidden_dims`), the stride, and the probability of subsampling.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor to be subsampled. Assumed to have batch and channel dimensions.
    stride : int, default=2
        The stride value to use when subsampling a given dimension.
        - A stride of 1 does not result in any subsampling.
        - A stride of 2 will reduce the elements of the selected dimension by 1/2.
    forbidden_dims : Sequence[int], default=(0, 1)
        A sequence of dimensions that should not be subsampled. If None, no dimensions
        are forbidden from subsampling.
    p : float, default=0.5
        The probability of selecting each dimension for subsampling. This probability
        is applied as an independent Bernoulli trial for each dimension.
    max_concurrent_subsamplings : int, default=None
        The maximum number of dimensions that can be subsampled simultaneously. If
        None, the number of concurrent subsamplings is set to the number of dimensions
        in `input_tensor`.

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
    >>> subsampled_tensor = subsample_random_dims(input_tensor)
    >>> print(subsampled_tensor)
    tensor([[[[ 0,  3],
              [10, 13],
              [20, 23]]]])
    >>> # Subsample by defining the stride range.
    >>> subsampled_tensor = subsample_random_dims(input_tensor, stride=4)
    >>> print(subsampled_tensor)
    tensor([[[[ 0,  4],
              [20, 24]]]])
    """

    if max_concurrent_subsamplings is None:
        max_concurrent_subsamplings = input_tensor.dim()

    elif max_concurrent_subsamplings <= input_tensor.dim():
        pass

    elif max_concurrent_subsamplings > input_tensor.dim():
        raise ValueError(f"Your tensor doesn't have {max_concurrent_subsamplings} dimensions!")

    dimensions_to_subsample = torch.randperm(input_tensor.dim())[:max_concurrent_subsamplings]

    if forbidden_dims is not None:
        forbidden_dims: torch.Tensor = torch.Tensor(forbidden_dims)
        mask = torch.isin(dimensions_to_subsample, forbidden_dims)
        dimensions_to_subsample = dimensions_to_subsample[~mask]

    # Apply Bernoulli mask to vary the number of dimensions subsampled each time
    dimensions_to_subsample = apply_bernoulli_mask(
        input_tensor=dimensions_to_subsample, p=p, returns='successes')

    for dimension in dimensions_to_subsample:
        # Adjust dimension index to account for batch and channel dims
        input_tensor = subsample(
            input_tensor=input_tensor, subsampling_dimension=int(dimension) - 2, stride=stride)

    return input_tensor


def upsample(
    input_tensor: torch.Tensor,
    scale_factor: Union[int, float, Sequence[Union[int, float]]] = 2,
    shape: Union[Sequence[int], None] = None,
    mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
) -> torch.Tensor:
    """
    Upsample 1D, 2D, or 3D tensors to a given `shape`.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor to be upsampled, with shape (B, C, *spatial).
    scale_factor : int, float, Sequence[int], or Sequence[float], default=2
        The factor by which to upsample each spatial dimension.
    shape : Sequence[int] or None, default=None
        Spatial dimensions (without batch or channel dimensions) to upsample `input_tensor` into.
    mode : {'linear', 'nearest', 'bicubic', 'area', 'nearest-exact'}, default='linear'
        Interpolation mode for upsampling.

    Returns
    -------
    torch.Tensor
        The upsampled tensor with shape (B, C, *upsampled_spatial).

    Examples
    --------
    >>> import torch
    # 2D Upsampling
    >>> input_tensor = torch.randn(1, 3, 32, 32)  # (B, C, H, W)
    >>> upsampled_tensor = upsample(input_tensor, shape=(64, 64))
    >>> print(upsampled_tensor.shape)
    torch.Size([1, 3, 64, 64])

    # 3D Upsampling
    >>> input_tensor = torch.randn(1, 3, 32, 32, 32)  # (B, C, D, H, W)
    >>> upsampled_tensor = upsample(input_tensor, shape=(64, 64, 64))
    >>> print(upsampled_tensor.shape)
    torch.Size([1, 3, 64, 64, 64])
    """
    return ne.upsample(
        input_tensor=input_tensor,
        scale_factor=scale_factor,
        size=shape,
        mode=mode,
        non_spatial_dims=(0, 1)
    )


def resample(
    input_tensor: torch.Tensor,
    resample_dimension: Union[int, Sequence[int], None] = None,
    downsample_stride: Union[int, Sequence[int]] = 2,
    upsample_scale_factor: Union[Union[int, float], Sequence[Union[int, float]]] = 2,
    mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
    shape: Union[Sequence[int], None] = None,
) -> torch.Tensor:
    """
    Subsample `input_tensor` by a factor `stride`, then upsample it by `scale_factor`.

    Combines `subsample` and `upsample` by first subsampling `input_tensor` along a
    given dimension by `stride`, then upsampling back to `shape`.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The tensor to resample.
    resample_dimension : int, Sequence[int], or None, default=None
        The dimension(s) that should be resampled. If None, all dimensions are resampled.
    downsample_stride : int or Sequence[int], default=2
        Factor by which to subsample.
    upsample_scale_factor : int, float, or Sequence[int or float], default=2
        Factor by which to upsample.
    mode : {'linear', 'nearest', 'bicubic', 'area', 'nearest-exact'}, default='linear'
        Interpolation mode for upsampling.
    shape : Sequence[int] or None, default=None
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
        input_tensor, subsampling_dimension=resample_dimension, stride=downsample_stride)

    resampled = upsample(resampled, shape=shape, mode=mode, scale_factor=upsample_scale_factor)

    return resampled


def random_clear_label(
    input_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    prob: Union[float, int] = 0.5,
    exclude_zero: bool = True,
    seed: Union[int, None] = None
) -> torch.Tensor:
    """
    Erase regions of an image from randomly selected regions in a label map.

    Identify unique labels within the `label_tensor` and, based on a specified probability,
    designate regions of the `input_tensor` to be erased (set to zero).

    Parameters
    ----------
    input_tensor : torch.Tensor
        Image or tensor to clear with shape (B, C, *spatial).
    label_tensor : torch.Tensor
        Label map corresponding to sampling domain from which to select regions for clearing.
    prob : float, default=0.5
        Probability of any label/region being selected for erasure as determined by iid Bernoulli
        trials.
    exclude_zero : bool, default=True
        Optionally exclude zero (usually background) from the list of potential regions to clear
        (never clear zero labels).
    seed : int, default=None
        A random seed to control the randomness of label clearing operations. If
        provided, it ensures reproducibility of the clearing process.

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
    return ne.random_clear_label(
        input_tensor, label_tensor, prob=prob, exclude_zero=exclude_zero, seed=seed
    )


def sample_image_from_labels(
    label_tensor: torch.Tensor,
    mean_range: Tuple[float, float] = (0.0, 1.0),
    noise_std: float = 0.5
) -> torch.Tensor:
    """
    Generate an image from a label map by sampling a random intensity for each label.

    Identify all unique integer labels in `label_tensor` and assigns each a mean intensity in the
    corresponding output image (`sampled_image`). The mean intensity serves as the mean for a noise
    distribution. Noise is sampled from a normal distribution with the specified standard deviation.

    Parameters
    ----------
    label_tensor : torch.Tensor
        A tensor with batch and channel dimensions containing integer labels defining distinct
        regions.
    mean_range : Tuple[float, float], default=(0.0, 1.0)
        Range (min, max) for sampling mean intensity for each region. Mean intensities are
        sampled uniformly from this range.
    noise_std : float, default=0.5
        Standard deviation of the Gaussian noise added to each region.

    Returns
    -------
    torch.Tensor
        A tensor of sampled image intensities with the same shape as `label_tensor`.
    """
    return ne.sample_image_from_labels(label_tensor, mean_range=mean_range, noise_std=noise_std)


def affine_to_dense_shift(
    affine_a: torch.Tensor,
    affine_b: torch.Tensor,
    grid_size: Sequence[int],
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
    grid_size : Sequence[int]
        Spatial size of the grid, e.g., (H, W) for 2D or (D, H, W) for 3D.
    device : str, default='cpu'
        Device for computations.
    dtype : torch.dtype, default=torch.float32
        Data type for computations.
    normalize : bool, default=True
        If True, grid coordinates are normalized to [-1, 1].

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
    size: Sequence[int],
    device: Union[str, torch.device] = "cpu",
    dtype: Union[str, torch.dtype] = torch.float32,
    normalize: bool = False,
    indexing: Literal["ij", "xy"] = "ij",
    stack: bool = False,
) -> torch.Tensor:
    """
    Generate a grid of spatial coordinates with (B, C, *spatial) format.

    Parameters
    ----------
    size : Sequence[int]
        Size of the spatial dimensions of the input tensor. e.g. (H, W) or (D, W, H)
    device : str or torch.device, default="cpu"
        The device on which the grid will reside.
    dtype : str or torch.dtype, default=torch.float32
        The data type of the tensor grid.
    normalize : bool, default=False
        Normalize each dimension of the grid to the range [-1, 1].
        Otherwise, the grid coords span from 0 to `size[i] - 1` for each dimension.
    indexing : {'ij', 'xy'}, default="ij"
        Indexing mode passed to ``torch.meshgrid``.
    stack : bool, default=False
        If True, stack the grid tensors along the last dimension to return a single tensor of
        shape `(*size, len(size))`. If False, return a tuple of tensors, each of shape
        `(*size)`.

    Returns
    -------
    torch.Tensor
        The meshgrid of spatial coordinates.
        if stack=False, a tuple of len(spatial) tensors of shape (B, C, *spatial)
        if stack=True, a tensor of shape (B, C, *spatial, len(spatial))

    Examples
    --------
    >>> # Make a 2d grid for a (2, 3, 4, 5) tensor
    >>> the_grid = volshape_to_ndgrid(size=(2, 3, 4, 5), stack=True)
    >>> print(the_grid.shape)
    torch.Size([2, 3, 4, 5, 2])
    """
    # Extract B, C, and spatial dimensions
    B, C = size[0], size[1]
    spatial_size = size[2:]

    # Get base grid (shape-agnostic)
    grid = ne.volshape_to_ndgrid(
        size=spatial_size,
        device=device,
        dtype=dtype,
        normalize=normalize,
        indexing=indexing,
        stack=stack
    )

    if stack:
        # Grid is shape (*spatial, len(spatial))
        # Add B and C dimensions: (B, C, *spatial, len(spatial))
        ndim = len(spatial_size)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, C, *[-1] * (ndim + 1))
    else:
        # Grid is tuple of tensors, each shape (*spatial) - add (B, C) to each
        grid = tuple(
            g.unsqueeze(0).unsqueeze(0).expand(B, C, *[-1] * len(spatial_size)) for g in grid
        )

    return grid


def constant_shift_field(
    shape: tuple = (1, 1, 16, 16),
    shift_size: Union[int, Sequence[Union[int, float]], torch.Tensor] = 1,
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
    shape : Sequence[int], default=(1, 1, 16, 16)
        Shape of the input tensor, expected as (B, C, *spatial_dims).
    shift_size : int, Sequence[int], or torch.Tensor, default=1
        Shift magnitude for each axis. If int, same shift on all axes. If Sequence[int], length must
        equal number of spatial dims. If Tensor, must have shape (n_spatial_dims,).
    normalize : bool, default=False
        If True, normalize the first spatial channel by (size - 1), where
        size is the extent of that axis.
    device : str, default='cpu'
        Device on which to create the tensor.

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
        shift_size: torch.Tensor = torch.tensor([shift_size] * n_spatial_dims)
    elif isinstance(shift_size, (Sequence)):
        shift_size = torch.tensor(shift_size)
    elif isinstance(shift_size, torch.Tensor):
        pass
    else:
        raise ValueError(f'shift_size must be a tensor, got {type(shift_size)}: {shift_size}')

    assert shift_size.shape[0] == n_spatial_dims, (
        f'shift_size must have {n_spatial_dims} elements. Got {shift_size.shape}: {shift_size}')

    flow_field = torch.zeros(shape[0], n_spatial_dims, *spatial_dims, device=device)

    # Reshape shift_size for broadcasting across spatial dimensions
    shift_size = shift_size.view(1, -1, *[1] * n_spatial_dims)
    flow_field += shift_size

    if normalize:
        flow_field[:, 0, ...] /= (spatial_dims[0] - 1)

    return flow_field


def cross_expand(
    x1: torch.Tensor, x2: torch.Tensor, return_batched: bool = True
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
    return_batched : bool, default=True
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
    raise NotImplementedError(
        "cross_expand() has been moved to neurite_sandbox. "
        "Please use: from neurite_sandbox.etienne_chollet.nn.functional import cross_expand"
    )


def filter_dim(tensor: torch.Tensor, dim: int = 0, verbose: bool = False) -> torch.Tensor:
    """
    Filter slices of a tensor that contain NaNs, infinite values, or are entirely zero.

    Parameters
    ----------
    tensor : torch.Tensor
        An n-dimensional tensor.
    dim : int, default=0
        The dimension along which to filter slices.
    verbose : bool, default=False
        If True, prints the number of elements filtered for each condition (NaNs, infinities,
        all-zeros).

    Returns
    -------
    torch.Tensor
        The filtered tensor with problematic slices removed along the specified dimension.

    Examples
    --------
    >>> # Create a tensor with some problematic slices along dim=0
    >>> tensor = torch.tensor([[1.0, 2.0], [float('nan'), 3.0], [0.0, 0.0], [4.0, 5.0]])
    >>> filtered = filter_dim(tensor, dim=0, verbose=True)
    N Batches with NaNs:  1
    N Batches with Inf:  0
    N Batches with Zero:  1
    >>> filtered.shape
    torch.Size([2, 2])
    """
    return ne.filter_dim(tensor, dim=dim, verbose=verbose)


def crop_to_nearest_multiple(tensor, multiple=128):
    """
    Crop the spatial dimensions of a tensor to the nearest multiple of
    `multiple`. Supports 1D, 2D, or 3D spatial dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor with shape (B, C, *spatial_dims), where `spatial_dims`
        can represent 1D, 2D, or 3D spatial dimensions.
    multiple : int, default=128
        The multiple to which spatial dimensions are cropped.

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
    raise NotImplementedError(
        "crop_to_nearest_multiple() has been moved to neurite_sandbox. "
        "Please use: from neurite_sandbox.etienne_chollet.nn.functional "
        "import crop_to_nearest_multiple"
    )


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
    slope : float, default=1.0
        The slope of the logistic function. A higher value results in a steeper transition between
        the asymptotic bounds.
    lower_asymptote : float, default=0.0
        The lower bound of output values (asymptote) as logits tend to infinity.
    upper_asymptote : float, default=1.0
        The maximum bound output values (asymptote) as logits tend to negative infinity.

    Returns
    -------
    torch.Tensor
        Result of the logistic function which can be interpreted as probabilities/normalized scores.
    """
    raise NotImplementedError(
        "logistic() has been moved to neurite_sandbox. "
        "Please use: from neurite_sandbox.etienne_chollet.nn.functional import logistic"
    )


def mse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Calculate mean squared error (MSE) between two tensors.

    Parameters
    ----------
    tensor1 : torch.Tensor
        An input tensor of any shape.
    tensor2 : torch.Tensor
        A tensor with the same shape as `tensor1`.

    Returns
    -------
    torch.Tensor
        Scalar mean squared error between `tensor1` and `tensor2`.

    Examples
    --------
    >>> import torch
    # Tensors with shape (B, C, H, W)
    >>> tensor1 = torch.randn(2, 3, 16, 16)
    >>> tensor2 = torch.randn(2, 3, 16, 16)
    # Calculate MSE
    >>> mse_value = mse(tensor1, tensor2)
    >>> print(mse_value.shape)
    torch.Size([])
    """
    return ne.mse(tensor1, tensor2)


def dice(
    *segs: torch.Tensor,
    smooth_numerator: float = 1e-12,
    smooth_denominator: float = 1e-12,
    reduction: Union[str, None] = 'mean',
    reduction_dim: Union[int, Tuple[int, ...]] = (0, 1),
    keepdims: bool = True,
) -> torch.Tensor:
    """
    Compute Dice score over multiple segmentation maps with shape (B, C, *spatial_dims).

    Parameters
    ----------
    *segs : torch.Tensor
        Two or more segmentation tensors of shape (B, C, *spatial_dims) with values in [0, 1].
    smooth_numerator : float, default=1e-12
        Smoothing constant added to the numerator.
    smooth_denominator : float, default=1e-12
        Smoothing constant added to the denominator.
    reduction : str or None, default='mean'
        The type of reduction to apply. Supported values for multidimensional reductions are:
        'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'; for single-dimension
        reductions: 'argmin', 'argmax', and all multidimensionals.
    reduction_dim : int or tuple of ints, default=(0, 1)
        Dimension(s) over which to apply the reduction. For multidimensional reductions, pass a
        tuple of dimensions; for single-dimension reductions, pass an integer.
    keepdims : bool, default=True
        Whether to retain reduced dimensions as a singleton.

    Returns
    -------
    torch.Tensor
        Dice score. If reduction=None, returns shape (B, C). Otherwise, reduced as specified.

    Examples
    --------
    >>> # Compute dice for 2 segmentation tensors (batch=2, classes=1, H=W=32) with no reduction
    >>> seg1 = torch.rand((2, 1, 32, 32))
    >>> seg2 = torch.rand((2, 1, 32, 32))
    >>> score = dice(seg1, seg2, reduction=None)
    >>> print(score.shape)
    torch.Size([2, 1])

    >>> # Compute the dice for three classes (batch=2, classes=3) with mean reduction
    >>> segs = [torch.rand((2, 3, 64, 64)) for _ in range(3)]
    >>> per_class = dice(*segs, reduction='mean')
    >>> print(per_class.shape)
    torch.Size([1, 1])
    """
    # Compute Dice using base implementation with (B, C) preserved
    dice_score = ne.dice(
        *segs,
        smooth_numerator=smooth_numerator,
        smooth_denominator=smooth_denominator,
        non_spatial_dims=(0, 1)
    )

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
    smooth_numerator : float, default=1e-12
        Smoothing constant added to the numerator to avoid log(0).
    smooth_denominator : float, default=1e-12
        Smoothing constant added to the denominator to avoid log(0).
    reduction : str, default='mean'
        The type of reduction to apply. Supported values for multidimensional reductions are:
        'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'; for single-dimension
        reductions: 'argmin', 'argmax', and all multidimensionals.
    reduction_dim : int or tuple of ints, default=(0, 1)
        Dimension(s) over which to apply the reduction. For multidimensional reductions, pass a
        tuple of dimensions; for single-dimension reductions, pass an integer.
    keepdims : bool, default=True
        Whether to retain reduced dimensions as a singleton.
    enforce_valid_probabilities : bool, default=False
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
    >>> seg1 = torch.randint(0, 2, (1, 1, 32, 32))
    >>> seg2 = torch.randint(0, 2, (1, 1, 32, 32))
    >>> log_dice = ne.utils.log_dice(seg1, seg2)
    >>> # Expecting log(0.5) ~= -0.69314
    >>> log_dice
    tensor([[-0.6970]])
    >>> # Converting to linear domain, should be about 0.5
    >>> torch.exp(log_dice)
    tensor([[0.4981]])
    """
    raise NotImplementedError(
        "log_dice() has been moved to neurite_sandbox. "
        "Please use: from neurite_sandbox.etienne_chollet.nn.functional import log_dice"
    )


def reduce(
    tensor: torch.Tensor,
    reduction: str = 'mean',
    dim: Union[Tuple[int, ...], int, None] = None,
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
    reduction : str, default='mean'
        The type of reduction to apply. Supported values for multidimensional reductions are:
        None, 'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'; for single
        dimension reductions: 'argmin', 'argmax', and all multidimensionals.
    dim : int or tuple of ints, default=None
        Dimension(s) over which to apply the reduction. For multidimensional reductions, pass a
        tuple of dimensions; for single-dimension reductions, pass an integer.
    keepdims : bool, default=False
        Whether to retain reduced dimensions as a singleton.

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
    return ne.reduce(tensor, reduction=reduction, dim=dim, keepdims=keepdims)


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
    normalization_type: Union[str, Type[nn.Module], nn.Module, None],
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
    normalization_type : str, Type[nn.Module], nn.Module, or None
        Type of normalization. Must be one of 'batch', 'instance', 'layer', 'group', or a custom
        `nn.Module` class.
        `batch` performs normalization per channel. The mean and variance are calculated
        across the B, and *spatial dimensions for each channel C.
    ndim : int, default=None
        Dimensionality for batch/instance normalization:
        - 1 -> *Norm1d
        - 2 -> *Norm2d
        - 3 -> *Norm3d
        Required for 'batch' or 'instance' normalizations.
    num_features : int, default=None
        Number of input features or channels. Required for 'batch', 'instance', 'layer', and 'group'
        normalizations. For layer normalization, this is the size of the normalized dimension. For
        batch and instance normalizations, this is typically the number of channels/features.
    num_groups : int, default=None
        Number of groups for GroupNorm. Required for 'group' normalization.
    eps : float, default=1e-5
        A value added to the denominator for numerical stability.
    affine : bool, default=True
        If True, the layer has learnable affine parameters.
    **kwargs : dict
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


def random_flip(dim: int, *args, prob: float = 0.5):
    """
    Randomly flip tensor(s) along the given dimension.

    Parameters
    ----------
    dim : int
        The dimension along which to flip (0-indexed). For tensors in (B, C, *spatial) format:
        dim=0 is batch, dim=1 is channel, dim=2 is first spatial dimension, etc.
    *args : torch.Tensor
        The image(s) to flip.
    prob : float, default=0.5
        The probability of flipping the image(s).

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor]
        The flipped tensor(s).

    Examples
    --------
    >>> import torch
    # Flip along spatial dimension (width) for (B, C, H, W) tensor
    >>> x = torch.randn(1, 1, 4, 4)
    >>> flipped = random_flip(dim=3, x, prob=1.0)  # dim=3 is width
    """
    return ne.random_flip(dim, *args, prob=prob)


def resize(
    image: torch.Tensor,
    scale_factor: Union[float, Sequence[float]] = None,
    shape: Sequence[int] = None,
    nearest: bool = False
) -> torch.Tensor:
    """
    Resize an image with the option of scaling and/or setting to a new shape.

    Parameters
    ----------
    image : torch.Tensor
        An input tensor with shape (C, H, W[, D]) to resize.
    scale_factor : float or Sequence[float], default=None
        Multiplicative factor(s) for scaling the input tensor. If a float, then the same
        scale factor is applied to all spatial dimensions. If a Sequence, then the scaling
        factor for each dimension should be provided.
    shape : Sequence[int], default=None
        Target shape of the output tensor.
    nearest : bool, default=False
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


def bw_grid(
    vol_shape: Sequence[int],
    spacing: Union[int, Sequence[int]],
    thickness: int = 1,
    indexing: str = 'ij'
) -> torch.Tensor:
    """
    Draw a black and white ND grid.

    Parameters
    ----------
    vol_shape : Sequence[int]
        Expected volume size (dimensions of the output grid).
    spacing : int or Sequence[int]
        Scalar or sequence the same size as vol_shape. Defines the spacing between grid lines in
        each dimension.
    thickness : int, default=1
        Line thickness in pixels.
    indexing : {'ij', 'xy'}, default='ij'
        Cartesian ('xy') or matrix ('ij') indexing of output.

    Returns
    -------
    grid_vol : torch.Tensor
        A volume with white lines (value=1) on black background (value=0).

    Examples
    --------
    >>> # Create a 2D grid with default 'ij' indexing
    >>> grid_2d = bw_grid((100, 100), spacing=10, thickness=2)
    >>> # Create a 3D grid with 'xy' indexing
    >>> grid_3d = bw_grid((50, 50, 50), spacing=[10, 10, 10], thickness=1, indexing='xy')
    """
    raise NotImplementedError(
        "bw_grid() has been moved to neurite_sandbox. "
        "Please use: from neurite_sandbox.etienne_chollet.nn.functional import bw_grid"
    )
