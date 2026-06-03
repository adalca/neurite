"""
Tensor operations and functions for Neurite.

A collection of functions for manipulating and analyzing PyTorch tensors,
with applications a focus on imaging.

Notes
-----
- All functions assume tensors follow the (B, C, *spatial_dims) convention.
"""

# Standard library imports
from typing import Union, Tuple, Literal, Sequence

# Third party imports
import torch
import numpy as np
import torch.nn.functional as F

# Custom imports
import neurite as ne


def identity(input_argument):
    "Returns the `input_argument`."
    return input_argument


def gaussian_smoothing(
    input_tensor: torch.Tensor,
    sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
    truncate: Union[int, float, Sequence[Union[int, float]]] = 3,
    normalize: Union[Literal["sum", "gaussian"], None] = "sum",
) -> torch.Tensor:
    """
    Apply Gaussian smoothing to the {1D, 2D, 3D} input tensor.

    Kernel size is automatically determined as 2 * int(truncate * sigma + 0.5) + 1 for each
    dimension. This ensures the kernel captures the appropriate number of standard deviations
    (default: 3 sigma, which captures ~99.7% of the Gaussian distribution).

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor, assumed to be 1D, 2D, or 3D with shape (B, C, *spatial).
    sigma : float, int, or Sequence[float or int], default=1
        Standard deviation of the Gaussian kernel. If float/int, same sigma is used
        for all dimensions. If Sequence, different sigmas can be specified per dimension.
    truncate : int, float, or Sequence[int or float], default=3
        Number of standard deviations at which to truncate the kernel. If scalar, same
        truncate value is used for all dimensions. If Sequence, different truncate values
        can be specified per dimension (must match sigma length).
    normalize : {'sum', 'gaussian'} or None, default='sum'
        How to normalize the Gaussian kernel. See `neurite.gaussian_kernel` for details.

    Returns
    -------
    smoothed_tensor : torch.Tensor
        The smoothed tensor with the same shape as input_tensor.

    Examples
    --------
    >>> import torch
    >>> # Make an input tensor
    >>> input_tensor = torch.rand(1, 1, 16, 16, 16)
    >>> # Smooth with default parameters (sigma=1, truncate=3)
    >>> smoothed_tensor = gaussian_smoothing(input_tensor)
    >>> # Smooth with per-dimension sigma values
    >>> smoothed_tensor = gaussian_smoothing(
    ...     input_tensor,
    ...     sigma=[0.5, 1.0, 1.5]
    ... )
    >>> # Smooth with custom truncate
    >>> smoothed_tensor = gaussian_smoothing(input_tensor, sigma=2.0, truncate=4)

    Notes
    -----
    The automatic kernel sizing follows the formula used in scipy and VoxelMorph:
    kernel_size = 2 * int(truncate * sigma + 0.5) + 1

    This ensures proper Gaussian kernel coverage regardless of sigma value, preventing
    the mathematical errors that occur when kernel_size is too small for the given sigma.
    """

    # Infer spatial dimensionality (subtract batch and channel dims)
    ndim = input_tensor.dim() - 2
    nchannels = input_tensor.shape[1]

    # Create Gaussian kernel with automatic sizing (on same device/dtype as input)
    kernel = ne.gaussian_kernel(
        sigma=sigma,
        truncate=truncate,
        ndim=ndim,
        normalize=normalize,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    # Add channel dimensions for depthwise convolution: (nchannels, 1, *spatial)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    if nchannels > 1:
        kernel = kernel.repeat(nchannels, 1, *([1] * ndim))

    # Depthwise: groups==nchannels ensures each channel is blurred independently
    # Use padding="same" to maintain output shape (zero-padding)
    conv_fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[ndim]
    smoothed_tensor = conv_fn(input=input_tensor, weight=kernel, padding="same", groups=nchannels)

    return smoothed_tensor


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
        # Apply subsampling using slice notation
        dim_idx = int(dimension)
        slices = [slice(None)] * input_tensor.dim()
        slices[dim_idx] = slice(None, None, stride)
        input_tensor = input_tensor[tuple(slices)]

    return input_tensor


def resample(
    input_tensor: torch.Tensor,
    size: Union[Sequence[int], None] = None,
    scale_factor: Union[int, float, Sequence[Union[int, float]], None] = None,
    mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
    antialias: bool = False
) -> torch.Tensor:
    """
    Resample 1D, 2D, or 3D tensors to a given size or scale factor.

    Wraps `neurite.functional.resample()` with automatic mode inference for linear interpolation.
    Handles both upsampling (scale > 1) and downsampling (scale < 1).

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor to be resampled, with shape (B, C, *spatial_dims).
    size : Sequence[int] or None, default=None
        Target spatial dimensions. If None, `scale_factor` must be specified.
    scale_factor : int, float, Sequence[int], Sequence[float], or None, default=None
        Factor by which to resample each spatial dimension. If None, `size` must be specified.
    mode : {'linear', 'nearest', 'bicubic', 'area', 'nearest-exact'}, default='linear'
        Interpolation mode. 'linear' is automatically converted to 'linear'/'bilinear'/'trilinear'
        based on spatial dimensionality.
    antialias : bool, default=False
        If True, apply antialiasing when downsampling. Only supported with 'bilinear' and
        'bicubic' modes.

    Returns
    -------
    torch.Tensor
        The resampled tensor with shape (B, C, *resampled_spatial_dims).

    Examples
    --------
    >>> import torch
    >>> # Downsample a 2D image by factor of 2
    >>> input_tensor = torch.randn(1, 3, 64, 64)
    >>> downsampled = resample(input_tensor, scale_factor=0.5)
    >>> print(downsampled.shape)
    torch.Size([1, 3, 32, 32])

    >>> # Upsample a 3D volume to specific size
    >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
    >>> upsampled = resample(input_tensor, size=(64, 64, 64))
    >>> print(upsampled.shape)
    torch.Size([1, 1, 64, 64, 64])

    >>> # Downsample with antialiasing
    >>> input_tensor = torch.randn(2, 1, 128, 128)
    >>> downsampled = resample(input_tensor, scale_factor=0.25, antialias=True)
    >>> print(downsampled.shape)
    torch.Size([2, 1, 32, 32])
    """
    return ne.resample(
        input_tensor=input_tensor,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        non_spatial_dims=(0, 1),
        antialias=antialias
    )


def resample_voxel_dimensions(
    input_tensor: torch.Tensor,
    downsample_scale: Union[Union[int, float], Sequence[Union[int, float]]] = 0.5,
    upsample_scale: Union[Union[int, float], Sequence[Union[int, float]]] = 2,
    mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
    shape: Union[Sequence[int], None] = None,
) -> torch.Tensor:
    """
    Resample tensor to simulate different voxel dimensions.

    Combines downsampling and upsampling by first subsampling `input_tensor` along
    specified dimensions by `downsample_stride`, then upsampling back to `shape`.
    This is useful for simulating anisotropic voxel dimensions in medical imaging.

    Parameters
    ----------
    input_tensor : torch.Tensor
        The tensor to resample, with shape (B, C, *spatial_dims).
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
        The resampled tensor with the same batch and channel dims as `input_tensor`
        and spatial dims equal to `shape`.

    Examples
    --------
    >>> import torch
    >>> input_tensor = torch.randn(1, 3, 32, 32)
    >>> # Subsample rows/cols by 2, then upsample to (64, 64)
    >>> res = resample_voxel_dimensions(
    ...     input_tensor, shape=(64, 64),
    ...     downsample_stride=2,
    ...     mode='linear'
    ... )
    >>> print(res.shape)
    torch.Size([1, 3, 64, 64])
    """

    resampled = resample(input_tensor, scale_factor=downsample_scale)
    resampled = resample(resampled, scale_factor=upsample_scale, size=shape, mode=mode)

    return resampled


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


def volshape_to_ndgrid(
    size: Sequence[int],
    device: Union[str, torch.device] = "cpu",
    dtype: Union[str, torch.dtype] = torch.float32,
    normalize: bool = False,
    indexing: Literal["ij", "xy"] = "ij",
    stack: bool = False,
) -> torch.Tensor:
    """
    Generate a grid of spatial coordinates for tensors in (B, C, *spatial) format.

    This is a convenience wrapper that extracts the spatial dimensions from a
    (B, C, *spatial) shape and returns the grid for just the spatial part.

    Parameters
    ----------
    size : Sequence[int]
        Size in (B, C, *spatial) format. B and C are ignored; only spatial dims are used.
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
        If True, stack the grid tensors along the first dimension to return a single tensor of
        shape `(len(spatial), *spatial)`. If False, return a tuple of tensors, each of shape
        `(*spatial)`.

    Returns
    -------
    torch.Tensor
        The meshgrid of spatial coordinates (without B, C dimensions).
        if stack=False, a tuple of len(spatial) tensors of shape (*spatial)
        if stack=True, a tensor of shape (len(spatial), *spatial) i.e. (ndim, *spatial)

    Examples
    --------
    >>> # Make a 2d grid for spatial part of a (2, 3, 4, 5) tensor
    >>> the_grid = volshape_to_ndgrid(size=(2, 3, 4, 5), stack=True)
    >>> print(the_grid.shape)
    torch.Size([2, 4, 5])
    """
    # Extract spatial dimensions only (skip B and C)
    spatial_size = size[2:]

    # Get base grid - returns (ndim, *spatial) when stacked
    return ne.volshape_to_ndgrid(
        size=spatial_size,
        device=device,
        dtype=dtype,
        normalize=normalize,
        indexing=indexing,
        stack=stack
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
    Compute Dice score over multiple segmentation maps with shape (B, L, *spatial_dims).

    Parameters
    ----------
    *segs : torch.Tensor
        Two or more segmentation tensors of shape (B, L, *spatial_dims) with values in [0, 1].
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
        Dice score. If reduction=None, returns shape (B, L). Otherwise, reduced as specified.

    Examples
    --------
    >>> # Compute dice for 2 segmentation tensors (batch=2, labels=1, H=W=32) with no reduction
    >>> seg1 = torch.rand((2, 1, 32, 32))
    >>> seg2 = torch.rand((2, 1, 32, 32))
    >>> score = dice(seg1, seg2, reduction=None)
    >>> print(score.shape)
    torch.Size([2, 1])

    >>> # Compute the dice for three labels (batch=2, labels=3) with mean reduction
    >>> segs = [torch.rand((2, 3, 64, 64)) for _ in range(3)]
    >>> per_label = dice(*segs, reduction='mean')
    >>> print(per_label.shape)
    torch.Size([1, 1])
    """
    # Compute Dice using base implementation with (B, L) preserved
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


def ncc(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    window_size: Union[int, Sequence[int]] = 9,
    eps: float = 1e-5,
    reduction: Union[str, None] = 'mean',
    reduction_dim: Union[int, Tuple[int, ...]] = (0, 1),
    keepdims: bool = True,
) -> torch.Tensor:
    """
    Compute local normalized cross-correlation (NCC) for tensors with shape (B, C, *spatial_dims).

    Parameters
    ----------
    tensor1 : torch.Tensor
        First input tensor with shape (B, C, *spatial_dims).
    tensor2 : torch.Tensor
        Second input tensor with same shape as tensor1.
    window_size : int or Sequence[int], default=9
        Size of local window for computing correlation. If int, same size for all
        spatial dimensions. If Sequence, per-dimension window sizes.
    eps : float, default=1e-5
        Small constant for numerical stability in division.
    reduction : str or None, default='mean'
        Reduction to apply over batch and channel dimensions. Supported values:
        'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'.
        If None, returns shape (B, C).
    reduction_dim : int or tuple of ints, default=(0, 1)
        Dimension(s) over which to apply the reduction.
    keepdims : bool, default=True
        Whether to retain reduced dimensions as singletons.

    Returns
    -------
    torch.Tensor
        NCC values (squared correlation coefficients) in range [0, 1].
        If reduction=None, returns shape (B, C). Otherwise, reduced as specified.

    Examples
    --------
    >>> import torch
    >>> import neurite.nn.functional as nef
    # Compute mean NCC across batch and channels
    >>> t1 = torch.rand(2, 3, 64, 64)
    >>> t2 = torch.rand(2, 3, 64, 64)
    >>> score = nef.ncc(t1, t2)
    >>> print(score.shape)
    torch.Size([1, 1])

    # Compute per-batch-and-channel NCC (no reduction)
    >>> score = nef.ncc(t1, t2, reduction=None)
    >>> print(score.shape)
    torch.Size([2, 3])

    # Compute NCC with custom window size
    >>> score = nef.ncc(t1, t2, window_size=5)
    >>> print(score.shape)
    torch.Size([1, 1])

    Notes
    -----
    The NCC is computed as the squared Pearson correlation coefficient:
        NCC = (cov(I, J))^2 / (var(I) * var(J))

    Values close to 1 indicate high similarity, values close to 0 indicate low similarity.

    References
    ----------
    .. [1] Balakrishnan et al., "VoxelMorph: A Learning Framework for Deformable
           Medical Image Registration", IEEE TMI, 2019.
    """
    # Compute NCC using base implementation with (B, C) preserved
    ncc_score = ne.ncc(
        tensor1=tensor1,
        tensor2=tensor2,
        window_size=window_size,
        non_spatial_dims=(0, 1),
        eps=eps,
    )

    if reduction is None:
        return ncc_score

    return reduce(
        tensor=ncc_score,
        reduction=reduction,
        dim=reduction_dim,
        keepdims=keepdims,
    )


def spatial_gradient(
    input_tensor: torch.Tensor,
    penalty: Literal['l1', 'l2'] = 'l2',
    reduction: Union[str, None] = 'mean',
    reduction_dim: Union[int, Tuple[int, ...], None] = None,
    keepdims: bool = False,
) -> torch.Tensor:
    """
    Compute spatial gradient penalty for tensors with shape (B, C, *spatial_dims).

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor with shape (B, C, *spatial_dims), e.g., displacement field.
    penalty : {'l1', 'l2'}, default='l2'
        Penalty type to apply to gradients:
        - 'l1': absolute value (promotes sparsity)
        - 'l2': squared value (promotes smoothness)
    reduction : str or None, default='mean'
        Reduction to apply. Supported values:
        'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var'.
        If None, returns the raw penalty values for each spatial dimension.
    reduction_dim : int, tuple of ints, or None, default=None
        Dimension(s) over which to apply the reduction. If None, reduces over
        all dimensions.
    keepdims : bool, default=False
        Whether to retain reduced dimensions as singletons.

    Returns
    -------
    torch.Tensor
        If reduction is None: list of penalty tensors (one per spatial dim).
        Otherwise: reduced scalar or tensor depending on reduction_dim.

    Examples
    --------
    >>> import torch
    >>> import neurite.nn.functional as nef
    # Compute L2 gradient penalty (smoothness loss)
    >>> displacement = torch.rand(2, 3, 64, 64, 64)  # (B, ndim, D, H, W)
    >>> loss = nef.spatial_gradient(displacement, penalty='l2')
    >>> print(loss.shape)
    torch.Size([])

    # Compute L1 gradient penalty
    >>> loss = nef.spatial_gradient(displacement, penalty='l1')
    >>> print(loss.shape)
    torch.Size([])

    References
    ----------
    .. [1] Balakrishnan et al., "VoxelMorph: A Learning Framework for Deformable
           Medical Image Registration", IEEE TMI, 2019.
    """
    if penalty not in ['l1', 'l2']:
        raise ValueError(f"penalty must be 'l1' or 'l2', got '{penalty}'")

    # Compute spatial gradients (list of tensors, one per spatial dim)
    grads = ne.spatial_gradient(input_tensor, non_spatial_dims=(0, 1))

    # Apply penalty
    if penalty == 'l1':
        penalties = [g.abs() for g in grads]
    else:  # l2
        penalties = [g * g for g in grads]

    if reduction is None:
        return penalties

    # Reduce each penalty tensor and average across spatial dimensions
    reduced = []
    for p in penalties:
        r = reduce(tensor=p, reduction=reduction, dim=reduction_dim, keepdims=keepdims)
        reduced.append(r)

    # Average across spatial dimensions
    return sum(reduced) / len(reduced)


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

        linear = ne.utils.infer_linear_interpolation_mode(image.ndim - 1)
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


def crop(
    input_tensor: torch.Tensor,
    size: Union[int, Sequence[int], None] = None,
    scale_factor: Union[float, Sequence[float], None] = None,
    offset: Union[int, Sequence[int]] = 0,
) -> torch.Tensor:
    """
    Crop tensor to specified size.

    Batch and channel dimensions are preserved. Mirrors torch.nn.functional.interpolate
    API: specify either `size` or `scale_factor` (mutually exclusive).

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor with shape (B, C, *spatial_dims).
    size : int, Sequence[int], or None, default=None
        Target spatial size(s). If int, same size for all spatial dims.
        If Sequence, per-dimension sizes. If None, `scale_factor` must be specified.
    scale_factor : float, Sequence[float], or None, default=None
        Multiplicative factor for spatial size. Output size = input size * scale_factor.
        If None, `size` must be specified.
    offset : int or Sequence[int], default=0
        Starting position for crop. If int, same offset for all spatial dimensions.
        If Sequence, per-dimension offsets.

    Returns
    -------
    torch.Tensor
        Cropped tensor with shape (B, C, *cropped_spatial_dims).
        Batch and channel dimensions preserved.

    Raises
    ------
    ValueError
        If both `size` and `scale_factor` are specified or both are None.

    Examples
    --------
    >>> import torch
    >>> import neurite.nn.functional as nef
    >>> x = torch.randn(2, 3, 64, 64)
    >>> # Crop from origin
    >>> cropped = nef.crop(x, size=32)
    >>> cropped.shape
    torch.Size([2, 3, 32, 32])
    >>> # Crop from offset
    >>> cropped = nef.crop(x, size=32, offset=16)
    >>> cropped.shape
    torch.Size([2, 3, 32, 32])
    >>> # Random crop (user controls randomness)
    >>> offset = torch.randint(0, 33, (2,)).tolist()
    >>> cropped = nef.crop(x, size=32, offset=offset)
    """
    return ne.crop(
        input_tensor=input_tensor,
        size=size,
        scale_factor=scale_factor,
        non_spatial_dims=(0, 1),
        offset=offset
    )


def clip(
    input_tensor: torch.Tensor,
    min: Union[float, int, None] = None,
    max: Union[float, int, None] = None,
) -> torch.Tensor:
    """
    Clip tensor values to specified range.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor with shape (B, C, *spatial_dims).
    min : float, int, or None, default=None
        Minimum value. If None, no lower bound.
    max : float, int, or None, default=None
        Maximum value. If None, no upper bound.

    Returns
    -------
    torch.Tensor
        Clipped tensor with same shape as input.

    Examples
    --------
    >>> import torch
    >>> import neurite.nn.functional as nef
    >>> x = torch.randn(2, 3, 32, 32) * 5
    >>> # Clip to [0, 1]
    >>> clipped = nef.clip(x, min=0, max=1)
    >>> clipped.shape
    torch.Size([2, 3, 32, 32])
    """
    return ne.clip(input_tensor, min=min, max=max)


def random_smoothed_noise(
    shape: Sequence[int],
    sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
    magnitude: float = 1.0,
    normalize: Union[Literal["sum", "gaussian"], None] = "sum",
    device: Union[torch.device, None] = None,
) -> torch.Tensor:
    """
    Generate smooth Gaussian noise in (B, C, *spatial) format.

    Parameters
    ----------
    shape : Sequence[int]
        Desired shape of output tensor in (B, C, *spatial). Must have at least 3 dimensions
        (batch, channel, and spatial). Examples: (1, 1, 64, 64) for 2d, (2, 3, 64, 64, 64) for 3d.
    sigma : float, int, or Sequence[float or int], default=1
        Standard deviation of the Gaussian kernel for smoothing. If float/int, same sigma
        is used for all spatial dimensions. If Sequence, different sigmas per dimension.
    magnitude : float, default=1.0
        Standard deviation of the noise after normalization.
    normalize : {'sum', 'gaussian'} or None, default='sum'
        How to normalize the Gaussian kernel. See `neurite.gaussian_kernel` for details.
    device : torch.device or None, default=None
        Device for tensor allocation. If None, defaults to CPU.

    Returns
    -------
    torch.Tensor
        Smooth Gaussian noise with shape (B, C, *spatial), zero mean, and standard
        deviation equal to magnitude.

    Examples
    --------
    >>> import neurite.nn.functional as nef
    >>> # Generate 2d noise field
    >>> noise_2d = nef.random_smoothed_noise(shape=(1, 1, 64, 64), sigma=2.0)
    >>> noise_2d.shape
    torch.Size([1, 1, 64, 64])

    >>> # Generate 3d noise field with multiple channels
    >>> noise_3d = nef.random_smoothed_noise(shape=(2, 3, 32, 32, 32), sigma=3.0, magnitude=2.0)
    >>> noise_3d.shape
    torch.Size([2, 3, 32, 32, 32])
    """
    return ne.random_smoothed_noise(
        shape=shape,
        sigma=sigma,
        magnitude=magnitude,
        non_spatial_dims=(0, 1),
        normalize=normalize,
        device=device,
    )


def upsample_noise(
    shape: Sequence[int],
    scale: Union[float, int],
    device: Union[torch.device, None] = None
) -> torch.Tensor:
    """
    Generate smooth noise by upsampling from a coarse grid in (B, C, *spatial) format.

    Parameters
    ----------
    shape : Sequence[int]
        Target shape in (B, C, *spatial) format. Must have at least 3 dimensions.
    scale : float or int
        Downsampling factor. Larger values produce smoother noise.
    device : torch.device or None, default=None
        Device for tensor allocation.

    Returns
    -------
    torch.Tensor
        Upsampled noise with shape (B, C, *spatial).

    Examples
    --------
    >>> import neurite.nn.functional as nef
    >>> # Generate 2d noise field
    >>> noise_2d = nef.upsample_noise(shape=(1, 1, 64, 64), scale=8.0)
    >>> noise_2d.shape
    torch.Size([1, 1, 64, 64])

    >>> # Generate 3d noise field with multiple channels
    >>> noise_3d = nef.upsample_noise(shape=(2, 3, 32, 32, 32), scale=4.0)
    >>> noise_3d.shape
    torch.Size([2, 3, 32, 32, 32])
    """
    return ne.upsample_noise(
        shape=shape,
        scale=scale,
        non_spatial_dims=(0, 1),
        device=device,
    )


def fractal_noise(
    shape: Sequence[int],
    scales: Union[float, int, Sequence[Union[float, int]], None] = None,
    magnitude: float = 1.0,
    weights: Union[Sequence[float], None] = None,
    normalize: Union[Literal["sum", "gaussian"], None] = "sum",
    device: Union[torch.device, None] = None,
    method: Literal['blur', 'upsample'] = 'blur'
) -> torch.Tensor:
    """
    Generate fractal noise in (B, C, *spatial) format.

    Parameters
    ----------
    shape : Sequence[int]
        Target shape in (B, C, *spatial) format. Must have at least 3 dimensions.
    scales : float, int, Sequence[float or int], or None, default=None
        Smoothing scale(s) for each octave. If None, defaults to powers of 2.
    magnitude : float, default=1.0
        Standard deviation of the final normalized noise.
    weights : Sequence[float] or None, default=None
        Weight for each scale. If None, uses linearly increasing weights.
    normalize : {'sum', 'gaussian'} or None, default='sum'
        How to normalize the Gaussian kernel. See `neurite.gaussian_kernel` for details.
        Only used when method='blur'.
    device : torch.device or None, default=None
        Device for tensor allocation.
    method : {'blur', 'upsample'}, default='blur'
        Noise generation method.

    Returns
    -------
    torch.Tensor
        Fractal noise with shape (B, C, *spatial).

    Examples
    --------
    >>> import neurite.nn.functional as nef
    >>> # Generate 2d fractal noise with default scales
    >>> noise_2d = nef.fractal_noise(shape=(1, 1, 64, 64))
    >>> noise_2d.shape
    torch.Size([1, 1, 64, 64])

    >>> # Generate 3d fractal noise with custom scales
    >>> noise_3d = nef.fractal_noise(shape=(1, 1, 32, 32, 32), scales=[2.0, 4.0, 8.0])
    >>> noise_3d.shape
    torch.Size([1, 1, 32, 32, 32])
    """
    return ne.fractal_noise(
        shape=shape,
        scales=scales,
        magnitude=magnitude,
        weights=weights,
        non_spatial_dims=(0, 1),
        normalize=normalize,
        device=device,
        method=method,
    )
