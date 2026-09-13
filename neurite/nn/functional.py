"""
Tensor operations and functions for Neurite.

A collection of functions for manipulating and analyzing PyTorch tensors,
with applications a focus on imaging.

Notes
-----
- All functions assume tensors follow the (B, C, *spatial_dims) convention.
"""

# Standard library imports
import math
from typing import Literal, Optional, Sequence, Tuple, Union

# Third party imports
import numpy as np
import torch
import torch.nn.functional as F

# Custom imports
from neurite.utils.utils import bernoulli, infer_linear_interpolation_mode


def identity(input_argument):
    "Returns the `input_argument`."
    return input_argument


def gaussian_kernel(
    sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
    truncate: Union[int, float, Sequence[Union[int, float]]] = 3,
    ndim: Optional[int] = None,
    normalize: Union[Literal["sum", "gaussian"], None] = "sum",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """
    Create a {1D, 2D, 3D} Gaussian kernel with automatic kernel sizing.

    Kernel size is automatically determined as 2 * int(truncate * sigma + 0.5) + 1 for each
    dimension. This ensures the kernel captures the appropriate number of standard deviations
    (default: 3 sigma, which captures ~99.7% of the Gaussian distribution).

    Shape-agnostic implementation that returns a kernel with only spatial dimensions,
    no batch or channel dimensions. Dimensionality is inferred from the length of `sigma`
    if it is a sequence, or from the `ndim` parameter if `sigma` is scalar.

    Parameters
    ----------
    sigma : float, int, or Sequence[float or int], optional
        Standard deviation of the Gaussian kernel. If float/int, same sigma is used for all
        dimensions. If Sequence, different sigmas can be specified per dimension and length
        determines dimensionality (1D, 2D, or 3D). Default is 1.
    truncate : int, float, or Sequence[int or float], optional
        Number of standard deviations at which to truncate the kernel. If scalar, same
        truncate value is used for all dimensions. If Sequence, different truncate values
        can be specified per dimension (must match sigma length). Default is 3.
    ndim : int, optional
        Number of spatial dimensions (1, 2, or 3). Only required when sigma is scalar.
        If sigma is a sequence, ndim is inferred from its length. Default is None.
    normalize : {'sum', 'gaussian'} or None, default='sum'
        How to normalize the kernel:
        - 'sum': divide by the discrete sum of kernel values so the kernel sums to 1.
        - 'gaussian': divide by the analytical Gaussian normalization constant
          (2*pi)^(ndim/2) * prod(sigmas). The center value equals the true PDF peak.
        - None: no normalization. Returns raw exp(-0.5 * (x/sigma)^2) values.
    device : torch.device, optional
        Device on which to create the kernel tensor. Default is None.
    dtype : torch.dtype, optional
        Data type of the kernel tensor. Default is torch.float32.

    Returns
    -------
    torch.Tensor
        Tensor representing the {1D, 2D, 3D} Gaussian kernel with automatically computed
        shape. No batch or channel dimensions.

    Examples
    --------
    >>> import torch
    # Make a 3D kernel with automatic sizing (normalized by sum, default)
    >>> gaussian_kernel_ = gaussian_kernel(sigma=1.0, ndim=3)
    >>> gaussian_kernel_.shape
    torch.Size([7, 7, 7])

    # Make a 2D kernel with different sigmas per dimension
    >>> gaussian_kernel_ = gaussian_kernel(sigma=(0.5, 2.0))
    # Kernel sizes: [5, 13]
    >>> gaussian_kernel_.shape
    torch.Size([5, 13])

    # Make a 1D kernel with custom truncate
    >>> gaussian_kernel_ = gaussian_kernel(sigma=2.0, truncate=4, ndim=1)
    >>> gaussian_kernel_.shape
    torch.Size([17])

    # Unnormalized kernel
    >>> gaussian_kernel_ = gaussian_kernel(sigma=1.0, ndim=2, normalize=None)
    >>> gaussian_kernel_.sum()  # Will NOT be 1.0

    Notes
    -----
    The automatic kernel sizing follows the formula used in scipy and VoxelMorph:
    kernel_size = 2 * int(truncate * sigma + 0.5) + 1

    This ensures the kernel is always odd-sized and captures the specified number of
    standard deviations. A truncate value of 3 captures ~99.7% of the Gaussian distribution.
    """
    if isinstance(sigma, (float, int)):
        assert ndim is not None, (
            "When sigma is a scalar, ndim must be specified to determine dimensionality"
        )
        assert ndim in [1, 2, 3], f"ndim must be 1, 2, or 3, got {ndim}"
        sigma_list = [float(sigma)] * ndim

    elif isinstance(sigma, Sequence):
        sigma_list = [float(s) for s in sigma]
        ndim = len(sigma_list)
        assert ndim in [1, 2, 3], (
            f"sigma length determines dimensionality and must be 1, 2, or 3. "
            f"Got length {ndim}"
        )
    else:
        raise TypeError(f"sigma must be a number or sequence, got {type(sigma)}")

    if isinstance(truncate, (int, float)):
        truncate_list = [float(truncate)] * ndim

    elif isinstance(truncate, Sequence):
        assert len(truncate) == ndim, (
            f"If truncate is a sequence, it must have length equal to sigma length "
            f"({ndim}). Got length {len(truncate)}"
        )
        truncate_list = [float(t) for t in truncate]
    else:
        raise TypeError(f"truncate must be a number or sequence, got {type(truncate)}")

    # Compute kernel size for each dimension: 2 * int(truncate * sigma + 0.5) + 1
    kernel_size_list = [2 * int(t * s + 0.5) + 1 for s, t in zip(sigma_list, truncate_list)]

    # Create coordinate grid centered at zero
    coords = [
        torch.arange(ks, device=device, dtype=dtype).float() - (ks - 1) / 2
        for ks in kernel_size_list
    ]

    grid = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
    sigma_tensor = torch.tensor(sigma_list, device=device, dtype=dtype)

    # Calculate the Gaussian function: exp(-0.5 * sum((x / sigma)^2))
    kernel = torch.exp(-0.5 * (grid ** 2 / sigma_tensor**2).sum(dim=-1))

    assert normalize in {"sum", "gaussian", None}, (
        f"normalize must be 'sum', 'gaussian', or None, got '{normalize}'"
    )

    if normalize == "sum":
        kernel /= kernel.sum()
    elif normalize == "gaussian":
        norm_const = (2 * torch.pi) ** (ndim / 2) * torch.prod(sigma_tensor)
        kernel /= norm_const
    return kernel


def gaussian_smoothing(
    input_tensor: torch.Tensor,
    sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
    truncate: Union[int, float, Sequence[Union[int, float]]] = 3,
    normalize: Union[Literal["sum", "gaussian"], None] = "sum",
    padding_mode: str = "constant",
    method: Literal["dense", "separable"] = "dense",
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
    padding_mode : {'constant', 'reflect', 'replicate', 'circular'}, default='constant'
        Boundary padding applied before convolution.
    method : {'dense', 'separable'}, default='dense'
        Convolution method. The dense method applies one multidimensional kernel. The
        separable method applies one one-dimensional kernel along each spatial axis.

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

    if padding_mode not in {"constant", "reflect", "replicate", "circular"}:
        raise ValueError(f"unsupported padding mode: {padding_mode}")
    if method not in {"dense", "separable"}:
        raise ValueError(f"unsupported Gaussian smoothing method: {method}")

    # Infer spatial dimensionality and select the matching convolution.
    ndim = input_tensor.dim() - 2
    nchannels = input_tensor.shape[1]
    conv_fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[ndim]

    if method == "separable":
        if isinstance(sigma, (float, int)):
            sigma_values = (float(sigma),) * ndim
        elif isinstance(sigma, Sequence):
            assert len(sigma) == ndim, "sigma must have one value per spatial dimension"
            sigma_values = tuple(float(value) for value in sigma)
        else:
            raise TypeError(f"sigma must be a number or sequence, got {type(sigma)}")

        if isinstance(truncate, (float, int)):
            truncate_values = (float(truncate),) * ndim
        elif isinstance(truncate, Sequence):
            assert len(truncate) == ndim, "truncate must have one value per spatial dimension"
            truncate_values = tuple(float(value) for value in truncate)
        else:
            raise TypeError(f"truncate must be a number or sequence, got {type(truncate)}")

        # Apply each one-dimensional factor independently along its spatial axis.
        smoothed_tensor = input_tensor
        axis_parameters = zip(sigma_values, truncate_values)
        for axis, (axis_sigma, axis_truncate) in enumerate(axis_parameters):
            kernel = gaussian_kernel(
                sigma=axis_sigma,
                truncate=axis_truncate,
                ndim=1,
                normalize=normalize,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
            kernel_shape = [1, 1] + [1] * ndim
            kernel_shape[axis + 2] = len(kernel)
            weight = kernel.reshape(kernel_shape)
            if nchannels > 1:
                weight = weight.repeat(nchannels, 1, *([1] * ndim))

            padding = [0] * (2 * ndim)
            reverse_axis = ndim - axis - 1
            radius = len(kernel) // 2
            padding[2 * reverse_axis:2 * reverse_axis + 2] = (radius, radius)
            padded = F.pad(smoothed_tensor, padding, mode=padding_mode)
            smoothed_tensor = conv_fn(input=padded, weight=weight, groups=nchannels)
        return smoothed_tensor

    # Build and apply one dense depthwise kernel per channel.
    kernel = gaussian_kernel(
        sigma=sigma,
        truncate=truncate,
        ndim=ndim,
        normalize=normalize,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    kernel = kernel.unsqueeze(0).unsqueeze(0)
    if nchannels > 1:
        kernel = kernel.repeat(nchannels, 1, *([1] * ndim))

    # Pad explicitly so callers can choose boundary behavior while preserving shape.
    padding = []
    for size in reversed(kernel.shape[2:]):
        radius = size // 2
        padding.extend((radius, radius))
    padded = F.pad(input_tensor, padding, mode=padding_mode)

    smoothed_tensor = conv_fn(input=padded, weight=kernel, groups=nchannels)

    return smoothed_tensor


def box_filter(
    input_tensor: torch.Tensor,
    window_size: Union[int, Sequence[int]],
) -> torch.Tensor:
    """
    Compute zero-padded local sums with a separable box filter.

    The input must follow the ``(B, C, *spatial_dims)`` convention. Filtering is
    independent across batches and channels, and the operation is differentiable with
    respect to ``input_tensor``.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor with one, two, or three spatial dimensions.
    window_size : int or Sequence[int]
        Width of the box window. An integer uses the same width along every spatial
        dimension. A sequence specifies one width per spatial dimension.

    Returns
    -------
    torch.Tensor
        Local window sums. Odd window sizes preserve the input shape. As with a dense
        convolution padded by ``window_size // 2``, even window sizes add one element
        along the corresponding spatial dimension.

    Examples
    --------
    >>> tensor = torch.ones(1, 1, 5, 5)
    >>> filtered = box_filter(tensor, window_size=3)
    >>> filtered[0, 0, 2, 2]
    tensor(9.)

    Notes
    -----
    A multidimensional box kernel is separable. Applying one one-dimensional pooling
    pass per spatial dimension reduces the work for a cubic window of width ``W`` from
    order ``W ** D`` to ``D * W`` operations per output element.
    """
    num_spatial = input_tensor.ndim - 2
    assert num_spatial in [1, 2, 3], (
        f"Only 1D, 2D, 3D spatial dimensions supported. Got {num_spatial}D"
    )

    if isinstance(window_size, int):
        window_size = [window_size] * num_spatial
    else:
        window_size = list(window_size)
        assert len(window_size) == num_spatial, (
            f'window_size length {len(window_size)} does not match spatial dims {num_spatial}'
        )

    assert all(size > 0 for size in window_size), (
        f'window_size values must be positive. Got {window_size}'
    )

    # Apply one local-sum pass per axis. Average pooling avoids constructing dense kernels.
    filtered = input_tensor
    if num_spatial == 1:
        size = window_size[0]
        padding = size // 2
        if filtered.shape[2] < size:
            filtered = F.pad(filtered, (padding, padding))
            padding = 0

        filtered = F.avg_pool1d(
            filtered,
            kernel_size=size,
            stride=1,
            padding=padding,
            count_include_pad=True,
        )
        return filtered * size

    pool_fn = {2: F.avg_pool2d, 3: F.avg_pool3d}[num_spatial]
    for dim, size in enumerate(window_size):
        kernel_size = [1] * num_spatial
        kernel_size[dim] = size
        padding = [0] * num_spatial
        padding[dim] = size // 2

        # Pooling rejects kernels wider than the input despite sufficient implicit padding.
        if filtered.shape[2 + dim] < size:
            explicit_padding = [0] * (2 * num_spatial)
            padding_offset = 2 * (num_spatial - dim - 1)
            explicit_padding[padding_offset:padding_offset + 2] = [size // 2] * 2
            filtered = F.pad(filtered, explicit_padding)
            padding[dim] = 0

        filtered = pool_fn(
            filtered,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            divisor_override=1,
        )

    return filtered


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
    bernoulli_mask = bernoulli(p=p, shape=input_tensor.shape)
    masked = torch.clone(input_tensor)

    if returns == 'successes':
        masked = masked[bernoulli_mask == 1]
    elif returns == 'failures':
        masked = masked[bernoulli_mask == 0]
    elif returns is None:
        masked[bernoulli_mask == 0] = 0
    else:
        assert returns in {'successes', 'failures', None}, f"{returns} isn't supported!"

    return masked


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
    assert size is not None or scale_factor is not None, (
        "Either size or scale_factor must be specified"
    )

    spatial_ndim = input_tensor.ndim - 2
    assert spatial_ndim in {1, 2, 3}, (
        f"Unsupported spatial dimensionality: {spatial_ndim} spatial dimensions. "
        "Only 1D, 2D, and 3D are supported."
    )

    if mode == 'linear':
        mode = infer_linear_interpolation_mode(spatial_ndim)

    if size is not None:
        return F.interpolate(input=input_tensor, size=size, mode=mode, antialias=antialias)

    return F.interpolate(
        input=input_tensor,
        scale_factor=scale_factor,
        mode=mode,
        antialias=antialias,
    )


def pad_to_multiple_of(
    input_tensor: torch.Tensor,
    multiple: int = 32,
    value: float = 0.0,
) -> torch.Tensor:
    """
    Pad spatial dimensions to multiples of a fixed value.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor with shape `(B, C, *spatial)`.
    multiple : int, default=32
        Positive value that each spatial output size must be divisible by.
    value : float, default=0.0
        Constant padding value.

    Returns
    -------
    torch.Tensor
        Padded tensor with shape `(B, C, *padded_spatial)`.

    Examples
    --------
    >>> import torch
    >>> import neurite.nn.functional as nef
    >>> tensor = torch.ones(2, 3, 5, 7)
    >>> padded = nef.pad_to_multiple_of(tensor, multiple=4)
    >>> padded.shape
    torch.Size([2, 3, 8, 8])
    """
    assert input_tensor.ndim >= 3, "input_tensor must have shape (B, C, *spatial)."
    assert isinstance(multiple, int) and multiple > 0, "multiple must be a positive integer."

    padding = []
    for size in reversed(input_tensor.shape[2:]):
        total_padding = (multiple - (size % multiple)) % multiple
        padding.extend([total_padding // 2, total_padding - total_padding // 2])

    return F.pad(input_tensor, padding, value=value)


def mask_border(
    mask: torch.Tensor,
    thickness: int,
    border_mode: Literal["inner", "outer"] = "inner",
) -> torch.Tensor:
    """
    Compute the inner or outer border of a binary mask.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask with shape `(B, C, *spatial)`.
    thickness : int
        Border thickness in voxels.
    border_mode : {'inner', 'outer'}, default='inner'
        Whether to return voxels inside the mask boundary or outside the mask boundary.

    Returns
    -------
    torch.Tensor
        Border mask with the same shape and dtype as `mask`.

    Examples
    --------
    >>> import torch
    >>> import neurite.nn.functional as nef
    >>> mask = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
    >>> mask[..., 1:4, 1:4] = True
    >>> border = nef.mask_border(mask, thickness=1)
    >>> border.shape
    torch.Size([1, 1, 5, 5])
    """
    assert mask.ndim in (3, 4, 5), "mask must have shape (B, C, *spatial) for 1d, 2d, or 3d."
    assert mask.shape[1] >= 1, "mask must have at least one channel."
    assert thickness >= 0, "thickness must be non-negative."
    assert border_mode in ("inner", "outer"), "border_mode must be 'inner' or 'outer'."

    num_spatial = mask.ndim - 2
    max_pool = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[num_spatial]
    mask_float = mask.float()

    if border_mode == "inner":
        mask_to_dilate = 1 - mask_float
        support_mask = mask_float
    else:
        mask_to_dilate = mask_float
        support_mask = 1 - mask_float

    dilated_mask = max_pool(
        mask_to_dilate,
        kernel_size=2 * thickness + 1,
        stride=1,
        padding=thickness,
    )
    border = dilated_mask * support_mask

    return border.to(dtype=mask.dtype)


def normalize_reference_intensity(
    image: torch.Tensor,
    reference_mask: torch.Tensor,
    target: Union[float, int, torch.Tensor] = 1.0,
    *,
    trim_fraction: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize image intensities using a masked reference-tissue mean.

    The reference statistic and multiplicative scale are computed independently for every batch
    and channel. A single-channel mask is broadcast across image channels. Symmetric trimming can
    make the statistic less sensitive to partial-volume voxels and outliers.

    Parameters
    ----------
    image : torch.Tensor
        Floating-point image with shape ``[B, C, *spatial]``.
    reference_mask : torch.Tensor
        Boolean reference-tissue mask with shape ``[B, 1, *spatial]`` or
        ``[B, C, *spatial]``.
    target : float, int, or torch.Tensor, default=1
        Positive target value for the trimmed reference mean. A tensor must be broadcastable to
        ``[B, C]``.
    trim_fraction : float, default=0
        Fraction removed from each tail after sorting the masked values. Must lie in ``[0, 0.5)``.
        Zero computes the ordinary masked mean.
    eps : float, default=1e-6
        Smallest permitted reference mean.

    Returns
    -------
    normalized : torch.Tensor
        Globally rescaled image with the same shape, dtype, and device as ``image``.
    scale : torch.Tensor
        Multiplicative scale with shape ``[B, C, 1, ..., 1]``.

    Raises
    ------
    TypeError
        If ``image`` is not floating point or ``reference_mask`` is not boolean.
    ValueError
        If shapes or parameters are invalid, a mask is empty, or a reference statistic is
        nonfinite or too small.

    Notes
    -----
    This operation does not clamp the normalized image or constrain it to a fixed range. Sorting
    and averaging remain differentiable with respect to the selected image values.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    >>> image = torch.tensor([1.0, 2.0, 3.0, 100.0]).reshape(1, 1, 2, 2)
    >>> mask = torch.ones_like(image, dtype=torch.bool)
    >>> normalized, scale = ne.nn.normalize_reference_intensity(
    ...     image, mask, target=5.0, trim_fraction=0.25
    ... )
    >>> normalized[mask].sort().values[1:-1].mean().item()
    5.0
    >>> scale.item()
    2.0
    """

    if not image.is_floating_point():
        raise TypeError("image must be a floating-point tensor.")
    if reference_mask.dtype != torch.bool:
        raise TypeError("reference_mask must have boolean dtype.")
    if image.ndim < 3:
        raise ValueError("image must have shape [B, C, *spatial].")
    if reference_mask.ndim != image.ndim:
        raise ValueError("reference_mask and image must have the same number of dimensions.")
    if reference_mask.shape[0] != image.shape[0]:
        raise ValueError("reference_mask and image batch dimensions must match.")
    if reference_mask.shape[1] not in (1, image.shape[1]):
        raise ValueError("reference_mask must have one channel or match the image channels.")
    if reference_mask.shape[2:] != image.shape[2:]:
        raise ValueError("reference_mask and image spatial dimensions must match.")
    if reference_mask.device != image.device:
        raise ValueError("reference_mask and image must be on the same device.")
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must lie in [0, 0.5).")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    expanded_mask = reference_mask.expand(
        image.shape[0], image.shape[1], *image.shape[2:]
    )
    reference_rows = []
    for batch_index in range(image.shape[0]):
        reference_channels = []
        for channel_index in range(image.shape[1]):
            values = image[batch_index, channel_index][
                expanded_mask[batch_index, channel_index]
            ]
            if values.numel() == 0:
                raise ValueError("Every batch and channel must contain reference-mask voxels.")

            trim_count = int(trim_fraction * values.numel())
            if trim_count:
                values = values.sort().values
                values = values[trim_count:-trim_count]
            reference_channels.append(values.mean())
        reference_rows.append(torch.stack(reference_channels))
    reference = torch.stack(reference_rows)  # [B, C]

    if not torch.isfinite(reference).all() or torch.any(reference <= eps):
        raise ValueError("Reference means must be finite and greater than eps.")
    target_tensor = torch.as_tensor(target, dtype=image.dtype, device=image.device)
    try:
        target_tensor = torch.broadcast_to(target_tensor, reference.shape)
    except RuntimeError as error:
        raise ValueError("target must be broadcastable to [B, C].") from error
    if not torch.isfinite(target_tensor).all() or torch.any(target_tensor <= 0):
        raise ValueError("target values must be finite and positive.")

    scale = target_tensor / reference  # [B, C]
    scale_shape = (*scale.shape, *((1,) * (image.ndim - 2)))
    scale = scale.reshape(scale_shape)  # [B, C, 1, ..., 1]
    return image * scale, scale


def sample_locations_in_mask(
    mask: torch.Tensor,
    nb_samples: Union[int, Sequence[int]],
    sample_method: Literal["uniform", "weighted"] = "uniform",
    replacement: bool = False,
) -> torch.Tensor:
    """
    Sample random spatial locations from each mask channel.

    Parameters
    ----------
    mask : torch.Tensor
        Mask with shape `(B, L, *spatial)`.
    nb_samples : int or Sequence[int]
        Number of samples per label channel. If a sequence, it must have one value per label.
    sample_method : {'uniform', 'weighted'}, default='uniform'
        Sampling distribution. `uniform` samples among positive voxels. `weighted` samples with
        probability proportional to mask values.
    replacement : bool, default=False
        Whether to sample with replacement.

    Returns
    -------
    torch.Tensor
        Integer locations with shape `(B, sum(nb_samples), D + 1)`. The last coordinate stores
        the label channel id.

    Examples
    --------
    >>> import torch
    >>> import neurite.nn.functional as nef
    >>> mask = torch.ones(2, 3, 4, 5)
    >>> locs = nef.sample_locations_in_mask(mask, nb_samples=2)
    >>> locs.shape
    torch.Size([2, 6, 3])
    """
    assert mask.ndim >= 3, "mask must have shape (B, L, *spatial)."
    assert sample_method in {"uniform", "weighted"}, "sample_method must be 'uniform' or 'weighted'"

    batch_size, nb_labels = mask.shape[:2]
    spatial_shape = tuple(mask.shape[2:])
    flat_mask = mask.reshape(batch_size * nb_labels, -1)

    if sample_method == "uniform":
        # TODO: Benchmark against a torch.nonzero-based implementation for uniform sampling.
        # torch.multinomial keeps the uniform and weighted paths batched and aligned, but may be
        # less efficient than sampling directly from positive indices.
        weights = (flat_mask > 0).float()
    else:
        weights = flat_mask.float()

    assert torch.all(weights.sum(dim=1) > 0), "Every mask channel must contain samples."

    if isinstance(nb_samples, int):
        samples_by_label = [nb_samples] * nb_labels
    else:
        samples_by_label = list(nb_samples)
        assert len(samples_by_label) == nb_labels, (
            "nb_samples must be an int or a sequence with one value per label channel."
        )

    assert all(count >= 0 for count in samples_by_label), "nb_samples values must be non-negative."

    if not replacement:
        available = (weights > 0).sum(dim=1).reshape(batch_size, nb_labels)
        required = torch.as_tensor(samples_by_label, device=mask.device, dtype=available.dtype)
        assert torch.all(available >= required[None, :]), (
            "Some mask channels contain fewer samples than requested."
        )

    locs = []
    for label_idx, count in enumerate(samples_by_label):
        if count == 0:
            empty_shape = (batch_size, 0, len(spatial_shape) + 1)
            locs.append(torch.empty(empty_shape, device=mask.device, dtype=torch.long))
            continue

        label_weights = weights[label_idx::nb_labels]
        flat_indices = torch.multinomial(label_weights, count, replacement=replacement)
        spatial_locs = torch.stack(torch.unravel_index(flat_indices, spatial_shape), dim=-1)
        labels = torch.full((batch_size, count, 1), label_idx, device=mask.device, dtype=torch.long)
        locs.append(torch.cat([spatial_locs.long(), labels], dim=-1))

    return torch.cat(locs, dim=1)


def sample_locations_on_border(
    mask: torch.Tensor,
    thickness: int,
    nb_samples: Union[int, Sequence[int]],
    border_mode: Literal["inner", "outer"] = "inner",
    sample_method: Literal["uniform", "weighted"] = "uniform",
    replacement: bool = False,
) -> torch.Tensor:
    """
    Sample random locations from mask borders.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask with shape `(B, L, *spatial)`.
    thickness : int
        Border thickness in voxels.
    nb_samples : int or Sequence[int]
        Number of samples per label channel. If a sequence, it must have one value per label.
    border_mode : {'inner', 'outer'}, default='inner'
        Whether to sample from the inner or outer border.
    sample_method : {'uniform', 'weighted'}, default='uniform'
        Sampling distribution passed to `sample_locations_in_mask`.
    replacement : bool, default=False
        Whether to sample with replacement.

    Returns
    -------
    torch.Tensor
        Integer locations with shape `(B, sum(nb_samples), D + 1)`.
    """
    border_mask = mask_border(mask, thickness=thickness, border_mode=border_mode)
    return sample_locations_in_mask(
        border_mask,
        nb_samples=nb_samples,
        sample_method=sample_method,
        replacement=replacement,
    )


def locs_to_mask(
    locs: torch.Tensor,
    vol_shape: Sequence[int],
    nb_labels: Union[int, None] = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create a label-channel mask from sampled locations.

    Parameters
    ----------
    locs : torch.Tensor
        Integer locations with shape `(B, N, D + 1)` or `(B, N, D)`. When the last dimension is
        `D + 1`, the final coordinate stores the label channel id.
    vol_shape : Sequence[int]
        Spatial shape of the output mask.
    nb_labels : int or None, default=None
        Number of label channels in the output. Use this to preserve trailing empty labels.
        If None, the value is inferred from `locs`.
    dtype : torch.dtype, default=torch.bool
        Output dtype.

    Returns
    -------
    torch.Tensor
        Mask with shape `(B, L, *vol_shape)`.
    """
    vol_shape = tuple(vol_shape)
    assert locs.ndim == 3, "locs must have shape (B, N, D + 1) or (B, N, D)."
    assert locs.shape[-1] in {len(vol_shape), len(vol_shape) + 1}, (
        "locs last dimension must match len(vol_shape) or len(vol_shape) + 1."
    )

    batch_size = locs.shape[0]
    locs = locs.to(dtype=torch.long)

    if locs.shape[-1] == len(vol_shape) + 1:
        label_idx = locs[..., -1]
        spatial_locs = locs[..., :-1]
        inferred_labels = int(label_idx.max().item()) + 1 if label_idx.numel() > 0 else 1
    else:
        label_idx = torch.zeros((batch_size, locs.shape[1]), device=locs.device, dtype=torch.long)
        spatial_locs = locs
        inferred_labels = 1

    if nb_labels is None:
        nb_labels = inferred_labels

    assert nb_labels >= inferred_labels, "nb_labels cannot be smaller than the largest label id."
    assert nb_labels > 0, "nb_labels must be positive."

    for dim_idx, dim_size in enumerate(vol_shape):
        coords = spatial_locs[..., dim_idx]
        assert torch.all((0 <= coords) & (coords < dim_size)), "locs contain out-of-bounds values."

    mask = torch.zeros((batch_size, nb_labels, *vol_shape), device=locs.device, dtype=dtype)
    if locs.shape[1] == 0:
        return mask

    batch_idx = torch.arange(batch_size, device=locs.device)[:, None].expand_as(label_idx)
    split_locs = spatial_locs.unbind(-1)
    mask[(batch_idx, label_idx, *split_locs)] = 1

    return mask


def extract_features_at_locs(
    feature_tensor: torch.Tensor,
    locs: torch.Tensor,
) -> torch.Tensor:
    """
    Extract feature vectors at spatial locations.

    Parameters
    ----------
    feature_tensor : torch.Tensor
        Feature tensor with shape `(B, C, *spatial)`.
    locs : torch.Tensor
        Integer spatial locations with shape `(B, N, D)`.

    Returns
    -------
    torch.Tensor
        Feature samples with shape `(B, C, N)`.
    """
    assert feature_tensor.ndim >= 3, "feature_tensor must have shape (B, C, *spatial)."
    assert locs.ndim == 3, "locs must have shape (B, N, D)."
    assert feature_tensor.shape[0] == locs.shape[0], (
        "feature_tensor and locs batch sizes must match."
    )
    assert locs.shape[-1] == feature_tensor.ndim - 2, (
        "locs last dimension must match the number of spatial dimensions."
    )

    batch_size = feature_tensor.shape[0]
    locs = locs.to(device=feature_tensor.device, dtype=torch.long)

    for dim_idx, dim_size in enumerate(feature_tensor.shape[2:]):
        coords = locs[..., dim_idx]
        assert torch.all((0 <= coords) & (coords < dim_size)), "locs contain out-of-bounds values."

    batch_idx = torch.arange(batch_size, device=feature_tensor.device)[:, None]
    batch_idx = batch_idx.expand_as(locs[..., 0])
    split_locs = locs.unbind(-1)
    features = feature_tensor[(batch_idx, slice(None), *split_locs)]

    return features.movedim(-1, 1)


def sample_features_at_mask_locs(
    feature_tensor: torch.Tensor,
    mask: torch.Tensor,
    nb_samples: Union[int, Sequence[int]],
    sample_method: Literal["uniform", "weighted"] = "uniform",
    replacement: bool = False,
    return_label_ids: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample feature vectors at random mask locations.

    Parameters
    ----------
    feature_tensor : torch.Tensor
        Feature tensor with shape `(B, C, *spatial)`.
    mask : torch.Tensor
        Mask with shape `(B, L, *spatial)`.
    nb_samples : int or Sequence[int]
        Number of locations to sample per label channel.
    sample_method : {'uniform', 'weighted'}, default='uniform'
        Sampling distribution passed to `sample_locations_in_mask`.
    replacement : bool, default=False
        Whether to sample with replacement.
    return_label_ids : bool, default=False
        If True, also return the sampled label ids.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Sampled features with shape `(B, C, sum(nb_samples))`. If `return_label_ids=True`, also
        returns label ids with shape `(B, sum(nb_samples))`.
    """
    locs = sample_locations_in_mask(
        mask,
        nb_samples=nb_samples,
        sample_method=sample_method,
        replacement=replacement,
    )
    features = extract_features_at_locs(feature_tensor, locs[..., :-1])

    if return_label_ids:
        return features, locs[..., -1]

    return features


def one_hot(
    label_tensor: torch.Tensor,
    num_classes: Union[int, None] = None,
    class_list: Union[Sequence[int], None] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert `(B, 1, *spatial)` integer labels to one-hot channels.

    Parameters
    ----------
    label_tensor : torch.Tensor
        Integer label tensor with shape `(B, 1, *spatial)`.
    num_classes : int or None, default=None
        Total number of classes. If None, it is inferred from `label_tensor.max() + 1`.
    class_list : Sequence[int] or None, default=None
        Class ids to keep in the output. If None, all classes are returned.
    dtype : torch.dtype, default=torch.float32
        Output dtype.

    Returns
    -------
    torch.Tensor
        One-hot tensor with shape `(B, C, *spatial)`.
    """
    assert label_tensor.ndim >= 3, "label_tensor must have shape (B, 1, *spatial)."
    assert label_tensor.shape[1] == 1, "label_tensor channel dimension must be singleton."

    labels = label_tensor[:, 0, ...].long()
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1

    encoded = F.one_hot(labels, num_classes=num_classes).to(dtype=dtype)
    encoded = encoded.movedim(-1, 1)

    if class_list is None:
        return encoded

    classes = torch.as_tensor(class_list, device=label_tensor.device, dtype=torch.long)
    assert classes.numel() > 0, "class_list must contain at least one class id."
    assert int(classes.max().item()) < num_classes, "class_list contains an out-of-range class id."
    assert int(classes.min().item()) >= 0, "class_list contains a negative class id."

    return encoded.index_select(1, classes)


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
    unique_labels = torch.unique(label_tensor)
    min_val, max_val = mean_range

    sampled_image = torch.zeros_like(label_tensor, dtype=torch.float32)
    uniform_dist = torch.distributions.Uniform(low=min_val, high=max_val)

    for label in unique_labels:
        mask = label_tensor == label
        num_elements = mask.sum().item()

        mean_region_intensity = uniform_dist.sample().item()
        texturized_region = mean_region_intensity + noise_std * torch.randn(num_elements)
        sampled_image[mask] = texturized_region

    return sampled_image


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
    normalized_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
    spatial_size = size[2:]

    if normalize:
        axes = []
        for sz in spatial_size:
            axes.append(torch.linspace(-1, 1, steps=sz, device=device, dtype=normalized_dtype))
    else:
        axes = []
        for sz in spatial_size:
            axes.append(torch.arange(0, sz, device=device, dtype=normalized_dtype))

    grid = torch.meshgrid(*axes, indexing=indexing)

    if stack:
        grid = torch.stack(grid, dim=0).contiguous()

    return grid


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
    dims_to_test = list(range(tensor.dim()))
    dims_to_test.remove(dim)

    nan_mask = ~torch.isnan(tensor).any(dim=dims_to_test)
    nan_mask = torch.nonzero(nan_mask, as_tuple=True)[0]
    filtered_tensor = torch.index_select(tensor, dim, nan_mask)

    inf_mask = ~torch.isinf(filtered_tensor).any(dim=dims_to_test)
    inf_mask = torch.nonzero(inf_mask, as_tuple=True)[0]
    filtered_tensor = torch.index_select(filtered_tensor, dim, inf_mask)

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

    has_zero_dim = torch.any(torch.tensor(filtered_tensor.shape) == 0)
    if has_zero_dim:
        zero_dims = []
        for d, size in enumerate(tensor.shape):
            if size == 0:
                zero_dims.append(d)
        assert not has_zero_dim, f"Dimension {zero_dims} of the filtered tensor has shape == 0."

    return filtered_tensor


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
    return torch.mean((tensor1 - tensor2) ** 2)


def dice(
    *segs: torch.Tensor,
    smooth_numerator: float = 1e-12,
    smooth_denominator: float = 1e-12,
    bounds_tolerance: float = 1e-6,
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
    bounds_tolerance : float, default=1e-6
        Absolute tolerance for segmentation values outside [0, 1].
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
    nsegs = len(segs)
    assert nsegs >= 2, 'Provide at least two segmentation tensors.'

    shapes_match = all(segs[0].shape == seg.shape for seg in segs)
    assert shapes_match, f'All segmentations must share shape; got {{seg.shape for seg in segs}}'

    for seg in segs:
        seg_min = seg.min()
        seg_max = seg.max()
        assert seg_min >= -bounds_tolerance, f'Minimum {seg_min} is too low.'
        assert seg_max <= 1 + bounds_tolerance, f'Maximum {seg_max} is too high.'

    stacked = torch.stack(segs, dim=0)
    spatial_dims = tuple(range(2, segs[0].ndim))
    intersection = stacked.prod(dim=0).sum(dim=spatial_dims)
    union_dims = (0,) + tuple(range(3, stacked.ndim))
    union = (stacked ** 2).sum(dim=union_dims)
    dice_score = (nsegs * intersection + smooth_numerator) / (union + smooth_denominator)

    if reduction is None:
        return dice_score

    return reduce(tensor=dice_score, reduction=reduction, dim=reduction_dim, keepdims=keepdims)


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
        Size of local window for computing correlation. If int, the same size is used
        for all spatial dimensions. If Sequence, specifies one size per dimension.
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
        Squared local correlation coefficients. If ``reduction=None``, returns shape
        ``(B, C)``. Otherwise, the requested reduction is applied.

    Examples
    --------
    >>> tensor1 = torch.rand(2, 3, 64, 64)
    >>> tensor2 = torch.rand(2, 3, 64, 64)
    >>> score = ncc(tensor1, tensor2)
    >>> score.shape
    torch.Size([1, 1])

    Notes
    -----
    The five local moment images are packed along the channel dimension and filtered
    together with :func:`box_filter`. The covariance and variances use their reduced
    algebraic forms:

    ``cross = sum(I * J) - sum(I) * sum(J) / N``

    ``variance(I) = sum(I ** 2) - sum(I) ** 2 / N``

    where ``N`` is the number of elements in the local window.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")

    num_spatial = tensor1.ndim - 2
    if num_spatial not in [1, 2, 3]:
        raise ValueError(f"Only 1-3D dimensions supported. Got {num_spatial}D")

    if isinstance(window_size, int):
        win = [window_size] * num_spatial
    else:
        win = list(window_size)
        if len(win) != num_spatial:
            raise ValueError(f'window_size len {len(win)} does not match dims {num_spatial}')

    if not all(size > 0 for size in win):
        raise ValueError(f'window_size values must be positive. Got {win}')

    # Flatten batch and channel so the five moments can be filtered as one tensor.
    batch_shape = tensor1.shape[:2]
    spatial_shape = tensor1.shape[2:]
    tensor1 = tensor1.reshape(-1, 1, *spatial_shape)  # [B * C, 1, *V]
    tensor2 = tensor2.reshape(-1, 1, *spatial_shape)  # [B * C, 1, *V]

    tensor1_squared = tensor1.square()
    tensor2_squared = tensor2.square()
    tensor_product = tensor1 * tensor2
    moments_list = (tensor1, tensor2, tensor1_squared, tensor2_squared, tensor_product)
    moments = torch.cat(moments_list, dim=1)  # [B * C, 5, *V]

    # Filter all moments together, then recover their individual local sums.
    moment_sums = box_filter(moments, window_size=win)  # [B * C, 5, *V]
    I_sum, J_sum, I2_sum, J2_sum, IJ_sum = moment_sums.chunk(5, dim=1)

    # Use the reduced covariance and variance formulas to avoid redundant volume operations.
    inv_win_size = 1 / math.prod(win)
    cross = IJ_sum - I_sum * J_sum * inv_win_size
    I_var = I2_sum - I_sum.square() * inv_win_size
    J_var = J2_sum - J_sum.square() * inv_win_size

    cc = cross.square() / (I_var * J_var + eps)
    spatial_dims = tuple(range(2, 2 + num_spatial))
    ncc_score = cc.mean(dim=spatial_dims).reshape(*batch_shape)

    if reduction is not None:
        ncc_score = reduce(ncc_score, reduction, dim=reduction_dim, keepdims=keepdims)

    return ncc_score


def spatial_gradients(input_tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Compute first-order forward differences over all spatial dimensions.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor with shape (B, C, *spatial).

    Returns
    -------
    tuple[torch.Tensor, ...]
        Gradient tensor for each spatial dimension. Each tensor is one element shorter along
        the dimension over which its difference was computed.
    """
    num_spatial = input_tensor.ndim - 2
    assert num_spatial >= 1, f"Need at least 1 spatial dim to compute gradients, got {num_spatial}"
    gradients = []
    for spatial_dim in range(num_spatial):
        gradients.append(torch.diff(input_tensor, dim=2 + spatial_dim))
    return tuple(gradients)


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
    assert penalty in ['l1', 'l2'], f"penalty must be 'l1' or 'l2', got '{penalty}'"

    grads = spatial_gradients(input_tensor)

    if penalty == 'l1':
        penalties = [g.abs() for g in grads]
    else:
        penalties = [g * g for g in grads]

    if reduction is None:
        return penalties

    reduced = []
    for penalty_tensor in penalties:
        reduced.append(
            reduce(tensor=penalty_tensor, reduction=reduction, dim=reduction_dim, keepdims=keepdims)
        )

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
    if reduction is None:
        return tensor

    torch_multidim_reductions = [
        'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'
    ]
    torch_singledim_reductions = ['argmin', 'argmax']

    if reduction in torch_multidim_reductions:
        return getattr(torch, reduction)(tensor, dim=dim, keepdim=keepdims)

    if reduction in torch_singledim_reductions:
        assert isinstance(dim, int), (
            f"Reduction type {reduction} is only compatible with one reduction dimension. Got "
            f"{dim}"
        )
        return getattr(torch, reduction)(tensor, dim=dim, keepdim=keepdims)

    assert reduction in [*torch_multidim_reductions, *torch_singledim_reductions], (
        f"reduce received an invalid `reduction`. Got {reduction}. Valid options"
        " are {'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean', 'argmin', "
        "'argmax'}"
    )
    return tensor


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
    if bernoulli(prob).item():
        result = tuple(arg.flip([dim]) for arg in args)
    else:
        result = args

    if len(result) == 1:
        return result[0]
    return result


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

        linear = infer_linear_interpolation_mode(image.ndim - 1)
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
    indexing: Literal["ij", "xy"] = "ij",
    device: Union[str, torch.device] = "cpu",
    dtype: Union[str, torch.dtype] = torch.float32,
) -> torch.Tensor:
    """
    Draw a black and white grid with white lines on a black background.

    Parameters
    ----------
    vol_shape : Sequence[int]
        Shape of the output tensor.
    spacing : int or Sequence[int]
        Legacy grid spacing. Line starts repeat every `spacing + 1` pixels. If an int, the same
        spacing is used for every dimension. If a sequence, it must have one value per dimension in
        `vol_shape`.
    thickness : int, default=1
        Line thickness in pixels.
    indexing : {'ij', 'xy'}, default='ij'
        Cartesian (`xy`) or matrix (`ij`) indexing mode passed to `torch.meshgrid`.
    device : str or torch.device, default='cpu'
        Device on which to create the grid.
    dtype : str or torch.dtype, default=torch.float32
        Data type of the output tensor.

    Returns
    -------
    torch.Tensor
        A tensor of shape `vol_shape` with white lines (value 1) on a black background (value 0).

    Examples
    --------
    >>> import neurite.nn.functional as nef
    >>> grid = nef.bw_grid((1, 1, 5, 5), spacing=1)
    >>> grid.shape
    torch.Size([1, 1, 5, 5])
    """
    shape_values = list(vol_shape)
    assert len(shape_values) > 0, "vol_shape must contain at least one dimension."
    assert all(isinstance(value, int) and value > 0 for value in shape_values), (
        f"vol_shape must contain positive integers, got {vol_shape}."
    )

    if isinstance(spacing, int):
        spacing = [spacing] * len(shape_values)
    else:
        spacing = list(spacing)

    assert len(spacing) == len(shape_values), "spacing and vol_shape must have the same length."
    assert all(isinstance(value, int) and value > 0 for value in spacing), (
        f"spacing must contain positive integers, got {spacing}."
    )
    assert isinstance(thickness, int) and thickness > 0, (
        f"thickness must be a positive integer, got {thickness}."
    )
    assert indexing in ("ij", "xy"), f"indexing must be 'ij' or 'xy', got {indexing}."

    normalized_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
    grid_image = torch.zeros(shape_values, device=device, dtype=normalized_dtype)

    for dim, size in enumerate(shape_values):
        ranges = []
        for axis_size in shape_values:
            ranges.append(torch.arange(0, axis_size, device=device, dtype=torch.long))

        for offset_value in range(thickness):
            line_coords = torch.arange(
                offset_value, size, spacing[dim] + 1, device=device, dtype=torch.long
            )
            last_coord = torch.tensor([size - 1], device=device, dtype=torch.long)
            ranges[dim] = torch.unique(torch.cat([line_coords, last_coord]))
            grid_image[torch.meshgrid(*ranges, indexing=indexing)] = 1

    return grid_image


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
    AssertionError
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
    assert size is not None or scale_factor is not None, (
        "Either size or scale_factor must be specified"
    )
    assert size is None or scale_factor is None, "size and scale_factor are mutually exclusive"

    num_spatial = input_tensor.dim() - 2
    spatial_dims = list(range(2, input_tensor.dim()))

    if size is not None:
        if isinstance(size, int):
            crop_sizes = [size] * num_spatial
        else:
            assert len(size) == num_spatial, (
                f"size length {len(size)} doesn't match number of spatial dims {num_spatial}"
            )
            crop_sizes = list(size)
    else:
        if isinstance(scale_factor, (int, float)):
            scale_factors = [scale_factor] * num_spatial
        else:
            assert len(scale_factor) == num_spatial, (
                f"scale_factor length {len(scale_factor)} doesn't match spatial"
                f"dims {num_spatial}"
            )
            scale_factors = list(scale_factor)

        crop_sizes = []
        for dim_idx, dim in enumerate(spatial_dims):
            input_size = input_tensor.shape[dim]
            crop_sizes.append(round(input_size * scale_factors[dim_idx]))

    for dim_idx, dim in enumerate(spatial_dims):
        input_size = input_tensor.shape[dim]
        assert crop_sizes[dim_idx] <= input_size, (
            f"Crop size {crop_sizes[dim_idx]} exceeds input size {input_size} at dim {dim}"
        )

    if isinstance(offset, int):
        offsets = [offset] * num_spatial
    else:
        assert len(offset) == num_spatial, (
            f"offset length {len(offset)} doesn't match number of spatial dims {num_spatial}"
        )
        offsets = list(offset)

    for dim_idx, dim in enumerate(spatial_dims):
        input_size = input_tensor.shape[dim]
        crop_size = crop_sizes[dim_idx]
        max_valid_offset = input_size - crop_size
        assert 0 <= offsets[dim_idx] <= max_valid_offset, (
            f"offset {offsets[dim_idx]} out of range [0, {max_valid_offset}] for dim {dim}"
        )

    slices = [slice(None)] * input_tensor.dim()
    for dim_idx, dim in enumerate(spatial_dims):
        crop_size = crop_sizes[dim_idx]
        dim_offset = offsets[dim_idx]
        slices[dim] = slice(dim_offset, dim_offset + crop_size)

    return input_tensor[tuple(slices)]


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
    return torch.clamp(input_tensor, min=min, max=max)


def random_smoothed_noise(
    shape: Sequence[int],
    sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
    magnitude: float = 1.0,
    normalize: Union[Literal["sum", "gaussian"], None] = "sum",
    device: Union[torch.device, None] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
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
    dtype : torch.dtype, default=torch.float32
        Data type for the noise field.
    generator : torch.Generator, optional
        Generator controlling random sampling.

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
    noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    noise = gaussian_smoothing(noise, sigma=sigma, truncate=3, normalize=normalize)

    # Normalize every batch-channel field independently over its spatial axes.
    spatial_dims = tuple(range(2, noise.ndim))
    noise -= noise.mean(dim=spatial_dims, keepdim=True)
    std = noise.std(dim=spatial_dims, keepdim=True).clamp_min(1e-8)
    noise *= magnitude / std

    return noise


def upsample_noise(
    shape: Sequence[int],
    scale: Union[float, int, Sequence[Union[float, int]]],
    device: Union[torch.device, None] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Generate smooth noise by upsampling from a coarse grid in (B, C, *spatial) format.

    Parameters
    ----------
    shape : Sequence[int]
        Target shape in (B, C, *spatial) format. Must have at least 3 dimensions.
    scale : float, int, or Sequence[float or int]
        Downsampling factor per spatial axis. Larger values produce smoother noise.
    device : torch.device or None, default=None
        Device for tensor allocation.
    dtype : torch.dtype, default=torch.float32
        Data type for the noise field.
    generator : torch.Generator, optional
        Generator controlling random sampling.

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
    spatial_shape = shape[2:]
    if isinstance(scale, (float, int)):
        scales = (float(scale),) * len(spatial_shape)
    else:
        scales = tuple(float(value) for value in scale)
    if len(scales) != len(spatial_shape):
        raise ValueError("scale must have one value per spatial dimension")
    if any(value <= 0 for value in scales):
        raise ValueError("scale values must be positive")

    coarse_spatial = []
    for size, axis_scale in zip(spatial_shape, scales):
        coarse_spatial.append(max(int(size // axis_scale), 2))
    coarse_shape = (*shape[:2], *coarse_spatial)
    noise = torch.randn(coarse_shape, device=device, dtype=dtype, generator=generator)

    mode = infer_linear_interpolation_mode(num_spatial=len(spatial_shape))
    return F.interpolate(noise, size=spatial_shape, mode=mode, align_corners=False)


def fractal_noise(
    shape: Sequence[int],
    scales: Union[
        float,
        int,
        Sequence[Union[float, int, Sequence[Union[float, int]]]],
        None,
    ] = None,
    magnitude: float = 1.0,
    weights: Union[Sequence[float], None] = None,
    normalize: Union[Literal["sum", "gaussian"], None] = "sum",
    device: Union[torch.device, None] = None,
    method: Literal['blur', 'upsample'] = 'blur',
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    standardize: bool = True,
) -> torch.Tensor:
    """
    Generate fractal noise in (B, C, *spatial) format.

    Parameters
    ----------
    shape : Sequence[int]
        Target shape in (B, C, *spatial) format. Must have at least 3 dimensions.
    scales : float, int, or sequence, optional
        Smoothing scale for each octave. An octave can contain one value per spatial axis.
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
    dtype : torch.dtype, default=torch.float32
        Data type for the noise field.
    generator : torch.Generator, optional
        Generator controlling random sampling.
    standardize : bool, default=True
        Whether to standardize each output field to zero mean and ``magnitude`` deviation.

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
    spatial_shape = shape[2:]

    if scales is None:
        scales = 2 ** np.arange(np.log2(max(spatial_shape)))[1:]

    if np.isscalar(scales):
        scales = [scales]
    else:
        scales = list(scales)

    if len(scales) == 1:
        weights = [1.0]
    elif weights is None:
        weights = list(np.arange(len(scales)) + 1)

    if len(weights) != len(scales):
        raise ValueError("weights length must match scales length")
    if method not in {"blur", "upsample"}:
        raise ValueError("method must be 'blur' or 'upsample'")

    noise = None
    for scale, weight in zip(scales, weights):
        if method == 'blur':
            sample = random_smoothed_noise(
                shape=shape,
                sigma=scale,
                magnitude=1.0,
                normalize=normalize,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        else:
            sample = upsample_noise(
                shape=shape,
                scale=scale,
                device=device,
                dtype=dtype,
                generator=generator,
            )

        sample *= weight

        if noise is None:
            noise = sample
        else:
            noise += sample

    if standardize:
        # Normalize every batch-channel field independently over its spatial axes.
        spatial_dims = tuple(range(2, noise.ndim))
        noise -= noise.mean(dim=spatial_dims, keepdim=True)
        std = noise.std(dim=spatial_dims, keepdim=True).clamp_min(1e-8)
        noise *= magnitude / std
    else:
        noise *= magnitude

    return noise
