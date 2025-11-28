"""
Single tensor operations (no B, C dimension assumption)
"""
# Standard library imports
from typing import Union, Sequence, Tuple, Literal, Optional

# Third party imports
import torch
import torch.nn.functional as F

# Custom imports
import neurite as ne

__all__ = [
    "mse",
    "dice",
    "ncc",
    "spatial_gradient",
    "reduce",
    "volshape_to_ndgrid",
    "apply_bernoulli_mask",
    "random_flip",
    "sample_image_from_labels",
    "resample",
    "filter_dim",
    "gaussian_kernel",
    "crop",
    "clip",
    "pad_for_vectorization",
    "unpad_from_vectorization",
    "smooth_gaussian",
    "upsample_noise",
    "fractal_noise",
]


def soft_quantize(
    input_tensor: torch.Tensor,
    nb_bins: int = 16,
    softness: Union[float, int] = 1.0,
    min_clip: Union[float, int] = -float('inf'),
    max_clip: Union[float, int] = float('inf'),
    return_log: bool = False
) -> torch.Tensor:
    """
    Deprecated: This function has been moved to the sandbox.

    The soft_quantize function has been deprecated and removed from the main neurite library.
    The implementation has been preserved in the sandbox at:
    neurite-sandbox/neurite_sandbox/etienne_chollet/notebooks/deprecated_from_dev.py
    """
    raise DeprecationWarning(
        "soft_quantize() has been deprecated and removed from neurite.functional. "
        "The implementation is available in the sandbox at "
        "neurite-sandbox/neurite_sandbox/etienne_chollet/notebooks/deprecated_from_dev.py"
    )


def mse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean squared error (MSE) between the elements of `tensor1` and `tensor2`.

    Parameters
    ----------
    tensor1 : torch.Tensor
        An input tensor of any shape.
    tensor2 : torch.Tensor
        A tensor with the same shape as `tensor1`.

    Returns
    -------
    torch.Tensor
        The mean squared error between `tensor1` and `tensor2`.

    Examples
    --------
    >>> import torch
    # First tensor with zero mean, unit variance
    >>> tensor1 = torch.randn(16, 16, 16)
    # Other tensor with zero mean, unit variance, and same shape as `tensor1`
    >>> tensor2 = torch.randn(16, 16, 16)
    # Calculate mse
    >>> mse_value = mse(tensor1, tensor2)
    # Print `mse_value` (should be approximately 2.0)
    >>> print(mse_value)
    """

    return torch.mean((tensor1 - tensor2) ** 2)


def dice(
    *segs: torch.Tensor,
    smooth_numerator: float = 1e-12,
    smooth_denominator: float = 1e-12,
    non_spatial_dims: Union[Tuple[int, ...], None] = None
) -> torch.Tensor:
    """
    Compute Dice score over multiple segmentation maps.

    Shape-agnostic implementation that can either compute a global Dice score
    (when non_spatial_dims=None) or preserve batch/channel structure.

    Parameters
    ----------
    *segs : torch.Tensor
        Two or more segmentation tensors with the same shape and values in [0, 1].
    smooth_numerator : float, optional
        Smoothing constant added to the numerator. Default is 1e-12.
    smooth_denominator : float, optional
        Smoothing constant added to the denominator. Default is 1e-12.
    non_spatial_dims : Tuple[int, ...] or None, optional
        Indices of non-spatial dimensions. Must be a contiguous sequence starting from 0.
        Valid values: `None`, `(0,)`, or `(0, 1)`. If None, assumes all dimensions are spatial
        and computes a single scalar Dice score. Default is None.

    Returns
    -------
    torch.Tensor
        Dice score. Shape depends on non_spatial_dims:
        - If None: scalar
        - If (0,): shape (B,)
        - If (0, 1): shape (B, C)

    Examples
    --------
    >>> import torch
    # Compute global dice for 2 segmentation tensors
    >>> seg1 = torch.rand(32, 32)
    >>> seg2 = torch.rand(32, 32)
    >>> score = dice(seg1, seg2)
    >>> print(score.shape)
    torch.Size([])

    # Compute per-batch dice
    >>> seg1 = torch.rand(4, 64, 64)
    >>> seg2 = torch.rand(4, 64, 64)
    >>> score = dice(seg1, seg2, non_spatial_dims=(0,))
    >>> print(score.shape)
    torch.Size([4])

    # Compute per-batch-and-channel dice
    >>> seg1 = torch.rand(2, 3, 64, 64)
    >>> seg2 = torch.rand(2, 3, 64, 64)
    >>> score = dice(seg1, seg2, non_spatial_dims=(0, 1))
    >>> print(score.shape)
    torch.Size([2, 3])
    """
    nsegs = len(segs)

    if nsegs < 2:
        raise ValueError('Provide at least two segmentation tensors.')

    if not all(segs[0].shape == seg.shape for seg in segs):
        shapes = {seg.shape for seg in segs}
        raise ValueError(f'All segmentations must share shape; got {shapes}')

    for seg in segs:
        if seg.min() < 0 or seg.max() > 1:
            raise AssertionError(
                f'Segmentations must be in [0,1]; got min {seg.min()}, max {seg.max()}'
            )

    # Parse and validate non_spatial_dims (handles None by setting num_non_spatial=0)
    num_non_spatial, num_spatial = _parse_non_spatial_dims(non_spatial_dims, segs[0].ndim)

    # Flatten spatial dimensions (when num_non_spatial=0, this flattens all dims)
    segs_flat = [seg.flatten(num_non_spatial) for seg in segs]

    # Stack segmentations: (nsegs, *non_spatial_dims, spatial_flat)
    stacked = torch.stack(segs_flat, dim=0)

    # Intersection: product across segs, sum over spatial
    intersection = stacked.prod(dim=0).sum(dim=-1)

    # Union: sum across segs and spatial
    union = stacked.sum(dim=(0, -1))

    # Dice for N tensors: N * intersection / union
    dice_score = (nsegs * intersection + smooth_numerator) / (union + smooth_denominator)

    return dice_score


def ncc(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    window_size: Union[int, Sequence[int]] = 9,
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute local normalized cross-correlation (NCC) between two tensors.

    Parameters
    ----------
    tensor1 : torch.Tensor
        First input tensor.
    tensor2 : torch.Tensor
        Second input tensor with same shape as tensor1.
    window_size : int or Sequence[int], default=9
        Size of local window for computing correlation. If int, same size for all
        spatial dimensions. If Sequence, per-dimension window sizes (must match
        number of spatial dimensions).
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Indices of non-spatial dimensions. Must be a contiguous sequence starting from 0.
        Valid values: None, (0,), or (0, 1). If None, assumes all dimensions are spatial
        and computes a single scalar NCC score.
    eps : float, default=1e-5
        Small constant for numerical stability in division.

    Returns
    -------
    torch.Tensor
        NCC values (squared correlation coefficients) in range [0, 1].
        Shape depends on non_spatial_dims:
        - If None: scalar
        - If (0,): shape (N,) where N is size of dim 0
        - If (0, 1): shape (N, M) where N, M are sizes of dims 0, 1

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    # Compute global NCC for two 2D tensors
    >>> t1 = torch.rand(64, 64)
    >>> t2 = torch.rand(64, 64)
    >>> score = ne.ncc(t1, t2)
    >>> print(score.shape)
    torch.Size([])

    # Compute per-batch NCC
    >>> t1 = torch.rand(4, 64, 64)
    >>> t2 = torch.rand(4, 64, 64)
    >>> score = ne.ncc(t1, t2, non_spatial_dims=(0,))
    >>> print(score.shape)
    torch.Size([4])

    # Compute per-batch-and-channel NCC with custom window
    >>> t1 = torch.rand(2, 3, 64, 64)
    >>> t2 = torch.rand(2, 3, 64, 64)
    >>> score = ne.ncc(t1, t2, window_size=5, non_spatial_dims=(0, 1))
    >>> print(score.shape)
    torch.Size([2, 3])

    Notes
    -----
    The NCC is computed as the squared Pearson correlation coefficient:
        NCC = (cov(I, J))^2 / (var(I) * var(J))

    where covariance and variance are computed locally over the specified window.
    Values close to 1 indicate high similarity, values close to 0 indicate low similarity.

    References
    ----------
    .. [1] Balakrishnan et al., "VoxelMorph: A Learning Framework for Deformable
           Medical Image Registration", IEEE TMI, 2019.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Tensors must have same shape. Got {tensor1.shape} and {tensor2.shape}"
        )

    # Parse non_spatial_dims
    num_non_spatial, num_spatial = _parse_non_spatial_dims(non_spatial_dims, tensor1.ndim)

    if num_spatial not in [1, 2, 3]:
        raise ValueError(
            f"Only 1D, 2D, 3D spatial dimensions supported. Got {num_spatial}D"
        )

    tensor1, orig_shape = pad_for_vectorization(tensor1, non_spatial_dims)
    tensor2, _ = pad_for_vectorization(tensor2, non_spatial_dims)

    # Parse window size
    if isinstance(window_size, int):
        win = [window_size] * num_spatial
    else:
        win = list(window_size)
        if len(win) != num_spatial:
            raise ValueError(
                f"window_size length {len(win)} doesn't match spatial dims {num_spatial}"
            )

    # Create sum filter: (1, 1, *win) - single channel since we use vectorization
    sum_filt = torch.ones(1, 1, *win, device=tensor1.device, dtype=tensor1.dtype)

    # Convolution parameters for "same" output size
    padding = [w // 2 for w in win]
    stride = [1] * num_spatial

    # Select conv function based on spatial dimensionality
    conv_fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[num_spatial]

    # Compute products
    Ii = tensor1
    Ji = tensor2
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji

    # Local sums using convolution (each batch element independent)
    I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    # Window size for normalization
    win_size = torch.tensor(win, device=tensor1.device, dtype=tensor1.dtype).prod()

    # Local means
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    # Cross-correlation: cov(I, J) * win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size

    # Variances: var(I) * win_size, var(J) * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    # Squared correlation coefficient
    cc = cross * cross / (I_var * J_var + eps)

    # Average over spatial dimensions (keep non-spatial structure)
    spatial_dims = tuple(range(2, 2 + num_spatial))
    ncc_score = cc.mean(dim=spatial_dims)

    # Remove added dimensions to restore original non-spatial structure
    ncc_score = unpad_from_vectorization(ncc_score, orig_shape)

    return ncc_score


def spatial_gradient(
    input_tensor: torch.Tensor,
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
) -> list[torch.Tensor]:
    """
    Compute spatial gradients using finite differences along each spatial dimension.

    Shape-agnostic implementation that computes first-order forward differences
    along each spatial dimension.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor (e.g., displacement field, velocity field, or image).
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Indices of non-spatial dimensions. Must be a contiguous sequence starting from 0.
        Valid values: None, (0,), or (0, 1). If None, assumes all dimensions are spatial.

    Returns
    -------
    list[torch.Tensor]
        List of gradient tensors, one per spatial dimension. Each tensor has shape
        reduced by 1 along the corresponding dimension (due to finite differences).

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    # Compute gradients of a 2D image (no batch/channel)
    >>> img = torch.rand(64, 64)
    >>> grads = ne.spatial_gradient(img, non_spatial_dims=None)
    >>> len(grads)
    2
    >>> grads[0].shape  # gradient along dim 0
    torch.Size([63, 64])
    >>> grads[1].shape  # gradient along dim 1
    torch.Size([64, 63])

    # Compute gradients with batch and channel dims
    >>> field = torch.rand(2, 3, 64, 64, 64)  # (B, C, D, H, W)
    >>> grads = ne.spatial_gradient(field, non_spatial_dims=(0, 1))
    >>> len(grads)
    3
    >>> grads[0].shape  # gradient along D
    torch.Size([2, 3, 63, 64, 64])

    References
    ----------
    .. [1] Balakrishnan et al., "VoxelMorph: A Learning Framework for Deformable
           Medical Image Registration", IEEE TMI, 2019.
    """
    # Parse non_spatial_dims
    num_non_spatial, num_spatial = _parse_non_spatial_dims(non_spatial_dims, input_tensor.ndim)

    if num_spatial < 1:
        raise ValueError("Need at least 1 spatial dimension to compute gradients")

    return [torch.diff(input_tensor, dim=num_non_spatial + i) for i in range(num_spatial)]


def reduce(
    tensor: torch.Tensor,
    reduction: str = 'mean',
    dim: Union[int, tuple[int, ...], None] = None,
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
        The input tensor of any shape to reduce.
    reduction : str, optional
        The type of reduction to apply. Supported values for multidimensional reductions are:
        None, 'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'; for single
        dimension reductions: 'argmin', 'argmax', and all multidimensionals. Default is 'mean'.
    dim : int, tuple of ints, or None, optional
        Dimension(s) over which to apply the reduction. For multidimensional reductions, pass a
        tuple of dimensions; for single-dimension reductions, pass an integer. If None, reduces
        over all dimensions. Default is None.
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
    >>> import torch
    # Make a random tensor
    >>> input_tensor = torch.randn(128, 128)
    # Getting the mean over all dimensions
    >>> reduce(input_tensor, reduction='mean')
    tensor(-0.0021)
    # Getting the largest value
    >>> reduce(input_tensor, reduction='amax')
    tensor(4.1831)
    """

    # Handle None reduction (return tensor unchanged)
    if reduction is None:
        return tensor

    # PyTorch multidimensional reductions (also work for single dimensions)
    torch_multidim_reductions = [
        'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'
    ]

    # PyTorch single-dimension-only reductions
    torch_singledim_reductions = ['argmin', 'argmax']

    if reduction in torch_multidim_reductions:
        return getattr(torch, reduction)(tensor, dim=dim, keepdim=keepdims)

    elif reduction in torch_singledim_reductions:

        assert isinstance(dim, int), (
            f"Reduction type {reduction} is only compatible with one reduction dimension. Got "
            f"{dim}"
        )

        return getattr(torch, reduction)(tensor, dim=dim, keepdim=keepdims)

    else:
        raise ValueError(
            f"reduce received an invalid `reduction`. Got {reduction}. Valid options"
            " are {'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean', 'argmin', "
            "'argmax'}"
        )


def volshape_to_ndgrid(
    size: Sequence[int],
    device: Union[str, torch.device] = "cpu",
    dtype: Union[str, torch.dtype] = torch.float32,
    normalize: bool = False,
    indexing: Literal["ij", "xy"] = "ij",
    stack: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Generate a grid of spatial coordinates.

    Define the coordinate axes by generating vectors for each spatial dimension represented by the
    elements of `shape`, then creates a grid representing all spatial coords.

    Parameters
    ----------
    size : Tuple[int]
        Size of the spatial dimensions. e.g. (H, W) or (D, W, H)
    device : Union[str, torch.device], optional
        The device on which the grid will reside. By default "cpu"
    dtype : Union[str, torch.dtype], optional
        The data type of the tensor grid, by default `torch.float32`
    indexing : Literal["ij", "xy"], optional
        Indexing mode passed to `torch.meshgrid`. Defaults to `"ij"`.
    normalize : bool, optional
        Normalize each dimension of the grid to the range [-1, 1].
        Otherwise, the grid coords span from 0 to `size[i] - 1` for each dimension.
        Default is False
    stack : bool, optional
        If True, stack the grid tensors along the first dimension to return a single tensor of
        shape `(len(size), *size)`. If False, return a tuple of tensors, each of shape
        `(*size)`. Default is False.

    Returns
    -------
    torch.Tensor
        The meshgrid of spatial coordinates
        if stack=False, a tuple of len(size) tensors of shape `*size`
        if stack=True, a tensor of shape `(len(size), *size)` i.e. `(ndim, *spatial)`

    Examples
    --------
    >>> import torch
    # Make a 2d grid of size (19, 32)
    >>> the_grid = volshape_to_ndgrid(size=(19, 32))
    >>> print(the_grid[0].shape)
    torch.Size([19, 32])

    >>> # Normalized grid
    >>> the_grid = volshape_to_ndgrid(size=(3, 2), normalize=True)
    >>> print(the_grid[0])
    tensor([[-1., -1.],
            [ 0.,  0.],
            [ 1.,  1.]])
    >>> # Stacked grid (channels-first: ndim, *spatial)
    >>> the_grid = volshape_to_ndgrid(size=(19, 32), stack=True)
    >>> print(the_grid.shape)
    torch.Size([2, 19, 32])
    """
    normalized_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)

    if normalize:
        axes = [
            torch.linspace(-1, 1, steps=sz, device=device, dtype=normalized_dtype) for sz in size]
    else:
        axes = [torch.arange(0, sz, device=device, dtype=normalized_dtype) for sz in size]

    grid = torch.meshgrid(*axes, indexing=indexing)

    if stack:
        grid = torch.stack(grid, dim=0).contiguous()

    return grid


def subsample(
    input_tensor: torch.Tensor,
    stride: Union[Sequence[int], int, None] = 2,
    subsampling_dimension: Union[list, int, None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = None
) -> torch.Tensor:
    """
    Deprecated: This function has been removed in favor of the `ne.functional.resample()` API.

    The `resample()` function provides a more comprehensive and flexible resampling interface
    with support for multiple interpolation modes, and antialiasing


    """
    raise DeprecationWarning(
        "subsample() has been removed in favor of the resample() API. "
        "Please use resample() for all downsampling/upsampling operations."
    )


def apply_bernoulli_mask(
    input_tensor,
    p: Union[float, int] = 0.5,
    returns: Union[str, None] = None
) -> torch.Tensor:
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
    input_tensor = torch.ones((32, 32, 32))

    # Mask the tensor.
    masked_tensor = apply_bernoulli_mask(input_tensor, p=0.9)

    # Return the average value of the tensor of ones, approximating the
    # expectation of the mask in this special case.
    masked_tensor.mean()
    ```

    #### Return successes only (as a flattened tensor representing elements from successful trials)
    ```python
    # Define input tensor.
    input_tensor = torch.ones((32, 32, 32))

    # Get masked tensor
    masked_tensor = apply_bernoulli_mask(input_tensor, p=0.9, returns='successes')

    # Compute original shape and masked shape
    original_shape, masked_shape = input_tensor.flatten().shape[0], masked_tensor.shape[0]

    # Compute difference in size as a percent. Should be ~= `p`
    print((masked_shape/original_shape))
    ```
    """
    # Sample the Bernoulli mask with parameter `p`
    bernoulli_mask = ne.utils.bernoulli(p=p, shape=input_tensor.shape)
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


def random_flip(dim: int, *args, prob: float = 0.5):
    """
    Randomly flip tensor(s) along the given dimension.

    Parameters
    ----------
    dim : int
        The dimension along which to flip (0-indexed).
    *args : torch.Tensor
        The tensor(s) to flip.
    prob : float
        The probability of flipping the tensor(s). By default 0.5.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor]
        The flipped tensor(s).

    Examples
    --------
    >>> import torch
    # Single tensor
    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> flipped = random_flip(dim=1, x, prob=1.0)
    >>> print(flipped)
    tensor([[3, 2, 1],
            [6, 5, 4]])

    # Multiple tensors
    >>> x = torch.tensor([[1, 2], [3, 4]])
    >>> y = torch.tensor([[5, 6], [7, 8]])
    >>> flipped_x, flipped_y = random_flip(dim=0, x, y, prob=1.0)
    """
    result = tuple([arg.flip([dim]) for arg in args]) if ne.utils.bernoulli(prob) else args
    if len(args) == 1:
        return result[0]
    return result


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
        A tensor containing integer labels defining distinct regions.
    mean_range : Tuple[float, float], default=(0.0, 1.0)
        Range (min, max) for sampling mean intensity for each region. Mean intensities are
        sampled uniformly from this range.
    noise_std : float, default=0.5
        Standard deviation of the Gaussian noise added to each region. The square root of
        the variance parameter.

    Returns
    -------
    torch.Tensor
        A tensor of sampled image intensities with the same shape as `label_tensor`.

    Examples
    --------
    >>> label_map = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1]]])
    >>> sampled = sample_image_from_labels(label_map, mean_range=(0.3, 0.7), noise_std=0.1)
    >>> sampled.shape
    torch.Size([1, 4, 4])
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


def resample(
    input_tensor: torch.Tensor,
    size: Union[Sequence[int], None] = None,
    scale_factor: Union[int, float, Sequence[Union[int, float]], None] = None,
    mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
    antialias: bool = False
) -> torch.Tensor:
    """
    Resample a tensor to a given size or scale factor.

    Shape-agnostic resampling that wraps F.interpolate. Handles both upsampling (scale > 1)
    and downsampling (scale < 1). The `non_spatial_dims` parameter specifies which leading
    dimensions are non-spatial (e.g., batch and channel).

    Parameters
    ----------
    input_tensor : torch.Tensor
        The input tensor to be resampled.
    size : Sequence[int] or None, default=None
        Target size for the spatial dimensions. If None, `scale_factor` must be specified.
    scale_factor : int, float, Sequence[int], Sequence[float], or None, default=None
        The factor by which to resample each spatial dimension. If None, `size` must be specified.
    mode : {'linear', 'nearest', 'bicubic', 'area', 'nearest-exact'}, default='linear'
        Interpolation mode for resampling. 'linear' will be automatically converted to the
        appropriate mode ('linear', 'bilinear', or 'trilinear') based on spatial dimensionality.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Indices of non-spatial dimensions. Must be a contiguous sequence starting from 0.
        Valid values: `()`, `(0,)`, or `(0, 1)`. If None, assumes all dimensions are spatial
        and will add 2 leading dimensions for batch and channel.
    antialias : bool, default=False
        If True, apply antialiasing when downsampling. Only supported with 'bilinear' and
        'bicubic' modes.

    Returns
    -------
    torch.Tensor
        The resampled tensor with the same number of dimensions as the input.

    Examples
    --------
    # Downsample a 3D tensor by factor of 2
    >>> tensor_3d = torch.randn(64, 64, 64)
    >>> downsampled = resample(tensor_3d, scale_factor=0.5)
    >>> print(downsampled.shape)
    torch.Size([32, 32, 32])

    # Upsample tensor with batch and channel dims
    >>> tensor_with_bc = torch.randn(2, 3, 32, 32)
    >>> upsampled = resample(tensor_with_bc, scale_factor=2, non_spatial_dims=(0, 1))
    >>> print(upsampled.shape)
    torch.Size([2, 3, 64, 64])

    # Resample to specific size with antialiasing
    >>> tensor_2d = torch.randn(10, 100, 100)
    >>> resampled = resample(tensor_2d, size=(50, 50), antialias=True, non_spatial_dims=(0,))
    >>> print(resampled.shape)
    torch.Size([10, 50, 50])
    """
    if size is None and scale_factor is None:
        raise ValueError("Either size or scale_factor must be specified")

    num_non_spatial, _ = _parse_non_spatial_dims(non_spatial_dims, input_tensor.ndim)
    dims_to_add = 2 - num_non_spatial

    # Add batch and/or channel dimensions if needed
    for _ in range(dims_to_add):
        input_tensor = input_tensor.unsqueeze(0)

    spatial_ndim = input_tensor.ndim - 2
    if spatial_ndim not in {1, 2, 3}:
        raise ValueError(
            f"Unsupported spatial dimensionality: {spatial_ndim} spatial dimensions. "
            "Only 1D, 2D, and 3D are supported."
        )

    # Infer interpolation mode for linear interpolation
    if mode == 'linear':
        mode = ne.utils.infer_linear_interpolation_mode(spatial_ndim)

    # F.interpolate requires exactly one of size or scale_factor
    if size is not None:
        resampled = F.interpolate(input=input_tensor, size=size, mode=mode, antialias=antialias)
    else:
        resampled = F.interpolate(
            input=input_tensor, scale_factor=scale_factor, mode=mode, antialias=antialias
        )

    # Remove added dimensions to match original tensor shape
    for _ in range(dims_to_add):
        resampled = resampled.squeeze(0)

    return resampled


def filter_dim(tensor: torch.Tensor, dim: int = 0, verbose: bool = False) -> torch.Tensor:
    """
    Filter slices of a tensor that contain NaNs, infinite values, or are entirely zero.

    Parameters
    ----------
    tensor : torch.Tensor
        An n-dimensional tensor.
    dim : int, optional
        The dimension along which to filter slices. Default is 0.
    verbose : bool, optional
        If True, prints the number of elements filtered for each condition (NaNs, infinities,
        all-zeros). Default is False.

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


def _parse_non_spatial_dims(
    non_spatial_dims: Union[Sequence[int], None],
    tensor_ndim: int
) -> Tuple[int, int]:
    """
    Validate and parse non_spatial_dims parameter.

    Parameters
    ----------
    non_spatial_dims : Sequence[int] or None
        Indices of non-spatial dimensions (dimensions to vectorize over). Can be any sequence of
        dimension indices. If None, assumes all dimensions are spatial.
    tensor_ndim : int
        Total number of dimensions in the tensor.

    Returns
    -------
    num_non_spatial : int
        Number of non-spatial dimensions.
    num_spatial : int
        Number of spatial dimensions.

    Raises
    ------
    ValueError
        If non_spatial_dims contains invalid indices or exceeds tensor dimensions.

    Examples
    --------
    >>> _parse_non_spatial_dims(None, 3)
    (0, 3)
    >>> _parse_non_spatial_dims((0,), 4)
    (1, 3)
    >>> _parse_non_spatial_dims((0, 1, 2), 5)
    (3, 2)
    """
    if non_spatial_dims is None:
        return 0, tensor_ndim

    non_spatial_dims = tuple(non_spatial_dims)
    num_non_spatial = len(non_spatial_dims)

    if num_non_spatial > tensor_ndim:
        raise ValueError(
            f"non_spatial_dims has {num_non_spatial} elements but tensor only has "
            f"{tensor_ndim} dimensions"
        )

    # Validate all indices are valid
    for dim in non_spatial_dims:
        if dim < 0 or dim >= tensor_ndim:
            raise ValueError(
                f"non_spatial_dims contains invalid index {dim} for tensor with "
                f"{tensor_ndim} dimensions"
            )

    num_spatial = tensor_ndim - num_non_spatial

    return num_non_spatial, num_spatial


def gaussian_kernel(
    sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
    truncate: Union[int, float, Sequence[Union[int, float]]] = 3,
    ndim: Optional[int] = None,
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
    # Make a 3D kernel with automatic sizing
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

    # Per-dimension truncate values
    >>> gaussian_kernel_ = gaussian_kernel(sigma=(1.0, 2.0, 3.0), truncate=(3, 4, 5))
    # Kernel sizes: [7, 17, 31]
    >>> gaussian_kernel_.shape
    torch.Size([7, 17, 31])

    Notes
    -----
    The automatic kernel sizing follows the formula used in scipy and VoxelMorph:
    kernel_size = 2 * int(truncate * sigma + 0.5) + 1

    This ensures the kernel is always odd-sized and captures the specified number of
    standard deviations. A truncate value of 3 captures ~99.7% of the Gaussian distribution.
    """
    if isinstance(sigma, (float, int)):
        if ndim is None:
            raise ValueError(
                "When sigma is a scalar, ndim must be specified to determine dimensionality"
            )
        if ndim not in [1, 2, 3]:
            raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
        sigma_list = [float(sigma)] * ndim

    elif isinstance(sigma, Sequence):
        sigma_list = [float(s) for s in sigma]
        ndim = len(sigma_list)
        if ndim not in [1, 2, 3]:
            raise ValueError(
                f"sigma length determines dimensionality and must be 1, 2, or 3. "
                f"Got length {ndim}"
            )
    else:
        raise TypeError(f"sigma must be a number or sequence, got {type(sigma)}")

    if isinstance(truncate, (int, float)):
        truncate_list = [float(truncate)] * ndim

    elif isinstance(truncate, Sequence):
        if len(truncate) != ndim:
            raise ValueError(
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
    kernel /= kernel.sum()

    return kernel


def crop(
    input_tensor: torch.Tensor,
    size: Union[int, Sequence[int], None] = None,
    scale_factor: Union[float, Sequence[float], None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
    offset: Union[int, Sequence[int]] = 0,
) -> torch.Tensor:
    """
    Crop tensor to specified size.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor to crop.
    size : int, Sequence[int], or None, default=None
        Target spatial size after cropping. If int, same size used for all spatial dimensions.
        If Sequence, per-dimension sizes. If None, `scale_factor` must be specified.
    scale_factor : float, Sequence[float], or None, default=None
        Multiplicative factor for spatial size. If float, same factor for all spatial dimensions.
        If Sequence, per-dimension factors. Output size = input size * scale_factor. If None,
        `size` must be specified.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Dimensions not to crop (e.g., (0, 1) for batch and channel). If None, crops all dimensions.
    offset : int or Sequence[int], default=0
        Starting position for crop. If int, same offset for all spatial dimensions.
        If Sequence, per-dimension offsets.

    Returns
    -------
    torch.Tensor
        Cropped tensor.

    Raises
    ------
    ValueError
        If both `size` and `scale_factor` are specified or both are None. If crop size exceeds
        input size for any dimension. If offset is out of valid range.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    >>> x = torch.randn(64, 64, 64)
    >>> # Crop from origin (0, 0, 0)
    >>> cropped = ne.crop(x, size=32)
    >>> cropped.shape
    torch.Size([32, 32, 32])
    >>> # Crop from offset (10, 10, 10)
    >>> cropped = ne.crop(x, size=32, offset=10)
    >>> cropped.shape
    torch.Size([32, 32, 32])
    >>> # Per-dimension offsets
    >>> cropped = ne.crop(x, size=(32, 48, 64), offset=(10, 5, 0))
    >>> cropped.shape
    torch.Size([32, 48, 64])
    >>> # Random crop (user controls randomness)
    >>> random_offset = torch.randint(0, 33, (3,)).tolist()
    >>> cropped = ne.crop(x, size=32, offset=random_offset)
    """
    if size is None and scale_factor is None:
        raise ValueError("Either size or scale_factor must be specified")
    if size is not None and scale_factor is not None:
        raise ValueError("size and scale_factor are mutually exclusive")

    # Handle non-spatial dimensions
    num_non_spatial, num_spatial = _parse_non_spatial_dims(non_spatial_dims, input_tensor.dim())
    spatial_dims = list(range(num_non_spatial, input_tensor.dim()))

    if size is not None:
        if isinstance(size, int):
            crop_sizes = [size] * num_spatial
        else:
            if len(size) != num_spatial:
                raise ValueError(
                    f"size length {len(size)} doesn't match number of spatial dims {num_spatial}"
                )
            crop_sizes = list(size)
    else:
        if isinstance(scale_factor, (int, float)):
            scale_factors = [scale_factor] * num_spatial
        else:
            if len(scale_factor) != num_spatial:
                raise ValueError(
                    f"scale_factor length {len(scale_factor)} doesn't match spatial"
                    f"dims {num_spatial}"
                )
            scale_factors = list(scale_factor)

        # Compute crop sizes from scale factors
        crop_sizes = []
        for dim_idx, dim in enumerate(spatial_dims):
            input_size = input_tensor.shape[dim]
            crop_size = round(input_size * scale_factors[dim_idx])
            crop_sizes.append(crop_size)

    # Validate crop sizes
    for dim_idx, dim in enumerate(spatial_dims):
        input_size = input_tensor.shape[dim]
        if crop_sizes[dim_idx] > input_size:
            raise ValueError(
                f"Crop size {crop_sizes[dim_idx]} exceeds input size {input_size} at dim {dim}"
            )

    # Parse offset
    if isinstance(offset, int):
        offsets = [offset] * num_spatial
    else:
        if len(offset) != num_spatial:
            raise ValueError(
                f"offset length {len(offset)} doesn't match number of spatial dims {num_spatial}"
            )
        offsets = list(offset)

    # Validate offsets
    for dim_idx, dim in enumerate(spatial_dims):
        input_size = input_tensor.shape[dim]
        crop_size = crop_sizes[dim_idx]
        max_valid_offset = input_size - crop_size
        if offsets[dim_idx] < 0 or offsets[dim_idx] > max_valid_offset:
            raise ValueError(
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

    Element-wise operation - works on any tensor shape. Thin wrapper around
    torch.clamp with consistent naming.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor to clip.
    min : float, int, or None, default=None
        Minimum value. If None, no lower bound.
    max : float, int, or None, default=None
        Maximum value. If None, no upper bound.

    Returns
    -------
    torch.Tensor
        Clipped tensor (non-inplace).

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    >>> x = torch.randn(10) * 5
    >>> # Clip to [-1, 1]
    >>> clipped = ne.clip(x, min=-1, max=1)
    >>> # Clip only minimum
    >>> clipped = ne.clip(x, min=0)
    >>> # Clip only maximum
    >>> clipped = ne.clip(x, max=1)
    """
    return torch.clamp(input_tensor, min=min, max=max)


def pad_for_vectorization(
    tensor: torch.Tensor,
    non_spatial_dims: Union[Sequence[int], None]
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Flatten non-spatial dims into batch and add singleton channel for PyTorch ops.

    Prepares a tensor for PyTorch operations that require (B, C, *spatial) format. All non-spatial
    dimensions are flattened into the batch dimension, ensuring each element is vectorized.
    A singleton channel dimension is added.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape (*non_spatial, *spatial).
    non_spatial_dims : Sequence[int] or None
        Indices of dims to vectorize over. These dimensions will be gathered into the batch dim.
        If None, tensor is treated as pure spatial and singleton batch and channel dims are added.

    Returns
    -------
    padded : torch.Tensor
        Tensor with shape (batch_flat, 1, *spatial) where batch_flat is the product of all
        non-spatial dimension sizes.
    original_non_spatial_shape : tuple[int, ...]
        Original shape of non-spatial dimensions, needed for unpad_from_vectorization. Empty tuple
        if non_spatial_dims was None.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    # Pure spatial tensor
    >>> t = torch.randn(64, 64, 64)
    >>> padded, shape = ne.pad_for_vectorization(t, non_spatial_dims=None)
    >>> padded.shape
    torch.Size([1, 1, 64, 64, 64])
    >>> shape
    ()

    # Single vectorization dimension
    >>> t = torch.randn(10, 64, 64)
    >>> padded, shape = ne.pad_for_vectorization(t, non_spatial_dims=(0,))
    >>> padded.shape
    torch.Size([10, 1, 64, 64])
    >>> shape
    (10,)

    # Multiple vectorization dimensions
    >>> t = torch.randn(2, 3, 64, 64)
    >>> padded, shape = ne.pad_for_vectorization(t, non_spatial_dims=(0, 1))
    >>> padded.shape
    torch.Size([6, 1, 64, 64])
    >>> shape
    (2, 3)

    # Arbitrary number of vectorization dimensions
    >>> t = torch.randn(2, 3, 4, 5, 32, 32)
    >>> padded, shape = ne.pad_for_vectorization(t, non_spatial_dims=(0, 1, 2, 3))
    >>> padded.shape
    torch.Size([120, 1, 32, 32])
    >>> shape
    (2, 3, 4, 5)

    See Also
    --------
    unpad_from_vectorization : Reverse operation to restore original shape.
    """
    if non_spatial_dims is None:
        num_non_spatial = 0
        original_non_spatial_shape = ()
    else:
        num_non_spatial = len(non_spatial_dims)
        original_non_spatial_shape = tuple(tensor.shape[:num_non_spatial])

    spatial_shape = tensor.shape[num_non_spatial:]

    # Compute flattened batch size
    if num_non_spatial == 0:
        batch_size = 1
    else:
        batch_size = 1
        for dim_size in original_non_spatial_shape:
            batch_size *= dim_size

    # Reshape to (batch_flat, *spatial), then add channel dim
    tensor = tensor.reshape(batch_size, *spatial_shape)
    tensor = tensor.unsqueeze(1)  # (batch_flat, 1, *spatial)

    return tensor, original_non_spatial_shape


def unpad_from_vectorization(
    tensor: torch.Tensor,
    original_non_spatial_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Restore original non-spatial shape after vectorized PyTorch operation.

    Reverses pad_for_vectorization by removing the singleton channel dimension
    and unflattening the batch dimension back to the original non-spatial shape.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor with shape (batch_flat, 1, *spatial) from a PyTorch operation.
    original_non_spatial_shape : tuple[int, ...]
        Original non-spatial shape from pad_for_vectorization. Empty tuple means input was
        pure spatial.

    Returns
    -------
    torch.Tensor
        Tensor with shape (*original_non_spatial_shape, *spatial).

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    # Round-trip for pure spatial
    >>> t = torch.randn(64, 64, 64)
    >>> padded, shape = ne.pad_for_vectorization(t, non_spatial_dims=None)
    >>> restored = ne.unpad_from_vectorization(padded, shape)
    >>> restored.shape
    torch.Size([64, 64, 64])

    # Round-trip with vectorization dims (spatial may change from operation)
    >>> t = torch.randn(2, 3, 64, 64)
    >>> padded, shape = ne.pad_for_vectorization(t, non_spatial_dims=(0, 1))
    >>> # Simulate operation that changes spatial dims
    >>> result = padded[..., ::2, ::2]  # (6, 1, 32, 32)
    >>> restored = ne.unpad_from_vectorization(result, shape)
    >>> restored.shape
    torch.Size([2, 3, 32, 32])

    See Also
    --------
    pad_for_vectorization : Prepare tensor for vectorized operations.
    """
    # Remove channel dimension: (batch_flat, 1, *spatial) -> (batch_flat, *spatial)
    tensor = tensor.squeeze(1)

    if len(original_non_spatial_shape) == 0:
        # Was pure spatial, squeeze the batch dim too
        tensor = tensor.squeeze(0)
    else:
        # Unflatten batch back to original non-spatial dims
        new_spatial_shape = tensor.shape[1:]
        tensor = tensor.reshape(*original_non_spatial_shape, *new_spatial_shape)

    return tensor


def smooth_gaussian(
    shape: Sequence[int],
    sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
    magnitude: float = 1.0,
    non_spatial_dims: Union[Sequence[int], None] = None,
    device: Union[torch.device, None] = None,
) -> torch.Tensor:
    """
    Generate smooth Gaussian noise.

    Creates noise with a smooth spatial structure by generating white noise and applying
    Gaussian smoothing. The result is normalized to have zero mean and specified standard
    deviation (magnitude).

    Parameters
    ----------
    shape : Sequence[int]
        Desired shape of output tensor. Interpretation depends on non_spatial_dims:
        - non_spatial_dims=None: (*spatial,) pure spatial tensor
        - non_spatial_dims=(0,): (C, *spatial) with channel dimension
        - non_spatial_dims=(0, 1): (B, C, *spatial) with batch and channel
    sigma : float, int, or Sequence[float or int], default=1
        Standard deviation of the Gaussian kernel for smoothing. If float/int, same sigma
        is used for all spatial dimensions. If Sequence, different sigmas per dimension.
    magnitude : float, default=1.0
        Standard deviation of the output noise after normalization.
    non_spatial_dims : Sequence of int or None, default=None
        Indices of non-spatial dimensions:
        - None: tensor is pure spatial (*spatial,)
        - (0,): first dim is non-spatial (C, *spatial)
        - (0, 1): first two dims are non-spatial (B, C, *spatial)
    device : torch.device or None, default=None
        Device for tensor allocation. If None, defaults to CPU.

    Returns
    -------
    torch.Tensor
        Smooth Gaussian noise with the specified shape, zero mean, and standard deviation
        equal to magnitude.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    >>> # Generate 2D noise field (pure spatial)
    >>> noise_2d = ne.smooth_gaussian(shape=(64, 64), sigma=2.0)
    >>> noise_2d.shape
    torch.Size([64, 64])

    >>> # Generate 3D noise with batch and channel dimensions
    >>> noise_3d = ne.smooth_gaussian(
    ...     shape=(2, 3, 32, 32, 32),
    ...     sigma=3.0,
    ...     magnitude=2.0,
    ...     non_spatial_dims=(0, 1)
    ... )
    >>> noise_3d.shape
    torch.Size([2, 3, 32, 32, 32])

    >>> # Per-dimension sigma values
    >>> noise = ne.smooth_gaussian(shape=(64, 64), sigma=[1.0, 2.0])
    """
    num_non_spatial, _ = _parse_non_spatial_dims(
        non_spatial_dims=non_spatial_dims,
        tensor_ndim=len(shape)
    )

    non_spatial_shape = shape[:num_non_spatial]
    spatial_shape = shape[num_non_spatial:]

    shape = (*non_spatial_shape, *spatial_shape)

    noise = torch.normal(0, 1, size=shape, device=device)
    noise, orig_shape = pad_for_vectorization(noise, non_spatial_dims)
    noise = ne.nn.functional.gaussian_smoothing(noise, sigma=sigma, truncate=3)

    # Normalize to zero mean and specified magnitude
    noise -= noise.mean()
    noise *= magnitude / noise.std()

    return unpad_from_vectorization(noise, orig_shape)


def upsample_noise(
    shape: Sequence[int],
    scale: Union[float, int],
    non_spatial_dims: Union[Sequence[int], None] = None,
    device: Union[torch.device, None] = None
) -> torch.Tensor:
    """
    Generate smooth noise by upsampling from a coarse grid.

    Creates noise at a downsampled resolution (determined by scale) and upsamples
    to the target shape using linear interpolation. This produces smooth, spatially
    correlated noise more efficiently than blurring full-resolution noise.

    Parameters
    ----------
    shape : Sequence[int]
        Target shape of output tensor. Interpretation depends on non_spatial_dims:
        - non_spatial_dims=None: (*spatial,) pure spatial tensor
        - non_spatial_dims=(0,): (C, *spatial) with channel dimension
        - non_spatial_dims=(0, 1): (B, C, *spatial) with batch and channel
    scale : float or int
        Downsampling factor. Larger values produce smoother noise. The coarse grid
        size along each spatial dimension is max(spatial_size // scale, 2).
    non_spatial_dims : Sequence of int or None, default=None
        Indices of non-spatial dimensions:
        - None: tensor is pure spatial (*spatial,)
        - (0,): first dim is non-spatial (C, *spatial)
        - (0, 1): first two dims are non-spatial (B, C, *spatial)
    device : torch.device or None, default=None
        Device for tensor allocation.

    Returns
    -------
    torch.Tensor
        Upsampled noise with the specified shape.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    >>> # Pure spatial 2d noise
    >>> noise = ne.upsample_noise(shape=(64, 64), scale=8.0)
    >>> noise.shape
    torch.Size([64, 64])

    >>> # With batch and channel dimensions
    >>> noise = ne.upsample_noise(shape=(2, 3, 64, 64), scale=8.0, non_spatial_dims=(0, 1))
    >>> noise.shape
    torch.Size([2, 3, 64, 64])
    """
    num_non_spatial, num_spatial = _parse_non_spatial_dims(
        non_spatial_dims=non_spatial_dims,
        tensor_ndim=len(shape)
    )

    non_spatial_shape = shape[:num_non_spatial]
    spatial_shape = shape[num_non_spatial:]

    # Compute downsampled noise
    coarse_spatial = tuple(max(int(s // scale), 2) for s in spatial_shape)
    coarse_shape = (*non_spatial_shape, *coarse_spatial)
    noise = torch.randn(coarse_shape, device=device)

    noise, orig_shape = pad_for_vectorization(noise, non_spatial_dims)

    # Interpolate to target spatial shape
    mode = ne.utils.infer_linear_interpolation_mode(num_spatial=num_spatial)
    noise = F.interpolate(noise, size=spatial_shape, mode=mode, align_corners=False)

    return unpad_from_vectorization(noise, orig_shape)


def fractal_noise(
    shape: Sequence[int],
    scales: Union[float, int, Sequence[Union[float, int]], None] = None,
    magnitude: float = 1.0,
    weights: Union[Sequence[float], None] = None,
    non_spatial_dims: Union[Sequence[int], None] = None,
    device: Union[torch.device, None] = None,
    method: Literal['blur', 'upsample'] = 'blur'
) -> torch.Tensor:
    """
    Generate fractal noise by combining multiple scales of smooth noise.

    Creates multi-scale noise by generating smooth noise at different scales and combining
    them with optional weighting. This produces natural-looking noise with features at
    multiple spatial frequencies.

    Parameters
    ----------
    shape : Sequence[int]
        Target shape of output tensor. Interpretation depends on non_spatial_dims:
        - non_spatial_dims=None: (*spatial,) pure spatial tensor
        - non_spatial_dims=(0,): (C, *spatial) with channel dimension
        - non_spatial_dims=(0, 1): (B, C, *spatial) with batch and channel
    scales : float, int, Sequence[float or int], or None, default=None
        Smoothing scale(s) for each octave. Interpretation depends on method:
        - method='blur': sigma values for Gaussian smoothing
        - method='upsample': downsampling factors for upsampled noise
        If None, defaults to powers of 2 up to max spatial dimension.
        If scalar, reduces to single-scale noise generation.
    magnitude : float, default=1.0
        Standard deviation of the final normalized noise.
    weights : Sequence[float] or None, default=None
        Weight for each scale. If None, uses linearly increasing weights [1, 2, 3, ...].
        Length must match scales if both are sequences.
    non_spatial_dims : Sequence of int or None, default=None
        Indices of non-spatial dimensions:
        - None: tensor is pure spatial (*spatial,)
        - (0,): first dim is non-spatial (C, *spatial)
        - (0, 1): first two dims are non-spatial (B, C, *spatial)
    device : torch.device or None, default=None
        Device for tensor allocation.
    method : {'blur', 'upsample'}, default='blur'
        Noise generation method:
        - 'blur': Generate noise at full resolution and apply Gaussian smoothing
        - 'upsample': Generate coarse noise and upsample (faster, lower memory)

    Returns
    -------
    torch.Tensor
        Fractal noise with the specified shape, zero mean, and std equal to magnitude.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    >>> # Pure spatial 2d fractal noise with default scales
    >>> noise = ne.fractal_noise(shape=(64, 64))
    >>> noise.shape
    torch.Size([64, 64])

    >>> # With batch and channel dimensions
    >>> noise = ne.fractal_noise(shape=(2, 3, 64, 64), non_spatial_dims=(0, 1))
    >>> noise.shape
    torch.Size([2, 3, 64, 64])

    >>> # Custom scales and weights
    >>> noise = ne.fractal_noise(shape=(64, 64), scales=[2.0, 4.0, 8.0], weights=[1.0, 0.5, 0.25])
    """
    import numpy as np

    num_non_spatial, _ = _parse_non_spatial_dims(
        non_spatial_dims=non_spatial_dims,
        tensor_ndim=len(shape)
    )
    spatial_shape = shape[num_non_spatial:]

    # Default scales: powers of 2 up to max spatial dimension
    if scales is None:
        scales = 2 ** np.arange(np.log2(max(spatial_shape)))[1:]

    # Convert scalar to list for uniform handling
    if np.isscalar(scales):
        scales = [scales]

    # Set default weights if not provided
    if len(scales) == 1:
        weights = [1.0]
    elif weights is None:
        weights = list(np.arange(len(scales)) + 1)

    if len(weights) != len(scales):
        raise ValueError(
            f'weights length ({len(weights)}) must match scales length ({len(scales)})'
        )

    # Generate noise at each scale
    noise = None
    for scale, weight in zip(scales, weights):
        if method == 'blur':
            sample = smooth_gaussian(
                shape=shape,
                sigma=scale,
                magnitude=1.0,
                non_spatial_dims=non_spatial_dims,
                device=device
            )
        else:  # method == 'upsample'
            sample = upsample_noise(
                shape=shape,
                scale=scale,
                non_spatial_dims=non_spatial_dims,
                device=device
            )

        sample *= weight

        if noise is None:
            noise = sample
        else:
            noise += sample

    # Normalize to target magnitude
    noise -= noise.mean()
    noise *= magnitude / noise.std()

    return noise
