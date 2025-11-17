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
    "soft_quantize",
    "mse",
    "dice",
    "reduce",
    "volshape_to_ndgrid",
    "apply_bernoulli_mask",
    "random_flip",
    "sample_image_from_labels",
    "resample",
    "filter_dim",
    "gaussian_kernel",
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
    Quantize continuous values into discrete bins.

    Instead of assigning each value to a single bin, use a soft assignment based on the distance
    between each value and the bin centers. Particularly useful for taking gradients during
    quantization.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor of any shape to softly quantize.
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
        Softly quantized tensor with the same shape as `input_tensor`.

    Examples
    --------
    >>> import torch
    >>> import matplotlib.pyplot as plt
    # Make a random 3D tensor with zero mean and unit variance.
    >>> input_tensor = torch.randn(32, 32, 32)
    # Compute the softly quantized tensor with a low softness to approximate (and visualize) a
    # pseudo-hard quantization.
    >>> softly_quantized_tensor = soft_quantize(input_tensor, nb_bins=4, softness=0.5)
    # Visualize the softly quantized tensor.
    >>> plt.imshow(softly_quantized_tensor[16])
    """

    # Invert softness to control sensitivity in softmax: higher input softness → sharper bins
    softness = 1 / softness
    input_tensor.clip_(min_clip, max_clip)

    bin_centers = torch.linspace(
        start=input_tensor.min(), end=input_tensor.max(), steps=nb_bins, device=input_tensor.device)

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
        If True, stack the grid tensors along the last dimension to return a single tensor of
        shape `(*size, len(size))`. If False, return a tuple of tensors, each of shape
        `(*size)`. Default is False.

    Returns
    -------
    torch.Tensor
        The meshgrid of spatial coordinates
        if stack=False, a tuple of len(size) tensors of shape `*size`
        if stack=True, a tensor of shape `*size, len(size)`

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
    >>> # Stacked grid
    >>> the_grid = volshape_to_ndgrid(size=(19, 32), stack=True)
    >>> print(the_grid.shape)
    torch.Size([19, 32, 2])
    """
    normalized_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)

    if normalize:
        axes = [
            torch.linspace(-1, 1, steps=sz, device=device, dtype=normalized_dtype) for sz in size]
    else:
        axes = [torch.arange(0, sz, device=device, dtype=normalized_dtype) for sz in size]

    grid = torch.meshgrid(*axes, indexing=indexing)

    if stack:
        grid = torch.stack(grid, dim=-1).contiguous()

    return grid


def subsample(
    input_tensor: torch.Tensor,
    stride: Union[Sequence[int], int, None] = 2,
    subsampling_dimension: Union[list, int, None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = None
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
    stride : int, Sequence[int], or None, default=2
        Factor by which to subsample (interleave dropouts). If int, same stride is used for all
        subsampled dimensions. If Sequence, must match the number of spatial dimensions.
    subsampling_dimension : int, Sequence[int], or None, default=None
        The spatial dimension(s) to subsample (0-indexed among spatial dims). If None, subsamples
        all spatial dimensions.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Indices of non-spatial dimensions. Must be a contiguous sequence starting from 0.
        Valid values: `()`, `(0,)`, or `(0, 1)`. If None, assumes all dimensions are spatial.

    Returns
    -------
    torch.Tensor
        Tensor that has been subsampled.

    Examples
    --------
    >>> import torch
    # Subsample a 2D tensor (no batch/channel dims)
    >>> input_tensor = torch.arange(25).view(5, 5)
    >>> subsampled = subsample(input_tensor, stride=2, subsampling_dimension=1)
    >>> print(subsampled.shape)
    torch.Size([5, 3])

    # Subsample with batch and channel dims
    >>> input_tensor = torch.randn(2, 3, 32, 32)
    >>> subsampled = subsample(input_tensor, stride=2, non_spatial_dims=(0, 1))
    >>> print(subsampled.shape)
    torch.Size([2, 3, 16, 16])
    """
    if isinstance(subsampling_dimension, torch.Tensor):
        raise TypeError(
            "subsampling_dimension must be an int, list, tuple, or None, not a Tensor"
        )

    num_non_spatial, num_spatial = _parse_non_spatial_dims(non_spatial_dims, input_tensor.ndim)
    ndim = input_tensor.ndim
    slices = [slice(None)] * ndim

    # Dimensions to subsample
    if subsampling_dimension is None:
        spatial_dims_to_subsample = list(range(num_spatial))
    elif isinstance(subsampling_dimension, int):
        spatial_dims_to_subsample = [subsampling_dimension]
    else:
        spatial_dims_to_subsample = list(subsampling_dimension)

    if isinstance(stride, int):
        strides = [stride] * num_spatial
    elif isinstance(stride, (tuple, list)):
        strides = list(stride)
        if len(strides) != num_spatial:
            raise ValueError(
                f"stride length {len(strides)} must match number of spatial dimensions "
                f"{num_spatial}"
            )
    else:
        strides = [2] * num_spatial  # Default stride

    # Convert spatial dimension indices to absolute tensor indices and apply subsampling
    for spatial_idx in spatial_dims_to_subsample:
        if spatial_idx < 0 or spatial_idx >= num_spatial:
            raise ValueError(
                f"subsampling_dimension index {spatial_idx} out of range for {num_spatial} "
                "spatial dims"
            )
        absolute_idx = num_non_spatial + spatial_idx
        slices[absolute_idx] = slice(None, None, strides[spatial_idx])

    return input_tensor[tuple(slices)]


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
    non_spatial_dims: Union[Tuple[int, ...], None],
    tensor_ndim: int
) -> Tuple[int, int]:
    """
    Validate and parse non_spatial_dims parameter.

    Parameters
    ----------
    non_spatial_dims : Tuple[int, ...] or None
        Indices of non-spatial dimensions. Must be None, (0,), or (0, 1).
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
        If non_spatial_dims is not valid.
    """
    valid_values = {None: 0, (0,): 1, (0, 1): 2}

    if non_spatial_dims not in valid_values:
        raise ValueError(
            f"non_spatial_dims must be None, (0,), or (0, 1), got {non_spatial_dims}"
        )

    num_non_spatial = valid_values[non_spatial_dims]
    num_spatial = tensor_ndim - num_non_spatial

    return num_non_spatial, num_spatial


def gaussian_kernel(
    kernel_size: Sequence[int],
    sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """
    Create a {1D, 2D, 3D} Gaussian kernel.

    Shape-agnostic implementation that returns a kernel with only spatial dimensions,
    no batch or channel dimensions. Dimensionality is inferred from the length of
    `kernel_size`.

    Parameters
    ----------
    kernel_size : Sequence[int]
        Size of each spatial dimension in the Gaussian kernel. Length determines
        dimensionality (1D, 2D, or 3D).
    sigma : float, int, or Sequence[float or int], optional
        Standard deviation of the Gaussian kernel. If float/int, same sigma is used for all
        dimensions. If Sequence, different sigmas can be specified per dimension. Default is 1.
    device : torch.device, optional
        Device on which to create the kernel tensor. Default is None.
    dtype : torch.dtype, optional
        Data type of the kernel tensor. Default is torch.float32.

    Returns
    -------
    torch.Tensor
        Tensor representing the {1D, 2D, 3D} Gaussian kernel with shape (*kernel_size).
        No batch or channel dimensions.

    Examples
    --------
    >>> import torch
    # Make a 3D kernel
    >>> gaussian_kernel_ = gaussian_kernel(kernel_size=(3, 3, 3), sigma=1)
    # Print shape (no batch/channel dimensions)
    >>> gaussian_kernel_.shape
    torch.Size([3, 3, 3])

    # Make a 2D kernel with different sizes per dimension
    >>> gaussian_kernel_ = gaussian_kernel(kernel_size=(3, 5), sigma=(0.5, 1.0))
    >>> gaussian_kernel_.shape
    torch.Size([3, 5])

    # Make a 1D kernel
    >>> gaussian_kernel_ = gaussian_kernel(kernel_size=(7,), sigma=2.0)
    >>> gaussian_kernel_.shape
    torch.Size([7])
    """
    # Validate kernel_size is a sequence
    if not isinstance(kernel_size, (list, tuple)):
        raise TypeError(
            f"kernel_size must be a sequence (list or tuple), got {type(kernel_size)}"
        )

    kernel_size_list = list(kernel_size)
    ndim = len(kernel_size_list)

    # Validate ndim
    if ndim not in [1, 2, 3]:
        raise ValueError(
            f"kernel_size length determines dimensionality and must be 1, 2, or 3. "
            f"Got length {ndim}"
        )

    # Create coordinate grid centered at zero
    coords = [
        torch.arange(ks, device=device, dtype=dtype).float() - (ks - 1) / 2
        for ks in kernel_size_list
    ]

    grid = torch.stack(
        torch.meshgrid(coords, indexing='ij'), dim=-1
    ).to(device=device, dtype=dtype)

    # Handle sigma parameter
    if isinstance(sigma, (float, int)):
        sigma_list = [sigma] * ndim
    elif isinstance(sigma, Sequence):
        if len(sigma) != ndim:
            raise ValueError(
                f"If sigma is a sequence, it must have length equal to kernel_size length "
                f"({ndim}). Got length {len(sigma)}"
            )
        sigma_list = list(sigma)
    else:
        raise TypeError(f"sigma must be a number or sequence, got {type(sigma)}")

    # Convert sigma to tensor on device
    sigma_tensor = torch.tensor(sigma_list, device=device, dtype=dtype)

    # Calculate the Gaussian function: exp(-0.5 * sum((x / sigma)^2))
    kernel = torch.exp(-0.5 * (grid ** 2 / sigma_tensor**2).sum(dim=-1))

    # Normalize the kernel so that it sums to 1
    kernel /= kernel.sum()

    return kernel
