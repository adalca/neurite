"""
Single tensor operations (no B, C dimension assumption)
"""
# Standard library imports
import importlib
import itertools
from typing import List, Literal, Optional, Sequence, Tuple, Union

# Third party imports
import torch
import torch.nn.functional as F

# Custom imports
import neurite.nn.functional as nef

__all__ = [
    "mse",
    "dice",
    "ncc",
    "spatial_gradient",
    "reduce",
    "zscore",
    "volshape_to_ndgrid",
    "bw_grid",
    "apply_bernoulli_mask",
    "random_flip",
    "sample_image_from_labels",
    "resample",
    "pad_to_multiple_of",
    "mask_border",
    "one_hot",
    "connected_components",
    "component_sizes",
    "fill_nearest",
    "filter_dim",
    "gaussian_kernel",
    "gaussian_smoothing",
    "crop",
    "clip",
    "batch_nonspatial",
    "unbatch_nonspatial",
    "random_smoothed_noise",
    "upsample_noise",
    "fractal_noise",
    "parse_non_spatial_dims",
]


def connected_components(
    mask: torch.Tensor,
    connectivity: int = 1,
    *,
    method: Literal["auto", "torch", "triton"] = "auto",
) -> Tuple[torch.Tensor, int]:
    """
    Label connected components in a spatial binary mask.

    Parameters
    ----------
    mask : torch.Tensor
        Spatial binary mask of shape [*V].
    connectivity : int
        Neighborhood connectivity from 1 through the number of spatial
        dimensions. One uses face connectivity; `mask.ndim` uses full
        connectivity.
    method : {'auto', 'torch', 'triton'}, default='auto'
        Labeling implementation. Auto uses Triton for supported CUDA tensors
        and otherwise uses portable Torch operations. Torch forces the portable
        implementation on any device. Triton requires both Triton and CUDA.

    Returns
    -------
    tuple[torch.Tensor, int]
        Integer component labels with shape [*V] and the number of components.
        Background is zero and components are numbered from one.

    Raises
    ------
    ValueError
        If the mask, connectivity, method, or forced Triton device and size are unsupported.
    RuntimeError
        If Triton is forced but unavailable, or if its labeling does not converge.

    Notes
    -----
    The Triton method supports tensors with at most ``int32`` elements. Auto
    falls back to Torch for larger tensors. Triton raises an error instead.
    The optional Triton backend is loaded only when explicitly requested or
    when an eligible CUDA tensor reaches automatic dispatch.
    """
    if mask.ndim not in (1, 2, 3):
        raise ValueError("mask must have one, two, or three spatial dimensions")
    if connectivity < 1 or connectivity > mask.ndim:
        raise ValueError("connectivity must be between 1 and mask.ndim")
    if method not in {"auto", "torch", "triton"}:
        raise ValueError("method must be 'auto', 'torch', or 'triton'")

    max_triton_elements = torch.iinfo(torch.int32).max
    if method == "triton":
        ne_triton = importlib.import_module("neurite.triton")

        return ne_triton.connected_components(mask, connectivity)
    if mask.numel() == 0:
        components = torch.zeros_like(mask, dtype=torch.long)
        return components, 0

    triton_candidate = mask.is_cuda and mask.numel() <= max_triton_elements
    if method == "auto" and triton_candidate:
        ne_triton = importlib.import_module("neurite.triton")

        if ne_triton.is_available():
            return ne_triton.connected_components(mask, connectivity)

    # Pad the mask so neighbor rolls cannot connect opposite spatial edges.
    padding = [1, 1] * mask.ndim
    foreground = F.pad(mask.bool(), pad=padding, value=False)  # [*Vp]
    flat_ids = torch.arange(foreground.numel(), device=mask.device, dtype=torch.long)
    flat_ids = flat_ids.reshape(foreground.shape)  # [*Vp]
    sentinel = foreground.numel()
    background = torch.full_like(flat_ids, sentinel)
    labels = torch.where(foreground, flat_ids, background)  # [*Vp]

    # Include offsets whose squared grid distance is within the requested
    # connectivity, matching the usual 1D-3D connectivity convention.
    offsets = []
    for offset in itertools.product((-1, 0, 1), repeat=mask.ndim):
        offset_distance = sum(step != 0 for step in offset)
        if 0 < offset_distance <= connectivity:
            offsets.append(offset)

    # Propagate the minimum flat index throughout each component.
    spatial_dims = tuple(range(mask.ndim))
    while True:
        updated = labels
        for offset in offsets:
            neighbor = torch.roll(labels, shifts=offset, dims=spatial_dims)  # [*Vp]
            updated = torch.minimum(updated, neighbor)  # [*Vp]
        updated = torch.where(foreground, updated, background)  # [*Vp]
        if torch.equal(updated, labels):
            break
        labels = updated

    # Remove padding and remap component roots to contiguous scan-order IDs.
    interior = (slice(1, -1),) * mask.ndim
    roots = labels[interior]  # [*V]
    foreground = mask.bool()  # [*V]
    root_values = roots[foreground]  # [Nfg]
    if root_values.numel() == 0:
        components = torch.zeros_like(mask, dtype=torch.long)
        return components, 0

    unique_roots, inverse = torch.unique(root_values, sorted=True, return_inverse=True)
    components = torch.zeros_like(mask, dtype=torch.long)
    components[foreground] = inverse + 1

    return components, int(unique_roots.numel())


def component_sizes(
    components: torch.Tensor,
    num_components: Optional[int] = None,
) -> torch.Tensor:
    """
    Count voxels in connected-component labels.

    Parameters
    ----------
    components : torch.Tensor
        Nonnegative component labels of shape [*V]. Background is zero.
    num_components : int, optional
        Number of foreground components. Defaults to the maximum component ID.

    Returns
    -------
    torch.Tensor
        Component sizes for IDs one through `num_components`, shape [N].
    """
    if torch.is_floating_point(components) or torch.is_complex(components):
        raise TypeError("components must have an integer dtype")
    if components.numel() and torch.any(components < 0):
        raise ValueError("components must be nonnegative")

    inferred_count = int(components.max().item()) if components.numel() else 0
    if num_components is None:
        num_components = inferred_count
    num_components = int(num_components)
    if num_components < inferred_count:
        message = "num_components cannot be smaller than the largest component ID"
        raise ValueError(message)
    if num_components < 0:
        raise ValueError("num_components must be nonnegative")

    flat_components = components.long().reshape(-1)
    counts = torch.bincount(flat_components, minlength=num_components + 1)

    return counts[1:num_components + 1]  # [N]


def fill_nearest(
    tensor: torch.Tensor,
    fill_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Fill masked spatial elements from their nearest retained element.

    Euclidean nearest-neighbor indices are computed exactly in voxel space with
    chunked Torch operations on the input tensor's device.

    Parameters
    ----------
    tensor : torch.Tensor
        Spatial values of shape [*V].
    fill_mask : torch.Tensor
        Boolean-like mask with the same shape and device as `tensor`. Nonzero
        elements are replaced and zero elements provide retained source values.

    Returns
    -------
    torch.Tensor
        Filled tensor with the same shape, dtype, and device as `tensor`.
    """
    if tensor.ndim not in (1, 2, 3):
        raise ValueError("tensor must have one, two, or three spatial dimensions")
    if fill_mask.shape != tensor.shape:
        raise ValueError("fill_mask must have the same shape as tensor")
    if fill_mask.device != tensor.device:
        raise ValueError("fill_mask must be on the same device as tensor")

    fill = fill_mask.bool()
    if not torch.any(fill):
        return tensor.clone()
    if torch.all(fill):
        raise ValueError("fill_mask must leave at least one retained source element")

    # Only retained pixels bordering the fill mask can be nearest donors. Full
    # connectivity preserves diagonal boundary donors in 2D and 3D.
    spatial_fill = fill[None, None].float()  # [1, 1, *V]
    if tensor.ndim == 1:
        neighboring_fill = F.max_pool1d(spatial_fill, kernel_size=3, stride=1, padding=1)
    elif tensor.ndim == 2:
        neighboring_fill = F.max_pool2d(spatial_fill, kernel_size=3, stride=1, padding=1)
    else:
        neighboring_fill = F.max_pool3d(spatial_fill, kernel_size=3, stride=1, padding=1)
    donor_mask = (~fill) & neighboring_fill[0, 0].bool()  # [*V]

    fill_coords = torch.nonzero(fill, as_tuple=False)  # [Nf, D]
    donor_coords = torch.nonzero(donor_mask, as_tuple=False)  # [Nd, D]
    fill_coords_float = fill_coords.to(dtype=torch.float64)  # [Nf, D]
    donor_coords_float = donor_coords.to(dtype=torch.float64)  # [Nd, D]

    # Bound pairwise distance matrices while preserving exact Euclidean choices.
    max_pairwise_elements = 4_000_000
    donor_chunk_size = min(donor_coords.shape[0], 4096)
    fill_chunk_size = max(1, max_pairwise_elements // donor_chunk_size)
    nearest_donor = torch.empty(fill_coords.shape[0], device=tensor.device, dtype=torch.long)

    for fill_start in range(0, fill_coords.shape[0], fill_chunk_size):
        fill_end = min(fill_start + fill_chunk_size, fill_coords.shape[0])
        fill_chunk = fill_coords_float[fill_start:fill_end]  # [Fc, D]
        fill_squared = (fill_chunk * fill_chunk).sum(dim=1)  # [Fc]
        best_distance = torch.full(
            (fill_chunk.shape[0],),
            torch.inf,
            device=tensor.device,
            dtype=torch.float64,
        )  # [Fc]
        best_donor = torch.zeros(fill_chunk.shape[0], device=tensor.device, dtype=torch.long)

        for donor_start in range(0, donor_coords.shape[0], donor_chunk_size):
            donor_end = min(donor_start + donor_chunk_size, donor_coords.shape[0])
            donor_chunk = donor_coords_float[donor_start:donor_end]  # [Dc, D]
            donor_squared = (donor_chunk * donor_chunk).sum(dim=1)  # [Dc]
            cross_term = fill_chunk @ donor_chunk.T  # [Fc, Dc]
            distances = fill_squared[:, None] + donor_squared[None, :] - 2 * cross_term
            chunk_distance, chunk_index = distances.min(dim=1)  # [Fc], [Fc]

            closer = chunk_distance < best_distance  # [Fc]
            best_distance = torch.where(closer, chunk_distance, best_distance)  # [Fc]
            global_index = donor_start + chunk_index
            best_donor = torch.where(closer, global_index, best_donor)  # [Fc]

        nearest_donor[fill_start:fill_end] = best_donor

    nearest_coords = donor_coords[nearest_donor]  # [Nf, D]
    fill_index = tuple(fill_coords[:, axis] for axis in range(tensor.ndim))
    donor_index = tuple(nearest_coords[:, axis] for axis in range(tensor.ndim))

    output = tensor.clone()
    output[fill_index] = tensor[donor_index]

    return output


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

    return nef.mse(tensor1, tensor2)


def dice(
    *segs: torch.Tensor,
    smooth_numerator: float = 1e-12,
    smooth_denominator: float = 1e-12,
    non_spatial_dims: Union[Tuple[int, ...], None] = None
) -> torch.Tensor:
    """
    Compute Dice score over multiple segmentation maps.

    Shape-agnostic implementation that can either compute a global Dice score
    (when non_spatial_dims=None) or preserve batch/label structure.

    Parameters
    ----------
    *segs : torch.Tensor
        Two or more segmentation tensors with the same shape and values in [0, 1].
    smooth_numerator : float, optional
        Smoothing constant added to the numerator. Default is 1e-12.
    smooth_denominator : float, optional
        Smoothing constant added to the denominator. Default is 1e-12.
    non_spatial_dims : Tuple[int, ...] or None, optional
        Indices of leading non-spatial dimensions. Must be a contiguous sequence starting from 0.
        If None, assumes all dimensions are spatial and computes a single scalar Dice score.
        Default is None.

    Returns
    -------
    torch.Tensor
        Dice score. Shape depends on non_spatial_dims:
        - If None: scalar
        - If (0,): shape (B,)
        - If (0, 1): shape (B, L)

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

    # Compute per-batch-and-label dice
    >>> seg1 = torch.rand(2, 3, 64, 64)
    >>> seg2 = torch.rand(2, 3, 64, 64)
    >>> score = dice(seg1, seg2, non_spatial_dims=(0, 1))
    >>> print(score.shape)
    torch.Size([2, 3])
    """
    tensor_batch = []
    original_shape = None
    for seg in segs:
        batched, original_shape = batch_nonspatial(seg, non_spatial_dims)
        tensor_batch.append(batched)

    dice_score = nef.dice(
        *tensor_batch,
        smooth_numerator=smooth_numerator,
        smooth_denominator=smooth_denominator,
        reduction=None,
    )

    return unbatch_nonspatial(dice_score, original_shape)


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
        Indices of leading non-spatial dimensions. Must be a contiguous sequence starting from 0.
        If None, assumes all dimensions are spatial and computes a single scalar NCC score.
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
    tensor1, orig_shape = batch_nonspatial(tensor1, non_spatial_dims)
    tensor2, _ = batch_nonspatial(tensor2, non_spatial_dims)

    ncc_score = nef.ncc(
        tensor1=tensor1,
        tensor2=tensor2,
        window_size=window_size,
        eps=eps,
        reduction=None,
    )

    return unbatch_nonspatial(ncc_score, orig_shape)


def spatial_gradient(
    input_tensor: torch.Tensor,
    non_spatial_dims: Union[Tuple[int, ...], None] = None,
) -> List[torch.Tensor]:
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
    input_tensor, orig_shape = batch_nonspatial(input_tensor, non_spatial_dims)
    gradients = nef.spatial_gradients(input_tensor)
    return [unbatch_nonspatial(gradient, orig_shape) for gradient in gradients]


def reduce(
    tensor: torch.Tensor,
    reduction: str = 'mean',
    dim: Union[int, Tuple[int, ...], None] = None,
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
    return nef.reduce(tensor, reduction=reduction, dim=dim, keepdims=keepdims)


def zscore(
    input_tensor: torch.Tensor,
    dim: Union[int, Tuple[int, ...], None] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Standardize a tensor to zero mean and unit standard deviation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Floating-point tensor to standardize.
    dim : int, tuple of ints, or None, default=None
        Dimension or dimensions over which to compute the mean and standard deviation. `None`
        standardizes the complete tensor.
    eps : float, default=1e-8
        Lower bound for the standard deviation, preventing division by zero.

    Returns
    -------
    torch.Tensor
        Standardized tensor with the same shape as `input_tensor`.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    >>> tensor = torch.randn(2, 3, 16, 16)
    >>> standardized = ne.zscore(tensor, dim=(1, 2, 3))
    >>> standardized.mean(dim=(1, 2, 3))
    tensor([0., 0.])
    """
    return nef.zscore(input_tensor, dim=dim, eps=eps)


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
    return nef.volshape_to_ndgrid(
        size=(1, 1, *size),
        device=device,
        dtype=dtype,
        normalize=normalize,
        indexing=indexing,
        stack=stack,
    )


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
    >>> import neurite as ne
    >>> grid = ne.bw_grid((5, 5), spacing=1)
    >>> grid
    tensor([[1., 1., 1., 1., 1.],
            [1., 0., 1., 0., 1.],
            [1., 1., 1., 1., 1.],
            [1., 0., 1., 0., 1.],
            [1., 1., 1., 1., 1.]])

    Notes
    -----
    This follows the original pystrum convention used by VoxelMorph: line starts repeat every
    `spacing + 1` pixels, and the last pixel along each axis is always set to a grid line.
    """
    return nef.bw_grid(
        vol_shape=vol_shape,
        spacing=spacing,
        thickness=thickness,
        indexing=indexing,
        device=device,
        dtype=dtype,
    )


def subsample(
    input_tensor: torch.Tensor,
    stride: Union[Sequence[int], int, None] = 2,
    subsampling_dimension: Union[list, int, None] = None,
    non_spatial_dims: Union[Tuple[int, ...], None] = None
) -> torch.Tensor:
    """
    Deprecated: This function has been removed in favor of the `ne.resample()` API.

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
    return nef.apply_bernoulli_mask(input_tensor, p=p, returns=returns)


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
    return nef.random_flip(dim, *args, prob=prob)


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
    return nef.sample_image_from_labels(
        label_tensor=label_tensor,
        mean_range=mean_range,
        noise_std=noise_std,
    )


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
        Indices of leading non-spatial dimensions. Must be a contiguous sequence starting from 0.
        If None or `()`, assumes all dimensions are spatial and will add batch and channel
        dimensions for interpolation.
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
    input_tensor, orig_shape = batch_nonspatial(input_tensor, non_spatial_dims)
    resampled = nef.resample(
        input_tensor=input_tensor,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        antialias=antialias,
    )
    return unbatch_nonspatial(resampled, orig_shape)


def pad_to_multiple_of(
    input_tensor: torch.Tensor,
    multiple: int = 32,
    non_spatial_dims: Union[Sequence[int], None] = None,
    value: float = 0.0,
) -> torch.Tensor:
    """
    Pad spatial dimensions to multiples of a fixed value.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor to pad.
    multiple : int, default=32
        Positive value that each spatial output size must be divisible by.
    non_spatial_dims : Sequence[int] or None, default=None
        Leading dimensions that are not padded. If None, every dimension is treated as spatial.
    value : float, default=0.0
        Constant padding value.

    Returns
    -------
    torch.Tensor
        Padded tensor with the same non-spatial shape and spatial sizes divisible by `multiple`.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    >>> tensor = torch.ones(3, 5)
    >>> padded = ne.pad_to_multiple_of(tensor, multiple=4)
    >>> padded.shape
    torch.Size([4, 8])

    >>> tensor = torch.ones(2, 3, 5, 7)
    >>> padded = ne.pad_to_multiple_of(tensor, multiple=4, non_spatial_dims=(0, 1))
    >>> padded.shape
    torch.Size([2, 3, 8, 8])
    """
    input_tensor, orig_shape = batch_nonspatial(input_tensor, non_spatial_dims)
    padded = nef.pad_to_multiple_of(input_tensor, multiple=multiple, value=value)
    return unbatch_nonspatial(padded, orig_shape)


def mask_border(
    mask: torch.Tensor,
    thickness: int,
    border_mode: Literal["inner", "outer"] = "inner",
    non_spatial_dims: Union[Sequence[int], None] = None,
) -> torch.Tensor:
    """
    Compute the inner or outer border of a binary mask.

    Parameters
    ----------
    mask : torch.Tensor
        Binary mask tensor.
    thickness : int
        Border thickness in voxels.
    border_mode : {'inner', 'outer'}, default='inner'
        Whether to return voxels inside the mask boundary or outside the mask boundary.
    non_spatial_dims : Sequence[int] or None, default=None
        Leading dimensions that index independent masks. If None, every dimension is spatial.

    Returns
    -------
    torch.Tensor
        Border mask with the same shape and dtype as `mask`.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    >>> mask = torch.zeros(5, 5, dtype=torch.bool)
    >>> mask[1:4, 1:4] = True
    >>> border = ne.mask_border(mask, thickness=1)
    >>> border.shape
    torch.Size([5, 5])
    """
    mask, orig_shape = batch_nonspatial(mask, non_spatial_dims)
    border = nef.mask_border(mask, thickness=thickness, border_mode=border_mode)
    return unbatch_nonspatial(border, orig_shape)


def one_hot(
    label_tensor: torch.Tensor,
    num_classes: Union[int, None] = None,
    class_list: Union[Sequence[int], None] = None,
    non_spatial_dims: Union[Sequence[int], None] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert integer labels to one-hot channels.

    The class axis is inserted immediately after the leading non-spatial dimensions.

    Parameters
    ----------
    label_tensor : torch.Tensor
        Integer label tensor with shape `(*non_spatial, *spatial)`.
    num_classes : int or None, default=None
        Total number of classes. If None, it is inferred from `label_tensor.max() + 1`.
    class_list : Sequence[int] or None, default=None
        Class ids to keep in the output. If None, all classes are returned.
    non_spatial_dims : Sequence[int] or None, default=None
        Leading dimensions that are not class labels or spatial dimensions.
    dtype : torch.dtype, default=torch.float32
        Output dtype.

    Returns
    -------
    torch.Tensor
        One-hot tensor with shape `(*non_spatial, C, *spatial)`.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    >>> labels = torch.tensor([[0, 1], [2, 1]])
    >>> encoded = ne.one_hot(labels, num_classes=3)
    >>> encoded.shape
    torch.Size([3, 2, 2])

    >>> labels = torch.randint(0, 3, (2, 5, 5))
    >>> encoded = ne.one_hot(labels, num_classes=3, non_spatial_dims=(0,))
    >>> encoded.shape
    torch.Size([2, 3, 5, 5])
    """
    num_non_spatial, _ = parse_non_spatial_dims(non_spatial_dims, label_tensor.ndim)
    labels = label_tensor.long()

    if num_classes is None:
        num_classes = int(labels.max().item()) + 1

    encoded = F.one_hot(labels, num_classes=num_classes).to(dtype=dtype)
    encoded = encoded.movedim(-1, num_non_spatial)

    if class_list is None:
        return encoded

    classes = torch.as_tensor(class_list, device=label_tensor.device, dtype=torch.long)
    assert classes.numel() > 0, "class_list must contain at least one class id."
    assert int(classes.max().item()) < num_classes, "class_list contains an out-of-range class id."
    assert int(classes.min().item()) >= 0, "class_list contains a negative class id."

    return encoded.index_select(num_non_spatial, classes)


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
    return nef.filter_dim(tensor, dim=dim, verbose=verbose)


def parse_non_spatial_dims(
    non_spatial_dims: Union[Sequence[int], None],
    tensor_ndim: int
) -> Tuple[int, int]:
    """
    Validate and parse non_spatial_dims parameter.

    Parameters
    ----------
    non_spatial_dims : Sequence[int] or None
        Indices of leading non-spatial dimensions to vectorize over. Must be a contiguous
        sequence starting at 0. If None, assumes all dimensions are spatial.
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
    AssertionError
        If non_spatial_dims contains invalid indices or exceeds tensor dimensions.

    Examples
    --------
    >>> parse_non_spatial_dims(None, 3)
    (0, 3)
    >>> parse_non_spatial_dims((0,), 4)
    (1, 3)
    >>> parse_non_spatial_dims((0, 1, 2), 5)
    (3, 2)
    """
    if non_spatial_dims is None:
        return 0, tensor_ndim

    non_spatial_dims = tuple(non_spatial_dims)
    num_non_spatial = len(non_spatial_dims)

    assert num_non_spatial <= tensor_ndim, (
        f"non_spatial_dims has {num_non_spatial} elements but tensor only has "
        f"{tensor_ndim} dimensions"
    )

    for dim in non_spatial_dims:
        assert 0 <= dim < tensor_ndim, (
            f"non_spatial_dims contains invalid index {dim} for tensor with "
            f"{tensor_ndim} dimensions"
        )

    expected_dims = tuple(range(num_non_spatial))
    assert non_spatial_dims == expected_dims, (
        "non_spatial_dims must be a leading contiguous sequence starting at 0. "
        f"Expected {expected_dims}, got {non_spatial_dims}."
    )

    num_spatial = tensor_ndim - num_non_spatial

    return num_non_spatial, num_spatial


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
    return nef.gaussian_kernel(
        sigma=sigma,
        truncate=truncate,
        ndim=ndim,
        normalize=normalize,
        device=device,
        dtype=dtype,
    )


def gaussian_smoothing(
    input_tensor: torch.Tensor,
    sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
    truncate: Union[int, float, Sequence[Union[int, float]]] = 3,
    normalize: Union[Literal["sum", "gaussian"], None] = "sum",
    padding_mode: str = "constant",
    method: Literal["dense", "separable"] = "dense",
    non_spatial_dims: Union[Sequence[int], None] = None,
) -> torch.Tensor:
    """Apply Gaussian smoothing while preserving leading non-spatial dimensions.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor of shape ``[*non_spatial, *spatial]`` with one to three spatial dimensions.
    sigma : float, int, or sequence
        Gaussian standard deviation per spatial dimension.
    truncate : float, int, or Sequence[float or int], default=3
        Kernel radius in multiples of `sigma` per spatial dimension.
    normalize : {'sum', 'gaussian'} or None, default='sum'
        Kernel normalization mode.
    padding_mode : {'constant', 'reflect', 'replicate', 'circular'}, default='constant'
        Boundary padding applied before convolution.
    method : {'dense', 'separable'}, default='dense'
        Apply one multidimensional kernel or one one-dimensional kernel per spatial axis.
    non_spatial_dims : sequence[int] or None, default=None
        Leading dimensions that index independent tensors. If None, every dimension is spatial.

    Returns
    -------
    torch.Tensor
        Smoothed tensor with the same shape, dtype, and device.

    Examples
    --------
    >>> import torch
    >>> image = torch.rand(64, 64)
    >>> smoothed = gaussian_smoothing(image, sigma=2)
    >>> images = torch.rand(2, 3, 64, 64)
    >>> smoothed = gaussian_smoothing(images, sigma=2, non_spatial_dims=(0, 1))
    """
    input_tensor, original_shape = batch_nonspatial(input_tensor, non_spatial_dims)
    smoothed = nef.gaussian_smoothing(
        input_tensor=input_tensor,
        sigma=sigma,
        truncate=truncate,
        normalize=normalize,
        padding_mode=padding_mode,
        method=method,
    )
    return unbatch_nonspatial(smoothed, original_shape)
    return unbatch_nonspatial(smoothed, original_shape)


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
    AssertionError
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
    input_tensor, orig_shape = batch_nonspatial(input_tensor, non_spatial_dims)
    cropped = nef.crop(
        input_tensor=input_tensor,
        size=size,
        scale_factor=scale_factor,
        offset=offset,
    )
    return unbatch_nonspatial(cropped, orig_shape)


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
    return nef.clip(input_tensor, min=min, max=max)


def batch_nonspatial(
        tensor: torch.Tensor,
        non_spatial_dims: Union[Sequence[int], None]
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Flatten non-spatial dims into batch and add singleton channel for PyTorch ops.

    Prepares a tensor for PyTorch operations that require (B, C, *spatial) format. All non-spatial
    dimensions are flattened into the batch dimension, ensuring each element is processed
    independently. A singleton channel dimension is added.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape (*non_spatial, *spatial).
    non_spatial_dims : Sequence[int] or None
        Indices of dims to flatten into batch. These dimensions will be collapsed into a single
        batch dimension. If None, tensor is treated as pure spatial and singleton batch and
        channel dims are added.

    Returns
    -------
    batched : torch.Tensor
        Tensor with shape (batch_flat, 1, *spatial) where batch_flat is the product of all
        non-spatial dimension sizes.
    original_non_spatial_shape : tuple[int, ...]
        Original shape of non-spatial dimensions, needed for unbatch_nonspatial. Empty tuple
        if non_spatial_dims was None.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    # Pure spatial tensor
    >>> t = torch.randn(64, 64, 64)
    >>> batched, shape = ne.batch_nonspatial(t, non_spatial_dims=None)
    >>> batched.shape
    torch.Size([1, 1, 64, 64, 64])
    >>> shape
    ()

    # Single non-spatial dimension
    >>> t = torch.randn(10, 64, 64)
    >>> batched, shape = ne.batch_nonspatial(t, non_spatial_dims=(0,))
    >>> batched.shape
    torch.Size([10, 1, 64, 64])
    >>> shape
    (10,)

    # Multiple non-spatial dimensions
    >>> t = torch.randn(2, 3, 64, 64)
    >>> batched, shape = ne.batch_nonspatial(t, non_spatial_dims=(0, 1))
    >>> batched.shape
    torch.Size([6, 1, 64, 64])
    >>> shape
    (2, 3)

    # Arbitrary number of non-spatial dimensions
    >>> t = torch.randn(2, 3, 4, 5, 32, 32)
    >>> batched, shape = ne.batch_nonspatial(t, non_spatial_dims=(0, 1, 2, 3))
    >>> batched.shape
    torch.Size([120, 1, 32, 32])
    >>> shape
    (2, 3, 4, 5)

    See Also
    --------
    unbatch_nonspatial : Reverse operation to restore original shape.
    """
    num_non_spatial, _ = parse_non_spatial_dims(non_spatial_dims, tensor.ndim)
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


def unbatch_nonspatial(
        tensor: torch.Tensor,
        original_non_spatial_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Restore original non-spatial shape after batched PyTorch operation.

    Reverses batch_nonspatial by removing the singleton channel dimension
    and unflattening the batch dimension back to the original non-spatial shape.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor with shape (batch_flat, 1, *spatial) from a PyTorch operation.
    original_non_spatial_shape : tuple[int, ...]
        Original non-spatial shape from batch_nonspatial. Empty tuple means input was
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
    >>> batched, shape = ne.batch_nonspatial(t, non_spatial_dims=None)
    >>> restored = ne.unbatch_nonspatial(batched, shape)
    >>> restored.shape
    torch.Size([64, 64, 64])

    # Round-trip with non-spatial dims (spatial may change from operation)
    >>> t = torch.randn(2, 3, 64, 64)
    >>> batched, shape = ne.batch_nonspatial(t, non_spatial_dims=(0, 1))
    >>> # Simulate operation that changes spatial dims
    >>> result = batched[..., ::2, ::2]  # (6, 1, 32, 32)
    >>> restored = ne.unbatch_nonspatial(result, shape)
    >>> restored.shape
    torch.Size([2, 3, 32, 32])

    See Also
    --------
    batch_nonspatial : Prepare tensor for batched operations.
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


def random_smoothed_noise(
    shape: Sequence[int],
    sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
    magnitude: float = 1.0,
    non_spatial_dims: Union[Sequence[int], None] = None,
    normalize: Union[Literal["sum", "gaussian"], None] = "sum",
    device: Union[torch.device, None] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
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
        Smooth Gaussian noise with the specified shape, zero mean, and standard deviation
        equal to magnitude.

    Examples
    --------
    >>> import torch
    >>> import neurite as ne
    >>> # Generate 2D noise field (pure spatial)
    >>> noise_2d = ne.random_smoothed_noise(shape=(64, 64), sigma=2.0)
    >>> noise_2d.shape
    torch.Size([64, 64])

    >>> # Generate 3D noise with batch and channel dimensions
    >>> noise_3d = ne.random_smoothed_noise(
    ...     shape=(2, 3, 32, 32, 32),
    ...     sigma=3.0,
    ...     magnitude=2.0,
    ...     non_spatial_dims=(0, 1)
    ... )
    >>> noise_3d.shape
    torch.Size([2, 3, 32, 32, 32])

    >>> # Per-dimension sigma values
    >>> noise = ne.random_smoothed_noise(shape=(64, 64), sigma=[1.0, 2.0])
    """
    num_non_spatial, _ = parse_non_spatial_dims(
        non_spatial_dims=non_spatial_dims,
        tensor_ndim=len(shape)
    )
    non_spatial_shape = shape[:num_non_spatial]
    spatial_shape = shape[num_non_spatial:]

    batch_size = 1
    for dim_size in non_spatial_shape:
        batch_size *= dim_size

    noise = nef.random_smoothed_noise(
        shape=(batch_size, 1, *spatial_shape),
        sigma=sigma,
        magnitude=magnitude,
        normalize=normalize,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return unbatch_nonspatial(noise, tuple(non_spatial_shape))


def upsample_noise(
    shape: Sequence[int],
    scale: Union[float, int, Sequence[Union[float, int]]],
    non_spatial_dims: Union[Sequence[int], None] = None,
    device: Union[torch.device, None] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
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
    scale : float, int, or sequence
        Downsampling factor per spatial axis. Larger values produce smoother noise.
    non_spatial_dims : Sequence of int or None, default=None
        Indices of non-spatial dimensions:
        - None: tensor is pure spatial (*spatial,)
        - (0,): first dim is non-spatial (C, *spatial)
        - (0, 1): first two dims are non-spatial (B, C, *spatial)
    device : torch.device or None, default=None
        Device for tensor allocation.
    dtype : torch.dtype, default=torch.float32
        Data type for the noise field.
    generator : torch.Generator, optional
        Generator controlling random sampling.

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
    num_non_spatial, _ = parse_non_spatial_dims(
        non_spatial_dims=non_spatial_dims,
        tensor_ndim=len(shape),
    )
    non_spatial_shape = shape[:num_non_spatial]
    spatial_shape = shape[num_non_spatial:]

    batch_size = 1
    for dim_size in non_spatial_shape:
        batch_size *= dim_size

    noise = nef.upsample_noise(
        shape=(batch_size, 1, *spatial_shape),
        scale=scale,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return unbatch_nonspatial(noise, tuple(non_spatial_shape))


def fractal_noise(
    shape: Sequence[int],
    scales: Optional[Union[float, int, Sequence[Union[float, int, Sequence[float]]]]] = None,
    magnitude: float = 1.0,
    weights: Union[Sequence[float], None] = None,
    non_spatial_dims: Union[Sequence[int], None] = None,
    normalize: Union[Literal["sum", "gaussian"], None] = "sum",
    device: Union[torch.device, None] = None,
    method: Literal['blur', 'upsample'] = 'blur',
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    standardize: bool = True,
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
    scales : float, int, or sequence, optional
        Smoothing scale for each octave. An octave may contain one value per spatial axis:
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
    normalize : {'sum', 'gaussian'} or None, default='sum'
        How to normalize the Gaussian kernel. See `neurite.gaussian_kernel` for details.
        Only used when method='blur'.
    device : torch.device or None, default=None
        Device for tensor allocation.
    method : {'blur', 'upsample'}, default='blur'
        Noise generation method:
        - 'blur': Generate noise at full resolution and apply Gaussian smoothing
        - 'upsample': Generate coarse noise and upsample (faster, lower memory)
    dtype : torch.dtype, default=torch.float32
        Data type for the noise field.
    generator : torch.Generator, optional
        Generator controlling random sampling.
    standardize : bool, default=True
        Whether to standardize each output field to zero mean and ``magnitude`` deviation.

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
    num_non_spatial, _ = parse_non_spatial_dims(
        non_spatial_dims=non_spatial_dims,
        tensor_ndim=len(shape),
    )
    non_spatial_shape = shape[:num_non_spatial]
    spatial_shape = shape[num_non_spatial:]

    batch_size = 1
    for dim_size in non_spatial_shape:
        batch_size *= dim_size

    noise = nef.fractal_noise(
        shape=(batch_size, 1, *spatial_shape),
        scales=scales,
        magnitude=magnitude,
        weights=weights,
        normalize=normalize,
        device=device,
        method=method,
        dtype=dtype,
        generator=generator,
        standardize=standardize,
    )
    return unbatch_nonspatial(noise, tuple(non_spatial_shape))
