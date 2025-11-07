"""
Single tensor operations (no B, C dimension assumption)
"""

# Standard library imports
from typing import Union, Sequence

# Third party imports
import torch
import torch.nn.functional as F


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
    *segs: Sequence[torch.Tensor],
    smooth_numerator: float = 1e-12,
    smooth_denominator: float = 1e-12,
) -> torch.Tensor:
    """
    Compute Dice score over multiple segmentation maps.

    Shape-agnostic implementation that flattens all dimensions and computes
    a single Dice score over the entire tensors.

    Parameters
    ----------
    *segs : torch.Tensor
        Two or more segmentation tensors with the same shape and values in [0, 1].
    smooth_numerator : float, optional
        Smoothing constant added to the numerator. Default is 1e-12.
    smooth_denominator : float, optional
        Smoothing constant added to the denominator. Default is 1e-12.

    Returns
    -------
    torch.Tensor
        Scalar Dice score between 0 and 1.

    Examples
    --------
    >>> import torch
    # Compute dice for 2 segmentation tensors
    >>> seg1 = torch.rand(32, 32)
    >>> seg2 = torch.rand(32, 32)
    >>> score = dice(seg1, seg2)
    >>> print(score)
    tensor(0.4523)

    # Works with any shape
    >>> seg1 = torch.rand(64, 64, 64)
    >>> seg2 = torch.rand(64, 64, 64)
    >>> score = dice(seg1, seg2)
    >>> print(score)
    tensor(0.3891)
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

    # Flatten all dimensions
    segs_flat = [seg.flatten() for seg in segs]

    # Compute intersection and union
    intersection = torch.stack(segs_flat, dim=0).prod(dim=0).sum()
    union = torch.stack(segs_flat, dim=0).sum()

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
