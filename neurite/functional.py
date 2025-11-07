"""
Single tensor operations (no B, C dimension assumption)
"""

# Standard library imports
from typing import Union

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
