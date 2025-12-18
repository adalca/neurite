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

# Standard library imports
from typing import Union, Literal, Type, Optional, Collection, Sequence, Type, Tuple, List
import inspect

# Third party imports
import torch
from torch import nn

# Custom imports
import neurite as ne


__all__ = [
    "bernoulli",
    "is_instantiated_normalization",
    "infer_linear_interpolation_mode",
    "downsampling_conv_blocks",
    "upsampling_conv_blocks",
    "build_normalization",
    "early_stopping",
]


def bernoulli(p: float = 0.5, shape: tuple = (1,)) -> torch.Tensor:
    """
    Sample from a Bernoulli distribution with a specified probability and shape.

    Parameters
    ----------
    p : float, optional
        Probability of realizing a successful trial (i.e., the probability of a 1) from the
        Bernoulli distribution. By default, 0.5. Must be in the range [0, 1].
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


def infer_linear_interpolation_mode(
    num_spatial: Literal[1, 2, 3]
) -> Literal['linear', 'bilinear', 'trilinear']:
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


def downsampling_conv_blocks(
    ndim: int,
    nb_features: Sequence[int],
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    padding_mode: Literal['zeros', 'replicate', 'reflect'] = 'zeros',
    normalizations: Union[str, nn.Module, None, Sequence[Union[str, nn.Module, None]]] = None,
    activations: Union[str, nn.Module, Sequence[Union[str, nn.Module, None]]] = "relu",
    pool_mode: str = "max",
    pool_kernel_size: int = 2,
    order: str = 'nca',
    return_skip: bool = False,
) -> tuple[nn.ModuleList, list[int]]:
    """
    Create an `nn.ModuleList` of downsampling conv blocks based the number of features per layer.

    Parameters
    ----------
    ndim : int
        Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
    nb_features : Sequence[int]
        Number of features at each level of the downsampling convs. Use `0` to indicate a
        pass-through level (pool only, no convolution). When `0` is used, the actual output
        channels equal the input channels (channels are preserved).
    kernel_size : int, default=3
        Size of the convolving kernel at each level.
    stride : int, default=1
        Stride of the convolution.
    padding : int, default=1
        Padding added to all sides of the input.
    normalizations : list, str, nn.Module, or None, default=None
        Normalization layers for each downsampling conv block.
    activations : list, str, nn.Module, or None, default='relu'
        Activation function for each downsampling conv block.
    pool_mode : str, default='max'
        Pooling mode ('max' or 'avg').
    pool_kernel_size : int, default=2
        Kernel size for pooling.
    order : str, default='nca'
        The order of operations in each downsampling conv block (normalization ->
        convolution -> activation). Each character in the string can be specified an arbitrary
        number of times in any order. Each character in the string represents one of the following:
        - `'c'`: Convolution
        - `'n'`: Normalization
        - `'a'`: Activation
    return_skip : bool, default=False
        If True, return skip connection features from each block's forward pass for use in
        UNet-style architectures.

    Returns
    -------
    tuple[nn.ModuleList, list[int]]
        A tuple of (blocks, actual_channels) where:
        - blocks: ModuleList of downsampling blocks (DownsampleConvBlock or DownsamplePassthrough)
        - actual_channels: List of actual output channel counts for each block, accounting for
          pass-through levels where channels are preserved

    Examples
    --------
    >>> # Standard usage
    >>> blocks, actual_ch = downsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[3, 16, 32],
    ...     kernel_size=3,
    ...     normalizations=["batch", "instance", "batch"],
    ...     activations="relu"
    ... )
    >>> actual_ch
    [16, 32]

    >>> # With pass-through (0) at first level
    >>> blocks, actual_ch = downsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[3, 0, 32],  # Skip conv at first level
    ...     return_skip=True
    ... )
    >>> actual_ch
    [3, 32]  # First level preserves input channels (3)
    """
    # Normalization layers
    if not isinstance(normalizations, list):
        normalizations = [normalizations] * len(nb_features)

    # Activation layers
    if not isinstance(activations, list):
        activations = [activations] * len(nb_features)

    # Init container for downsampling convs and track actual channels
    blocks = nn.ModuleList()

    # Important for pass through connections
    actual_channels = [nb_features[0]]  # Start with input channels

    # Make downsampling conv block and append to list of them
    for i in range(len(nb_features) - 1):
        in_ch = actual_channels[-1]  # Use actual channels from previous block
        out_ch = nb_features[i + 1]
        passthrough = (out_ch == 0)

        block = ne.nn.modules.DownsampleConvBlock(
            ndim=ndim,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            normalization=normalizations[i],
            activation=activations[i],
            pool_mode=pool_mode,
            pool_kernel_size=pool_kernel_size,
            order=order,
            return_skip=return_skip,
            passthrough=passthrough,
        )
        blocks.append(block)

        # Track actual output channels (passthrough preserves input channels)
        actual_channels.append(in_ch if passthrough else out_ch)

    # Return blocks and actual output channels (excluding input channels)
    return blocks, actual_channels[1:]


def upsampling_conv_blocks(
    ndim: int,
    nb_features: Sequence[int],
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    padding_mode: Literal['zeros', 'replicate', 'reflect'] = 'zeros',
    upsample_mode: Literal['linear', 'transposed', 'nearest'] = 'linear',
    upsample_kernel_size: int = 4,
    upsample_stride: int = 2,
    upsample_padding: int = 1,
    scale_factor: int = 2,
    normalizations: Union[str, nn.Module, None, Sequence[Union[str, nn.Module, None]]] = None,
    activations: Union[str, nn.Module, Sequence[Union[str, nn.Module, None]]] = "relu",
    order: str = 'nca',
    accepts_skip: bool = True,
    skip_channels: Union[Sequence[int], None] = None,
    in_channels: int | None = None,
) -> tuple[nn.ModuleList, list[int]]:
    """
    Create an `nn.ModuleList` of upsampling conv blocks based the number of features per layer/
    level.

    Parameters
    ----------
    ndim : int
        Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
    nb_features : Sequence[int]
        Number of features at each upsampling conv block. Use `0` to indicate a pass-through
        level (upsample + skip concat only, no convolution). When `0` is used, the actual
        output channels equal the upsampled channels plus skip channels (if accepts_skip=True).
    kernel_size : int, default=3
        Size of the convolving kernel at each level.
    stride : int, default=1
        Stride of the convolution at each level.
    padding : int, default=1
        Padding added to all sides of the input at each level.
    upsample_kernel_size : int, default=4
        Kernel size for the transposed convolution at each level.
    upsample_stride : int, default=2
        Stride for the transposed convolution at each upsampling conv block.
    upsample_padding : int, default=1
        Padding for the transposed convolution at each level.
    scale_factor : int, default=2
        Scale factor for interpolation upsampling (used when upsample_mode is not 'transposed').
    normalizations : list, str, nn.Module, or None, default=None
        Normalization layers for each upsampling conv block at each level. If a list, must have the
        same length as nb_features.
    activations : list, str, nn.Module, or None, default='relu'
        Activation functions for each upsampling conv block at each level. If a list, must have the
        same length as nb_features.
    order : str, default='nca'
        The order of operations in the block (normalization -> convolution ->
        activation). Each character in the string can be specified an arbitrary number of times
        in any order. Each character in the string represents one of the following:
        - `'c'`: Convolution
        - `'n'`: Normalization
        - `'a'`: Activation
    accepts_skip : bool, default=True
        If True, the blocks are configured to accept skip connections (UNet-style). This allows
        the blocks to concatenate skip features with the main input.
    skip_channels : Sequence[int] or None, default=None
        List of channel counts for skip connections, one per upsampling block. If provided and
        accepts_skip=True, enables asymmetric downsampling/upsampling architectures. If None and
        accepts_skip=True, assumes symmetric architecture (skip channels equal upsampled
        channels).
    in_channels : int or None, default=None
        Actual input channels to the first upsampling block (from bottleneck). If None, uses
        nb_features[0]. This is needed when nb_features[0] is 0 (pass-through) but the actual
        input comes from a bottleneck layer with non-zero channels.

    Returns
    -------
    tuple[nn.ModuleList, list[int]]
        A tuple of (blocks, actual_channels) where:
        - blocks: ModuleList of upsampling blocks (UpsampleConvBlock or UpsamplePassthrough)
        - actual_channels: List of actual output channel counts for each block, accounting for
          pass-through levels where output = upsampled + skip channels

    Notes
    -----
    - If `normalizations` or `activations` are a list, they must be the same length as the
    nb_features.

    Examples
    --------
    >>> # Standard usage
    >>> blocks, actual_ch = upsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[32, 16, 4],
    ...     upsample_kernel_size=4,
    ...     normalizations=["batch", "instance", "batch"],
    ...     activations="relu"
    ... )
    >>> actual_ch
    [16, 4, 4]

    >>> # With pass-through (0) at last level
    >>> blocks, actual_ch = upsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[32, 16, 0],  # Pass-through at last level
    ...     skip_channels=[16, 8, 4],
    ...     accepts_skip=True
    ... )
    >>> actual_ch
    [16, 0 + skip, ...]  # Last level: upsampled + skip channels
    """
    # Normalization layers
    if not isinstance(normalizations, list):
        normalizations = [normalizations] * len(nb_features)

    # Activation layers
    if not isinstance(activations, list):
        activations = [activations] * len(nb_features)

    # Init upsampling conv blocks container and track actual channels
    blocks = nn.ModuleList()
    # Use in_channels if provided, otherwise use nb_features[0]
    # This handles the case where nb_features[0] is 0 (pass-through) but actual input
    # comes from a bottleneck with non-zero channels
    initial_channels = in_channels if in_channels is not None else nb_features[0]
    actual_channels = [initial_channels]

    # make the number of features for the upsampling conv blocks
    nb_features_extended = [*nb_features, nb_features[-1]]

    # Validate skip_channels length if provided
    num_blocks = len(nb_features_extended) - 1
    assert skip_channels is None or len(skip_channels) == num_blocks, (
        f"skip_channels must have length {num_blocks} "
        f"(one per upsampling block), got {len(skip_channels)}"
    )

    # Make upsampling conv block and append to list of them
    for i in range(num_blocks):
        in_ch = actual_channels[-1]  # Use actual channels from previous block
        out_ch = nb_features_extended[i + 1]
        skip_ch = skip_channels[i] if skip_channels is not None else None
        passthrough = (out_ch == 0)

        block = ne.nn.modules.UpsampleConvBlock(
            ndim=ndim,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            upsample_mode=upsample_mode,
            upsample_kernel_size=upsample_kernel_size,
            upsample_stride=upsample_stride,
            upsample_padding=upsample_padding,
            scale_factor=scale_factor,
            normalization=normalizations[-i - 1],
            activation=activations[-i - 1],
            order=order,
            accepts_skip=accepts_skip,
            skip_channels=skip_ch,
            passthrough=passthrough,
        )
        blocks.append(block)

        if not passthrough:
            actual_out = out_ch
        elif accepts_skip:
            actual_out = in_ch + (skip_ch or in_ch)
        else:
            actual_out = in_ch
        actual_channels.append(actual_out)

    # Return blocks and actual output channels (excluding input channels)
    return blocks, actual_channels[1:]


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
) -> Union[nn.Module, None]:

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
    # Normalization object has been instantiated with parameters
    if ne.utils.is_instantiated_normalization(normalization_type):
        return normalization_type

    # Normalization object has been provided but not instantiated
    if isinstance(normalization_type, type) and issubclass(normalization_type, nn.Module):

        # Assume user provided a custom normalization class directly
        if num_features is None:
            raise ValueError("`num_features` must be specified for custom normalizations.")

        return normalization_type(
            num_features=num_features, eps=eps, affine=affine, **kwargs
        )

    # Handle known norm_types
    if normalization_type not in NORMALIZATION_MAP:

        raise ValueError(
            f"Invalid normalization_type '{normalization_type}'. Must be one of "
            f"{list(NORMALIZATION_MAP.keys())} or a custom nn.Module subclass."
        )

    # Batch and instance normalization require an input dimensionality
    if normalization_type in ("batch", "instance"):

        if ndim not in (1, 2, 3):

            raise ValueError(
                "For 'batch' or 'instance' normalization, ndim must be 1, 2, or 3."
            )

        # They also require the number of features
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


def early_stopping(
    loss_history: Sequence[float],
    patience: int = 100,
    threshold: float = 0.0,
    warm_start_steps: int = 0
) -> bool:
    """
    Check if training should stop based on loss history.

    Parameters
    ----------
    loss_history : Sequence[float]
        History of loss values (lower is better).
    patience : int
        Number of steps without improvement before stopping.
    threshold : float
        Minimum decrease to count as improvement.
    warm_start_steps : int
        Number of initial steps to skip.

    Returns
    -------
    bool
        True if training should stop.

    Examples
    --------
    >>> history = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
    >>> early_stopping(history, patience=3)
    True
    >>> early_stopping(history, patience=10)
    False
    """
    relevant = loss_history[warm_start_steps:]
    if len(relevant) <= patience:
        return False

    best_before = min(relevant[:-patience])
    best_recent = min(relevant[-patience:])

    return best_recent >= best_before - threshold
