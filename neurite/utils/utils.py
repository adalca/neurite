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
from typing import Union, List, Literal, Type, Optional, Collection
import inspect

# Third party imports
import torch
from torch import isin, nn

# Custom imports
import neurite as ne
from neurite.samplers import Sampler


__all__ = [
    "gaussian_kernel",
    "bernoulli",
    "make_range",
    "is_instantiated_normalization",
    "infer_linear_interpolation_mode",
    "build_normalization",
    "downsampling_conv_blocks",
    "upsampling_conv_blocks"
]


def gaussian_kernel(
    kernel_size: int = 3,
    sigma: Union[float, int] = 1,
    ndim: int = 3,
    nchannels: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """
    Create a {1D, 2D, 3D} Gaussian kernel.

    Parameters
    ----------
    kernel_size : int, optional
        Size of each dimension in the Gaussian kernel. Default is 3.
    sigma : float, int, or Sampler, optional
        Standard deviation of the Gaussian kernel. Default is 1.
    ndim : int
        Dimensionality of the gaussian kernel. Default is 3.

    Returns
    -------
    torch.Tensor
        Tensor representing the {1D, 2D, 3D} Gaussian kernel with batch and channel dimensions.

    Examples
    --------
    >>> import torch
    # Make the kernel!
    >>> gaussian_kernel_ = gaussian_kernel(3, 1, 3)
    # Print shape (should have batch and channel dimensions)
    >>> gaussian_kernel_.shape
    torch.Size([1, 1, 3, 3, 3])
    """

    # Create a coordinate grid centered at zero
    if isinstance(kernel_size, list):
        if len(kernel_size) != ndim:
            raise ValueError("If coords is a list, it must have length equal to `ndim`.")
        coords = [
            torch.arange(
                ks, device=device, dtype=dtype
            ).float() - (ks - 1) / 2
            for ks in kernel_size
        ]
    else:
        coords = [torch.arange(kernel_size, device=device, dtype=dtype).float() - (kernel_size - 1) / 2] * ndim
        kernel_size = [kernel_size] * ndim

    grid = torch.stack(
        torch.meshgrid(coords, indexing='ij'), -1
    ).to(device=device, dtype=dtype)

    # print(f'ndim: {ndim}, grid.shape: {grid.shape}, sigma: {sigma}')
    # Per-dimension sigma
    if isinstance(sigma, (float, int)):
        sigma = [sigma] * ndim

    elif isinstance(sigma, Collection):
        if len(sigma) != ndim:
            raise ValueError("If sigma is a collection, it must have length equal to `ndim`.")

    # Make the sigmas on device
    sigma = torch.tensor(sigma, device=device, dtype=dtype)

    # Calculate the Gaussian function. Make the kernel by integrating over all spatial dimensions.
    kernel = torch.exp(-0.5 * (grid ** 2 / sigma**2).sum(-1))

    # Normalize the kernel so that it sums to 1
    kernel /= kernel.sum()

    # Reshape to 5D tensor for conv3d
    kernel = kernel.view(1, 1, *kernel_size)

    # Repeat the kernel for each channel (depth-wise convolution)
    if nchannels > 1:
        kernel = kernel.repeat(nchannels, nchannels, *([1] * ndim))

    return kernel


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


def make_range(*args, **kwargs) -> tuple:
    """
    Creates a tuple specigying the bounds for a range of numbers `(min, max)`.

    Generate a tuple containing the min and max values for a range. The range can be defined through
    positional and/or keyword arguments. If only one positional argument is provided, it is
    interpreted as `max` with `min` defaulting to 0. Keyword arguments can be used to explicitly set
    `min` and/or `max`, overriding positional arguments.

    Parameters
    ----------
    min : int or float, optional
        The minimum value of the range.
    max : int or float, optional
        The maximum value of the range.

    Returns
    -------
    tuple of (int or float, int or float)
        A tuple containing the minimum and maximum values `(min, max)`.

    Examples
    --------
    >>> # Using two positional arguments
    >>> rng = range_endpoints(0, 19.7)
    >>> print(rng)
    (0, 19.7)
    >>> # Using one positional argument
    >>> rng = range_endpoints(5)
    >>> print(rng)
    (0, 5)
    >>> # Using keyword arguments
    >>> rng = range_endpoints(min=0.6, 1)
    >>> print(rng)
    (0.6, 1)
    """
    # Return arguments of type {Sampler, list, tuple} as-is
    for arg in args:
        if isinstance(arg, Sampler):
            return arg
        elif isinstance(arg, (list, tuple)):
            return arg

    # Return keyword arguments of type {Sampler, list, tuple} as-is
    for arg in kwargs.values():
        if isinstance(arg, Sampler):
            return arg
        elif isinstance(arg, (list, tuple)):
            return arg

    # Setting default values
    min, max = 0, 1
    # Handle positional arguments
    if len(args) == 2:
        min, max = args

    elif len(args) == 1:
        if isinstance(args[0], list | tuple):
            # if the argument is a list, unpack it:
            min, max = args[0]
        else:
            # if only one input arg is defined, interpret it as `max`
            min, max = 0, args[0]

    # Handle kwargs (if they exist)
    if 'min' in kwargs:
        min = kwargs['min']
    if 'max' in kwargs:
        max = kwargs['max']

    # Thrown an error if min is greater than max
    if max <= min:
        raise ValueError("`max` must be greater than `min`.")

    return (min, max)


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

    # Normalization object has been instantiated with parameters
    if ne.utils.is_instantiated_normalization(normalization_type):
        normalization = normalization_type
        return

    # Normalization object has been provided but not instantiated
    if isinstance(normalization_type, type) and issubclass(normalization_type, nn.Module):

        # Assume user provided a custom normalization class directly
        if num_features is None:
            raise ValueError("`num_features` must be specified for custom normalizations.")

        normalization = normalization_type(
            num_features=num_features, eps=eps, affine=affine, **kwargs
        )
        return

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


def downsampling_conv_blocks(
    ndim: int,
    nb_features: List[int],
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    padding_mode: Literal['zeros', 'replicate', 'reflect'] = 'zeros',
    normalizations: Union[str, nn.Module, None] = None,
    activations: Union[str, nn.Module, None] = "relu",
    pool_mode: str = "max",
    pool_kernel_size: int = 2,
    order: str = 'nca',
    return_residual: bool = False,
) -> nn.ModuleList:
    """
    Create an `nn.ModuleList` of downsampling conv blocks based the number of features per layer.

    Parameters
    ----------
    ndim : int
        Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
    nb_features : List[int]
        Number of features at each level of the downsampling convs.
    kernel_size : int, optional
        Size of the convolving kernel at each level. Default is 3.
    stride : int, optional
        Stride of the convolution. Default is 1.
    padding : int, optional
        Padding added to all sides of the input. Default is 1.
    normalizations : list, str, nn.Module, or None, optional
        Normalization layers for each downsampling conv block.
    activations : list, str, nn.Module, or None, optional
        Activation function for each downsampling conv block.
    pool_mode : str, optional
        Pooling mode ('max' or 'avg'). Default is 'max'.
    pool_kernel_size : int, optional
        Kernel size for pooling. Default is 2.
    order : str, optional
        The order of operations in each downsampling conv block. Default is 'nca' (normalization ->
        convolution -> activation). Each character in the string can be specified an arbitrary
        number of times in any order. Each character in the string represents one of the following:
        - `'c'`: Convolution
        - `'n'`: Normalization
        - `'a'`: Activation
    return_residual : bool
        Optionally return a residual (skip connection) from the output of the forward pass.

    Returns
    -------
    nn.ModuleList
        A list of downsampling convolutional blocks.

    Examples
    --------
    >>> downsampling_conv_blocks = downsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[3, 16, 32],
    ...     kernel_size=3,
    ...     normalizations=["batch", "instance", "batch"],
    ...     activations="relu"
    ... )
    >>> print(downsampling_conv_blocks)
    ModuleList(...)
    """

    # Normalization layers
    if not isinstance(normalizations, list):
        normalizations = [normalizations] * len(nb_features)

    # Activation layers
    if not isinstance(activations, list):
        activations = [activations] * len(nb_features)

    # Init container for downsampling convs
    downsampling_conv_blocks = nn.Sequential()

    # Make downsampling conv block and append to list of them
    for i in range(len(nb_features) - 1):

        downsampling_conv_block = ne.nn.modules.DownsampleConvBlock(
            ndim=ndim,
            in_channels=nb_features[i],
            out_channels=nb_features[i + 1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            normalization=normalizations[i],
            activation=activations[i],
            pool_mode=pool_mode,
            pool_kernel_size=pool_kernel_size,
            order=order,
            return_residual=return_residual,
        )

        downsampling_conv_blocks.append(downsampling_conv_block)

    return downsampling_conv_blocks


def upsampling_conv_blocks(
    ndim: int,
    nb_features: List[int],
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    padding_mode: Literal['zeros', 'replicate', 'reflect'] = 'zeros',
    upsample_mode: Literal['linear', 'transposed', 'nearest'] = 'linear',
    upsample_kernel_size: int = 4,
    upsample_stride: int = 2,
    upsample_padding: int = 1,
    normalizations: List[Union[str, nn.Module, None]] = None,
    activations: List[Union[str, nn.Module, None]] = None,
    order: str = 'nca',
    accepts_residuals: bool = True,
) -> nn.ModuleList:
    """
    Create an `nn.ModuleList` of upsampling conv blocks based the number of features per layer/
    level.

    Parameters
    ----------
    ndim : int
        Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
    nb_features : List[int]
        Number of features at each upsampling conv block.
    kernel_size : int, optional
        Size of the convolving kernel at each level. Default is 3.
    stride : int, optional
        Stride of the convolution at each level. Default is 1.
    padding : int, optional
        Padding added to all sides of the input at each level. Default is 1.
    upsample_kernel_size : int, optional
        Kernel size for the transposed convolution at each level. Default is 4.
    upsample_stride : int, optional
        Stride for the transposed convolution at each upsampling conv block. Default is 2.
    upsample_padding : int, optional
        Padding for the transposed convolution at each level. Default is 1.
    normalizations : list, str, nn.Module, or None, optional
        Normalization layers for each upsampling conv block at each level. If a list, must have the
        same length as nb_features.
    activations : list, str, nn.Module, or None, optional
        Activation functions for each upsampling conv block at each level. If a list, must have the
        same length as nb_features.
    order : str, optional
        The order of operations in the block. Default is 'nca' (normalization -> convolution ->
        activation). Each character in the string can be specified an arbitrary number of times
        in any order. Each character in the string represents one of the following:
        - `'c'`: Convolution
        - `'n'`: Normalization
        - `'a'`: Activation
    accepts_residuals : bool
        If True, the blocks are configured to accept residual connections. This doubles the
        expected number of input channels, allowing the blocks to concatenate skip features with
        the main input.

    Returns
    -------
    nn.ModuleList
        A list of upsampling convolutional blocks.

    Notes
    -----
    - If `normalizations` or `activations` are a list, they must be the same length as the
    nb_features.

    Examples
    --------
    >>> upsampling_conv_blocks = upsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[32, 16, 4],
    ...     upsample_kernel_size=4,
    ...     normalizations=["batch", "instance", "batch"],
    ...     activations="relu"
    ... )
    >>> print(upsampling_conv_blocks)
    ModuleList(...)
    """

    # Normalization layers
    if not isinstance(normalizations, list):
        normalizations = [normalizations] * len(nb_features)

    # Activation layers
    if not isinstance(activations, list):
        activations = [activations] * len(nb_features)

    # Init upsampling conv blocks container
    upsampling_conv_blocks = nn.Sequential()

    # make the number of features for the upsampling conv blocks
    nb_features = [*nb_features, nb_features[-1]]

    # Make upsampling conv block and append to list of them
    for i in range(len(nb_features) - 1):

        upsampling_conv_block = ne.nn.modules.UpsampleConvBlock(
            ndim=ndim,
            in_channels=nb_features[i],
            out_channels=nb_features[i + 1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            upsample_mode=upsample_mode,
            upsample_kernel_size=upsample_kernel_size,
            upsample_stride=upsample_stride,
            upsample_padding=upsample_padding,
            normalization=normalizations[-i],
            activation=activations[-i],
            order=order,
            accepts_residuals=accepts_residuals,
        )

        upsampling_conv_blocks.append(upsampling_conv_block)

    return upsampling_conv_blocks
