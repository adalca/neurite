"""
Modules are simple operations containing learnable parameters. The `modules` module contains general
nD building blocks for neural networks.

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
implied. Seee License for the specific language governing permissions and limitations under
the License.
"""

# Standard library imports
from typing import Union, Type, Tuple, Literal, Optional, Sequence
import importlib

# Third party imports
import torch
from torch import nn
import torch.nn.functional as F

# Custom imports
import neurite as ne
import neurite.nn.functional as nef


class Dice(nn.Module):
    """
    Compute the Dice score between two segmentation tensors (e.g. ground truth, predictions, etc...)

    Examples
    --------
    # Example 1: Computing the hard dice score with binary seg maps
    >>> # Instantiate the dice score module
    >>> dice_module = ne.nn.modules.Dice()
    >>> # Randomly sample binary tensors with 3 batches and 4 channels
    >>> seg1 = torch.randint(0, 2, (3, 4, 128, 128))
    >>> seg2 = torch.randint(0, 2, (3, 4, 128, 128))
    >>> # Compute the dice score and return
    >>> dice_module(seg1, seg2)
    tensor([[0.5003]])

    # Example 2: Computing the soft dice score with continuious seg maps and no reduction
    >>> dice_module = Dice(reduction=None)
    >>> # Randomly sample continuious "logits"
    >>> seg1 = torch.randn(3, 4, 128, 128)
    >>> seg2 = torch.randn(3, 4, 128, 128)
    >>> # Activation functions
    >>> seg1 = torch.sigmoid(seg1)
    >>> seg2 = torch.sigmoid(seg2)
    >>> # Compute the dice score and return
    >>> dice_module(seg1, seg2)
    tensor([[0.4982, 0.5022, 0.4984, 0.5024],
            [0.5016, 0.5035, 0.5021, 0.5001],
            [0.5001, 0.4998, 0.4990, 0.4996]])
    """

    def __init__(
        self,
        smooth_numerator: float = 1e-12,
        smooth_denominator: float = 1e-12,
        reduction: str = 'mean',
        reduction_dim: Union[int, Tuple] = (0, 1),
        keepdims: bool = True,
    ) -> None:
        """
        Initialize `Dice`.

        Parameters
        ----------
        smooth_numerator : float, default=1e-12
            Smoothing constant added to the numerator.
        smooth_denominator : float, default=1e-12
            Smoothing constant added to the denominator.
        reduction : str, default='mean'
            The type of reduction to apply. Supported values for multidimensional reductions are:
            'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'; for single-dimension
            reductions: 'argmin', 'argmax', and all multidimensionals.
        reduction_dim : int or tuple of ints, default=(0, 1)
            Dimension(s) over which to apply the reduction. For multidimensional reductions, pass a
            tuple of dimensions; for single-dimension reductions, pass an integer.
        keepdims : bool, default=True
            Whether to retain reduced dimensions as a singleton.
        """
        super().__init__()

        self.smooth_numerator = smooth_numerator
        self.smooth_denominator = smooth_denominator
        self.reduction = reduction
        self.reduction_dim = reduction_dim
        self.keepdims = keepdims

    def forward(self, *segs) -> torch.Tensor:
        """
        Compute the Dice coefficient between two or more segmentation tensors.

        Parameters
        ----------
        *segs : torch.Tensor
            Two or more segmentation tensors of shape (B, C, *spatial_dims) with values in [0, 1].

        Returns
        -------
        torch.Tensor
            The computed Dice coefficient, optionally reduced according to object initialization.
        """
        return nef.dice(
            *segs,
            smooth_numerator=self.smooth_numerator,
            smooth_denominator=self.smooth_denominator,
            reduction=self.reduction,
            reduction_dim=self.reduction_dim,
            keepdims=self.keepdims
        )


class Activation(nn.Module):
    """
    Dynamically construct an activation/nonlinearity based on the specified type.
    """
    def __init__(
        self,
        activation_type: Union[str, Type[nn.Module], None] = None,
        inplace: bool = True,
        negative_slope: Union[float, int] = 0.01,
        alpha: Union[float, int] = 1.0
    ) -> None:
        """
        Initialize `Activation`.

        Parameters
        ----------
        activation_type : str, Type[nn.Module], or None, default=None
            Type of activation function. Supported values: 'relu', 'leaky_relu', 'elu', or an
            nn.Module class/instance.
        inplace : bool, default=True
            Whether to perform the operation in-place.
        negative_slope : float, default=0.01
            Negative slope for 'leaky_relu'.
        alpha : float, default=1.0
            Alpha value for 'elu'.
        """
        super(Activation, self).__init__()

        if activation_type is None:
            self.activation = nn.Identity()

        elif activation_type == "None":
            self.activation = nn.Identity()

        elif isinstance(activation_type, torch.nn.Module):
            self.activation = activation_type

        elif isinstance(activation_type, type) and issubclass(activation_type, nn.Module):
            self.activation = activation_type()

        elif activation_type == "relu":
            self.activation = nn.ReLU(inplace=inplace)

        elif activation_type == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

        elif activation_type == "elu":
            self.activation = nn.ELU(alpha=alpha, inplace=inplace)

        elif "." in activation_type:
            module_name, _, cls_name = activation_type.rpartition(".")
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            if issubclass(cls, nn.Module):
                self.activation = cls()
            else:
                self.activation = nn.Identity()

        else:
            raise ValueError(
                f"Unsupported activation_type '{activation_type}'. "
                f"Supported types: 'relu', 'leaky_relu', 'elu'."
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the activation/nonlinearity.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Input tensor with activation/nonlinearity applied.
        """
        return self.activation(input_tensor)


class ConvBlock(nn.Sequential):
    """
    Convolutional block comprising a conv, and optionally, an activation and/or normalization.

    The default sequence of operations in this block is:

    1. **Convolution**: Apply an nD convolution over the input.
    2. **Normalization**: Normalize the output of the conv.
    3. **Activation Function**: Introduce non-linearity to the model.

    Attributes
    ----------
    conv : nn.Conv*d
        The convolutional layer.
    batch_norm : nn.BatchNorm*d
        The batch normalization layer.
    activation : nn.Module
        The activation function.

    Examples
    --------
    >>> import torch.nn as nn
    >>> conv_block = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU()
        )
    >>> input_tensor = torch.randn(16, 64, 32, 32)
    >>> output = conv_block(input_tensor)
    >>> print(output.shape)
    torch.Size([16, 128, 32, 32])
    """
    conv_dim_map = {1: '1d', 2: '2d', 3: '3d'}

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        padding_mode: Literal['zeros', 'replicate', 'reflect'] = 'zeros',
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        normalization: Union[str, nn.Module, None] = None,
        activation: Union[Sequence, str, nn.Module, None] = None,
        order: str = 'cna',
    ):
        """
        Initialize `ConvBlock`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple, default=3
            Size of the convolving kernel.
        stride : int or tuple, default=1
            Stride of the convolution.
        padding : int or tuple, default=1
            Padding added to all sides of the input.
        padding_mode : {'zeros', 'replicate', 'reflect'}, default='zeros'
            Padding mode for the convolution.
        dilation : int or tuple, default=1
            Spacing between kernel elements. Every `dilation`-th element is used.
        groups : int, default=1
            Number of blocked connections from input to output channels.
        bias : bool, default=True
            If True, a learnable bias is added to the output.
        normalization : str, nn.Module, or None, default=None
            Defines the normalization layer. Can be one of:
            - A string: Supported options are 'batch', 'instance', 'layer', or 'group'.
            - A `Normalization` module: Instantiated or uninstantiated `Normalization` layer.
                e.g. nn.InstanceNorm3d(16) or nn.InstanceNorm3d
            - `None`: No normalization is applied.
        activation : Sequence, str, nn.Module, or None, default=None
            Defines the activation layer. Can be one of:
            - A string: Supported options are 'relu', 'leaky_relu', or 'elu'.
            - A Sequence: A sequence containing any of these optinons. Must have the same number
                of elements as the number of activations specified in `order`.
            - A `nn.Module`: Instantiated or uninstantiated activation module.
                e.g. nn.Sigmoid(), nn.Sigmoid
            - `None`: No activation is applied.
        order : str, default='cna'
            The order of operations in the block (normalization -> convolution -> activation).
            Each character in the string represents one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation

        Examples
        --------
        ### Basic usage with default options
        >>> conv_block = ConvBlock(
                ndim=2,
                in_channels=16,
                out_channels=32,
                normalization="batch",
                activation="relu"
            )
        >>> input_tensor = torch.randn(1, 16, 64, 64)
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])

        ### Use pre-initialized `Normalization` and `Activation` modules
        >>> norm_layer = nn.BatchNorm2d(32)
        >>> activation_layer = nn.ReLU()
        >>> conv_block = ConvBlock(
                ndim=2,
                in_channels=16,
                out_channels=32,
                normalization=norm_layer,
                activation=activation_layer
            )
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])

        ### Initialize `ConvBlock` without normalizations nor activations
        >>> conv_block = ConvBlock(
            ndim=2,
            in_channels=16,
            out_channels=32,
            normalization=None,
            activation=None
        )
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])
        """

        super().__init__()

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = list(order)

        valid_operations = {'c', 'n', 'a'}
        if not set(order).issubset(valid_operations):
            raise ValueError(f"Invalid order. Must be a subset of {valid_operations}.")

        if ndim not in ConvBlock.conv_dim_map:
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3.")

        n_activations = order.count('a')
        if not isinstance(activation, (list, tuple)):
            activation = [activation] * n_activations
        else:
            assert len(activation) == n_activations, (
                "The total number of activations passed to `activation` must be the same number "
                f"defined in `order`. Got activation={activation}, order={order}"
            )

        conv_cls_name = f"Conv{ConvBlock.conv_dim_map[ndim]}"
        conv_cls = getattr(nn, conv_cls_name)

        # Build layers
        layers = nn.ModuleDict()
        conv_id, norm_id, act_id = 0, 0, 0

        for operation in self.order:
            if operation == 'c':
                layers[f"conv{conv_id}"] = conv_cls(
                    in_channels, out_channels, kernel_size, stride,
                    padding, dilation, groups, bias, padding_mode=padding_mode
                )
                in_channels = out_channels
                conv_id += 1

            elif operation == 'n' and normalization is not None:
                layers[f'normalization{norm_id}'] = ne.utils.build_normalization(
                    normalization_type=normalization,
                    ndim=ndim,
                    num_features=in_channels,
                )
                norm_id += 1

            elif operation == 'a' and activation is not None:
                layers[f'activation{act_id}'] = Activation(activation[act_id])
                act_id += 1

        for name, layer in layers.items():
            self.add_module(name, layer)


class TransposedConv(nn.Module):
    """
    Dynamically construct a transposed convolution {`ConvTranspose1d`,
    `ConvTranspose2d`, `ConvTranspose3d`} based on the number of input dimensions.

    TransposeConv can lead to checkerboard artifacts. Consider using linear interpolation instead.
    https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        Initialize `TransposedConv`.

        Parameters
        ----------
        ndim : int
            Spatial dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
            - 1: Uses `torch.nn.ConvTranspose1d` and expects input tensors of shape `(N, C, L)`,
            where `N` is the batch size, `C` is the number of input channels, and `L` is the length
            of the input sequence.

            - 2: Uses `torch.nn.ConvTranspose2d` and expects input tensors of shape `(N, C, H, W)`,
            where `H` and `W` are the spatial dimensions of the input image or feature map.

            - 3: Uses `torch.nn.ConvTranspose3d` and expects input tensors of shape
            `(N, C, D, H, W)`, where `D`, `H`, and `W` are the spatial dimensions of the input
            image or feature map.

        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple, default=4
            Size of the convolving kernel.
        stride : int or tuple, default=2
            Stride of the convolution.
        padding : int or tuple, default=1
            Padding added to all sides of the input.
        output_padding : int or tuple, default=0
            Additional size added to one side of each dimension in the output shape.
        dilation : int or tuple, default=1
            Spacing between kernel elements.
        groups : int, default=1
            Number of blocked connections from input to output channels.
        bias : bool, default=True
            If True, a learnable bias is added to the output.
        """
        super(TransposedConv, self).__init__()

        conv_dim_map = {1: '1d', 2: '2d', 3: '3d'}

        if ndim not in conv_dim_map:
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3.")

        conv_cls_name = f"ConvTranspose{conv_dim_map[ndim]}"
        conv_cls = getattr(nn, conv_cls_name)

        self.conv = conv_cls(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transposed convolution.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after transposed convolution.
        """
        return self.conv(input_tensor)


class Pool(nn.Module):
    """
    nD Pooling layer.

    Attributes
    ----------
    pool : nn.Module
        The pooling operation to apply. It is one of `MaxPool`, `AvgPool`,
        or `LPPool` for 1D, 2D, or 3D inputs.
    """

    def __init__(self, ndim: int, pool_mode: str = 'max', kernel_size=2):
        """
        Initialize `Pool`.

        Parameters
        ----------
        ndim : int
            The spatial dimensionality of the pooling operation. Must be 1, 2, or 3.
        pool_mode : str, default='max'
            The pooling mode to use. Options are 'max' for max pooling, 'avg' for average pooling,
            and 'lp' for LP pooling.
        kernel_size : int or tuple, default=2
            The size of the pooling kernel.
        """
        super(Pool, self).__init__()

        pool_map = {'max': 'MaxPool', 'avg': 'AvgPool', 'lp': 'LPPool'}
        if pool_mode not in pool_map:
            raise ValueError(f"Unsupported pool_mode={pool_mode}. Must be `max`, `avg`, or `lp`.")

        pool_dim_map = {1: '1d', 2: '2d', 3: '3d'}
        if ndim not in pool_dim_map:
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3.")

        pool_cls_name = f"{pool_map[pool_mode]}{pool_dim_map[ndim]}"
        pool_cls = getattr(nn, pool_cls_name)
        self.pool = pool_cls(kernel_size=kernel_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to be pooled.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the pooling operation.
        """
        return self.pool(input_tensor)


class DownsampleConvBlock(nn.Module):
    """
    Apply `ConvBlock` followed by `Pool` to extract features and reduce spatial shape.

    When `passthrough=True`, skips the convolution and only pools. This is useful for
    UNet architectures where you want to skip convolutions at certain levels (e.g., to
    avoid expensive full-resolution convolutions).

    Attributes
    ----------
    conv_block : ConvBlock or None
        The convolutional block applying a series of convolutions, normalization, and activation.
        None when passthrough=True.
    pool : Pool
        The pooling layer to downsample the feature maps.
    passthrough : bool
        If True, skip convolution and only pool.
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        padding_mode: Literal['zeros', 'replicate', 'reflect'] = 'zeros',
        normalization: Union[str, nn.Module, None] = None,
        activation: Union[str, nn.Module, None] = "relu",
        pool_mode: str = "max",
        pool_kernel_size: int = 2,
        order: str = 'nca',
        return_skip: bool = False,
        passthrough: bool = False,
    ):
        """
        Initialize `DownsampleConvBlock`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
        in_channels : int
            Number of input channels. Ignored when passthrough=True.
        out_channels : int
            Number of output channels. Ignored when passthrough=True.
        kernel_size : int, default=3
            Size of the convolving kernel.
        stride : int, default=1
            Stride of the convolution.
        padding : int, default=1
            Padding added to all sides of the input.
        padding_mode : {'zeros', 'replicate', 'reflect'}, default='zeros'
            Padding mode for the convolution.
        normalization : str, nn.Module, or None, default=None
            Normalization type.
        activation : str, nn.Module, or None, default='relu'
            Activation type.
        pool_mode : str, default='max'
            Pooling mode ('max' or 'avg').
        pool_kernel_size : int, default=2
            Kernel size for pooling.
        order : str, default='nca'
            The order of operations in the block (normalization -> convolution -> activation).
            Each character in the string can be specified an arbitrary number of times in any order.
            Each character in the string represents one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation
        return_skip : bool, default=False
            If True, return skip connection features from the forward pass for use in UNet-style
            architectures.
        passthrough : bool, default=False
            If True, skip convolution and only pool. Output channels equal input channels.
            When True, in_channels and out_channels are ignored.
        """
        super().__init__()
        self.return_skip = return_skip
        self.passthrough = passthrough

        if not passthrough:
            self.conv_block = ConvBlock(
                ndim=ndim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                normalization=normalization,
                activation=activation,
                order=order,
                padding_mode=padding_mode,
            )
        else:
            self.conv_block = None

        self.pool = Pool(ndim=ndim, pool_mode=pool_mode, kernel_size=pool_kernel_size)

    def forward(self, input_tensor: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the downsampling convolution.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Downsampled tensor after applying convolution (if not passthrough) and pooling.
            If return_skip=True, returns (pooled, skip) where skip is pre-pool features.
        """
        if self.passthrough:
            conv_result = input_tensor
        else:
            conv_result = self.conv_block(input_tensor)

        pooled_result = self.pool(conv_result)

        if self.return_skip:
            return pooled_result, conv_result
        return pooled_result


class UpsampleConvBlock(nn.Module):
    """
    Apply upsampling followed by `ConvBlock` to increase spatial shape and extract features.

    When `passthrough=True`, skips the convolution and only upsamples (+ concatenates skip
    if accepts_skip=True). This is useful for UNet architectures where you want to skip
    convolutions at certain levels.

    Attributes
    ----------
    upsample : TransposedConv or nn.Upsample
        Module to be used for upsampling. One of:
         - `neurite.modules.TransposedConv`: nD transposed convolution.
         - `torch.nn.Upsample`: Interpolation upsampler from PyTorch.
    conv_block : ConvBlock or None
        The convolutional block applying a series of convolutions, normalizations, and activations.
        None when passthrough=True.
    passthrough : bool
        If True, skip convolution after upsampling.
    accepts_skip : bool
        If True, concatenate skip connection in forward pass.
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        padding_mode: Literal['zeros', 'replicate', 'reflect'] = 'zeros',
        upsample_mode: Literal['linear', 'transposed', 'nearest'] = 'linear',
        upsample_kernel_size: int = 4,
        upsample_stride: int = 2,
        upsample_padding: int = 1,
        scale_factor: int = 2,
        normalization: Union[str, nn.Module, None] = None,
        activation: Union[str, nn.Module, None] = "relu",
        order: str = 'nca',
        accepts_skip: bool = True,
        skip_channels: Union[int, None] = None,
        passthrough: bool = False,
    ):
        """
        Initialize `UpsampleConvBlock`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels. Ignored when passthrough=True.
        kernel_size : int, default=3
            Size of the convolving kernel.
        stride : int, default=1
            Stride of the convolution.
        padding : int, default=1
            Padding added to all sides of the input.
        padding_mode : {'zeros', 'replicate', 'reflect'}, default='zeros'
            Padding mode for the convolution.
        upsample_mode : {'linear', 'transposed', 'nearest'}, default='linear'
            Upsampling mode.
        upsample_kernel_size : int, default=4
            Kernel size for the transposed convolution.
        upsample_stride : int, default=2
            Stride for the transposed convolution.
        upsample_padding : int, default=1
            Padding for the transposed convolution.
        scale_factor : int, default=2
            Scale factor for upsampling.
        normalization : str, nn.Module, or None, default=None
            Normalization type.
        activation : str, nn.Module, or None, default='relu'
            Activation type.
        order : str, default='nca'
            The order of operations in the block (normalization -> convolution -> activation).
            Each character in the string can be specified an arbitrary number of times in any order.
            Each character in the string represents one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation
        accepts_skip : bool, default=True
            If True, the block is configured to accept skip connections (UNet-style). This allows
            the block to concatenate skip features with the main input.
        skip_channels : int or None, default=None
            Number of channels in the skip connection. If provided and accepts_skip=True,
            the actual concatenated input will be in_channels + skip_channels. If None and
            accepts_skip=True, defaults to in_channels (symmetric assumption).
        passthrough : bool, default=False
            If True, skip convolution and only upsample (+ concat skip if accepts_skip=True).
            Output channels = in_channels + skip_channels (if accepts_skip) or in_channels.
            When True, out_channels is ignored.
        """
        super().__init__()
        self.passthrough = passthrough
        self.accepts_skip = accepts_skip

        if upsample_mode == 'transposed':
            self.upsample = TransposedConv(
                ndim=ndim,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=upsample_kernel_size,
                stride=upsample_stride,
                padding=upsample_padding
            )
        else:
            if upsample_mode == 'linear':
                upsample_mode = ne.utils.infer_linear_interpolation_mode(ndim)

            align = None if upsample_mode == 'nearest' else True
            self.upsample = nn.Upsample(
                scale_factor=scale_factor,
                mode=upsample_mode,
                align_corners=align
            )

        if not passthrough:
            # Compute input channels to conv (after skip concatenation)
            conv_in_channels = in_channels
            if accepts_skip:
                if skip_channels is not None:
                    conv_in_channels += skip_channels
                else:
                    # Backward compatibility when assuming symmetric architecture
                    conv_in_channels += in_channels

            # Build convolutional block
            self.conv_block = ConvBlock(
                ndim=ndim,
                in_channels=conv_in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                normalization=normalization,
                activation=activation,
                order=order,
                padding_mode=padding_mode,
            )
        else:
            self.conv_block = None

    def forward(
        self,
        input_tensor: torch.Tensor,
        skip: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """
        Forward pass of the upsampling convolutional block.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.
        skip : torch.Tensor, optional
            Skip connection features from downsampling path to concatenate with upsampled input.

        Returns
        -------
        torch.Tensor
            Upsampled tensor after applying upsampling operation and conv blocks (if not
            passthrough). If passthrough=True, returns upsampled + skip concatenated (if
            accepts_skip=True).
        """
        features = self.upsample(input_tensor)

        if self.accepts_skip and isinstance(skip, torch.Tensor):
            features = torch.cat([features, skip], dim=1)

        if self.passthrough:
            return features
        else:
            return self.conv_block(features)


class RescaleValues(nn.Module):
    """
    Scale each element of the input tensor by a multiplicative factor.
    """

    def __init__(self, scale_factor: Union[float, int]):
        """
        Initialize the `RescaleValues` module.

        Parameters
        ----------
        scale_factor : float or int
            Factor by which to rescale the values of the input tensor.
        """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RescaleValues` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Tensor to be resacled.

        Returns
        -------
        torch.Tensor
            Rescaled tensor.
        """
        return input_tensor * self.scale_factor


class Resize(nn.Module):
    """
    Resize the input tensor.
    """
    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
        antialias: bool = False,
    ):
        """
        Initialize the `Resize` module.

        Parameters
        ----------
        size : int, Tuple[int, int], or None, default=None
            The desired output size. If None, uses `scale_factor`.
        scale_factor : float, Tuple[float, float], or None, default=None
            Scaling factor for resizing. If None, uses `size`.
        mode : {'linear', 'nearest', 'bicubic', 'area', 'nearest-exact'}, default='linear'
            Interpolation mode for upsampling. When 'linear' is specified, the mode is
            automatically converted to 'linear', 'bilinear', or 'trilinear' based on the
            input tensor's spatial dimensionality.
        align_corners : bool or None, default=None
            Alignment for "linear", "bilinear", or "trilinear" modes.
        recompute_scale_factor : bool or None, default=None
            If True, recomputes the scale factor for interpolation.
        antialias : bool, default=False
            Applies anti-aliasing if `scale_factor` < 1.0.

        Examples
        --------
        >>> # Get a random tensor ~N(0, 1)
        >>> input_tensor = torch.randn(1, 1, 32, 32, 32)

        ### Resize with fixed `scale_factor`
        >>> resize_module = Resize(scale_factor=2)
        >>> resized_tensor = resize_module(input_tensor)
        >>> print(resized_tensor.shape)
        torch.Size([1, 1, 64, 64, 64])

        ### Resize with a sampled `scale_factor`
        >>> resize_module = Resize(scale_factor=Uniform(0.5, 4))
        >>> resized_tensor = resize_module(input_tensor)
        >>> print(resized_tensor)
        torch.Size([1, 1, 74, 74, 74])

        ### Resize to a specific shape
        >>> resize_module = Resize(size=(96, 96, 96))
        >>> resized_tensor = resize_module(input_tensor)
        >>> print(resized_tensor)
        torch.Size([1, 1, 96, 96, 96])

        Notes
        -----
        - This class assumes the input tensor has batch and channel dimensions.
        - It is not possible to define `size` and `scale_factor` simultaneously. Only one can be
        defined for a given instatntiation of `Resize`.
        - When defining `size` do not include batch or channel dimensions, only spatial dims.
        """
        super().__init__()
        if size is None and scale_factor is None:
            scale_factor = 1

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `Resize` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to be resized. Must have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            The resized tensor.
        """
        # Infer interpolation mode for linear interpolation
        mode = self.mode
        if mode == 'linear':
            spatial_ndim = input_tensor.ndim - 2
            mode = ne.utils.infer_linear_interpolation_mode(spatial_ndim)

        resized_tensor = F.interpolate(
            input=input_tensor,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,
        )

        return resized_tensor


class SoftQuantize(nn.Module):
    """
    Map continuous values to discrete bins.

    Map continuous values to discrete bins while retaining some smoothness/continuity
    which is parametrized by a softening parameter. It is especially useful in the context of
    machine learning, where it is desirable to have a differentiable version of a quantized
    quantity, allowing for backprop. Hard quantization is non-differentiable and creates gradients
    of zero, making gradient-based optimization impossible.
    """

    def __init__(
        self,
        nb_bins: int = 16,
        softness: Union[float, int] = 1.0,
        min_clip: Union[float, int] = -float('inf'),
        max_clip: Union[float, int] = float('inf'),
        return_log: bool = False,
    ):
        """
        Initialize `SoftQuantize`.

        Parameters
        ----------
        nb_bins : int, default=16
            The number of discrete bins to softly quantize the input values into.
        softness : float or int, default=1.0
            The softness factor for quantization. A higher value gives smoother quantization.
        min_clip : float or int, default=-inf
            Clip data lower than this value before calculating bin centers.
        max_clip : float or int, default=inf
            Clip data higher than this value before calculating bin centers.
        return_log : bool, default=False
            Optionally return the log of the softly quantized tensor.

        Examples
        --------
        >>> # Make 3D tensor ~N(0, 1).
        >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
        >>> # Initialize & apply the SoftQuantize instance.
        >>> soft_quantizer = SoftQuantize(nb_bins=4, softness=0.5)
        >>> softly_quantized_tensor = soft_quantizer(input_tensor)
        >>> # Visualize the softly quantized tensor.
        >>> plt.imshow(softly_quantized_tensor[0, 0, 16])

        """
        super().__init__()
        self.nb_bins = nb_bins
        self.softness = softness
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.return_log = return_log

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of `SoftQuantize`.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor to softly quantize.

        Returns
        -------
        torch.Tensor
            Softly quantized tensor with the same dimensions as `input_tensor`.
        """
        return ne.utils.soft_quantize(
            input_tensor=input_tensor,
            nb_bins=self.nb_bins,
            softness=self.softness,
            min_clip=self.min_clip,
            max_clip=self.max_clip,
            return_log=self.return_log
        )


class MSE(nn.Module):
    """
    Calculate the mean squared error (MSE).
    """

    def __init__(self):
        """
        Initialize `MSE`.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE between two tensors.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The tensor representing the model's prediction(s).
        target_tensor : torch.Tensor
            The target or ground truth values.

        Returns
        -------
        torch.Tensor
            The mean squared error between `input_tensor` and `target_tensor`.
        """
        return nef.mse(tensor1=input_tensor, tensor2=target_tensor)


class NCC(nn.Module):
    """
    Compute local normalized cross-correlation (NCC) between two tensors.

    Examples
    --------
    # Example 1: Computing NCC with default reduction
    >>> ncc_module = NCC()
    >>> t1 = torch.rand(2, 1, 64, 64)
    >>> t2 = torch.rand(2, 1, 64, 64)
    >>> score = ncc_module(t1, t2)
    >>> print(score.shape)
    torch.Size([1, 1])

    # Example 2: Computing NCC without reduction
    >>> ncc_module = NCC(reduction=None)
    >>> t1 = torch.rand(2, 3, 64, 64)
    >>> t2 = torch.rand(2, 3, 64, 64)
    >>> score = ncc_module(t1, t2)
    >>> print(score.shape)
    torch.Size([2, 3])
    """

    def __init__(
        self,
        window_size: Union[int, Sequence[int]] = 9,
        eps: float = 1e-5,
        reduction: Union[str, None] = 'mean',
        reduction_dim: Union[int, Tuple[int, ...]] = (0, 1),
        keepdims: bool = True,
    ) -> None:
        """
        Initialize `NCC`.

        Parameters
        ----------
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
        """
        super().__init__()

        self.window_size = window_size
        self.eps = eps
        self.reduction = reduction
        self.reduction_dim = reduction_dim
        self.keepdims = keepdims

    def forward(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NCC between two tensors.

        Parameters
        ----------
        tensor1 : torch.Tensor
            First input tensor with shape (B, C, *spatial_dims).
        tensor2 : torch.Tensor
            Second input tensor with same shape as tensor1.

        Returns
        -------
        torch.Tensor
            NCC values (squared correlation coefficients) in range [0, 1].
            Shape depends on reduction settings.
        """
        return nef.ncc(
            tensor1=tensor1,
            tensor2=tensor2,
            window_size=self.window_size,
            eps=self.eps,
            reduction=self.reduction,
            reduction_dim=self.reduction_dim,
            keepdims=self.keepdims,
        )


class SpatialGradient(nn.Module):
    """
    Compute spatial gradient penalty for smoothness regularization.

    Computes finite differences along each spatial dimension and applies L1 or L2
    penalty.

    Examples
    --------
    # Example 1: L2 smoothness for displacement field
    >>> gradient_func = SpatialGradient(penalty='l2')
    >>> displacement = torch.rand(2, 3, 64, 64, 64)  # (B, ndim, D, H, W)
    >>> gradient = gradient_func(displacement)
    >>> print(gradient.shape)
    torch.Size([])

    # Example 2: L1 penalty (promotes sparse gradients)
    >>> gradient_func = SpatialGradient(penalty='l1')
    >>> gradient = gradient_func(displacement)
    """

    def __init__(
        self,
        penalty: Literal['l1', 'l2'] = 'l2',
        reduction: Union[str, None] = 'mean',
        reduction_dim: Union[int, Tuple[int, ...], None] = None,
        keepdims: bool = False,
    ) -> None:
        """
        Initialize `SpatialGradient`.

        Parameters
        ----------
        penalty : {'l1', 'l2'}, default='l2'
            Penalty type to apply to gradients:
            - 'l1': absolute value (promotes sparsity)
            - 'l2': squared value (promotes smoothness)
        reduction : str or None, default='mean'
            Reduction to apply. Supported values:
            'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var'.
            If None, returns raw penalty values for each spatial dimension.
        reduction_dim : int, tuple of ints, or None, default=None
            Dimension(s) over which to apply the reduction. If None, reduces
            over all dimensions.
        keepdims : bool, default=False
            Whether to retain reduced dimensions as singletons.
        """
        super().__init__()

        self.penalty = penalty
        self.reduction = reduction
        self.reduction_dim = reduction_dim
        self.keepdims = keepdims

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial gradient

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor with shape (B, C, *spatial_dims).

        Returns
        -------
        torch.Tensor
            Gradient (scalar by default, or shaped by reduction settings).
        """
        return nef.spatial_gradient(
            input_tensor=input_tensor,
            penalty=self.penalty,
            reduction=self.reduction,
            reduction_dim=self.reduction_dim,
            keepdims=self.keepdims,
        )


class GaussianBlur(nn.Module):
    """
    Apply a {1D, 2D, 3D} gaussian blur to the input tensor by convolving it with a Gaussian kernel.

    Kernel size is automatically determined as 2 * int(truncate * sigma + 0.5) + 1 for each
    dimension. This ensures the kernel captures the appropriate number of standard deviations
    (default: 3 sigma, which captures ~99.7% of the Gaussian distribution).
    """

    def __init__(
        self,
        sigma: Union[float, int, Sequence[Union[float, int]]] = 1,
        truncate: Union[int, float, Sequence[Union[int, float]]] = 3,
    ):
        """
        Initialize `GaussianBlur`.

        Parameters
        ----------
        sigma : float, int, or Sequence[float or int], default=1
            Standard deviation of the Gaussian kernel. If float/int, same sigma is used
            for all dimensions. If Sequence, different sigmas can be specified per dimension.
        truncate : int, float, or Sequence[int or float], default=3
            Number of standard deviations at which to truncate the kernel. If scalar, same
            truncate value is used for all dimensions. If Sequence, different truncate values
            can be specified per dimension (must match sigma length).

        Notes
        -----
        The automatic kernel sizing follows the formula used in scipy and VoxelMorph:
        kernel_size = 2 * int(truncate * sigma + 0.5) + 1

        This ensures proper Gaussian kernel coverage regardless of sigma value.
        """
        super().__init__()
        self.sigma = sigma
        self.truncate = truncate

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of `GaussianBlur`.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor, assumed to have 1, 2, or 3 spatial dimensions with
            shape (B, C, *spatial).

        Returns
        -------
        torch.Tensor
            The smoothed tensor with the same shape as input_tensor.
        """
        return nef.gaussian_smoothing(
            input_tensor=input_tensor,
            sigma=self.sigma,
            truncate=self.truncate
        )


class ResampleVoxelDimensions(nn.Module):
    """
    Resample tensor to simulate different voxel dimensions.

    This module resamples the input tensor by downsampling along specified spatial
    dimensions, then upsampling back to restore the original or target dimensions.
    This is useful for simulating anisotropic voxel dimensions in medical imaging.
    """

    def __init__(
        self,
        resample_dimension: Union[int, Sequence[int], None] = None,
        downsample_stride: Union[int, Sequence[int]] = 2,
        upsample_scale_factor: Union[int, Sequence[int]] = 2,
        mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
        shape: Union[tuple, None] = None,
    ):
        """
        Initialize `ResampleVoxelDimensions`.

        Parameters
        ----------
        resample_dimension : int, Sequence[int], or None, default=None
            The dimension(s) that should be resampled. If None, all dimensions are resampled.
        downsample_stride : int or Sequence[int], default=2
            Factor by which to subsample.
        upsample_scale_factor : int or Sequence[int], default=2
            Factor by which to upsample.
        mode : {'linear', 'nearest', 'bicubic', 'area', 'nearest-exact'}, default='linear'
            Interpolation mode for upsampling.
        shape : tuple or None, default=None
            Spatial dimensions (without batch or channel dims) to upsample the subsampled tensor
            into.

        Examples
        --------
        ### Subsample with custom stride
        >>> # Make a 2D tensor ~N(0, 1) with batch and channel dims
        >>> input_tensor = torch.randn(1, 1, 128, 128)
        >>> # Downsample 2x in 1st dim and 4x in second dim. Upsample the same way
        >>> resampled_tensor = ResampleVoxelDimensions(
        ...    downsample_stride=(2, 4),
        ...    upsample_scale_factor=(2, 4)
        ... )(input_tensor)
        >>> # Ensure spatial dimensions are the same
        >>> print(resampled_tensor.shape)
        torch.Size([1, 1, 128, 128])

        ### Upsample with custom stride and trilinear interpolation
        >>> # Make a 3D tensor ~N(0, 1) with batch and channel dims
        >>> input_tensor = torch.randn(1, 1, 32, 32, 32)
        >>> # Downsample 2x then upsample 6x
        >>> resampled_tensor = ResampleVoxelDimensions(
        ...    downsample_stride=2,
        ...    upsample_scale_factor=6,
        ...    mode='linear'
        ... )(input_tensor)
        >>> # Ensure dimensions are (1, 1, 96, 96, 96)
        >>> print(resampled_tensor.shape)
        torch.Size([1, 1, 96, 96, 96])
        """

        super().__init__()
        self.resample_dimension = resample_dimension
        self.downsample_stride = downsample_stride
        self.upsample_scale_factor = upsample_scale_factor
        self.mode = mode
        self.shape = shape

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of `ResampleVoxelDimensions`.
        """

        return nef.resample_voxel_dimensions(
            input_tensor=input_tensor,
            downsample_scale=self.downsample_stride,
            upsample_scale=self.upsample_scale_factor,
            mode=self.mode,
            shape=self.shape
        )


class Crop(nn.Module):
    """
    Crop layer that crops spatial dimensions.

    Parameters
    ----------
    size : int, Sequence[int], or None, default=None
        Target spatial size(s). If None, `scale_factor` must be specified.
    scale_factor : float, Sequence[float], or None, default=None
        Multiplicative factor for spatial size. If None, `size` must be specified.
    offset : int or Sequence[int], default=0
        Starting position for crop. If int, same offset for all spatial dimensions. If Sequence,
        per-dimension offsets.

    Examples
    --------
    >>> import torch
    >>> from neurite.nn.modules import Crop
    >>> # Crop from origin
    >>> crop_layer = Crop(size=32)
    >>> x = torch.randn(2, 3, 64, 64)
    >>> cropped = crop_layer(x)
    >>> cropped.shape
    torch.Size([2, 3, 32, 32])
    >>> # Crop from offset
    >>> crop_layer = Crop(size=32, offset=16)
    >>> cropped = crop_layer(x)
    >>> cropped.shape
    torch.Size([2, 3, 32, 32])
    """

    def __init__(
        self,
        size: Union[int, Sequence[int], None] = None,
        scale_factor: Union[float, Sequence[float], None] = None,
        offset: Union[int, Sequence[int]] = 0,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.offset = offset

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return nef.crop(input_tensor, self.size, self.scale_factor, self.offset)


class Clip(nn.Module):
    """
    Clip layer that clamps values to a specified range.

    Parameters
    ----------
    min : float, int, or None, default=None
        Minimum value. If None, no lower bound.
    max : float, int, or None, default=None
        Maximum value. If None, no upper bound.

    Examples
    --------
    >>> import torch
    >>> from neurite.nn.modules import Clip
    >>> clip_layer = Clip(min=0, max=1)
    >>> x = torch.randn(2, 3, 32, 32) * 5
    >>> clipped = clip_layer(x)
    >>> float(clipped.min()), float(clipped.max())
    (0.0, 1.0)
    """

    def __init__(
        self,
        min: Union[float, int, None] = None,
        max: Union[float, int, None] = None,
    ):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return ne.clip(input_tensor, self.min, self.max)


class RandomIntensityLookup(nn.Module):
    """
    Randomly augment the contrast of a single-channel tensor.

    Compute a smoothly varying lookup table to map the original single-channel tensor (usually a
    greyscale image) to a tensor with a new contrast.
    """

    def __init__(self):
        """
        Initialize the `RandomIntensityLookup` module.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomIntensityLookup` module.
        """
        raise NotImplementedError("The `RandomIntensityLookup` module isn't ready yet :(")


class SampleImageFromLabels(nn.Module):
    """
    Generate an image from a label map by sampling a random intensity for each label.

    Identify all unique integer labels in `label_tensor` and assigns each a mean intensity in the
    corresponding output image (`sampled_image`). The mean intensity serves as the mean for a noise
    distribution. Noise is sampled from a normal distribution with the specified standard deviation.
    """

    def __init__(
        self,
        mean_range: Tuple[float, float] = (0.0, 1.0),
        noise_std: Union[float, int] = 0.5,
    ):
        """
        Initialize `SampleImageFromLabels`.

        Parameters
        ----------
        mean_range : Tuple[float, float], default=(0.0, 1.0)
            Range (min, max) for sampling mean intensity for each region. Mean intensities are
            sampled uniformly from this range.
        noise_std : float or int, default=0.5
            Standard deviation of the Gaussian noise added to each region.
        """
        super().__init__()
        self.mean_range = mean_range
        self.noise_std = noise_std

    def forward(self, label_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of `SampleImageFromLabels`.

        Parameters
        ----------
        label_tensor : torch.Tensor
            A tensor with batch and channel dimensions containing integer labels defining distinct
            regions.

        Returns
        -------
        torch.Tensor
            A tensor of sampled image intensities with the same shape as `label_tensor`.
        """

        return ne.sample_image_from_labels(
            label_tensor,
            mean_range=self.mean_range,
            noise_std=self.noise_std
        )


class CategoricalCrossentropy(nn.Module):
    """
    Compute the Categorical Crossentropy between two tensors.
    """
    def __init__(self):
        """
        Initialize `CategoricalCrossentropy`.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `CategoricalCrossentropy` module.
        """
        raise NotImplementedError("The `CategoricalCrossentropy` module isn't ready yet :(")


class MeanSquaredErrorProb(nn.Module):
    """
    Compute the Mean Squared Error between two tensors.
    """
    def __init__(self):
        """
        Initialize `MeanSquaredErrorProb`.
        """
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `CLASSNAME` module.
        """
        raise NotImplementedError("The `CLASSNAME` module isn't ready yet :(")
