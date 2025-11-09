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
from typing import List, Union, Type, Tuple, Literal, Optional
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
    >>> dice_module = ne.losses.Dice()
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
    >>> seg1 = ne.utils.logistic(seg1)
    >>> seg2 = ne.utils.logistic(seg2)
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
        smooth_numerator : float, optional
            Smoothing constant added to the numerator.
        smooth_denominator : float, optional
            Smoothing constant added to the denominator.
        reduction : str, optional
            The type of reduction to apply. Supported values for multidimensional reductions are:
            'mean', 'sum', 'median', 'amax', 'amin', 'std', 'var', 'var_mean'; for single-dimension
            reductions: 'argmin', 'argmax', and all multidimensionals. Default is 'mean'.
        reduction_dim : int or tuple of ints, optional
            Dimension(s) over which to apply the reduction. For multidimensional reductions, pass a
            tuple of dimensions; for single-dimension reductions, pass an integer. Default is (0, 1)
        keepdims : bool, optional
            Whether to retain reduced dimensions as a singleton. Default is True.
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
        negative_slope: float = 0.01,
        alpha: float = 1.0
    ) -> nn.Module:
        """
        Initialize `Activation`.

        Parameters
        ----------
        activation_type : str
            Type of activation function. Supported values: 'relu',
            'leaky_relu',
            'elu'.
        inplace : bool, optional
            Whether to perform the operation in-place. Default is True.
        negative_slope : float, optional
            Negative slope for 'leaky_relu'. Default is 0.01.
        alpha : float, optional
            Alpha value for 'elu'. Default is 1.0.
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
        if self.activation is None:
            return nn.Identity()(input_tensor)
        else:
            return self.activation(input_tensor)


class ConvBlock(nn.Sequential):
    """
    Convolutional block comprising a conv, and optionally, an activation and/or normalization.

    The default sequence of operations in this block is:

    1. **Convolution**: Apply an nD convolution over the input.
    2. **Normalization**: Normalize the output of the convolution to stabilize and accelerate
    training.
    3. **Activation Function**: Introduce non-linearity to the model.

    Attributes
    ----------
    conv : nn.Conv2d
        The convolutional layer.
    batch_norm : nn.BatchNorm2d
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

    # Mapping of spatial dimensions for convolutions
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
        activation: Union[List, str, nn.Module, None] = None,
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
        kernel_size : int or tuple, optional
            Size of the convolving kernel. Default is 3.
        stride : int or tuple, optional
            Stride of the convolution. Default is 1.
        padding : int or tuple, optional
            Padding added to all sides of the input. Default is 1.
        dilation : int or tuple, optional
            Spacing between kernel elements. Every `dilation`-th element is used. Default is 1.
        groups : int, optional
            Number of blocked connections from input to output channels. Default is 1.
        bias : bool, optional
            If True, a learnable bias is added to the output. Default is True.
        normalization : str, nn.Module, or None, optional
            Defines the normalization layer. Can be one of:
            - A string: Supported options are 'batch', 'instance', 'layer', or 'group'.
            - A `Normalization` module: Instantiated or uninstantiated `Normalization` layer.
                e.g. nn.InstanceNorm3d(16) or nn.InstanceNorm3d
            - `None`: No normalization is applied. Default is `None`.

        activation : str, nn.Module, or None, optional
            Defines the activation layer. Can be one of:
            - A string: Supported options are 'relu', 'leaky_relu', or 'elu'.
            - A list: A list containing any of these optinons. Must have the same number of elements
                as the number of activations specified in `order`.
            - A `nn.Module`: Instantiated or uninstantiated activation module.
                e.g. nn.Sigmoid(), nn.Sigmoid
            - `None`: No activation is applied. Default is `None`.

        order : str, optional
            The order of operations in the block. Default is 'cna'
            (normalization -> convolution -> activation).
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

        layers = nn.ModuleDict()
        self.order = list(order)
        valid_operations = ['c', 'n', 'a']

        # Count activations to validate that activation list matches order specification
        n_activations = order.count('a')

        if not isinstance(activation, (list, tuple)):
            activation = [activation] * n_activations
        else:
            assert len(activation) == n_activations, (
                "The total number of activations passed to `activation` must be the same number ",
                f"defined in `order`. Got activation={activation}, order={order}"
            )

        if not set(order).issubset(valid_operations):
            raise ValueError(f"Invalid order. Must be a subset of {valid_operations}.")

        if ndim not in ConvBlock.conv_dim_map:
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3.")

        conv_cls_name = f"Conv{ConvBlock.conv_dim_map[ndim]}"
        conv_cls = getattr(nn, conv_cls_name)

        conv_id, norm_id, act_id = 0, 0, 0

        for operation in self.order:

            if operation == 'c':
                layers[f"conv{conv_id}"] = conv_cls(
                    in_channels, out_channels, kernel_size, stride,
                    padding, dilation, groups, bias, padding_mode=padding_mode
                )

                in_channels = out_channels  # Subsequent convs have this many
                conv_id += 1

            elif operation == 'n' and normalization is not None:
                layers[f'normalization{norm_id}'] = ne.utils.utils.build_normalization(
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
        kernel_size : int or tuple
            Size of the convolving kernel.
        stride : int or tuple, optional
            Stride of the convolution. Default is 2.
        padding : int or tuple, optional
            Padding added to all sides of the input. Default is 1.
        output_padding : int or tuple, optional
            Additional size added to one side of each dimension in the output
            shape. Default is 0.
        dilation : int or tuple, optional
            Spacing between kernel elements. Default is 1.
        groups : int, optional
            Number of blocked connections from input to output channels.
            Default is 1.
        bias : bool, optional
            If True, a learnable bias is added to the output. Default is True.
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
        pool_mode : str, optional
            The pooling mode to use. Options are 'max' for max pooling,
            'avg' for average pooling, and 'lp' for LP pooling. Default is
            'max'.
        kernel_size : int or tuple, optional
            The size of the pooling kernel. Default is 2.
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

    Attributes
    ----------
    conv_block : ConvBlock
        The convolutional block applying a series of convolutions, normalization, and activation.
    pool : Pool
        The pooling layer to downsample the feature maps.
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
        order='nca',
        return_skip: bool = False,
    ):
        """
        Initialize `DownsampleConvBlock`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolving kernel. Default is 3.
        stride : int, optional
            Stride of the convolution. Default is 1.
        padding : int, optional
            Padding added to all sides of the input. Default is 1.
        padding_mode : {'zeros', 'replicate', 'reflect'}, optional
            Padding mode for the convolution. Default is 'zeros'.
        normalization : str, nn.Module, or None, optional
            Normalization type. Default is 'batch'.
        activation : str, nn.Module, or None, optional
            Activation type. Default is 'relu'.
        pool_mode : str, optional
            Pooling mode ('max' or 'avg'). Default is 'max'.
        pool_kernel_size : int, optional
            Kernel size for pooling. Default is 2.
        order : str, optional
            The order of operations in the block. Default is 'nca' (normalization -> convolution ->
            activation). Each character in the string can be specified an arbitrary number of times
            in any order. Each character in the string represents one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation
        return_skip : bool
            If True, return skip connection features from the forward pass for use in UNet-style
            architectures. Default is False.
        """
        super().__init__()
        self.return_skip = return_skip

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

        self.pool = Pool(ndim=ndim, pool_mode=pool_mode, kernel_size=pool_kernel_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the downsampling convolution.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Downsampled tensor after applying convolution and pooling.
        """

        if self.return_skip:
            conv_result = self.conv_block(input_tensor)
            return self.pool(conv_result), conv_result
        else:
            return self.pool(self.conv_block(input_tensor))


class UpsampleConvBlock(nn.Module):
    """
    Apply `ConvBlock` followed by an upsampling operation to extract features and increase spatial
    shape.

    Attributes
    ----------
    upsample : TransposedConv
        Module to be used for upsampling. One of:
         - `neurite.modules.TransposedConv`: nD transposed convolution.
         - `torch.nn.Upsample`: Interpolation upsampler form PyTorch.
    conv_block : ConvBlock
        The convolutional block applying a series of convolutions, normalizations, and activations.
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
            Number of output channels.
        kernel_size : int, optional
            Size of the convolving kernel. Default is 3.
        stride : int, optional
            Stride of the convolution. Default is 1.
        padding : int, optional
            Padding added to all sides of the input. Default is 1.
        upsample_kernel_size : int, optional
            Kernel size for the transposed convolution. Default is 4.
        upsample_stride : int, optional
            Stride for the transposed convolution. Default is 2.
        upsample_padding : int, optional
            Padding for the transposed convolution. Default is 1.
        normalization : str, nn.Module, or None, optional
            Normalization type. Default is 'batch'.
        activation : str, nn.Module, or None, optional
            Activation type. Default is 'relu'.
        order : str, optional
            The order of operations in the block. Default is 'nca' (normalization -> convolution ->
            activation). Each character in the string can be specified an arbitrary number of times
            in any order. Each character in the string represents one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation
        accepts_skip : bool
            If True, the block is configured to accept skip connections (UNet-style). This allows
            the block to concatenate skip features with the main input. Default is True.
        skip_channels : int or None, optional
            Number of channels in the skip connection. If provided and accepts_skip=True,
            the actual concatenated input will be in_channels + skip_channels. If None and
            accepts_skip=True, defaults to in_channels (symmetric assumption). Default is None.
        """

        super().__init__()

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
                upsample_mode = ne.utils.utils.infer_linear_interpolation_mode(ndim)
            align = None if upsample_mode == 'nearest' else True

            self.upsample = nn.Upsample(
                scale_factor=scale_factor,
                mode=upsample_mode,
                align_corners=align
            )

        if accepts_skip:
            if skip_channels is not None:
                in_channels += skip_channels
            else:
                # Backward compatibility when assumimg symmetric architecture
                in_channels += in_channels

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

    def forward(self, input_tensor: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
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
            Upsampled tensor after applying upsampling operation and conv blocks.
        """
        if isinstance(skip, torch.Tensor):

            features = self.upsample(input_tensor)
            features = torch.cat([features, skip], dim=1)

            return self.conv_block(features)
        else:
            return self.conv_block(self.upsample(input_tensor))


class ContextCrossConv(nn.Module):
    """
    nD Convolutional layer that performs cross convolutions to interact a query image with a context
    set (examples) defining a task.

    Examples
    --------
    ### 2D cross convolution on CPU
    >>> # Create random 2D inputs: 3 sets of 2 query images and 6 context pairs.
    >>> query_image = torch.randn(3, 2, 1, 64, 64)
    >>> context_images = torch.randn(3, 6, 1, 64, 64)
    >>> context_segmentations = torch.randint(0, 10, (3, 6, 1, 64, 64))
    >>> # Concat along the channel dimension
    >>> context = torch.cat([context_images, context_segmentations], dim=1)
    >>> # Define the number of query image channels and context set channels seperately:
    >>> in_channels = (1, 2)
    >>> # Define cross convolution block
    >>> cross_conv_block = ContextCrossConv(
    ...     ndim=2, in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
    ... )
    >>> # Forward pass of cross convolutiom block, returning new query and context representations
    >>> new_query_image, new_context_image = cross_conv_block(query_image, context_image)
    >>> # Expected output shapes: (3, 2, 16, 64, 64), (3, 6, 16, 64, 64)
    >>> print(new_query_image.shape, new_context_image.shape)
    torch.Size([3, 2, 16, 64, 64]) torch.Size([3, 6, 16, 64, 64])

    ### 3D cross convolution on GPU
    >>> # Create random 3D inputs: 1 set of 1 query image and 9 context pairs.
    >>> query_image = torch.randn(1, 1, 1, 128, 128, 128)
    >>> context_images = torch.randn(1, 9, 1, 128, 128, 128)
    >>> context_segmentations = torch.randint(0, 10, (1, 9, 1, 128, 128, 128))
    >>> # Concat along the channel dimension
    >>> context = torch.cat([context_images, context_segmentations], dim=1)
    >>> # Define the number of query image channels and context set channels seperately:
    >>> in_channels = (1, 2)
    >>> # Define cross convolution block
    >>> cross_conv_block = ContextCrossConv(
    ...     ndim=3, in_channels=in_channels, out_channels=32, kernel_size=3, padding=1
    ... )
    >>> # Forward pass of cross convolutiom block, returning new query and context representations
    >>> new_query_image, new_context_image = cross_conv_block(query_image, context_image)
    >>> # Expected output shapes: (1, 1, 32, 128, 128, 128), (1, 9, 32, 128, 128, 128)
    >>> print(new_query_image.shape, new_context_image.shape)
    torch.Size([3, 1, 32, 128, 128, 128]) torch.Size([3, 9, 32, 128, 128, 128])

    Notes
    -----
    Modified from the original description on [GitHub](https://github.com/JJGO/UniverSeg):
    The pairwise convolution is computed by first forming a Cartesian product of the slices in `x1`
    and `x2`. For example, if `x1` has Sx1 slices and `x2` has Sx2 slices, then the concatenated
    tensor has shape (B, Sx1, Sx2, Cx1 + Cx2, ...). This tensor is reshaped to combine the first
    three dimensions so that the standard nn.Conv*d can be applied. Finally, the output is reshaped
    back to separate the batch and slice indices.

    References
    ----------
    J. G. Ortiz et al., "UniverSeg: Universal Medical Image Segmentation,"
    GitHub repository, 2023. Available: https://github.com/JJGO/UniverSeg
    """

    def __init__(
        self,
        ndim: int,
        in_channels: Tuple[int, int],
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        padding_mode: Literal['zeros', 'replicate', 'reflect'] = 'zeros',
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        normalization: Union[str, nn.Module, None] = None,
        activation: Union[str, nn.Module, None] = None,
        order: str = 'cna',
    ):
        """
        Initialize the `ContextCrossConv` module.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
        in_channels : Tuple of int
            Number of channels in `x1` and x2`, respectively
        out_channels : int
            Number of output channels for the cross convolution.
        kernel_size : int or tuple, optional
            Size of the convolving kernel. Default is 3.
        stride : int or tuple, optional
            Stride of the convolution. Default is 1.
        padding : int or tuple, optional
            Padding added to all sides of the input. Default is 1.
        dilation : int or tuple, optional
            Spacing between kernel elements. Every `dilation`-th element is used. Default is 1.
        groups : int, optional
            Number of blocked connections from input to output channels. Default is 1.
        bias : bool, optional
            If True, a learnable bias is added to the output. Default is True.

        normalization : str, nn.Module, or None, optional
            Defines the normalization layer. Can be one of:
            - A string: Supported options are 'batch', 'instance', 'layer', or 'group'.
            - A `Normalization` module: Instantiated or uninstantiated `Normalization` layer.
                e.g. nn.InstanceNorm3d(16) or nn.InstanceNorm3d
            - `None`: No normalization is applied. Default is `None`.

        activation : str, nn.Module, or None, optional
            Defines the activation layer. Can be one of:
            - A string: Supported options are 'relu', 'leaky_relu', or 'elu'.
            - A `nn.Module`: Instantiated or uninstantiated activation module.
                e.g. nn.Sigmoid(), nn.Sigmoid
            - `None`: No activation is applied. Default is `None`.

        order : str, optional
            The order of operations in the block. Default is 'cna'
            (normalization -> convolution -> activation).
            Each character in the string represents one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation
        """
        raise NotImplementedError(
            "ContextCrossConv has been moved to neurite_sandbox. "
            "Please use: from neurite_sandbox.etienne_chollet.nn.modules import ContextCrossConv"
        )


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
        size : int or Tuple[int, int], optional
            The desired output size. If None, uses `scale_factor`.
        scale_factor : float or Tuple[float, float], optional
            Scaling factor for resizing. If None, uses `size`.
        mode : str, optional
            Interpolation mode for upsampling. Options include 'nearest', 'linear',
            'bicubic', 'area', and 'nearest-exact'. Default is 'linear'.
        align_corners : bool, optional
            Alignment for "linear", "bilinear", or "trilinear" modes.
        recompute_scale_factor : bool, optional
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
        resized_tensor = F.interpolate(
            input=input_tensor,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
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
        nb_bins : int, optional
            The number of discrete bins to softly quantize the input values into. By default 16
        softness : float or int, optional
            The softness factor for quantization. A higher value gives smoother quantization.
            By default 1.0
        min_clip : float or int, optional
            Clip data lower than this value before calculating bin centers. By default -float('inf')
        max_clip : float or int, optional
            Clip data higher than this value before calculating bin centers. By default float('inf')
        return_log : bool, optional
            Optionally return the log of the softly quantized tensor. By default False

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
        return ne.nn.functional.mse(tensor1=input_tensor, tensor2=target_tensor)


class GaussianBlur(nn.Module):
    """
    Apply a {1D, 2D, 3D} gaussian blur to the input tensor by convolving it with a Gaussian kernel.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        sigma: float = 1,
    ):
        """
        Initialize `GaussianBlur`.

        Parameters
        ----------
        kernel_size : int, optional
            Size of the Gaussian kernel, default is 3.
        sigma : float or int, optional
            Standard deviation of the Gaussian kernel, default is 1.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of `GaussianBlur`.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor, assumed to have 1, 2, or 3 spatial dimensions.

        Returns
        -------
        torch.Tensor
            The smoothed tensor.
        """
        return nef.gaussian_smoothing(
            input_tensor=input_tensor,
            kernel_size=self.kernel_size,
            sigma=self.sigma
        )


class GaussianAntialiasing(nn.Module):
    """
    Apply Gaussian antialiasing by combining Gaussian blur with downsampling.

    This module reduces aliasing artifacts when downsampling by first applying
    a Gaussian blur filter followed by subsampling. This is particularly important
    in medical imaging to preserve structural information during downsampling operations.
    """
    def __init__(
        self,
        stride: Union[int, List[int]] = 2,
        kernel_size: Union[int, List[int], None] = None,
        sigma: Union[float, int, List[float], List[int], None] = None,
        subsampling_dimension: Union[List[int], int, None] = None
    ):
        """
        Initialize `GaussianAntialiasing`.

        Parameters
        ----------
        stride : int or List[int], optional
            Downsampling stride. If int, the same stride is applied to all spatial dimensions.
            If List[int], different strides can be specified per dimension. Default is 2.
        kernel_size : int or List[int], optional
            Size of the Gaussian kernel for antialiasing. If int, same size is used for all
            dimensions. If List[int], different sizes can be specified per dimension.
            If None, automatically computed as 2 * stride + 1 per dimension. Default is None.
        sigma : float, int, List[float], or List[int], optional
            Standard deviation of the Gaussian kernel. If float/int, same sigma is used for
            all dimensions. If List, different sigmas can be specified per dimension.
            If None, automatically computed as stride / 2 per dimension. Default is None.
        subsampling_dimension : List[int], int, or None, optional
            Dimensions to apply antialiasing and subsampling. If None, applies to all
            spatial dimensions. Default is None.

        Examples
        --------
        >>> import torch
        >>> # Create a 3D medical image tensor
        >>> input_tensor = torch.randn(1, 1, 64, 64, 64)
        >>> # Apply Gaussian antialiasing with 2x downsampling
        >>> antialiasing_layer = GaussianAntialiasing(stride=2)
        >>> antialiased_tensor = antialiasing_layer(input_tensor)
        >>> print(antialiased_tensor.shape)
        torch.Size([1, 1, 32, 32, 32])

        >>> # Apply different strides per dimension with custom sigma
        >>> antialiasing_layer = GaussianAntialiasing(stride=[2, 2, 4], sigma=1.5)
        >>> antialiased_tensor = antialiasing_layer(input_tensor)
        >>> print(antialiased_tensor.shape)
        torch.Size([1, 1, 32, 32, 16])

        >>> # Apply per-dimension antialiasing parameters
        >>> antialiasing_layer = GaussianAntialiasing(
        ...     stride=[2, 2, 4],
        ...     kernel_size=[5, 5, 9],
        ...     sigma=[1.0, 1.0, 2.0]
        ... )
        >>> antialiased_tensor = antialiasing_layer(input_tensor)
        >>> print(antialiased_tensor.shape)
        torch.Size([1, 1, 32, 32, 16])
        """
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.subsampling_dimension = subsampling_dimension

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of `GaussianAntialiasing`.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to be downsampled with antialiasing, assumed to be 1D, 2D, or 3D.

        Returns
        -------
        torch.Tensor
            Antialiased and downsampled tensor with reduced spatial dimensions.
        """
        return nef.gaussian_antialiasing(
            input_tensor=input_tensor,
            stride=self.stride,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            subsampling_dimension=self.subsampling_dimension
        )


class Resample(nn.Module):
    """
    Spatially {subsample, resample} the input tensor.

    This module resamples the input tensor by a factor of `stride` along the specified spatial
    dimension(s) by interleaving dropouts (keeping every `stride`'th element), then upsamples to
    restore it to its original dimensions.
    """

    def __init__(
        self,
        resample_dimension: Union[int, List[int]] = None,
        downsample_stride: Union[int, List[int]] = 2,
        upsample_scale_factor: Union[int, List[int]] = 2,
        mode: Literal['linear', 'nearest', 'bicubic', 'area', 'nearest-exact'] = 'linear',
        shape: tuple = None,
    ):
        """
        Initialize `Resample`.

        Parameters
        ----------
        resample_dimension : int or list of ints, optional
            The dimension(s) that should be resampled. If None, all dimensions are resampled.
            Default is None.
            dimensions.
        downsample_stride : int or list of ints, optional
            Factor by which to subsample. Default is 2.
        upsample_scale_factor : int, float or list of ints or floats, optional
            Factor by which to upsample. Default is 2.
        mode : str, optional
            Interpolation mode for upsampling. Options include 'nearest', 'linear',
            'bicubic', 'area', and 'nearest-exact'. Default is 'linear'.
        shape : tuple
            Spatial dimensions (without batch or channel dims) to upsample the subsampled tensor
            into.

        Examples
        --------
        ### Subsample with custom stride
        >>> # Make a 2D tensor ~N(0, 1) with batch and channel dims
        >>> input_tensor = torch.randn(1, 1, 128, 128)
        >>> # Downsample 2x in 1st dim and 4x in second dim. Upsample the same way 
        >>> resampled_tensor = Resample(
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
        >>> resampled_tensor = Resample(
        ...    downsample_stride=2,
        ...    upsample_scale_factor=6,
        ...    mode='trilinear'
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
        Perform the forward pass of the `Resample`.
        """

        return ne.utils.utils.resample(
            input_tensor=input_tensor,
            resample_dimension=self.resample_dimension,
            downsample_stride=self.downsample_stride,
            upsample_scale_factor=self.upsample_scale_factor,
            mode=self.mode,
            shape=self.shape
        )


class RandomCrop(nn.Module):
    """
    Randomly crop the input tensor along allowed dimensions.

    This module randomly selects a subset of the allowed dimensions (excluding `forbidden_dims`)
    and crops each independently by a proportion that is randomly drawn from a distribution. The
    proportion to crop can either be fixed or sampled from a specified distribution. Each allowed
    dimension has a probability `prob` of being cropped based on the results of independent
    Bernoulli trials.
    """

    def __init__(
        self,
        crop_proportion: Union[float, int] = 0.5,
        prob: Union[float, int] = 1,
        forbidden_dims: Union[Tuple, List] = (0, 1),
        seed: Union[int, None] = None,
    ):
        """
        Initialize the `RandomCrop` module.

        Parameters
        ----------
        crop_proportion : float, optional
            The proportion that is randomly cropped from any allowed dimension. By default 0.5.
            Represents the maximum proportion (0 to 1) to crop, sampled from independent uniform
            distributions for each allowed dimension. A value of `0.5` means up to 50% of each
            dimension can be cropped.
        prob : float, optional
            The probability of cropping each allowed dimension. By default 1.0.
            Used as a fixed probability for all eligible dimensions.
        forbidden_dims : Union[Tuple[int, ...], List[int]], optional
            Dimensions that should never be cropped. By default `(0, 1)` (batch and channel
            dimensions)
        seed : int, optional
            A random seed to control the randomness of cropping operations. If provided,
            it ensures reproducibility of the cropping. Defaults to `None`.
        """
        super().__init__()
        self.crop_proportion = crop_proportion
        self.prob = prob
        self.forbidden_dims = forbidden_dims
        self.seed = seed

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the `RandomCrop` module.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The tensor to be randomly cropped. It is assumed to have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            The tensor that has been randomly cropped.
        """

        return ne.utils.augment.random_crop(
            input_tensor=input_tensor,
            crop_proportion=self.crop_proportion,
            prob=self.prob,
            forbidden_dims=self.forbidden_dims,
            seed=self.seed
        )


class RandomClip(nn.Module):
    """
    Randomly clip the intensities of the input tensor.
    """

    def __init__(
        self,
        clip_min: Union[float, int] = 0,
        clip_max: Union[float, int] = 1,
        clip_prob: Union[float, int] = 0.5,
        seed: Union[int, None] = None,
    ):
        """
        Initialize `RandomClip` with specified clipping bounds and sampling probability.

        Parameters
        ----------
        clip_min : float or int, optional
            The lower bound for clipping. Elements less than `clip_min` are set to `clip_min`.
            Defaults to 0.
        clip_max : float or int, optional
            The upper bound for clipping. Elements greater than `clip_max` are set to `clip_max`.
            Defaults to 1.
        clip_prob : float or int, optional
            Probability of applying this operation. Defaults to 0.5.
        seed : int, optional
            Seed for random number generation to ensure reproducibility. Defaults to None.

        Examples
        --------
        ### Initialize `RandomClip` and apply it deterministically to a tensor:
        >>> random_clip = RandomClip(clip_min=0.1, clip_max=0.9, clip_prob=0.5)
        >>> input_tensor = torch.randn(3, 3)
        >>> output_tensor = random_clip(input_tensor)
        >>> print(output_tensor)
        """
        super().__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clip_prob = clip_prob
        self.seed = seed

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of `RandomClip`.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The tensor to be clipped.

        Returns
        -------
        torch.Tensor
            The clipped tensor (if Bernoulli trial defined by parameter `clip_prob` is successful).
        """

        return ne.utils.augment.random_clip(
            input_tensor=input_tensor,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            clip_prob=self.clip_prob,
            seed=self.seed
        )


class RandomGamma(nn.Module):
    """
    Apply a randomized or deterministic nonlinear gamma scaling operation.

    Adjust the contrast of the input tensor by applying a non-linear scaling operation.
    Each element in the tensor is raised to the power of `gamma`, which is a quantity sampled from
    a distribution.
    """

    def __init__(
        self,
        gamma: Union[float, int] = 1,
        prob: Union[float, int] = 1,
        seed: Union[int, None] = None,
    ):
        """
        Initialize `RandomGamma`.

        Parameters
        ----------
        gamma : float or int, optional
            The gamma value to apply for the scaling operation. Represents a fixed gamma value.
            By default `1.0`, which leaves the tensor unchanged.
        prob : float, optional
            The probability of applying the gamma operation. Used as a fixed probability for
            the operation. Default is `1.0` (always apply).
        seed : int, optional
            A random seed to control the randomness of the gamma scaling operation. If
            provided, it ensures reproducibility of the operation. Defaults to `None`.

        Examples
        --------
        ### Fixed gamma scaling operation
        >>> gamma_module = RandomGamma(gamma=2.0, prob=1.0)
        >>> tensor = torch.tensor([0.25, 0.5, 0.75])
        >>> gamma_tensor = gamma_module(tensor)
        >>> print(gamma_tensor)
        tensor([0.0625, 0.2500, 0.5625])

        ### Applying gamma operation with reproducibility
        >>> gamma_module1 = RandomGamma(gamma=2.0, prob=1.0, seed=42)
        >>> gamma_module2 = RandomGamma(gamma=2.0, prob=1.0, seed=42)
        >>> tensor = torch.tensor([0.25, 0.5, 0.75])
        >>> gamma_tensor1 = gamma_module1(tensor)
        >>> gamma_tensor2 = gamma_module2(tensor)
        >>> print(torch.equal(gamma_tensor1, gamma_tensor2))
        True
        """
        super().__init__()
        self.gamma = gamma
        self.prob = prob
        self.seed = seed

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of `RandomGamma`.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to which the gamma operation will be applied. It is assumed to
            have a range suitable for gamma correction (typically normalized between 0 and 1).

        Returns
        -------
        torch.Tensor
            The tensor after applying the gamma scaling operation. If operation is not applied
            (based on `prob`), the original `input_tensor` is returned unchanged.
        """

        return ne.utils.augment.random_gamma(
            input_tensor=input_tensor,
            gamma=self.gamma,
            prob=self.prob,
            seed=self.seed
        )


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


class RandomClearLabel(nn.Module):
    """
    Erase regions of an image from randomly selected regions in a label map.

    Identify unique labels within the `label_tensor` and, based on a specified probability,
    designate regions of the `input_tensor` to be erased (set to zero).

    Examples
    --------
    ### Clearing labels with a fixed probability
    >>> clear_label = RandomClearLabel(prob=0.75)
    >>> input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> label_tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> cleared_tensor = clear_label(input_tensor, label_tensor)
    >>> print(cleared_tensor)
    tensor([0.0000, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000])

    ### Reproducibility with a seed
    >>> clear_label1 = RandomClearLabel(seed=32)
    >>> clear_label2 = RandomClearLabel(seed=32)
    >>> input_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    >>> label_tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> cleared_tensor1 = clear_label1(input_tensor, label_tensor)
    >>> cleared_tensor2 = clear_label2(input_tensor, label_tensor)
    >>> print(torch.equal(cleared_tensor1, cleared_tensor2))
    True
    """

    def __init__(
        self,
        prob: Union[float, int] = 0.5,
        exclude_zero: bool = True,
        seed: Union[int, None] = None,
    ):
        """
        Initialize `RandomClearLabel`.

        Parameters
        ----------
        prob : float, optional
            Probability of any label/region being selected for erasure as determined by iid
            Bernoulli trials, by default 0.5.
        exclude_zero : bool, optional
            Optionally exclude zero (uaually background) from the list of potential regions to clear
            (never clear zero labels), by default True.
        seed : int, optional
            A random seed to control the randomness of label clearing operations. If
            provided, it ensures reproducibility of the clearing process. By default, None.
        """
        super().__init__()
        self.prob = prob
        self.exclude_zero = exclude_zero
        self.seed = seed

    def forward(
        self,
        input_tensor: torch.Tensor,
        label_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of `RandomClearLabel`.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Image or tensor to clear.
        label_tensor : torch.Tensor
            Label map corresponding to sampling domain from which to select regions for clearing.

        Returns
        -------
        torch.Tensor
            The modified tensor with specified labels cleared (set to zero). If no labels are
            cleared, the original `input_tensor` is returned unchanged.
        """

        return ne.utils.random_clear_label(
            input_tensor=input_tensor,
            label_tensor=label_tensor,
            prob=self.prob,
            exclude_zero=self.exclude_zero,
            seed=self.seed
        )


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

        return ne.functional.sample_image_from_labels(
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
