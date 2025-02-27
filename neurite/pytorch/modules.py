"""
Modules are simple operations containing learnable parameters. The `modules` module contains general
nD building blocks for neural networks.
"""
__all__ = [
    "Norm",
    "Activation",
    "ConvBlock",
    "TransposedConv",
    "Pool",
    "DownsampleConvBlock",
    "UpsampleConvBlock",
    "CrossConvBlock"
]

from typing import Union, Type, Optional, Tuple
import einops
import torch
from torch import nn
import neurite as ne


class Norm(nn.Module):
    """
    Dynamically constructs a normalization layer based on `norm_type` and `ndim`.

    Supports
    --------
    - 'batch': BatchNorm1d, BatchNorm2d, BatchNorm3d (requires `ndim`)
    - 'instance': InstanceNorm1d, InstanceNorm2d, InstanceNorm3d (requires `ndim`)
    - 'layer': LayerNorm (does not use `ndim`)
    - 'group': GroupNorm (requires `num_groups`)

    The user can also provide a custom `nn.Module` class in `norm_type`.
    """

    # Map normalization types and dimensions for their corresponding PyTorch classes
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
        "group": nn.GroupNorm
    }

    def __init__(
        self,
        norm_type: Union[str, Type[nn.Module], None],
        ndim: Optional[int] = None,
        num_features: Optional[int] = None,
        num_groups: Optional[int] = None,
        eps: float = 1e-5,
        affine: bool = True,
        **kwargs
    ):
        """
        Initialize the `Norm` module.

        Parameters
        ----------
        norm_type : str or nn.Module
            Type of normalization. Must be one of 'batch', 'instance', 'layer',
            'group', or a custom `nn.Module` class.
        ndim : int, optional
            Dimensionality for batch/instance normalization:
            - 1 -> *Norm1d
            - 2 -> *Norm2d
            - 3 -> *Norm3d
            Required for 'batch' or 'instance' norms.

        num_features : int, optional
            Number of input features or channels. Required for 'batch', 'instance',
            'layer', and 'group' normals. For layer norm, this is the size of the
            normalized dimension. For batch and instance norms, this is typically the
            number of channels/features.
        num_groups : int, optional
            Number of groups for GroupNorm. Required for 'group' normalization.
        eps : float, optional
            A value added to the denominator for numerical stability. Default is 1e-5.
        affine : bool, optional
            If True, the layer has learnable affine parameters. Default is True.
        **kwargs : dict, optional
            Additional keyword arguments are passed directly to the normalization class
            constructor. This enables further customization without modifying this class.

        Examples
        --------
        >>> # Dummy input of shape (B, C, H, W)
        >>> x = torch.randn(1, 16, 32, 32)

        ### Using a custom, pre-instantiated normalization layer
        >>> norm_a = nn.InstanceNorm2d(16)
        >>> norm_A = Norm(norm_a)
        >>> norm_A(x)
        ...

        ### Using a custom normalization layer that has not been instantiated
        >>> norm_b = nn.InstanceNorm2d
        >>> norm_B = Norm(norm_b, num_features=16)
        >>> norm_B(x)
        ...

        ### Using the wrapper to define a normalization layer
        >>> norm_C = Norm(norm_type='instance', ndim=2, num_features=16)
        >>> norm_C(x)
        ...
        """
        super().__init__()

        # Norm object has been instantiated with parameters
        if ne.utils.utils.is_instantiated_normalization(norm_type):
            self.norm = norm_type
            return

        # Norm object has been provided but not instantiated
        if isinstance(norm_type, type) and issubclass(norm_type, nn.Module):
            # Assume user provided a custom normalization class directly
            if num_features is None:
                raise ValueError("`num_features` must be specified for custom norms.")
            self.norm = norm_type(num_features=num_features, eps=eps, affine=affine, **kwargs)
            return

        # Handle known norm_types
        if norm_type not in self.NORMALIZATION_MAP:
            raise ValueError(
                f"Invalid norm_type '{norm_type}'. Must be one of "
                f"{list(self.NORMALIZATION_MAP.keys())} or a custom nn.Module subclass."
            )

        # Batch and instance norm require an input dimensionality
        if norm_type in ("batch", "instance"):
            if ndim not in (1, 2, 3):
                raise ValueError(
                    "For 'batch' or 'instance' normalization, ndim must be 1, 2, or 3."
                )
            # They also require the number of features
            if num_features is None:
                raise ValueError("`num_features` must be specified for 'batch' or 'instance' norm.")

            norm_class = self.NORMALIZATION_MAP[norm_type][ndim]
            self.norm = norm_class(
                num_features=num_features, eps=eps, affine=affine, **kwargs
            )

        elif norm_type == "layer":
            if num_features is None:
                raise ValueError(
                    "`num_features` (normalized shape) must be specified for 'layer' norm."
                )
            self.norm = nn.LayerNorm(
                num_features, eps=eps, elementwise_affine=affine, **kwargs
            )

        elif norm_type == "group":
            if num_groups is None:
                raise ValueError("For 'group' normalization, `num_groups` must be specified.")
            if num_features is None:
                raise ValueError("`num_features` must be specified for 'group' norm.")
            self.norm = nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine, **kwargs)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the normalization layer.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor to be normalized. The shape depends on the normalization type:
            - For 1D norms: (N, C, L)
            - For 2D norms: (N, C, H, W)
            - For 3D norms: (N, C, D, H, W)
            - For layer/group norms: shape can vary, but typically (N, *)

        Returns
        -------
        torch.Tensor
            Normalized output tensor.
        """
        return self.norm(input_tensor)


class Activation(nn.Module):
    """
    Dynamically constructs an activation layer based on the specified type.
    """
    def __init__(
        self,
        activation_type: Union[str, Type[nn.Module], None] = None,
        inplace: bool = True,
        negative_slope: float = 0.01,
        alpha: float = 1.0
    ):
        """
        Initialize the `Activation` module.

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
            self.activation = None

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

        else:
            raise ValueError(
                f"Unsupported activation_type '{activation_type}'. "
                f"Supported types: 'relu', 'leaky_relu', 'elu'."
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the activation layer.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated output tensor.
        """
        if self.activation is None:
            return nn.Identity()(input_tensor)
        else:
            return self.activation(input_tensor)


class ConvBlock(nn.Sequential):
    """
    Convolutional Block comprising a convolutional layer, and optionally, an activation function and
    normalization.

    The default sequence of operations in this block is:
        1. **Convolution**: Applies a 2D convolution over the input.
        2. **Normalization**: Normalizes the output of the convolution to stabilize and accelerate
        training.
        3. **Activation Function**: Introduces non-linearity to the model.

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
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: Union[str, nn.Module, None] = None,
        activation: Union[str, nn.Module, None] = None,
        order: str = 'cna',
    ):
        """
        Initialize the `ConvBlock`.

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
        norm : str, nn.Module, or None, optional
            Defines the normalization layer. Can be one of:
            - A string: Supported options are 'batch', 'instance', 'layer', or 'group'.
            - A `Norm` module: Instantiated or uninstantiated `Norm` layer.
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

        Examples
        --------
        ### Basic usage with default options:
        >>> conv_block = ConvBlock(
                ndim=2,
                in_channels=16,
                out_channels=32,
                norm="batch",
                activation="relu"
            )
        >>> input_tensor = torch.randn(1, 16, 64, 64)
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])

        ### Using a pre-instantiated `Norm` and `Activation` module:
        >>> norm_layer = nn.BatchNorm2d(32)
        >>> activation_layer = nn.ReLU()
        >>> conv_block = ConvBlock(
                ndim=2,
                in_channels=16,
                out_channels=32,
                norm=norm_layer,
                activation=activation_layer
            )
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])

        ### Omitting normalization or activation:
        >>> conv_block = ConvBlock(
            ndim=2,
            in_channels=16,
            out_channels=32,
            norm=None,
            activation=None
        )
        >>> output_tensor = conv_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 32, 64, 64])
        """

        super().__init__()

        layers = nn.ModuleDict()
        self.order = list(order)  # make string of letters into list of letters
        valid_operations = ['c', 'n', 'a']

        # Validate the operations
        if not set(order).issubset(valid_operations):
            raise ValueError(f"Invalid order. Must be a subset of {valid_operations}.")

        # Validate the dimensions
        if ndim not in ConvBlock.conv_dim_map:
            # This only supports 1, 2, and 3 dimensions!
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3.")

        # Dynamically retreive the appropriate `Conv*` class
        conv_cls_name = f"Conv{ConvBlock.conv_dim_map[ndim]}"
        conv_cls = getattr(nn, conv_cls_name)

        # Initialize trackers for the order in the conv
        conv_id, norm_id, act_id = 0, 0, 0

        # Collect layers in the appropriate order
        for operation in self.order:

            if operation == 'c':
                # Init the conv with appropriate params
                layers[f"conv{conv_id}"] = conv_cls(
                    in_channels, out_channels, kernel_size, stride,
                    padding, dilation, groups, bias
                )

                in_channels = out_channels  # All future convs and stuff will have this many in
                conv_id += 1  # Increment conv id for easy tracking/accessing

            # Dynamically construct the normalization and assign it to a named key in layers
            elif operation == 'n' and norm is not None:
                layers[f'norm{norm_id}'] = Norm(norm, ndim, in_channels)
                norm_id += 1

            # Construct the activation and assign it to a named key in layers
            elif operation == 'a' and activation is not None:
                layers[f'activation{act_id}'] = Activation(activation)
                act_id += 1

        # Add layers to `Sequential`
        for name, layer in layers.items():
            self.add_module(name, layer)


class TransposedConv(nn.Module):
    """
    Dynamically constructs a transposed convolutional layer (ConvTranspose1d,
    ConvTranspose2d, or ConvTranspose3d) based on the input dimensionality
    `ndim`.

    This module enables flexible definition of transposed convolutional layers
    for 1D, 2D, or 3D data, by internally selecting the appropriate PyTorch
    transposed convolution operation (`torch.nn.ConvTranspose1d`,
    `torch.nn.ConvTranspose2d`, or `torch.nn.ConvTranspose3d`).
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
        Initialize the `TransposedConv` module.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
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

        # Mapping of spatial dimensions for convolutions
        conv_dim_map = {1: '1d', 2: '2d', 3: '3d'}

        # Determine if `ndim` is valid
        if ndim not in conv_dim_map:
            # This only supports 1, 2, and 3 dimensions!
            raise ValueError(f"Unsupported ndim={ndim}. Must be 1, 2, or 3.")

        # Dynamically retreive nn.convXd
        conv_cls_name = f"ConvTranspose{conv_dim_map[ndim]}"
        conv_cls = getattr(nn, conv_cls_name)

        # Construct the transposed convolution
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
        Forward pass of the transposed convolutional layer.

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
    A pooling layer that dynamically constructs a pooling operation based
    on the dimensionality and pooling mode specified.

    Attributes
    ----------
    pool : nn.Module
        The pooling operation to apply. It is one of `MaxPool`, `AvgPool`,
        or `LPPool` for 1D, 2D, or 3D inputs.
    """

    def __init__(self, ndim: int, pool_mode: str = 'max', kernel_size=2):
        """
        Initialize the `Pool` module.

        Parameters
        ----------
        ndim : int
            The dimensionality of the pooling operation. Must be 1, 2, or 3.
        pool_mode : str, optional
            The pooling mode to use. Options are 'max' for max pooling,
            'avg' for average pooling, and 'lp' for LP pooling. Default is
            'max'.
        kernel_size : int or tuple, optional
            The size of the pooling kernel. Default is 2.
        """
        super(Pool, self).__init__()

        # Mapping of pooling operations
        pool_map = {
            'max': 'MaxPool', 'avg': 'AvgPool', 'lp': 'LPPool'
            }

        # Determine if pooling operation is supported
        if pool_mode not in pool_map:
            raise ValueError(f"Unsupported pool_mode={pool_mode}. Must be `max`, `avg`, or `lp`.")

        # Mapping of spatial dimensions for pooling operation
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
    Downsampling convoultional block consisting of a `ConvBlock` followed by a pooling layer.

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
        norm: Union[str, nn.Module, None] = None,
        activation: Union[str, nn.Module, None] = "relu",
        pool_mode: str = "max",
        pool_kernel_size: int = 2,
        order='nca',
        return_residual: bool = False,
    ):
        """
        Initialize the `DownsampleConvBlock`.

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
        norm : str, nn.Module, or None, optional
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
        return_residual : bool
            Optionally return a residual (skip connection) from the output of the forward pass.
        """
        super().__init__()
        self.return_residual = return_residual

        self.conv_block = ConvBlock(
            ndim=ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=activation,
            order=order,
        )

        self.pool = Pool(ndim=ndim, pool_mode=pool_mode, kernel_size=pool_kernel_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the downsampling convolutional block.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Downsampled tensor after applying convolution and pooling.
        """

        if self.return_residual:
            conv_resultant = self.conv_block(input_tensor)
            return self.pool(conv_resultant), conv_resultant
        else:
            return self.pool(self.conv_block(input_tensor))


class UpsampleConvBlock(nn.Module):
    """
    Upsampling convoultional block consisting of a `TransposedConvBlock` followed by a
    `ConvBlock`.

    Attributes
    ----------
    upsample : TransposedConv
        The transposed convolutional layer to upsample the feature maps.
    conv_block : ConvBlock
        The convolutional block applying a series of convolutions, normalization, and activation.
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        upsample_kernel_size: int = 4,
        upsample_stride: int = 2,
        upsample_padding: int = 1,
        norm: Union[str, nn.Module, None] = None,
        activation: Union[str, nn.Module, None] = "relu",
        order: str = 'nca',
        accepts_residuals: bool = True,
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
        norm : str, nn.Module, or None, optional
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
        accepts_residuals : bool
            If True, the block is configured to accept residual connections. This doubles the
            expected number of input channels, allowing the block to concatenate skip features with
            the main input.
        """

        super().__init__()

        self.upsample = TransposedConv(
            ndim=ndim,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            padding=upsample_padding,
        )

        if accepts_residuals:
            in_channels += in_channels

        self.conv_block = ConvBlock(
            ndim=ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            activation=activation,
            order=order
        )

    def forward(self, input_tensor: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the upsampling convolutional block.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Upsampled tensor after applying transposed convolution and further convolutions.
        """
        if isinstance(residual, torch.Tensor):

            features = self.upsample(input_tensor)
            features = torch.cat([features, residual], dim=1)

            return self.conv_block(features)
        else:
            return self.conv_block(self.upsample(input_tensor))


class CrossConvBlock(ConvBlock):
    """
    nD Convolutional layer that performs pairwise convolutions between slice elements of two input
    tensors.

    Notes
    -----
    Modified from the original description in https://github.com/JJGO/UniverSeg:
    The pairwise convolution is computed by first forming a Cartesian product of the slices in `x1`
    and `x2`. For example, if `x1` has Sx1 slices and `x2` has Sx2 slices, then the concatenated
    tensor has shape (B, Sx1, Sx2, Cx1 + Cx2, ...). This tensor is reshaped to combine the first
    three dimensions so that the standard nn.Conv*d can be applied. Finally, the output is reshaped
    back to separate the batch and slice indices.

    References
    ----------
    J. G. Ortiz et al., "UniverSeg: Universal Medical Image Segmentation,"
    GitHub repository, 2023. Available: https://github.com/JJGO/UniverSeg

    Examples
    --------
    ### 2D pairwise convolution on CPU
    >>> # Create random 2D inputs: 10 slices (target) and 3 slices (support).
    >>> target_image = torch.randn(2, 10, 4, 64, 64)
    >>> support_image = torch.randn(2, 3, 1, 64, 64)
    >>> cross_conv_block = CrossConvBlock(
    ...     ndim=2, in_channels=(4, 1), out_channels=16, kernel_size=3, padding=1
    ... )
    >>> new_target_image, new_support_image = cross_conv_block(target_image, support_image)
    >>> # Expected output shapes: (2, 10, 16, 64, 64), (2, 3, 16, 64, 64)
    >>> print(new_target_image.shape, new_support_image.shape)
    torch.Size([2, 10, 16, 64, 64]) torch.Size([2, 3, 16, 64, 64])

    ### 3D pairwise convolution on GPU (if available)
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> # Create random 3D inputs: 13 slices (target) and 6 slices (support).
    >>> target_image = torch.randn(1, 13, 1, 64, 64, 64, device=device)
    >>> support_image = torch.randn(1, 6, 1, 64, 64, 64, device=device)
    >>> cross_conv_block = CrossConvBlock(
    ...     ndim=3, in_channels=(1, 1), out_channels=16, kernel_size=3, padding=1
    ... )
    >>> new_target_image, new_support_image = cross_conv_block(target_image, support_image)
    >>> # Expected output shapes: (1, 13, 16, 64, 64, 64), (1, 6, 16, 64, 64, 64)
    >>> print(new_target_image.shape, new_support_image.shape)
    torch.Size([1, 13, 16, 64, 64, 64]) torch.Size([1, 6, 16, 64, 64, 64])
    """

    def __init__(
        self,
        ndim: int,
        in_channels: Tuple[int, int],
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: Union[str, nn.Module, None] = None,
        activation: Union[str, nn.Module, None] = None,
        order: str = 'cna',
    ):
        """
        Initialize the `CrossConvBlock` module.

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

        norm : str, nn.Module, or None, optional
            Defines the normalization layer. Can be one of:
            - A string: Supported options are 'batch', 'instance', 'layer', or 'group'.
            - A `Norm` module: Instantiated or uninstantiated `Norm` layer.
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

        super().__init__(
            ndim=ndim, in_channels=sum(in_channels), out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, norm=norm, activation=activation, order=order
        )

        # Separate ConvBlock to further process the aggregated features
        self.target_conv_block = ConvBlock(
            ndim=ndim, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, norm=norm,
            activation=activation, order=order
        )

        # Separate ConvBlock to further process the aggregated features
        self.support_conv_block = ConvBlock(
            ndim=ndim, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, norm=norm,
            activation=activation, order=order
        )

    def forward(
        self,
        target: torch.Tensor,
        support: torch.Tensor,
    ):
        """
        Compute the cross convolution between target and support inputs.

        This method computes the pairwise convolution between all slices of the
        `target` and `support` tensors. The steps are as follows:
          1. Expand the inputs into all possible slice pairs using `utils.cross_expand`.
          2. Flatten the slice dimensions into the batch dimension and apply the inherited conv
            operation.
          3. Rearrange the convolved output back into separate target and support slice dimensions
            using `einops.rearrange`.
          4. Aggregate the outputs by computing:
            - The new target representation as the average over support slices.
            - The new support representation as the average over target slices.
          5. Refine each branch by processing through their respective ConvBlock modules.

        Parameters
        ----------
        target : torch.Tensor
            Input tensor representing the target of shape (B, Sx1, Cx1, ...), where Sx1 is the
            number of slices or subimages in the target.
        support : torch.Tensor
            Input tensor representing the support with shape (B, Sx2, Cx2, ...), where Sx2 is the
            number of slices or subimages in the support.

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing:
              - new_target: Target features after cross convolution, average over support slices,
                and further convolutions. Has shape (B, S_target, out_channels, ...).
              - new_support: Support features after cross convolution, average over support slices,
                and further convolutions. Has shape (B, S_support, out_channels, ...).
        """

        # Compute all pairs of slices and patch into batch dimension
        batched_paired_tensors = ne.utils.utils.cross_expand(target, support)

        # perform the cross convolution
        batched_output = super().forward(batched_paired_tensors)

        cross_conv_output = einops.rearrange(
            batched_output,
            "(B Sx1 Sx2) C ... -> B Sx1 Sx2 C ...",
            B=target.size(0),
            Sx1=target.size(1),
            Sx2=support.size(1)
        )

        # Average over the support slices/subimages to get the new target
        new_target = cross_conv_output.mean(dim=2)  # New shape: (B, S_t, C, ...)

        # Average over the target slices/subimages to get the new support
        new_support = cross_conv_output.mean(dim=1)  # New shape: (B, S_s, C, ...)

        # Process each branch with more convs!
        new_target = self.target_conv_block(
            new_target.flatten(0, 1)
        ).unflatten(0, new_target.shape[:2])

        new_support = self.support_conv_block(
            new_support.flatten(0, 1)
        ).unflatten(0, new_support.shape[:2])

        return new_target, new_support
