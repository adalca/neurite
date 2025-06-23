"""
Modules are simple operations containing learnable parameters. The `modules` module contains general
nD building blocks for neural networks.
"""

__all__ = [
    "Activation",
    "ConvBlock",
    "TransposedConv",
    "Pool",
    "DownsampleConvBlock",
    "UpsampleConvBlock",
    "ContextCrossConv"
]

from typing import List, Union, Type, Tuple, Literal
import importlib
import einops
import torch
from torch import nn
from torch.nn.modules import activation
import neurite as ne


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

        # Assign required instance attributes
        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = nn.ModuleDict()
        self.order = list(order)  # make string of letters into list of letters
        valid_operations = ['c', 'n', 'a']

        # Get the number of activations to validate `activation` argument
        n_activations = order.count('a')

        # If `activation` is a single object, make it into a list with `n_activation` elements
        if not isinstance(activation, (list, tuple)):
            activation = [activation] * n_activations

        else:
            # Hmm. Not sure what the user passed!
            assert len(activation) == n_activations, (
                "The total number of activations passed to `activation` must be the same number ",
                f"defined in `order`. Got activation={activation}, order={order}"
            )

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
                    padding, dilation, groups, bias, padding_mode=padding_mode
                )

                in_channels = out_channels  # All future convs and stuff will have this many in
                conv_id += 1  # Increment conv id for easy tracking/accessing

            # Dynamically construct the normalization and assign it to a named key in layers
            elif operation == 'n' and normalization is not None:
                layers[f'normalization{norm_id}'] = ne.utils.utils.build_normalization(
                    normalization_type=normalization,
                    ndim=ndim,
                    num_features=in_channels,
                )
                norm_id += 1

            # Construct the activation and assign it to a named key in layers
            elif operation == 'a' and activation is not None:
                layers[f'activation{act_id}'] = Activation(activation[act_id])
                act_id += 1

        # Add layers to `Sequential`
        for name, layer in layers.items():
            self.add_module(name, layer)


class TransposedConv(nn.Module):
    """
    Dynamically construct a transposed convolution {`ConvTranspose1d`,
    `ConvTranspose2d`, `ConvTranspose3d`} based on the number of input dimensions.
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
        return_residual: bool = False,
    ):
        """the
        Initialize  `DownsampleConvBlock`.

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

        if self.return_residual:
            conv_resultant = self.conv_block(input_tensor)
            return self.pool(conv_resultant), conv_resultant
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
        accepts_residuals : bool
            If True, the block is configured to accept residual connections. This doubles the
            expected number of input channels, allowing the block to concatenate skip features with
            the main input.
        """

        super().__init__()

        # Choose upsampling strategy
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

            # align_corners only applies to non-nearest modes
            align = None if upsample_mode == 'nearest' else True

            self.upsample = nn.Upsample(
                scale_factor=scale_factor,
                mode=upsample_mode,
                align_corners=align
            )

        # Double channels if there's a residual connection
        if accepts_residuals:
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
            Upsampled tensor after applying upsampling operation and conv blocks.
        """
        if isinstance(residual, torch.Tensor):

            features = self.upsample(input_tensor)
            features = torch.cat([features, residual], dim=1)

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
    >>> context_segmentations = ne.samplers.RandInt()((3, 6, 1, 64, 64))
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
    >>> context_segmentations = ne.samplers.RandInt()((1, 9, 1, 128, 128, 128))
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
        super().__init__()

        self.cross_conv = ConvBlock(
            ndim=ndim, in_channels=sum(in_channels), out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, normalization=normalization, activation=activation, order=order,
            padding_mode=padding_mode,
        )

        # Separate ConvBlock to further process the aggregated features
        self.query_conv_block = ConvBlock(
            ndim=ndim, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, normalization=normalization,
            activation=activation, order=order, padding_mode=padding_mode,
        )

        # Separate ConvBlock to further process the aggregated features
        self.context_conv_block = ConvBlock(
            ndim=ndim, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, normalization=normalization,
            activation=activation, order=order, padding_mode=padding_mode,
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cross convolution between the query image and the context set.

        This method computes the cross convolution between the `query` image and the members of the
        `context` set. The steps are as follows:

        1. Interact the inputs and gather in the batch dimension using
        [`cross_expand`][neurite.pytorch.utils.utils.cross_expand]
        2. Perform the [cross] convolution operation.
        3. Rearrange the output back into separate query features and context set features, but
            with `out_channels` number of features.
        4. Reduce the outputs such that:
            - The new query representation as the average over context representations.
            - The new context representation as the average over query representations.
        5. Refine each branch by processing through their respective ConvBlock modules.

        Parameters
        ----------
        query : torch.Tensor
            Input tensor representing a query image of shape (B, Sq, Cq, ...), where Sq always
            equals 1, and Cq represents the number of image features.
        context : torch.Tensor
            Tensor representing the context set of shape (B, Sc, Cc, ...), where Sc is the
            number of members in the context set (usually 2 for image and label).

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing:
              - `new_query`: query features after cross convolution and reducing over context dim,
                followed by further convolutions. Has shape (B, Sq, Cq, out_channels, ...).
              - `new_context`: Support features after cross convolution and reducing over query dim,
                followed by further convolutions. Has shape (B, Sc, Cc, out_channels, ...).
        """

        # Prepare the features to be interacted by crossing them
        batched_paired_tensors = ne.utils.cross_expand(       # (B*Sq*Sc, Cq+Cc, ...)
            query,
            context
        )

        # Interact the features by performing cross convolution and taking advantage of batch dim
        cross_conv_output = self.cross_conv(                        # (B*Sq*Sc, out_channels, ...)
            batched_paired_tensors
        )

        # Rearrange the response into separate query features and context set features
        cross_conv_output = einops.rearrange(                       # (B, Sq, Sc, out_channels, ...)
            cross_conv_output,
            "(B Sq Sc) C ... -> B Sq Sc C ...",
            B=query.size(0),
            Sq=query.size(1),
            Sc=context.size(1)
        )

        # Reduce output to obtain new query representation by averageing over the context set
        new_query = cross_conv_output.mean(dim=2)                   # (B, Sq, out_channels, ...)

        # Reduce output to obtain new context representation by averageing over the query image(s)
        new_context = cross_conv_output.mean(dim=1)                 # (B, Sc, out_channels, ...)

        # Process each branch with more convs!
        new_query = self.query_conv_block(                          # (B, Sq, out_channels, ...)
            new_query.flatten(0, 1)
        ).unflatten(0, new_query.shape[:2])

        new_context = self.context_conv_block(                      # (B, Sc, out_channels, ...)
            new_context.flatten(0, 1)
        ).unflatten(0, new_context.shape[:2])

        return new_query, new_context
