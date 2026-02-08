"""
Prebuilt yet flexible neural network architectures designed for specific tasks, such as image
segmentation, registration, or classification. models leverage layers and modules from other
components of the neurite for streamlined object construction.
"""

# Standard library imports
from __future__ import annotations
from typing import cast, Union, Callable, Literal, Sequence, Tuple, List

# Third party imports
import torch
from torch import nn

# Custom imports
import neurite as ne


class BasicUNet(nn.Module):
    """
    Flexible UNet with many configuration options.

    Attributes
    ----------
    downsampling_conv_blocks : nn.ModuleList
        Downsampling convolutional blocks.
    lowest_resolution_conv_block : nn.Module
        Central convolutional block at the lowest spatial resolution.
    upsampling_conv_blocks : nn.ModuleList
        Upsampling convolutional blocks.
    out_layer : nn.Module
        Final output layer.
    down_actual_channels : list[int]
        Actual output channel counts for each downsampling block (accounts for pass-through levels).

    Notes
    -----
    `BasicUNet` is derived from the architecture of the UNet described in
    [Olaf Ronneberger](https://arxiv.org/pdf/1505.04597)

    Use `0` in `nb_features` to indicate pass-through levels that skip convolutions:
    - Downsampling `0`: Pool only, no convolution. Channels preserved from input.
    - Upsampling `0`: Upsample + concat skip only, no conv. Output = upsampled + skip channels.

    Examples
    --------
    >>> model = BasicUNet(
    ...     ndim=2, in_channels=1, out_channels=1,
    ...     nb_features=[16, 32, 64],
    ...     normalizations='instance', activations=nn.ReLU
    ... )
    >>> input_tensor = torch.randn(1, 1, 128, 128)
    >>> output = model(input_tensor)
    >>> output.shape
    torch.Size([1, 1, 128, 128])

    >>> # Skip full-resolution convolutions using 0
    >>> model = BasicUNet(
    ...     ndim=2, in_channels=1, out_channels=1,
    ...     nb_features=[0, 32, 64],  # Pass-through at full resolution
    ... )
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        nb_features: Union[Sequence[int], Sequence[Sequence[int]]] = (16, 16, 16, 16, 16),
        padding_mode: Literal['zeros', 'replicate', 'reflect'] = 'zeros',
        upsample_mode: Literal['linear', 'transposed', 'nearest'] = 'linear',
        normalizations: Union[Sequence[Union[Callable, str]], Callable, str, None] = None,
        activations: Union[Sequence[Union[Callable, str]], Callable, str, None] = nn.ReLU,
        order: str = 'ca',
        final_activation: Union[str, nn.Module, None] = None,
        skip_connections: bool = True,
    ):
        """
        Initialize `BasicUNet`

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions of the input (1, 2, or 3).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        nb_features : Union[Sequence[int], Sequence[Sequence[int]]], default=(16, 16, 16, 16, 16)
            Number of features at each level of the unet. Can be:
            - Single sequence: [16, 32, 64] (symmetric - downsampling uses [16, 32, 64],
              upsampling uses [64, 32, 16])
            - Sequence of sequences: [[downsampling_features], [upsampling_features]] for complete
              asymmetry
            Use `0` to indicate pass-through levels (no convolution):
            - Downsampling: [0, 32, 64] skips conv at first level (pool only, channels preserved)
            - Upsampling: [64, 32, 0] skips conv at last level (upsample + skip concat only)
        padding_mode : {'zeros', 'replicate', 'reflect'}, default='zeros'
            Padding mode for convolutional layers.
        upsample_mode : {'linear', 'transposed', 'nearest'}, default='linear'
            Upsampling mode for upsampling path.
        normalizations : Sequence[Union[Callable, str]], Callable, str, or None, default=None
            Normalization layers to use in each block. Can be a string or a sequence
            of strings specifying normalizations for each layer, or `None` for no normalization.
        activations : Sequence[Union[Callable, str]], Callable, str, or None, default=nn.ReLU
            Activation functions to use in each block. Can be a callable,
            a string, or a sequence of strings/callables.
        order : str, default='ca'
            Order of operations in each convolutional block (e.g., 'ncaca').
        final_activation : Union[str, nn.Module, None], default=None
            Activation function applied after the final output layer.
        skip_connections : bool, default=True
            Enable skip connections to concatenate features from downsampling path with upsampling
            path at matching resolutions.

        Examples
        --------
        >>> # Symmetric UNet (default behavior)
        >>> model = BasicUNet(
        ...     ndim=2, in_channels=1, out_channels=1,
        ...     nb_features=[16, 32, 64],
        ...     activations=nn.ReLU
        ... )
        >>> # Asymmetric UNet with different downsampling/upsampling features
        >>> model = BasicUNet(
        ...     ndim=2, in_channels=1, out_channels=1,
        ...     nb_features=[[16, 32, 64], [128, 64, 32]],  # Lowest resolution: 64 -> 128
        ...     skip_connections=True,  # Fully supports asymmetric architectures!
        ...     activations=nn.ReLU
        ... )
        >>> # Skip full-resolution convolutions
        >>> model = BasicUNet(
        ...     ndim=2, in_channels=1, out_channels=1,
        ...     nb_features=[0, 32, 64],  # First level: pool only, no conv
        ... )
        """
        super().__init__()

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_connections = skip_connections

        # Asymmetric: [[down], [up]]
        if isinstance(nb_features[0], Sequence) and isinstance(nb_features[1], Sequence):
            assert len(nb_features) == 2, (
                f"Asymmetric nb_features requires 2 sequences, got {len(nb_features)}"
            )
            downsampling_features = tuple(nb_features[0])
            upsampling_features = tuple(nb_features[1])
        else:
            # Symmetric: reversed for upsampling
            downsampling_features = tuple(nb_features)
            upsampling_features = tuple(reversed(nb_features))

        assert all(isinstance(f, int) for f in downsampling_features), (
            "All elements in nb_features must be integers"
        )

        # Store feature specs as immutables
        self.downsampling_features = tuple(downsampling_features)
        self.upsampling_features = tuple(upsampling_features)

        if not isinstance(normalizations, list):
            normalizations = [normalizations] * len(self.downsampling_features)

        if not isinstance(activations, list):
            activations = [activations] * len(self.downsampling_features)

        self.normalizations = normalizations
        self.activations = activations
        self.nb_features_fwd: list = [in_channels, *self.downsampling_features]

        # Downsampling conv blocks
        self.downsampling_conv_blocks, down_actual_channels = ne.utils.downsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.nb_features_fwd,
            normalizations=self.normalizations,
            activations=self.activations,
            order=order,
            return_skip=skip_connections,
            padding_mode=padding_mode,
        )

        # Store actual downsampling channels for reference
        self.down_actual_channels = down_actual_channels

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        down_output_channels = down_actual_channels[-1]

        # Determine first upsampling feature (0 is falsy)
        bottleneck_out_channels = self.upsampling_features[0] or down_output_channels

        self.lowest_resolution_conv_block = ne.nn.modules.ConvBlock(
            ndim=ndim,
            in_channels=down_output_channels,
            out_channels=bottleneck_out_channels,
            order=order,
            padding_mode=padding_mode,
        )

        # Compute skip connection channel counts from actual downsampling outputs (reversed)
        skip_channels_list = list(reversed(down_actual_channels)) if skip_connections else None

        # Upsampling convolutional blocks (returns blocks and actual channel counts)
        self.upsampling_conv_blocks, up_actual_channels = ne.utils.upsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.upsampling_features,
            normalizations=self.normalizations,
            activations=self.activations,
            order=order,
            upsample_kernel_size=2,
            upsample_stride=2,
            upsample_padding=0,
            accepts_skip=skip_connections,
            skip_channels=skip_channels_list,
            padding_mode=padding_mode,
            upsample_mode=upsample_mode,
            in_channels=bottleneck_out_channels,
        )

        # Store actual upsampling channels for reference
        self.up_actual_channels = up_actual_channels

        # Final convolutional block - use actual upsampling output channels
        if up_actual_channels:
            up_output_channels = up_actual_channels[-1]
        else:
            up_output_channels = bottleneck_out_channels

        self.out_layer = ne.nn.modules.ConvBlock(
            ndim=ndim,
            in_channels=up_output_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=final_activation,
            padding_mode=padding_mode,
        )

    def forward(self, feature_tensor: torch.Tensor):
        """
        Forward pass through the `BasicUNet` model.

        Parameters
        ----------
        feature_tensor : torch.Tensor
            Tensor to be passed through the model. Assumed to have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            Result of forward pass of the model.
        """
        skip_connections = []

        # Downsampling path
        if self.skip_connections:
            for block in self.downsampling_conv_blocks:
                feature_tensor, skip = block(feature_tensor)
                skip_connections.append(skip)
        else:
            for block in self.downsampling_conv_blocks:
                feature_tensor = block(feature_tensor)

        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)

        # Upsampling path
        if self.skip_connections:
            for block, skip in zip(self.upsampling_conv_blocks, reversed(skip_connections)):
                feature_tensor = block(feature_tensor, skip)
        else:
            for block in self.upsampling_conv_blocks:
                feature_tensor = block(feature_tensor)

        return self.out_layer(feature_tensor)


class BasicAutoencoder(nn.Module):
    """
    Flexible autoencoder.

    Attributes
    ----------
    downsampling_conv_blocks : nn.ModuleList
        Downsampling convolutional blocks.
    lowest_resolution_conv_block : nn.Module
        Central convolutional block at the lowest spatial resolution.
    upsampling_conv_blocks : nn.ModuleList
        Upsampling convolutional blocks.
    out_layer : nn.Module
        Final output layer.

    Examples
    --------
    >>> autoencoder = BasicAutoencoder(
    ...    ndim=3,
    ...    in_channels=1,
    ...    latent_features=4,
    ...    out_channels=1,
    ...    activations="elu"
    ... )
    >>> input_tensor = torch.randn(1, 1, 64, 64, 64)
    >>> output = model(input_tensor)
    >>> output.shape
    torch.Size([1, 1, 64, 64, 64])
    """

    def __init__(
        self,
        ndim: Literal[1, 2, 3],
        in_channels: int,
        latent_features: int,
        out_channels: int,
        nb_features: Sequence[int] = (16, 16, 16, 16, 16),
        normalizations: Union[Sequence[Union[Callable, str]], Callable, str, None] = None,
        activations: Union[Sequence[Union[Callable, str]], Callable, str, None] = nn.ReLU,
        order: str = 'caca',
        final_activation: Union[str, nn.Module, None] = None,
        padding_mode: str = 'zeros',
    ):
        """
        Instantiate `BasicAutoencoder`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the input (1, 2, or 3).
        in_channels : int
            Number of input channels.
        latent_features : int
            Number of features/channels in the latent space.
        out_channels : int
            Number of output channels.
        nb_features : Sequence[int], default=(16, 16, 16, 16, 16)
            Number of features at each level of the autoencoder. Must be a sequence of positive
            integers.
        normalizations : Sequence[Union[Callable, str]], Callable, str, or None, default=None
            Normalization layers to use in each block. Can be a string or a sequence
            of strings specifying normalizations for each layer, or `None` for no normalization.
        activations : Sequence[Union[Callable, str]], Callable, str, or None, default=nn.ReLU
            Activation functions to use in each block. Can be a callable,
            a string, or a sequence of strings/callables.
        order : str, default='caca'
            Order of operations in each convolutional block (e.g., 'ncaca').
        final_activation : Union[str, nn.Module, None], default=None
            Activation function applied after the last convolution.
        padding_mode : str, default='zeros'
            Padding mode for convolutional layers.
        """

        super().__init__()

        # Normalization layers
        if not isinstance(normalizations, list):
            self.normalizations = [normalizations] * len(nb_features)

        # Activation layers
        if not isinstance(activations, list):
            self.activations = [activations] * len(nb_features)

        # Downsampling network
        self.downsampling_conv_blocks = ne.utils.downsampling_conv_blocks(
            ndim=ndim,
            nb_features=[in_channels, *nb_features],
            normalizations=self.normalizations,
            activations=self.activations,
            order=order,
            return_skip=False,
        )

        # Latent space layer (lowest resolution, highest feature dimension)
        latent_layer = ne.nn.modules.ConvBlock(
            ndim=ndim,
            in_channels=nb_features[-1],
            out_channels=latent_features,
            kernel_size=1,
            padding=0,
            activation=activations if callable(activations) else nn.ReLU(),
            order=order,
            padding_mode=padding_mode,
        )

        # Add latent layer to downsampling_conv_blocks so users can easily predict the latent space.
        self.downsampling_conv_blocks.append(latent_layer)

        # Upsampling network
        self.upsampling_conv_blocks = ne.utils.upsampling_conv_blocks(
            ndim=ndim,
            nb_features=[latent_features, *reversed(nb_features[1:])],
            normalizations=self.normalizations,
            activations=self.activations,
            accepts_skip=False,
            order=order,
        )

        # Output layer
        self.out_layer = ne.nn.modules.ConvBlock(
            ndim=ndim,
            in_channels=nb_features[1],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=final_activation,
            order=order,
            padding_mode=padding_mode,
        )

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the `BasicAutoencoder` model.

        Parameters
        ----------
        feature_tensor : torch.Tensor
            Tensor to be passed through the model. Assumed to have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            Result of forward pass of the model.
        """

        # Downsampling path
        for downsampling_conv_block in self.downsampling_conv_blocks:
            feature_tensor = downsampling_conv_block(feature_tensor)

        # Decode
        for upsampling_conv_block in self.upsampling_conv_blocks:
            feature_tensor = upsampling_conv_block(feature_tensor)

        # Output layer
        feature_tensor = self.out_layer(feature_tensor)

        return feature_tensor
