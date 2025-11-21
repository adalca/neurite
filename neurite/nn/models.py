"""
Prebuilt yet flexible neural network architectures designed for specific tasks, such as image
segmentation, registration, or classification. models leverage layers and modules from other
components of the neurite for streamlined object construction.
"""

# Standard library imports
from __future__ import annotations
from typing import List, Union, Callable, Literal, Sequence

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

    Notes
    -----
    `BasicUNet` is derived from the architecture of the UNet described in
    [Olaf Ronneberger](https://arxiv.org/pdf/1505.04597)

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
        padding_mode : {'zeros', 'replicate', 'reflect'}, default='zeros'
            Padding mode for convolutional layers.
        upsample_mode : {'linear', 'transposed', 'nearest'}, default='linear'
            Upsampling mode for decoder path.
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
        ...     nb_features=[[16, 32, 64], [128, 64, 32]],  # Lowest resolution: 64 → 128
        ...     skip_connections=True,  # Fully supports asymmetric architectures!
        ...     activations=nn.ReLU
        ... )
        """

        super().__init__()

        # Storing some attributes that might be useful later on
        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Make `skip_connections` an attribute as we will need it later in forward pass
        self.skip_connections = skip_connections

        # Asymmetric: [[downsampling_features], [upsampling_features]]
        if isinstance(nb_features[0], (list, tuple)):

            if len(nb_features) != 2:
                raise ValueError(
                    f"Asymmetric nb_features must have exactly 2 lists "
                    f"(downsampling and upsampling), got {len(nb_features)}"
                )

            downsampling_features = list(nb_features[0])
            upsampling_features = list(nb_features[1])

        else:
            # Symmetric: [features] used for both downsampling and upsampling
            downsampling_features = list(nb_features)
            upsampling_features = list(reversed(nb_features))

        # Store feature specifications as immutable attributes
        self.downsampling_features = tuple(downsampling_features)
        self.upsampling_features = tuple(upsampling_features)

        # Normalization layers
        if not isinstance(normalizations, list):
            self.normalizations = [normalizations] * len(downsampling_features)

        # Activation layers
        if not isinstance(activations, list):
            self.activations = [activations] * len(downsampling_features)

        # Original sequence for downsampling conv blocks
        self.nb_features = [in_channels, *downsampling_features]

        # Inverted sequence for upsampling conv blocks
        self.reversed_features = upsampling_features

        # Downsampling convolutional blocks
        self.downsampling_conv_blocks = ne.utils.downsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.nb_features,
            normalizations=self.normalizations,
            activations=self.activations,
            order=order,
            return_skip=skip_connections,
            padding_mode=padding_mode,
        )

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        self.lowest_resolution_conv_block = ne.nn.modules.ConvBlock(
            ndim=ndim,
            in_channels=downsampling_features[-1],
            out_channels=upsampling_features[0],
            order=order,
            padding_mode=padding_mode,
        )

        # Compute skip connection channel counts (downsampling features in reverse order)
        skip_channels_list = list(reversed(downsampling_features)) if skip_connections else None

        # Upsampling convolutional blocks
        self.upsampling_conv_blocks = ne.utils.upsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.reversed_features,
            normalizations=self.normalizations,
            activations=self.activations,
            order=order,
            upsample_kernel_size=2,
            upsample_stride=2,
            upsample_padding=0,
            accepts_skip=skip_connections,
            skip_channels=skip_channels_list,
            padding_mode=padding_mode,
            upsample_mode=upsample_mode
        )

        # Final convolutional block
        self.out_layer = ne.nn.modules.ConvBlock(
            ndim=ndim,
            in_channels=upsampling_features[-1],
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

        # Downsampling path
        skip_connections = []

        for downsampling_conv_block in self.downsampling_conv_blocks:
            if self.skip_connections:
                feature_tensor, skip = downsampling_conv_block(feature_tensor)
                skip_connections.append(skip)  # Save for skip connection
            else:
                feature_tensor = downsampling_conv_block(feature_tensor)

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)

        # Upsampling path
        for i, upsampling_conv_block in enumerate(self.upsampling_conv_blocks):
            if self.skip_connections:
                skip = skip_connections[-(i + 1)]
                feature_tensor = upsampling_conv_block(feature_tensor, skip)
            else:
                feature_tensor = upsampling_conv_block(feature_tensor)

        # Output layer
        feature_tensor = self.out_layer(feature_tensor)
        return feature_tensor


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
        ndim: int,
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

        # Encoder network
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

        # Decoder network
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
