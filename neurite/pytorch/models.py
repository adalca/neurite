"""
Prebuilt yet flexible neural network architectures designed for specific tasks, such as image
segmentation, registration, or classification. models leverage layers and modules from other
components of the neurite for streamlined object construction.
"""

from __future__ import annotations

__all__ = [
    "BasicUNet",
    "BasicAutoencoder",
]

from typing import List, Union, Callable
import torch
from torch import nn
import neurite.pytorch as ne


class BasicUNet(nn.Module):
    """
    Flexible unet with many configuration options!

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
    This UNet is based on the architecture of the UNet found in the paper by
    [Olaf Ronneberger](https://arxiv.org/pdf/1505.04597)

    Examples
    --------
    >>> model = BasicUNet(
    ...     ndim=2, in_channels=1, out_channels=1,
    ...     nb_features=[16, 32, 64],
    ...     norms='instance', activations=nn.ReLU
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
        nb_features: List[int] = (16, 16, 16, 16, 16),
        norms: Union[List[Union[Callable, str]], Callable, str, None] = None,
        activations: Union[List[Union[Callable, str]], Callable, str, None] = nn.ReLU,
        order: str = 'caca',
        final_activation: Union[str, nn.Module, None] = nn.Sigmoid(),
        residual_connections: bool = True,
    ):

        """
        Instantiate `BasicUNet`

        Parameters
        ----------
        ndim : int
            Dimensionality of the input (1, 2, or 3).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        nb_features : List[int]
            Number of features at each level of the unet. Must be a list of
            positive integers.
        norms : Union[List[str], str, None], optional
            Normalization layers to use in each block. Can be a string or a list
            of strings specifying norms for each layer, or `None` for no norm.
        activations : Union[List[str], str, Callable], optional
            Activation functions to use in each block. Can be a callable,
            a string, or a list of strings/callables.
        order : str, optional
            Order of operations in each convolutional block (e.g., 'ncaca').
        residual_connections : bool
            Enable residual connections to communicate information between levels of the
            downsampling and upsampling paths.
        """

        super().__init__()

        # Storing some attributes that might be useful later on
        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Make `residual_connections` an attribute as we will need it later in forward pass
        self.residual_connections = residual_connections

        # Normalization layers
        if not isinstance(norms, list):
            self.norms = [norms] * len(nb_features)

        # Activation layers
        if not isinstance(activations, list):
            self.activations = [activations] * len(nb_features)

        # Original sequence for downsampling conv blocks
        self.nb_features = [in_channels, *nb_features]

        # Inverted sequence for upsampling conv blocks
        self.reversed_features = list(reversed(nb_features))

        # Downsampling convolutional blocks
        self.downsampling_conv_blocks = ne.utils.make_downsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.nb_features,
            norms=self.norms,
            activations=self.activations,
            order=order,
            return_residual=residual_connections,
        )

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        self.lowest_resolution_conv_block = ne.modules.ConvBlock(
            ndim=ndim,
            in_channels=self.nb_features[-1],
            out_channels=self.nb_features[-1],
            order=order,
        )

        # Upsampling convolutional blocks
        self.upsampling_conv_blocks = ne.utils.make_upsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.reversed_features,
            norms=self.norms,
            activations=self.activations,
            order=order,
            upsample_kernel_size=2,
            upsample_stride=2,
            upsample_padding=0,
            accepts_residuals=residual_connections,
        )

        # Final convolutional block
        self.out_layer = ne.modules.ConvBlock(
            ndim=ndim,
            in_channels=nb_features[0],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=final_activation,
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
            if self.residual_connections:
                feature_tensor, residual = downsampling_conv_block(feature_tensor)
                skip_connections.append(residual)  # Save for skip connection
            else:
                feature_tensor = downsampling_conv_block(feature_tensor)

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)

        # Upsampling path
        for i, upsampling_conv_block in enumerate(self.upsampling_conv_blocks):
            if self.residual_connections:
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
        nb_features: List[int] = [16, 16, 16, 16, 16],
        norms: Union[List[Union[Callable, str]], Callable, str, None] = None,
        activations: Union[List[Union[Callable, str]], Callable, str, None] = nn.ReLU,
        order: str = 'caca',
        final_activation: Union[str, nn.Module, None] = nn.Sigmoid(),
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
        nb_features : List[int]
            Number of features at each level of the unet. Must be a list of positive integers.
        norms : Union[List[str], str, None], optional
            Normalization layers to use in each block. Can be a string or a list
            of strings specifying norms for each layer, or `None` for no norm.
        activations : Union[List[str], str, Callable], optional
            Activation functions to use in each block. Can be a callable,
            a string, or a list of strings/callables.
        order : str, optional
            Order of operations in each convolutional block (e.g., 'ncaca').
        final_activation : Union[str, nn.Module, None], optional
            Activation function applied after the last convolution.
        """

        super().__init__()

        # Normalization layers
        if not isinstance(norms, list):
            self.norms = [norms] * len(nb_features)

        # Activation layers
        if not isinstance(activations, list):
            self.activations = [activations] * len(nb_features)

        # Encoder network
        self.downsampling_conv_blocks = ne.utils.make_downsampling_conv_blocks(
            ndim=ndim,
            nb_features=[in_channels, *nb_features],
            norms=self.norms,
            activations=self.activations,
            order=order,
            return_residual=False,
        )

        # Bottleneck layer (latent space)
        bottleneck = ne.modules.ConvBlock(
            ndim=ndim,
            in_channels=nb_features[-1],
            out_channels=latent_features,
            kernel_size=1,
            padding=0,
            activation=activations if callable(activations) else nn.ReLU(),
            order=order,
        )

        # Add bottleneck to downsampling_conv_blocks so users can easily predict the latent space.
        self.downsampling_conv_blocks = self.downsampling_conv_blocks.append(bottleneck)

        # Decoder network
        self.upsampling_conv_blocks = ne.utils.make_upsampling_conv_blocks(
            ndim=ndim,
            nb_features=[latent_features, *reversed(nb_features[1:])],
            norms=self.norms,
            activations=self.activations,
            accepts_residuals=False,
            order=order,
        )

        # Output layer
        self.out_layer = ne.modules.ConvBlock(
            ndim=ndim,
            in_channels=nb_features[1],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=final_activation,
            order=order,
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
