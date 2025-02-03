"""
Prebuilt yet flexible neural network architectures designed for specific tasks, such as image
segmentation, registration, or classification. models leverage layers and modules from other
components of the neurite for streamlined object construction.
"""
__all__ = [
    "BasicUNet",
    "VxmDeformable",
]

from typing import List, Union, Tuple
import torch
from torch import nn
from . import modules, utils, layers, random


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
    - This UNet is based on the architecture of the UNet found in the paper by Olaf Ronneberger:
    https://arxiv.org/pdf/1505.04597

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
        norms: Union[List[str], str] = None,
        activations: Union[List[str], str] = nn.ReLU,
        order: str = 'ncaca',
        final_activation: Union[str, nn.Module, None] = nn.Sigmoid(),
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

        Raises
        ------

        """
        super().__init__()
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
        self.downsampling_conv_blocks = utils.make_downsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.nb_features,
            norms=self.norms,
            activations=self.activations,
            order=order
        )

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        self.lowest_resolution_conv_block = modules.ConvBlock(
            ndim=ndim,
            in_channels=self.nb_features[-1],
            out_channels=self.nb_features[-1],
            order=order,
        )

        # Upsampling convolutional blocks
        self.upsampling_conv_blocks = utils.make_upsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.reversed_features,
            norms=self.norms,
            activations=self.activations,
            order=order,
            upsample_kernel_size=2,
            upsample_stride=2,
            upsample_padding=0,
        )

        # Final convolutional block
        self.out_layer = modules.ConvBlock(
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
            feature_tensor, residual = downsampling_conv_block(feature_tensor, return_residual=True)
            skip_connections.append(residual)  # Save for skip connection

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)

        # Upsampling path
        for i, upsampling_conv_block in enumerate(self.upsampling_conv_blocks):
            skip = skip_connections[-(i + 1)]
            feature_tensor = upsampling_conv_block(feature_tensor, skip)

        # Output layer
        feature_tensor = self.out_layer(feature_tensor)
        return feature_tensor


class VxmDeformable(BasicUNet):
    '''
    A network archetecture built on `BasicUNet` to perform nD image registration using a flow field.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).
    in_channels : int
        Number of input channels in the source and target images.
    out_channels : int
        Number of output channels in the flow field.
    *args : list
        Additional positional arguments for the `BasicUNet` constructor.
    nb_features : List[int], optional
        List of integers specifying the number of features in each
        level of the UNet architecture. Default is `[16, 16, 16, 16, 16]`.
    norms : Union[List[str], str], optional
        Normalization layers for the UNet. Can be a list of normalization
        types or a single normalization type. Default is `None`.
    activations : Union[List[str], str], optional
        Activation functions for the UNet layers. Can be a list of
        activation functions or a single function. Default is `nn.ReLU`.
    order : str, optional
        The order of operations in each UNet block. Default is `'ncaca'`.
    final_activation : Union[str, nn.Module, None], optional
        The activation applied to the final output of the network. Default is `None`.
    flow_initializer : ne.random.Sampler, optional
        A custom sampler for initializing the weights of the flow layer.
        If not provided, it defaults to a normal distribution
        with mean 0 and standard deviation `1e-5`.
    integration_steps : int, optional
        Number of steps to take in integrating the flow field. Default is 1.
    **kwargs : dict
        Additional keyword arguments passed to the `BasicUNet` constructor.

    Attributes
    ----------
    flow_layer : nn.Module
        A custom convolutional block used to generate the flow field
        from the combined source and target features.

    Methods
    -------
    forward(source, target)
        Combines source and target images, processes them through the
        UNet and the flow layer, and returns the resulting flow field.
    '''

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        nb_features: List[int] = (16, 16, 16, 16, 16),
        norms: Union[List[str], str] = None,
        activations: Union[List[str], str] = nn.ReLU,
        order: str = 'caca',
        final_activation: Union[str, nn.Module, None] = None,
        flow_initializer: Union[float, random.Sampler] = random.Normal(0, 1e-5),
        bidirectional_cost: bool = False,
        integration_steps: int = 0,
        resize_integrated_fields: bool = False,
        device: str = "cpu",
    ):

        """
        Initialize the `VxmDeformable`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the input (1, 2, or 3).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        expected_moving_shape : tuple[int]
            The expected shape of the `moving_tensor` input to the forward method of this class.
            without batch or channel dimensions. Used to initialize the `VecInt` integrator.
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
            The order of operations in each convolutional block. Default is 'cna'
            (normalization -> convolution -> activation). Each character in the string represents
            one of the following:
            - `'c'`: Convolution
            - `'n'`: Normalization
            - `'a'`: Activation
        bidirectional_cost : bool, optional
            Enable calculation of the cost-function bidirectionally. Default is False
        integration_steps : int, optional
            Number of scaling and squaring steps for integrating the flow field.
            Default is 0 (no integration).
        device : str, optional
            Device identifier (e.g., 'cpu' or 'cuda') to place/run the model on.
        """

        super().__init__(
            ndim=ndim, in_channels=in_channels, out_channels=out_channels, nb_features=nb_features,
            norms=norms, activations=activations, order=order, final_activation=final_activation
        )

        # Configure bidirectional cost/training
        self.integration_steps = integration_steps
        self.bidirectional_cost = bidirectional_cost
        self.resize_integrated_fields = resize_integrated_fields
        self.device = device

        self._init_flow_layer(ndim, out_channels, flow_initializer)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        register: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of `VxmDeformable`.

        The forward pass concatenates the `source` and `target` images, passes them through the
        `BasicUNet` backbone, applies a flow layer to obtain the flow (velocity) field, then warps
        the images according to the `register` argument.

        Parameters
        ----------
        source : torch.Tensor
            2D or 3D Source image tensor with batch and channel dimensions to be registered/warped
            to the target image.
        target : torch.Tensor
            Image to which `source` is registered/warped. Same shape as `source`.
        register : bool, optional
            If `True`, returns the registered source image along with the predicted positive flow
            field.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            - If `register=True`, returns a tuple of
                - `warped_source`
                - `pos_flow`
            - If `register=False` and `bidirectional_cost=False`, returns a tuple of:
                - `warped_source`
                - `preintegrated_flow`
            - If `register=False` and `bidirectional_cost=True`, returns a tuple of:
                - `warped_source`
                - `preintegrated_flow`
                - `warped_target`
        """

        if not hasattr(self, 'flow_layer'):
            raise RuntimeError(
                "The `flow_layer` is not initialized. Ensure a valid `flow_initializer` "
                "is passed during initialization or load a trained model from a checkpoint."
            )

        # Concat the source and target along channel dimension
        combined_features = torch.cat([source, target], dim=1)

        # Pass combined features through the `BasicUNet` backbone
        combined_features = super().forward(combined_features)

        # Apply flow layer to get the positive flow field `pos_flow`
        pos_flow = self.flow_layer(combined_features)

        # Keep a copy of the flow before it's integrated
        preintegrated_flow = torch.clone(pos_flow)

        # For bidirectional cost mode, prepare negative flow (target->source)
        neg_flow = -pos_flow if self.bidirectional_cost else None

        # Optionally integrate
        if self.integration_steps > 0:
            pos_flow, neg_flow = self._integrate_velocity_fields(pos_flow, neg_flow)

        # Perform the warping operations for the source and target
        warped_source = self._spatial_transform(source, pos_flow)

        # Warp the target image using the negative flow if needed
        warped_target = self._spatial_transform(
            target, neg_flow
        ) if self.bidirectional_cost else None

        # Prepare the output based on the 'register' flag and cost mode
        output_list = [warped_source]

        if register:
            # output_list: [warped_source, pos_flow]
            output_list.append(pos_flow)
        else:
            # output_list: [warped_source, preintegrated_flow]
            output_list.append(preintegrated_flow)
            if self.bidirectional_cost:
                # output_list: [warped_source, preintegrated_flow, warped_target]
                output_list.append(warped_target)

        return output_list

    def _init_flow_layer(
        self,
        ndim: int,
        features: int,
        flow_initializer: Union[float, random.Sampler] = random.Normal(0, 1e-5)
    ):

        """
        Initialize the flow layer with custom weight initialization (by sampling
        `flow_initializer`).

        This layer is a convolutional block that produces a displacement (flow)
        field. The weights of its initial convolution are sampled using the
        provided flow_initializer, and biases are set to zero.

        Parameters
        ----------
        ndim : int
            **Spatial** dimensionality of the input (1, 2, or 3).
        features : int
            Number of input and output features for the flow layer.
        flow_initializer :  Union[float, ne.random.Sampler], optional
            Sampler for initializing the *weights* of the flow layer. Default is
            `ne.random.Normal(0, 1e-5)`.
        """

        # Initialize the conv ("flow") layer with congruent in and out features
        flow_layer = modules.ConvBlock(ndim, features, features).to(self.device)

        # Optionally, apply custom initialization if `flow_initializer`` is provided
        if flow_initializer is not None:

            # Make the distribution to sample the flow parameters
            flow_initializer = random.Fixed.make(flow_initializer)

            # Sample the weight parameters from the distribution for first (and only) conv
            flow_layer.conv0.weight = nn.Parameter(
                flow_initializer(flow_layer.conv0.weight.shape)
            ).to(self.device)

            # Set the bias term(s) to zero for the first (and only) conv
            flow_layer.conv0.bias = nn.Parameter(
                torch.zeros(flow_layer.conv0.bias.shape)
            ).to(self.device)

        # Register the flow layer as a submodule
        self.add_module("flow_layer", flow_layer)

    def _integrate_velocity_fields(
        self,
        pos_flow: torch.Tensor,
        neg_flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate the velocity fields to obtain diffeomorphic warp (displacement) fields.

        Derive a smooth and invertable displacement field by integrating a velocity field via the
        scaling and squaring method. If no `IntegrateVelocityField` object exists in the model's
        state dictionary, instantiate it with the correct size of the input and insert it. This will
        only happen once upon initial call of this method.

        Parameters
        ----------
        pos_flow : torch.Tensor
            Positive flow (velocity) field (source -> target).
        neg_flow : torch.Tensor
            Negative flow (velocity) field (target -> source).

        Returns
        -------
        torch.Tensor
            Displacement field obtained by integrating the velocity field via scaling and squaring.
        """

        # If the velocity integrator is not defined, dynamically construct it
        if not hasattr(self, "velocity_field_integrator"):

            # Dynamically construct the integrator based on the spatial shape
            velocity_field_integrator = layers.IntegrateVelocityField(
                shape=pos_flow.shape[2:], steps=self.integration_steps, device=self.device
            )

            # Add it to the module
            self.add_module("velocity_field_integrator", velocity_field_integrator)

        # Integrate the positive flow
        pos_flow = self.velocity_field_integrator(pos_flow)

        # Integrate the negative velocity field if bidirectional cost is enabled
        neg_flow = self.velocity_field_integrator(neg_flow) if self.bidirectional_cost else None

        return pos_flow, neg_flow

    def _spatial_transform(
        self,
        moving_image: torch.Tensor,
        deformation_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp an image tensor using a deformation/displacement field.

        This method applies a spatial transformation to the provided image tensor based on the
        deformation field using a SpatialTransformer. If no `IntegrateVelocityField` object exists
        in the model's state dictionary, instantiate one with the correct size of the input and
        register it as a submodule. This will only happen once upon the initial call of this method.

        Parameters
        ----------
        moving_image : torch.Tensor
            Image tensor to be warped, with shape (B, C, ...).
        deformation_field : torch.Tensor
            Displacement field used for warping, with shape matching the spatial dimensions of
            `moving_image`.

        Returns
        -------
        torch.Tensor
            The warped image tensor.
        """

        if not hasattr(self, "spatial_transformer"):
            # Dynamically construct the spatial transformer with the correct spatial shape
            spatial_transformer = layers.SpatialTransformer(
                size=moving_image.shape[2:], device=self.device
            )

            # Register it as a submodule
            self.add_module("spatial_transformer", spatial_transformer)

        # Warp the moving image with the deformation field
        warped_image = self.spatial_transformer(moving_image, deformation_field)

        return warped_image
