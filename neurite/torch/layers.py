import torch
from torch import nn
from torch.nn.modules.utils import _pair, _triple  # _single
from torch.nn.common_types import _size_2_t, _size_3_t  # , _size_1_t
from torch.nn.modules.conv import _ConvNd
from typing import Tuple, Union, Optional
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import math


##########################################
# HyperMorph Layers
# Reproduces the functionality of the corresponding Tensorflow HyperMorph layers,
# but inherits from and is structured according to the class hierarchy of the builtin
# PyTorch Convolutional Layers
##########################################

class HyperConv(_ConvNd):
    """
    Private, abstract N-D hyper-convolution layer for use in hypernetworks.
    This layer has no trainable weights, as it performs a convolution
    using external kernel (and bias) weights that are provided as
    input tensors. The expected layer input is a tensor list:
        [input_features, kernel_weights, bias_weights]
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None,
                 ):

        if padding == 'causal':
            raise ValueError('Causal padding is not supported for HyperConv')

        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         transposed=transposed,
                         output_padding=output_padding,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode,
                         device=device,
                         dtype=dtype)

        self.weight.requires_grad = False
        self.weight: Optional[torch.Tensor] = None
        if bias:
            self.bias.requires_grad = False
            self.bias = None


    def reset_parameters(self) -> None:
        pass


class HyperConvFromDense(HyperConv):
    """
    Private, abstract N-D hyper-convolution wrapping layer that
    includes the dense mapping from a final hypernetwork layer to the
    internal kernel/bias weights. The expected layer input is a
    tensor list:

        [input_features, last_hypernetwork_output]

    Parameters:
        rank: Rank of the convolution.
        filters: The dimensionality of the output space.
        kernel_size: An int or int list specifying the convolution window size.
        hyperkernel_use_bias: Enable bias in hyper-kernel mapping. Default is True.
        hyperbias_use_bias: Enable bias in hyper-bias mapping. Default is True.
        hyperkernel_activation: Activation for the hyper-kernel mapping. Default is None
        hyperbias_activation: Activation for the hyper-bias mapping. Default is None.
        name: Layer name.
        kwargs: Forwarded to the HyperConv constructor.
    """

    def __init__(self,
                 hyp_inputs: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 hyperkernel_use_bias=True,
                 hyperbias_use_bias=True,
                 hyperkernel_activation=None,
                 hyperbias_activation=None,
                 device=None,
                 dtype=None,
                 **kwargs):

        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         transposed=transposed,
                         output_padding=output_padding,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode,
                         device=device,
                         dtype=dtype)

        self.use_bias = bias  # actual bias weight is deleted, so record preference here
        self.hyp_inputs = hyp_inputs
        self.hyperkernel_use_bias = hyperkernel_use_bias
        self.hyperbias_use_bias = hyperbias_use_bias
        self.hyperkernel_activation = get_activation(hyperkernel_activation)
        self.hyperbias_activation = get_activation(hyperbias_activation)

        self._build_hyper_weights(device, dtype)

    def _build_hyper_weights(self, device, dtype):
        """
        Builds a hyper-conv layer from a tensor with two internal dense operations,
        'pseudo dense layers', that predict convolutional kernel (and optional bias
        weights, if the parameter use_bias is set to True).
        """
        nb_hyp_features = self.hyp_inputs  # int(input_shape[1][-1])
        kernel_shape = (self.out_channels, self.in_channels) + self.kernel_size

        # builds the internal dense layer (kernel and bias weights that
        # create the hyper-conv kernel weights)
        self.hyperkernel = HyperWeight(
            nb_hyp_features, kernel_shape, use_bias=self.hyperkernel_use_bias,
            activation=self.hyperbias_activation, device=device, dtype=dtype)

        # builds the internal dense layer (kernel and bias weights that
        # create the hyper-conv bias weights)
        self.hyperbias = None
        if self.use_bias:
            self.hyperbias = HyperWeight(
                nb_hyp_features, [self.out_channels], use_bias=self.hyperbias_use_bias,
                activation=self.hyperbias_activation, device=device, dtype=dtype)

    def forward(self, inputs: Tensor, hyp_tensor) -> Tensor:
        kernel = self.hyperkernel(hyp_tensor)

        bias = None
        if self.use_bias:
            bias = self.hyperbias(hyp_tensor)

        return self._conv_forward(inputs, weight=kernel, bias=bias)



class HyperConv2dFromDense(HyperConvFromDense):
    """
    2D hyper-convolution dense wrapping layer for use in hypernetworks.
    """

    def __init__(self,
                 hyp_inputs: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',  # TODO: refine this type
                 device=None,
                 dtype=None,
                 hyperkernel_use_bias=True,
                 hyperbias_use_bias=True,
                 hyperkernel_activation=None,
                 hyperbias_activation=None,
                 ) -> None:
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            hyp_inputs=hyp_inputs, in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size_, stride=stride_, padding=padding_,
            dilation=dilation_, transposed=False, output_padding=_pair(0),
            groups=groups, bias=bias, padding_mode=padding_mode,
            hyperkernel_use_bias=hyperkernel_use_bias,
            hyperbias_use_bias=hyperbias_use_bias,
            hyperkernel_activation=hyperkernel_activation,
            hyperbias_activation=hyperbias_activation,
            device=device, dtype=dtype)

    def _conv_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(inputs, self._reversed_padding_repeated_twice,
                                  mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(inputs, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)


class HyperConv3dFromDense(HyperConvFromDense):
    """
    3D hyper-convolution dense wrapping layer for use in hypernetworks.
    """

    def __init__(self,
                 hyp_inputs: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = 1,
                 padding: Union[str, _size_3_t] = 0,
                 dilation: _size_3_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 hyperkernel_use_bias=True,
                 hyperbias_use_bias=True,
                 hyperkernel_activation=None,
                 hyperbias_activation=None,
                 ) -> None:
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            hyp_inputs=hyp_inputs, in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size_, stride=stride_, padding=padding_,
            dilation=dilation_, transposed=False, output_padding=_pair(0),
            groups=groups, bias=bias, padding_mode=padding_mode,
            hyperkernel_use_bias=hyperkernel_use_bias,
            hyperbias_use_bias=hyperbias_use_bias,
            hyperkernel_activation=hyperkernel_activation,
            hyperbias_activation=hyperbias_activation,
            device=device, dtype=dtype,
        )

    def _conv_forward(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv3d(F.pad(inputs, self._reversed_padding_repeated_twice,
                                  mode=self.padding_mode),
                            weight, bias, self.stride, _triple(0), self.dilation,
                            self.groups)
        return F.conv3d(inputs, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)


class HyperWeight(nn.Module):
    """
    Class to contain the kernel (and optional bias) for a HyperConv layer
    """

    def __init__(self, nb_hyp_features, target_shape, use_bias, activation, device,
                 dtype):
        """
        Creates weights for an internal dense 'pseudo-layer' described
        in the build() documentation.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # target_shape = tf.TensorShape(target_shape)
        units = np.prod(target_shape)

        # create dense kernel weights
        kernel = nn.Parameter(torch.empty((nb_hyp_features, units), **factory_kwargs))

        # create dense bias weights
        if use_bias:
            bias = nn.Parameter(torch.empty(units, **factory_kwargs))
        else:
            bias = None

        self.kernel = kernel
        self.bias = bias
        if activation is not None:
            activation = activation()
        self.activation = activation
        self.target_shape = target_shape
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        if self.bias is not None:
            # noinspection PyProtectedMember
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernel)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.is_sparse:
            outputs = torch.sparse.mm(inputs, self.kernel)
        else:
            outputs = torch.mm(inputs, self.kernel)

        if self.bias is not None:
            outputs += self.bias

        if self.activation is not None:
            outputs = self.activation(outputs)

        output_weight = torch.reshape(outputs, self.target_shape)
        return output_weight


def get_activation(act):
    if act is None:
        return None
    else:
        return getattr(nn, act)
