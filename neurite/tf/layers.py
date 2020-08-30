"""
tensorflow/keras utilities for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

or for the transformation/integration functions:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

# internal python imports
import sys
import itertools

# third party
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputLayer, Input
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend
from tensorflow import roll as _roll

# local imports
from . import utils


class Negate(Layer):
    """ 
    Keras Layer: negative of the input
    """

    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Negate, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -x

    def compute_output_shape(self, input_shape):
        return input_shape


class RescaleValues(Layer):
    """ 
    Very simple Keras layer to rescale data values (e.g. intensities) by fixed factor
    """

    def __init__(self, resize, **kwargs):
        self.resize = resize
        super(RescaleValues, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'resize': self.resize})
        return config

    def build(self, input_shape):
        super(RescaleValues, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.resize 

    def compute_output_shape(self, input_shape):
        return input_shape


class Resize(Layer):
    """
    N-D Resize Tensorflow / Keras Layer
    Note: this is not re-shaping an existing volume, but resizing, like scipy's "Zoom"

    If you find this function useful, please cite:
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,Dalca AV, Guttag J, Sabuncu MR
        CVPR 2018  

    Since then, we've re-written the code to be generalized to any 
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 zoom_factor,
                 interp_method='linear',
                 **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
        """
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        super(Resize, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'zoom_factor': self.zoom_factor,
            'interp_method': self.interp_method,
        })
        return config

    def build(self, input_shape):
        """
        input_shape should be an element of list of one inputs:
        input1: volume
                should be a *vol_shape x N
        """

        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('Resize must be called on a list of length 1.')

        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        # set up number of dimensions
        self.ndims = len(input_shape) - 2
        self.inshape = input_shape
        if not isinstance(self.zoom_factor, (list, tuple)):
            self.zoom_factor = [self.zoom_factor] * self.ndims
        else:
            assert len(self.zoom_factor) == self.ndims, \
                'zoom factor length {} does not match number of dimensions {}'\
                    .format(len(self.zoom_factor), self.ndims)

        # confirm built
        self.built = True

        super(Resize, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, inputs):
        """
        Parameters
            inputs: volume of list with one volume
        """

        # check shapes
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs has to be len 1. found: %d" % len(inputs)
            vol = inputs[0]
        else:
            vol = inputs

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[1:]])

        # map transform across batch
        return tf.map_fn(self._single_resize, vol, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        
        output_shape = [input_shape[0]]
        output_shape += [int(input_shape[1:-1][f] * self.zoom_factor[f]) for f in range(self.ndims)]
        output_shape += [input_shape[-1]]
        return tuple(output_shape)

    def _single_resize(self, inputs):
        return utils.resize(inputs, self.zoom_factor, interp_method=self.interp_method)

# Zoom naming of resize, to match scipy's naming
Zoom = Resize


class MSE(Layer):
    """ 
    Keras Layer: mean squared error
    """

    def __init__(self, **kwargs):
        super(MSE, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MSE, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.mean(K.batch_flatten(K.square(x[0] - x[1])), -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], )


class GaussianBlur(Layer):
    """ 
    Applies gaussian blur to an input image.
    TODO: switch from 'level' based to 'sigma' based
    """

    def __init__(self, level=None, sigma=None, **kwargs):
        assert sigma is not None or level is not None, 'sigma or level must be provided'
        assert not (sigma is not None and level is not None), 'only sigma or level must be provided'

        if level is not None:
            if level < 1:
                raise ValueError('Gaussian blur level must not be less than 1')

            self.sigma = (level - 1) ** 2

        else:
            if sigma < 0:
                raise ValueError('Gaussian blur sigma must not be less than 0')
                
            self.sigma = sigma

        super().__init__(**kwargs)

    def build(self, input_shape):
        ndims = len(input_shape) - 2
        
        # prepare kernel
        kernel = utils.gaussian_kernel([self.sigma] * ndims)
        kernel = tf.reshape(kernel, kernel.shape.as_list() + [1, 1])

        # prepare convoperation
        convnd = getattr(tf.nn, 'conv%dd' % ndims)
        self.conv = lambda x : convnd(tf.expand_dims(x, -1), kernel, [1] * (ndims + 2), padding='SAME')
        self.nfeat = input_shape[-1]

    def call(self, x):
        # TODO: switch to mutli-line if statement
        return x if self.sigma == 0 else tf.concat([self.conv(x[..., n]) for n in range(self.nfeat)], -1)

    def compute_output_shape(self, input_shape):
        return input_shape



#########################################################
# Sparse layers
#########################################################

class SpatiallySparse_Dense(Layer):
    """ 
    Spatially-Sparse Dense Layer (great name, huh?)
    This is a Densely connected (Fully connected) layer with sparse observations.

    # layer can (and should) be used when going from vol to embedding *and* going back.
    # it will account for the observed variance and maintain the same weights

    # if going vol --> enc:
    # tensor inputs should be [vol, mask], and output will be a encoding tensor enc
    # if going enc --> vol:
    # tensor inputs should be [enc], and output will be vol
    """

    def __init__(self, input_shape, output_len, use_bias=False, 
                 kernel_initializer='RandomNormal',
                 bias_initializer='RandomNormal', **kwargs):
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.output_len = output_len
        self.cargs = 0
        self.use_bias = use_bias
        self.orig_input_shape = input_shape  # just the image size
        super(SpatiallySparse_Dense, self).__init__(**kwargs)

    def build(self, input_shape):



        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='mult-kernel',
                                    shape=(np.prod(self.orig_input_shape),
                                           self.output_len),
                                    initializer=self.kernel_initializer,
                                    trainable=True)

        M = K.reshape(self.kernel, [-1, self.output_len])  # D x d
        mt = K.transpose(M) # d x D
        mtm_inv = tf.matrix_inverse(K.dot(mt, M))  # d x d
        self.W = K.dot(mtm_inv, mt) # d x D

        if self.use_bias:
            self.bias = self.add_weight(name='bias-kernel',
                                        shape=(self.output_len, ),
                                        initializer=self.bias_initializer,
                                        trainable=True)

        # self.sigma_sq = self.add_weight(name='bias-kernel',
        #                                 shape=(1, ),
        #                                 initializer=self.initializer,
        #                                 trainable=True)

        super(SpatiallySparse_Dense, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, args):

        if not isinstance(args, (list, tuple)):
            args = [args]
        self.cargs = len(args)

        # flatten
        if len(args) == 2:  # input y, m
            # get inputs
            y, y_mask = args 
            a_fact = int(y.get_shape().as_list()[-1] / y_mask.get_shape().as_list()[-1])
            y_mask = K.repeat_elements(y_mask, a_fact, -1)
            y_flat = K.batch_flatten(y)  # N x D
            y_mask_flat = K.batch_flatten(y_mask)  # N x D

            # prepare switching matrix
            W = self.W # d x D

            w_tmp = K.expand_dims(W, 0)  # 1 x d x D
            Wo = K.permute_dimensions(w_tmp, [0, 2, 1]) * K.expand_dims(y_mask_flat, -1)  # N x D x d
            WoT = K.permute_dimensions(Wo, [0, 2, 1])    # N x d x D
            WotWo_inv = tf.matrix_inverse(K.batch_dot(WoT, Wo))  # N x d x d
            pre = K.batch_dot(WotWo_inv, WoT) # N x d x D
            res = K.batch_dot(pre, y_flat)  # N x d

            if self.use_bias:
                res += K.expand_dims(self.bias, 0)

        else:
            x_data = args[0]
            shape = K.shape(x_data)

            x_data = K.batch_flatten(x_data)  # N x d

            if self.use_bias:
                x_data -= self.bias

            res = K.dot(x_data, self.W)

            # reshape
            # Here you can mix integers and symbolic elements of `shape`
            pool_shape = tf.stack([shape[0], *self.orig_input_shape])
            res = K.reshape(res, pool_shape)

        return res

    def compute_output_shape(self, input_shape):
        # print(self.cargs, input_shape, self.output_len, self.orig_input_shape)
        if self.cargs == 2:
            return (input_shape[0][0], self.output_len)
        else:
            return (input_shape[0], *self.orig_input_shape)


#########################################################
# "Local" layers -- layers with parameters at each voxel
#########################################################

class LocalBias(Layer):
    """ 
    Local bias layer: each pixel/voxel has its own bias operation (one parameter)
    out[v] = in[v] + b
    """

    def __init__(self, my_initializer='RandomNormal', biasmult=1.0, **kwargs):
        self.initializer = my_initializer
        self.biasmult = biasmult
        super(LocalBias, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalBias, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x + self.kernel * self.biasmult  # weights are difference from input

    def compute_output_shape(self, input_shape):
        return input_shape


class LocalLinear(Layer):
    """ 
    Local linear layer: each pixel/voxel has its own linear operation (two parameters)
    out[v] = a * in[v] + b
    """

    def __init__(self, initializer='RandomNormal', **kwargs):
        self.initializer = initializer
        super(LocalLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.mult = self.add_weight(name='mult-kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias-kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalLinear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.mult + self.bias 

    def compute_output_shape(self, input_shape):
        return input_shape
 

class LocallyConnected3D(Layer):
    """
    Code based on LocallyConnected2D from TensorFLow/Keras:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/local.py

	Locally-connected layer for 3D inputs.
	  The `LocallyConnected3D` layer works similarly
	  to the `Conv3D` layer, except that weights are unshared,
	  that is, a different set of filters is applied at each
	  different patch of the input.
	  Note: layer attributes cannot be modified after the layer has been called
	  once (except the `trainable` attribute).
	  Examples:
	  ```python
		  # apply a 3x3x3 unshared weights convolution with 64 output filters on a
		  32x32x32 image
		  # with `data_format="channels_last"`:
		  model = Sequential()
		  model.add(LocallyConnected3D(64, (3, 3, 3), input_shape=(32, 32, 32, 3)))
		  # now model.output_shape == (None, 30, 30, 30, 64)
		  # notice that this layer will consume (30*30*30)*(3*3*3*64) + (30*30*30)*64
		  parameters
		  # add a 3x3x3 unshared weights convolution on top, with 32 output filters:
		  model.add(LocallyConnected3D(32, (3, 3, 3)))
		  # now model.output_shape == (None, 28, 28, 28, 32)
	  ```
	  Arguments:
		  filters: Integer, the dimensionality of the output space
			  (i.e. the number of output filters in the convolution).
		  kernel_size: An integer or tuple/list of 3 integers, specifying the
			  width and height of the 3D convolution window.
			  Can be a single integer to specify the same value for
			  all spatial dimensions.
		  strides: An integer or tuple/list of 3 integers,
			  specifying the strides of the convolution along the width, height
			  and depth. Can be a single integer to specify the same value for
			  all spatial dimensions.
		  padding: Currently only support `"valid"` (case-insensitive).
			  `"same"` will be supported in future.
			  `"valid"` means no padding.
		  data_format: A string,
			  one of `channels_last` (default) or `channels_first`.
			  The ordering of the dimensions in the inputs.
			  `channels_last` corresponds to inputs with shape
			  `(batch, height, width, depth, channels)` while `channels_first`
			  corresponds to inputs with shape
			  `(batch, channels, height, width, height)`.
			  It defaults to the `image_data_format` value found in your
			  Keras config file at `~/.keras/keras.json`.
			  If you never set it, then it will be "channels_last".
		  activation: Activation function to use.
			  If you don't specify anything, no activation is applied
			  (ie. "linear" activation: `a(x) = x`).
		  use_bias: Boolean, whether the layer uses a bias vector.
		  kernel_initializer: Initializer for the `kernel` weights matrix.
		  bias_initializer: Initializer for the bias vector.
		  kernel_regularizer: Regularizer function applied to
			  the `kernel` weights matrix.
		  bias_regularizer: Regularizer function applied to the bias vector.
		  activity_regularizer: Regularizer function applied to
			  the output of the layer (its "activation").
		  kernel_constraint: Constraint function applied to the kernel matrix.
		  bias_constraint: Constraint function applied to the bias vector.
		  implementation: implementation mode, either `1`, `2`, or `3`.
			  `1` loops over input spatial locations to perform the forward pass.
			  It is memory-efficient but performs a lot of (small) ops.
			  `2` stores layer weights in a dense but sparsely-populated 2D matrix
			  and implements the forward pass as a single matrix-multiply. It uses
			  a lot of RAM but performs few (large) ops.
			  `3` stores layer weights in a sparse tensor and implements the forward
			  pass as a single sparse matrix-multiply.
			  How to choose:
			  `1`: large, dense models,
			  `2`: small models,
			  `3`: large, sparse models,
			  where "large" stands for large input/output activations
			  (i.e. many `filters`, `input_filters`, large `np.prod(input_size)`,
			  `np.prod(output_size)`), and "sparse" stands for few connections
			  between inputs and outputs, i.e. small ratio
			  `filters * input_filters * np.prod(kernel_size) / (np.prod(input_size)
			  * np.prod(strides))`, where inputs to and outputs of the layer are
			  assumed to have shapes `input_size + (input_filters,)`,
			  `output_size + (filters,)` respectively.
			  It is recommended to benchmark each in the setting of interest to pick
			  the most efficient one (in terms of speed and memory usage). Correct
			  choice of implementation can lead to dramatic speed improvements (e.g.
			  50X), potentially at the expense of RAM.
			  Also, only `padding="valid"` is supported by `implementation=1`.
	  Input shape:
		  5D tensor with shape:
		  `(samples, channels, rows, cols, z)` if data_format='channels_first'
		  or 5D tensor with shape:
		  `(samples, rows, cols, z, channels)` if data_format='channels_last'.
	  Output shape:
		  5D tensor with shape:
		  `(samples, filters, new_rows, new_cols, new_z)` if data_format='channels_first'
		  or 5D tensor with shape:
		  `(samples, new_rows, new_cols, new_z, filters)` if data_format='channels_last'.
		  `rows`, `cols` and `z` values might have changed due to padding.
	"""

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 implementation=1,
                 **kwargs):
        super(LocallyConnected3D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 3, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if self.padding != 'valid' and implementation == 1:
            raise ValueError('Invalid border mode for LocallyConnected3D '
                             '(only "valid" is supported if implementation is 1): '
                             + padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.implementation = implementation
        self.input_spec = InputSpec(ndim=5)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if self.data_format == 'channels_last':
            input_row, input_col, input_z = input_shape[1:-1]
            input_filter = input_shape[4]
        else:
            input_row, input_col, input_z = input_shape[2:]
            input_filter = input_shape[1]
        if input_row is None or input_col is None or input_z is None:
            raise ValueError('The spatial dimensions of the inputs to '
                             ' a LocallyConnected3D layer '
                             'should be fully-defined, but layer received '
                             'the inputs shape ' + str(input_shape))
        output_row = conv_utils.conv_output_length(
            input_row, self.kernel_size[0], self.padding, self.strides[0])
        output_col = conv_utils.conv_output_length(
            input_col, self.kernel_size[1], self.padding, self.strides[1])
        output_z = conv_utils.conv_output_length(
            input_z, self.kernel_size[2], self.padding, self.strides[2])
        self.output_row = output_row
        self.output_col = output_col
        self.output_z = output_z

        if self.implementation == 1:
            self.kernel_shape = (
                output_row * output_col * output_z,
                self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * input_filter,
                self.filters)

            self.kernel = self.add_weight(
                shape=self.kernel_shape,
                initializer=self.kernel_initializer,
                name='kernel',
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)

        elif self.implementation == 2:
            if self.data_format == 'channels_first':
              self.kernel_shape = (input_filter, input_row, input_col, input_z,
                                   self.filters, self.output_row, self.output_col, self.output_z)
            else:
              self.kernel_shape = (input_row, input_col, input_z, input_filter,
                                   self.output_row, self.output_col, self.output_z, self.filters)

            self.kernel = self.add_weight(shape=self.kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

            self.kernel_mask = LocallyConnected3D.get_locallyconnected_mask(
                input_shape=(input_row, input_col, input_z),
                kernel_shape=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format
            )

        elif self.implementation == 3:
            self.kernel_shape = (self.output_row * self.output_col * self.output_z * self.filters,
                                 input_row * input_col * input_z * input_filter)

            self.kernel_idxs = sorted(
                LocallyConnected3D.conv_kernel_idxs(
                    input_shape=(input_row, input_col, input_z),
                    kernel_shape=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    filters_in=input_filter,
                    filters_out=self.filters,
                    data_format=self.data_format)
            )

            self.kernel = self.add_weight(
                shape=(len(self.kernel_idxs),),
                initializer=self.kernel_initializer,
                name='kernel',
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)

        else:
            raise ValueError('Unrecognized implementation mode: %d.'
                             % self.implementation)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(output_row, output_col, output_z, self.filters),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        if self.data_format == 'channels_first':
            self.input_spec = InputSpec(ndim=5, axes={1: input_filter})
        else:
            self.input_spec = InputSpec(ndim=5, axes={-1: input_filter})
        self.built = True

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            z = input_shape[4]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            z = input_shape[3]

        rows = conv_utils.conv_output_length(
            rows, self.kernel_size[0], self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(
            cols, self.kernel_size[1], self.padding, self.strides[1])
        z = conv_utils.conv_output_length(
            z, self.kernel_size[2], self.padding, self.strides[2])

        if self.data_format == 'channels_first':
            return (input_shape[0], self.filters, rows, cols, z)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, z, self.filters)

    def call(self, inputs):
        if self.implementation == 1:
            output = LocallyConnected3D.local_conv(inputs, self.kernel,
                self.kernel_size, self.strides,
                (self.output_row, self.output_col, self.output_z), self.data_format)

        elif self.implementation == 2:
            output = LocallyConnected3D.local_conv_matmul(inputs, self.kernel,
                self.kernel_mask, self.compute_output_shape(inputs.shape))

        elif self.implementation == 3:
            output = LocallyConnected3D.local_conv_sparse_matmul(inputs,
                self.kernel, self.kernel_idxs, self.kernel_shape,
                self.compute_output_shape(inputs.shape))

        else:
            raise ValueError('Unrecognized implementation mode: %d.'
                             % self.implementation)

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)

        output = self.activation(output)
        return output

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'implementation': self.implementation
        }
        base_config = super(LocallyConnected3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    @staticmethod
    def local_conv(inputs,
                   kernel,
                   kernel_size,
                   strides,
                   output_shape,
                   data_format=None):
        """Apply N-D convolution with un-shared weights.
        Arguments:
            inputs: (N+2)-D tensor with shape
                (batch_size, channels_in, d_in1, ..., d_inN)
                if data_format='channels_first', or
                (batch_size, d_in1, ..., d_inN, channels_in)
                if data_format='channels_last'.
            kernel: the unshared weight for N-D convolution,
                with shape (output_items, feature_dim, channels_out), where
                feature_dim = np.prod(kernel_size) * channels_in,
                output_items = np.prod(output_shape).
            kernel_size: a tuple of N integers, specifying the
                spatial dimensions of the N-D convolution window.
            strides: a tuple of N integers, specifying the strides
                of the convolution along the spatial dimensions.
            output_shape: a tuple of (d_out1, ..., d_outN) specifying the spatial
                dimensionality of the output.
            data_format: string, "channels_first" or "channels_last".
        Returns:
            An (N+2)-D tensor with shape:
            (batch_size, channels_out) + output_shape
            if data_format='channels_first', or:
            (batch_size,) + output_shape + (channels_out,)
            if data_format='channels_last'.
        Raises:
            ValueError: if `data_format` is neither
            `channels_last` nor `channels_first`.
        """
        if data_format is None:
            data_format = image_data_format()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: ' + str(data_format))

        kernel_shape = K.int_shape(kernel)
        feature_dim = kernel_shape[1]
        channels_out = kernel_shape[-1]
        ndims = len(output_shape)
        spatial_dimensions = list(range(ndims))

        xs = []
        output_axes_ticks = [range(axis_max) for axis_max in output_shape]
        for position in itertools.product(*output_axes_ticks):
            slices = [slice(None)]

            if data_format == 'channels_first':
                slices.append(slice(None))

            slices.extend([slice(position[d] * strides[d],
                                 position[d] * strides[d] + kernel_size[d])
                           for d in spatial_dimensions])

            if data_format == 'channels_last':
                slices.append(slice(None))

            xs.append(K.reshape(inputs[slices], (1, -1, feature_dim)))

        x_aggregate = K.concatenate(xs, axis=0)
        output = K.batch_dot(x_aggregate, kernel)
        output = K.reshape(output, output_shape + (-1, channels_out))

        if data_format == 'channels_first':
            permutation = [ndims, ndims + 1] + spatial_dimensions
        else:
            permutation = [ndims] + spatial_dimensions + [ndims + 1]

        return K.permute_dimensions(output, permutation)


    @staticmethod
    def get_locallyconnected_mask(input_shape,
                                  kernel_shape,
                                  strides,
                                  padding,
                                  data_format):
        """Return a mask representing connectivity of a locally-connected operation.
        This method returns a masking numpy array of 0s and 1s (of type `np.float32`)
        that, when element-wise multiplied with a fully-connected weight tensor, masks
        out the weights between disconnected input-output pairs and thus implements
        local connectivity through a sparse fully-connected weight tensor.
        Assume an unshared convolution with given parameters is applied to an input
        having N spatial dimensions with `input_shape = (d_in1, ..., d_inN)`
        to produce an output with spatial shape `(d_out1, ..., d_outN)` (determined
        by layer parameters such as `strides`).
        This method returns a mask which can be broadcast-multiplied (element-wise)
        with a 2*(N+1)-D weight matrix (equivalent to a fully-connected layer between
        (N+1)-D activations (N spatial + 1 channel dimensions for input and output)
        to make it perform an unshared convolution with given `kernel_shape`,
        `strides`, `padding` and `data_format`.
        Arguments:
          input_shape: tuple of size N: `(d_in1, ..., d_inN)`
                       spatial shape of the input.
          kernel_shape: tuple of size N, spatial shape of the convolutional kernel
                        / receptive field.
          strides: tuple of size N, strides along each spatial dimension.
          padding: type of padding, string `"same"` or `"valid"`.
          data_format: a string, `"channels_first"` or `"channels_last"`.
        Returns:
          a `np.float32`-type `np.ndarray` of shape
          `(1, d_in1, ..., d_inN, 1, d_out1, ..., d_outN)`
          if `data_format == `"channels_first"`, or
          `(d_in1, ..., d_inN, 1, d_out1, ..., d_outN, 1)`
          if `data_format == "channels_last"`.
        Raises:
          ValueError: if `data_format` is neither `"channels_first"` nor
                      `"channels_last"`.
        """
        mask = conv_utils.conv_kernel_mask(
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            strides=strides,
            padding=padding
        )

        ndims = int(mask.ndim / 2)

        if data_format == 'channels_first':
            mask = np.expand_dims(mask, 0)
            mask = np.expand_dims(mask, -ndims - 1)

        elif data_format == 'channels_last':
            mask = np.expand_dims(mask, ndims)
            mask = np.expand_dims(mask, -1)

        else:
            raise ValueError('Unrecognized data_format: ' + str(data_format))

        return mask


    @staticmethod
    def local_conv_matmul(inputs, kernel, kernel_mask, output_shape):
        """Apply N-D convolution with un-shared weights using a single matmul call.
        This method outputs `inputs . (kernel * kernel_mask)`
        (with `.` standing for matrix-multiply and `*` for element-wise multiply)
        and requires a precomputed `kernel_mask` to zero-out weights in `kernel` and
        hence perform the same operation as a convolution with un-shared
        (the remaining entries in `kernel`) weights. It also does the necessary
        reshapes to make `inputs` and `kernel` 2-D and `output` (N+2)-D.
        Arguments:
            inputs: (N+2)-D tensor with shape
                `(batch_size, channels_in, d_in1, ..., d_inN)`
                or
                `(batch_size, d_in1, ..., d_inN, channels_in)`.
            kernel: the unshared weights for N-D convolution,
                an (N+2)-D tensor of shape:
                `(d_in1, ..., d_inN, channels_in, d_out2, ..., d_outN, channels_out)`
                or
                `(channels_in, d_in1, ..., d_inN, channels_out, d_out2, ..., d_outN)`,
                with the ordering of channels and spatial dimensions matching
                that of the input.
                Each entry is the weight between a particular input and
                output location, similarly to a fully-connected weight matrix.
            kernel_mask: a float 0/1 mask tensor of shape:
                 `(d_in1, ..., d_inN, 1, d_out2, ..., d_outN, 1)`
                 or
                 `(1, d_in1, ..., d_inN, 1, d_out2, ..., d_outN)`,
                 with the ordering of singleton and spatial dimensions
                 matching that of the input.
                 Mask represents the connectivity pattern of the layer and is
                 precomputed elsewhere based on layer parameters: stride,
                 padding, and the receptive field shape.
            output_shape: a tuple of (N+2) elements representing the output shape:
                `(batch_size, channels_out, d_out1, ..., d_outN)`
                or
                `(batch_size, d_out1, ..., d_outN, channels_out)`,
                with the ordering of channels and spatial dimensions matching that of
                the input.
        Returns:
            Output (N+2)-D tensor with shape `output_shape`.
        """
        inputs_flat = K.reshape(inputs, (K.shape(inputs)[0], -1))

        kernel = kernel_mask * kernel
        kernel = LocallyConnected3D.make_2d(kernel, split_dim=K.ndim(kernel) // 2)

        output_flat = tf.linalg.matmul(inputs_flat, kernel, b_is_sparse=True)
        output = K.reshape(output_flat,
            [K.shape(output_flat)[0],] + output_shape.as_list()[1:])
        return output


    @staticmethod
    def local_conv_sparse_matmul(inputs, kernel, kernel_idxs, kernel_shape,
                                 output_shape):
        """Apply N-D convolution with un-shared weights using a single sparse matmul.
        This method outputs `inputs . tf.sparse.SparseTensor(indices=kernel_idxs,
        values=kernel, dense_shape=kernel_shape)`, with `.` standing for
        matrix-multiply. It also reshapes `inputs` to 2-D and `output` to (N+2)-D.
        Arguments:
            inputs: (N+2)-D tensor with shape `(batch_size, channels_in, d_in1, ...,
              d_inN)` or `(batch_size, d_in1, ..., d_inN, channels_in)`.
            kernel: a 1-D tensor with shape `(len(kernel_idxs),)` containing all the
              weights of the layer.
            kernel_idxs:  a list of integer tuples representing indices in a sparse
              matrix performing the un-shared convolution as a matrix-multiply.
            kernel_shape: a tuple `(input_size, output_size)`, where `input_size =
              channels_in * d_in1 * ... * d_inN` and `output_size = channels_out *
              d_out1 * ... * d_outN`.
            output_shape: a tuple of (N+2) elements representing the output shape:
              `(batch_size, channels_out, d_out1, ..., d_outN)` or `(batch_size,
              d_out1, ..., d_outN, channels_out)`, with the ordering of channels and
              spatial dimensions matching that of the input.
        Returns:
            Output (N+2)-D dense tensor with shape `output_shape`.
        """
        inputs_flat = K.reshape(inputs, (K.shape(inputs)[0], -1))
        output_flat = backend.sparse_ops.sparse_tensor_dense_mat_mul(
            kernel_idxs, kernel, kernel_shape, inputs_flat, adjoint_b=True)
        output_flat_transpose = K.transpose(output_flat)

        output_reshaped = K.reshape(
            output_flat_transpose,
            [K.shape(output_flat_transpose)[0],] + output_shape.as_list()[1:]
        )
        return output_reshaped


    @staticmethod
    def conv_kernel_idxs(input_shape, kernel_shape, strides, padding, filters_in,
                         filters_out, data_format):
        """Yields output-input tuples of indices in a CNN layer.
        The generator iterates over all `(output_idx, input_idx)` tuples, where
          `output_idx` is an integer index in a flattened tensor representing a single
          output image of a convolutional layer that is connected (via the layer
          weights) to the respective single input image at `input_idx`
        Example:
          >>> input_shape = (2, 2)
          >>> kernel_shape = (2, 1)
          >>> strides = (1, 1)
          >>> padding = "valid"
          >>> filters_in = 1
          >>> filters_out = 1
          >>> data_format = "channels_last"
          >>> list(conv_kernel_idxs(input_shape, kernel_shape, strides, padding,
          ...                       filters_in, filters_out, data_format))
          [(0, 0), (0, 2), (1, 1), (1, 3)]
        Args:
          input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
            input.
          kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
            receptive field.
          strides: tuple of size N, strides along each spatial dimension.
          padding: type of padding, string `"same"` or `"valid"`.
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
          filters_in: `int`, number if filters in the input to the layer.
          filters_out: `int', number if filters in the output of the layer.
          data_format: string, "channels_first" or "channels_last".
        Yields:
          The next tuple `(output_idx, input_idx)`, where
          `output_idx` is an integer index in a flattened tensor representing a single
          output image of a convolutional layer that is connected (via the layer
          weights) to the respective single input image at `input_idx`.
        Raises:
            ValueError: if `data_format` is neither
            `"channels_last"` nor `"channels_first"`, or if number of strides, input,
            and kernel number of dimensions do not match.
            NotImplementedError: if `padding` is neither `"same"` nor `"valid"`.
        """
        if padding not in ('same', 'valid'):
             raise NotImplementedError('Padding type %s not supported. '
                                      'Only "valid" and "same" '
                                      'are implemented.' % padding)

        in_dims = len(input_shape)
        if isinstance(kernel_shape, int):
            kernel_shape = (kernel_shape,) * in_dims
        if isinstance(strides, int):
            strides = (strides,) * in_dims

        kernel_dims = len(kernel_shape)
        stride_dims = len(strides)
        if kernel_dims != in_dims or stride_dims != in_dims:
            raise ValueError('Number of strides, input and kernel dimensions must all '
                             'match. Received: %d, %d, %d.' %
                             (stride_dims, in_dims, kernel_dims))

        output_shape = LocallyConnected3D.conv_output_shape(input_shape,
            kernel_shape, strides, padding)
        output_axes_ticks = [range(dim) for dim in output_shape]

        if data_format == 'channels_first':
            concat_idxs = lambda spatial_idx, filter_idx: (filter_idx,) + spatial_idx
        elif data_format == 'channels_last':
            concat_idxs = lambda spatial_idx, filter_idx: spatial_idx + (filter_idx,)
        else:
            raise ValueError('Data format %s not recognized.'
                             '`data_format` must be "channels_first" or '
                             '"channels_last".' % data_format)

        for output_position in itertools.product(*output_axes_ticks):
            input_axes_ticks = LocallyConnected3D.conv_connected_inputs(input_shape,
                kernel_shape, output_position, strides, padding)
            for input_position in itertools.product(*input_axes_ticks):
                for f_in in range(filters_in):
                    for f_out in range(filters_out):
                        out_idx = np.ravel_multi_index(
                            multi_index=concat_idxs(output_position, f_out),
                            dims=concat_idxs(output_shape, filters_out))
                        in_idx = np.ravel_multi_index(
                            multi_index=concat_idxs(input_position, f_in),
                            dims=concat_idxs(input_shape, filters_in))
                        yield (out_idx, in_idx)


    @staticmethod
    def conv_connected_inputs(input_shape, kernel_shape, output_position, strides,
                              padding):
        """Return locations of the input connected to an output position.
        Assume a convolution with given parameters is applied to an input having N
        spatial dimensions with `input_shape = (d_in1, ..., d_inN)`. This method
        returns N ranges specifying the input region that was convolved with the
        kernel to produce the output at position
        `output_position = (p_out1, ..., p_outN)`.
        Example:
          >>> input_shape = (4, 4)
          >>> kernel_shape = (2, 1)
          >>> output_position = (1, 1)
          >>> strides = (1, 1)
          >>> padding = "valid"
          >>> conv_connected_inputs(input_shape, kernel_shape, output_position,
          ...                       strides, padding)
          [range(1, 3), range(1, 2)]
        Args:
          input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
            input.
          kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
            receptive field.
          output_position: tuple of size N: `(p_out1, ..., p_outN)`, a single position
            in the output of the convolution.
          strides: tuple of size N, strides along each spatial dimension.
          padding: type of padding, string `"same"` or `"valid"`.
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        Returns:
          N ranges `[[p_in_left1, ..., p_in_right1], ...,
                    [p_in_leftN, ..., p_in_rightN]]` specifying the region in the
          input connected to output_position.
        """
        ranges = []

        ndims = len(input_shape)
        for d in range(ndims):
            left_shift = int(kernel_shape[d] / 2)
            right_shift = kernel_shape[d] - left_shift
            center = output_position[d] * strides[d]
            if padding == 'valid':
                center += left_shift
            start = max(0, center - left_shift)
            end = min(input_shape[d], center + right_shift)
            ranges.append(range(start, end))

        return ranges


    @staticmethod
    def conv_output_shape(input_shape, kernel_shape, strides, padding):
        """Return the output shape of an N-D convolution.
        Forces dimensions where input is empty (size 0) to remain empty.
        Args:
          input_shape: tuple of size N: `(d_in1, ..., d_inN)`, spatial shape of the
            input.
          kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
            receptive field.
          strides: tuple of size N, strides along each spatial dimension.
          padding: type of padding, string `"same"` or `"valid"`.
            `"valid"` means no padding. `"same"` results in padding evenly to
            the left/right or up/down of the input such that output has the same
            height/width dimension as the input.
        Returns:
          tuple of size N: `(d_out1, ..., d_outN)`, spatial shape of the output.
        """
        dims = range(len(kernel_shape))
        output_shape = [
            conv_utils.conv_output_length(input_shape[d], kernel_shape[d], padding, strides[d])
            for d in dims
        ]
        output_shape = tuple(
            [0 if input_shape[d] == 0 else output_shape[d] for d in dims])
        return output_shape


    @staticmethod
    def make_2d(tensor, split_dim):
        """Reshapes an N-dimensional tensor into a 2D tensor.
        Dimensions before (excluding) and after (including) `split_dim` are grouped
        together.
        Arguments:
        tensor: a tensor of shape `(d0, ..., d(N-1))`.
        split_dim: an integer from 1 to N-1, index of the dimension to group
            dimensions before (excluding) and after (including).
        Returns:
        Tensor of shape
        `(d0 * ... * d(split_dim-1), d(split_dim) * ... * d(N-1))`.
        """
        shape = K.shape(tensor)
        in_dims = shape[:split_dim]
        out_dims = shape[split_dim:]

        in_size = tf.math.reduce_prod(in_dims)
        out_size = tf.math.reduce_prod(out_dims)

        return K.reshape(tensor, (in_size, out_size))


class LocalCrossLinear(Layer):
    """ 
    Local cross mult layer

    input: [batch_size, *vol_size, nb_feats_1]
    output: [batch_size, *vol_size, nb_feats_2]
    
    at each spatial voxel, there is a different linear relation learned.
    """

    def __init__(self, output_features, 
                 mult_initializer=None,
                 bias_initializer=None,
                 mult_regularizer=None,
                 bias_regularizer=None,
                 use_bias=True,
                 **kwargs):
        
        self.output_features = output_features
        self.mult_initializer = mult_initializer
        self.bias_initializer = bias_initializer
        self.mult_regularizer = mult_regularizer
        self.bias_regularizer = bias_regularizer
        self.use_bias = use_bias
        
        super(LocalCrossLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        mult_shape = [1] + list(input_shape)[1:] + [self.output_features]
        
        
        # verify initializer
        if self.mult_initializer is None:
            mean = 1/input_shape[-1]
            stddev = 0.01
            self.mult_initializer = keras.initializers.RandomNormal(mean=mean, stddev=stddev)
        
        self.mult = self.add_weight(name='mult-kernel', 
                                      shape=mult_shape,
                                      initializer=self.mult_initializer,
                                      regularizer=self.mult_regularizer,
                                      trainable=True)

        if self.use_bias:
            if self.bias_initializer is None:
                mean = 1/input_shape[-1]
                stddev = 0.01
                self.bias_initializer = keras.initializers.RandomNormal(mean=mean, stddev=stddev)
            
            bias_shape = [1] + list(input_shape)[1:-1] + [self.output_features]
            self.bias = self.add_weight(name='bias-kernel', 
                                          shape=bias_shape,
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          trainable=True)
        super(LocalCrossLinear, self).build(input_shape)

    def call(self, x):
        map_fn = lambda z: self._single_matmul(z, self.mult[0, ...])
        y = tf.stack(tf.map_fn(map_fn, x, dtype=tf.float32), 0)
        
        if self.use_bias:
            y = y + self.bias
        
        return y

    def _single_matmul(self, x, mult):
        x = K.expand_dims(x, -2)
        y = tf.matmul(x, mult)[...,0,:]
        return y

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1] + [self.output_features])
 

class LocalCrossLinearTrf(Layer):
    """ 
    Local cross mult layer with transform

    input: [batch_size, *vol_size, nb_feats_1]
    output: [batch_size, *vol_size, nb_feats_2]
    
    at each spatial voxel, there is a different linear relation learned.
    """

    def __init__(self, output_features, 
                 mult_initializer=None,
                 bias_initializer=None,
                 mult_regularizer=None,
                 bias_regularizer=None,
                 use_bias=True,
                 trf_mult=1,
                 **kwargs):
        
        self.output_features = output_features
        self.mult_initializer = mult_initializer
        self.bias_initializer = bias_initializer
        self.mult_regularizer = mult_regularizer
        self.bias_regularizer = bias_regularizer
        self.use_bias = use_bias
        self.trf_mult = trf_mult
        self.interp_method = 'linear'
        
        super(LocalCrossLinearTrf, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        mult_shape = list(input_shape)[1:] + [self.output_features]
        ndims = len(list(input_shape)[1:-1])
        
        
        # verify initializer
        if self.mult_initializer is None:
            mean = 1/input_shape[-1]
            stddev = 0.01
            self.mult_initializer = keras.initializers.RandomNormal(mean=mean, stddev=stddev)
        
        self.mult = self.add_weight(name='mult-kernel', 
                                      shape=mult_shape,
                                      initializer=self.mult_initializer,
                                      regularizer=self.mult_regularizer,
                                      trainable=True)

        self.trf = self.add_weight(name='def-kernel', 
                                      shape=mult_shape + [ndims],
                                      initializer=keras.initializers.RandomNormal(mean=0, stddev=0.001),
                                      trainable=True)

        if self.use_bias:
            if self.bias_initializer is None:
                mean = 1/input_shape[-1]
                stddev = 0.01
                self.bias_initializer = keras.initializers.RandomNormal(mean=mean, stddev=stddev)
            
            bias_shape = list(input_shape)[1:-1] + [self.output_features]
            self.bias = self.add_weight(name='bias-kernel', 
                                          shape=bias_shape,
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          trainable=True)
        
        super(LocalCrossLinearTrf, self).build(input_shape)

    def call(self, x):
        

        # for each element in the batch
        y = tf.map_fn(self._single_batch_trf, x, dtype=tf.float32)
        
        return y
    
    def _single_batch_trf(self, vol):
        # vol should be vol_shape + [nb_features]
        # self.trf should be vol_shape + [nb_features] + [ndims]

        vol_shape = vol.shape.as_list()
        nb_input_dims = vol_shape[-1]

        # this is inefficient...
        new_vols = [None] * self.output_features
        for j in range(self.output_features):
            new_vols[j] = tf.zeros(vol_shape[:-1], dtype=tf.float32)
            for i in range(nb_input_dims):
                trf_vol = transform(vol[..., i], self.trf[..., i, j, :] * self.trf_mult, interp_method=self.interp_method)
                trf_vol = tf.reshape(trf_vol, vol_shape[:-1])
                new_vols[j] += trf_vol * self.mult[..., i, j]

                if self.use_bias:
                    new_vols[j] += self.bias[..., j]
        
        return tf.stack(new_vols, -1)


    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1] + [self.output_features])
 

class LocalParamLayer(Layer):
    """ 
    Local Parameter layer: each pixel/voxel has its own parameter (one parameter)
    out[v] = b

    using code from 
    https://github.com/YerevaNN/R-NET-in-Keras/blob/master/layers/SharedWeight.py
    and
    https://github.com/keras-team/keras/blob/ee02d256611b17d11e37b86bd4f618d7f2a37d84/keras/engine/input_layer.py
    """

    def __init__(self,
                 shape,
                 my_initializer='RandomNormal',
                 dtype=None,
                 name=None,
                 mult=1.0,
                 **kwargs):

        
        # some input checking
        if not name:
            prefix = 'local_param'
            name = prefix + '_' + str(backend.get_uid(prefix))
            
        if not dtype:
            dtype = backend.floatx()
        
        self.shape = [1, *shape]
        self.my_initializer = my_initializer
        self.mult = mult

        if not name:
            prefix = 'param'
            name = '%s_%d' % (prefix, K.get_uid(prefix))
        Layer.__init__(self, name=name, **kwargs)

        # Create a trainable weight variable for this layer.
        with K.name_scope(self.name):
            self.kernel = self.add_weight(name='kernel', 
                                            shape=shape,
                                            initializer=self.my_initializer,
                                            dtype=dtype,
                                            trainable=True)

        # prepare output tensor, which is essentially the kernel.
        output_tensor = K.expand_dims(self.kernel, 0) * self.mult
        output_tensor._keras_shape = self.shape
        output_tensor._uses_learning_phase = False
        output_tensor._keras_history = base_layer.KerasHistory(self, 0, 0)
        output_tensor._batch_input_shape = self.shape

        self.trainable = True
        self.built = True    
        self.is_placeholder = False

        # create new node
        tensorflow.python.keras.engine.base_layer.node_module.Node(self,
            inbound_layers=[],
            node_indices=[],
            tensor_indices=[],
            input_tensors=[],
            output_tensors=[output_tensor],
            input_masks=[],
            output_masks=[None],
            input_shapes=[],
            output_shapes=self.shape)

    def get_config(self):
        config = {
                'dtype': self.dtype,
                'sparse': self.sparse,
                'name': self.name
        }
        return config


class LocalParamWithInput(Layer):
    """ 
    Update 9/29/2019 - TODO: should try ne.layers.LocalParam() again after update.

    The neuron.layers.LocalParam has an issue where _keras_shape gets lost upon calling get_output :(

    tried using call() but this requires an input (or i don't know how to fix it)
    the fix was that after the return, for every time that tensor would be used i would need to do something like
    new_vec._keras_shape = old_vec._keras_shape

    which messed up the code. Instead, we'll do this quick version where we need an input, but we'll ignore it.

    this doesn't have the _keras_shape issue since we built on the input and use call()
    """

    def __init__(self, shape, initializer='RandomNormal', mult=1.0, **kwargs):
        self.shape = shape
        self.initializer = initializer
        self.biasmult = mult
        print('LocalParamWithInput: Consider using neuron.layers.LocalParam()')
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=self.shape,  # input_shape[1:]
                                      initializer=self.initializer,
                                      trainable=True)
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # want the x variable for it's keras properties and the batch.
        xslice = K.batch_flatten(x)[:, 0:1]
        b = xslice * tf.zeros((1,)) + tf.ones((1,))
        # b = K.batch_flatten(0 * x)[:, 0:1] + 1
        params = K.flatten(self.kernel * self.biasmult)[tf.newaxis, ...]
        return K.reshape(K.dot(b, params), [-1, *self.shape])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.shape)


def LocalParam(    # pylint: disable=invalid-name
        shape,
        batch_size=None,
        name=None,
        dtype=None,
        **kwargs):
    """
    `LocalParam()` is used to instantiate a Keras tensor.
    A Keras tensor is a tensor object from the underlying backend
    (Theano or TensorFlow), which we augment with certain
    attributes that allow us to build a Keras model
    just by knowing the inputs and outputs of the model.
    For instance, if a, b and c are Keras tensors,
    it becomes possible to do:
    `model = Model(input=[a, b], output=c)`
    The added Keras attribute is:
            `_keras_history`: Last layer applied to the tensor.
                    the entire layer graph is retrievable from that layer,
                    recursively.
    Arguments:
            shape: A shape tuple (integers), not including the batch size.
                    For instance, `shape=(32,)` indicates that the expected input
                    will be batches of 32-dimensional vectors. Elements of this tuple
                    can be None; 'None' elements represent dimensions where the shape is
                    not known.
            batch_size: optional static batch size (integer).
            name: An optional name string for the layer.
                    Should be unique in a model (do not reuse the same name twice).
                    It will be autogenerated if it isn't provided.
            dtype: The data type expected by the input, as a string
                    (`float32`, `float64`, `int32`...)
            **kwargs: deprecated arguments support.
    Returns:
        A `tensor`.
    Example:
    ```python
    # this is a logistic regression in Keras
    x = Input(shape=(32,))
    y = Dense(16, activation='softmax')(x)
    model = Model(x, y)
    ```
    Note that even if eager execution is enabled,
    `Input` produces a symbolic tensor (i.e. a placeholder).
    This symbolic tensor can be used with other
    TensorFlow ops, as such:
    ```python
    x = Input(shape=(32,))
    y = tf.square(x)
    ```
    Raises:
        ValueError: in case of invalid arguments.
    """   
    input_layer = LocalParamLayer(shape, name=name, dtype=dtype)

    # Return tensor including `_keras_history`.
    # Note that in this case train_output and test_output are the same pointer.
    outputs = input_layer._inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


##########################################
## Stream layers
##########################################


class MeanStream(Layer):
    """ 
    Maintain stream of data mean. 

    cap refers to mainting an approximation of up to that number of subjects -- that is,
    any incoming datapoint will have at least 1/cap weight.
    """

    def __init__(self, cap=100, **kwargs):
        self.cap = K.variable(cap, dtype='float32')
        super(MeanStream, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create mean and count
        # These are weights because just maintaining variables don't get saved with the model, and we'd like
        # to have these numbers saved when we save the model.
        # But we need to make sure that the weights are untrainable.
        self.mean = self.add_weight(name='mean', 
                                      shape=input_shape[1:],
                                      initializer='zeros',
                                      trainable=False)
        self.count = self.add_weight(name='count', 
                                      shape=[1],
                                      initializer='zeros',
                                      trainable=False)

        # self.mean = K.zeros(input_shape[1:], name='mean')
        # self.count = K.variable(0.0, name='count')
        super(MeanStream, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # get new mean and count
        this_bs_int = K.shape(x)[0]
        new_mean, new_count = _mean_update(self.mean, self.count, x, self.cap)
        
        # update op
        updates = [(self.count, new_count), (self.mean, new_mean)]
        self.add_update(updates, x)

        # prep for broadcasting :(
        p = tf.concat((K.reshape(this_bs_int, (1,)), K.shape(self.mean)), 0)
        z = tf.ones(p)
        
        # the first few 1000 should not matter that much towards this cost
        return K.minimum(1., new_count/self.cap) * (z * K.expand_dims(new_mean, 0))

    def compute_output_shape(self, input_shape):
        return input_shape


class CovStream(Layer):
    """ 
    Maintain stream of data mean. 

    cap refers to mainting an approximation of up to that number of subjects -- that is,
    any incoming datapoint will have at least 1/cap weight.
    """

    def __init__(self, cap=100, **kwargs):
        self.cap = K.variable(cap, dtype='float32')
        super(CovStream, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create mean, cov and and count
        # See note in MeanStream.build()
        self.mean = self.add_weight(name='mean', 
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=False)
        v = np.prod(input_shape[1:])
        self.cov = self.add_weight(name='cov', 
                                 shape=[v, v],
                                 initializer='zeros',
                                 trainable=False)
        self.count = self.add_weight(name='count', 
                                      shape=[1],
                                      initializer='zeros',
                                      trainable=False)

        super(CovStream, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        x_orig = x

        # x reshape
        this_bs_int = K.shape(x)[0]
        this_bs = tf.cast(this_bs_int, 'float32')  # this batch size
        prev_count = self.count
        x = K.batch_flatten(x)  # B x N

        # update mean
        new_mean, new_count = _mean_update(self.mean, self.count, x, self.cap)        

        # new C update. Should be B x N x N
        x = K.expand_dims(x, -1)
        C_delta = K.batch_dot(x, K.permute_dimensions(x, [0, 2, 1]))

        # update cov
        prev_cap = K.minimum(prev_count, self.cap)
        C = self.cov * (prev_cap - 1) + K.sum(C_delta, 0)
        new_cov = C / (prev_cap + this_bs - 1)

        # updates
        updates = [(self.count, new_count), (self.mean, new_mean), (self.cov, new_cov)]
        self.add_update(updates, x_orig)

        # prep for broadcasting :(
        p = tf.concat((K.reshape(this_bs_int, (1,)), K.shape(self.cov)), 0)
        z = tf.ones(p)

        return K.minimum(1., new_count/self.cap) * (z * K.expand_dims(new_cov, 0))

    def compute_output_shape(self, input_shape):
        v = np.prod(input_shape[1:])
        return (input_shape[0], v, v)


def _mean_update(pre_mean, pre_count, x, pre_cap=None):

    # compute this batch stats
    this_sum = tf.reduce_sum(x, 0)
    this_bs = tf.cast(K.shape(x)[0], 'float32')  # this batch size
    
    # increase count and compute weights
    new_count = pre_count + this_bs
    alpha = this_bs/K.minimum(new_count, pre_cap)
    
    # compute new mean. Note that once we reach self.cap (e.g. 1000), the 'previous mean' matters less
    new_mean = pre_mean * (1-alpha) + (this_sum/this_bs) * alpha

    return (new_mean, new_count)

##########################################
## FFT Layers
##########################################

class FFT(Layer):
    """
    fft layer, assuming the real/imag are input/output via two features
    Input: tf.complex of size [batch_size, ..., nb_feats]
    Output: tf.complex of size [batch_size, ..., nb_feats]
    """

    def __init__(self, **kwargs):
        super(FFT, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1, 2, 3], 'only 1D, 2D or 3D supported'

        # super
        super(FFT, self).build(input_shape)

    def call(self, inputx):
        
        if not inputx.dtype in [tf.complex64, tf.complex128]:
            print('Warning: inputx is not complex. Converting.', file=sys.stderr)
        
            # if inputx is float, this will assume 0 imag channel
            inputx = tf.cast(inputx, tf.complex64)

        # get the right fft
        if self.ndims == 1:
            fft = tf.fft
        elif self.ndims == 2:
            fft = tf.fft2d
        else:
            fft = tf.fft3d

        perm_dims = [0, self.ndims + 1] + list(range(1, self.ndims + 1))
        invert_perm_ndims = [0] + list(range(2, self.ndims + 2)) + [1]
        
        perm_inputx = K.permute_dimensions(inputx, perm_dims)  # [batch_size, nb_features, *vol_size]
        fft_inputx = fft(perm_inputx)
        return K.permute_dimensions(fft_inputx, invert_perm_ndims)

    def compute_output_shape(self, input_shape):
        return input_shape


class IFFT(Layer):
    """
    ifft layer, assuming the real/imag are input/output via two features
    Input: tf.complex of size [batch_size, ..., nb_feats]
    Output: tf.complex of size [batch_size, ..., nb_feats]
    """

    def __init__(self, **kwargs):
        super(IFFT, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1, 2, 3], 'only 1D, 2D or 3D supported'

        # super
        super(IFFT, self).build(input_shape)

    def call(self, inputx):
        
        if not inputx.dtype in [tf.complex64, tf.complex128]:
            print('Warning: inputx is not complex. Converting.', file=sys.stderr)
        
            # if inputx is float, this will assume 0 imag channel
            inputx = tf.cast(inputx, tf.complex64)
        
        # get the right fft
        if self.ndims == 1:
            ifft = tf.ifft
        elif self.ndims == 2:
            ifft = tf.ifft2d
        else:
            ifft = tf.ifft3d

        perm_dims = [0, self.ndims + 1] + list(range(1, self.ndims + 1))
        invert_perm_ndims = [0] + list(range(2, self.ndims + 2)) + [1]
        
        perm_inputx = K.permute_dimensions(inputx, perm_dims)  # [batch_size, nb_features, *vol_size]
        ifft_inputx = ifft(perm_inputx)
        return K.permute_dimensions(ifft_inputx, invert_perm_ndims)

    def compute_output_shape(self, input_shape):
        return input_shape


class ComplexToChannels(Layer):

    def __init__(self, **kwargs):
        super(ComplexToChannels, self).__init__(**kwargs)

    def build(self, input_shape):
        # super
        super(ComplexToChannels, self).build(input_shape)

    def call(self, inputx):
        
        assert inputx.dtype in [tf.complex64, tf.complex128], 'inputx is not complex.'
        
        return tf.concat([tf.real(inputx), tf.imag(inputx)], -1)

    def compute_output_shape(self, input_shape):
        i_s = list(input_shape)
        i_s[-1] *= 2
        return tuple(i_s)


class ChannelsToComplex(Layer):

    def __init__(self, **kwargs):
        super(ChannelsToComplex, self).__init__(**kwargs)

    def build(self, input_shape):
        # super
        super(ChannelsToComplex, self).build(input_shape)

    def call(self, inputx):
        nb_channels = inputx.shape[-1] // 2
        return tf.complex(inputx[...,:nb_channels], inputx[...,nb_channels:])
        
    def compute_output_shape(self, input_shape):
        i_s = list(input_shape)
        i_s[-1] = i_s[-1] // 2
        return tuple(i_s)


class FFTShift(Layer):
    """
    fftshift for keras tensors (so only inner dimensions get shifted)

    modified from
    https://gist.github.com/Gurpreetsingh9465/f76cc9e53107c29fd76515d64c294d3f

    Shift the zero-frequency component to the center of the spectrum.
    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.
    Parameters
    ----------
    x : array_like, Tensor
        Input array.
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.
    Returns
    -------
    y : Tensor.
    """

    def __init__(self, axes=None, **kwargs):
        self.axes = axes
        super(FFTShift, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1, 2, 3], 'only 1D, 2D or 3D supported'

        # super
        super(FFTShift, self).build(input_shape)

    def call(self, x):
        axes = self.axes
        if axes is None:
            axes = tuple(range(K.ndim(x)))
            shift = [0] + [dim // 2 for dim in x.shape] + [0]
        elif isinstance(axes, int):
            shift = x.shape[axes] // 2
        else:
            shift = [x.shape[ax] // 2 for ax in axes]

        return _roll(x, shift, axes)

    def compute_output_shape(self, input_shape):
        return input_shape


class IFFTShift(Layer):
    """
    ifftshift for keras tensors (so only inner dimensions get shifted)

    modified from
    https://gist.github.com/Gurpreetsingh9465/f76cc9e53107c29fd76515d64c294d3f

    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.
    Parameters
    ----------
    x : array_like, Tensor.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.
    Returns
    -------
    y : Tensor.
    """

    def __init__(self, axes=None, **kwargs):
        self.axes = axes
        super(IFFTShift, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1, 2, 3], 'only 1D, 2D or 3D supported'

        # super
        super(IFFTShift, self).build(input_shape)

    def call(self, x):
        axes = self.axes
        if axes is None:
            axes = tuple(range(K.ndim(x)))
            shift = [0] + [-(dim // 2) for dim in x.shape.as_list()[1:-1]] + [0]
        elif isinstance(axes, int):
            shift = -(x.shape[axes] // 2)
        else:
            shift = [-(x.shape[ax] // 2) for ax in axes]

        return _roll(x, shift, axes)

    def compute_output_shape(self, input_shape):
        return input_shape


##########################################
## Stochastic Sampling layers
##########################################

class SampleNormalLogVar(Layer):
    """ 
    Keras Layer: Gaussian sample given mean and log_variance
    
    inputs: list of Tensors [mu, log_var]
    outputs: Tensor sample from N(mu, sigma^2)
    """

    def __init__(self, **kwargs):
        super(SampleNormalLogVar, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SampleNormalLogVar, self).build(input_shape)

    def call(self, x):
        return self._sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def _sample(self, args):
        """
        sample from a normal distribution

        args should be [mu, log_var], where log_var is the log of the squared sigma

        This is probably equivalent to 
            K.random_normal(shape, args[0], exp(args[1]/2.0))
        """
        mu, log_var = args

        # sample from N(0, 1)
        noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # make it a sample from N(mu, sigma^2)
        z = mu + tf.exp(log_var/2.0) * noise
        return z
