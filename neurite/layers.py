"""
layers for the neuron project

If you use this code, please cite the following, and read function docs for further info/citations
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018. https://arxiv.org/abs/1903.03148


Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

# internal python imports
import sys
import itertools
import warnings

# third party
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputLayer, Input, InputSpec

# keras internal utils
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils

# tensorflow ops (direct import required)
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_sparse_ops

# local imports
from . import utils
from .. import py


class Negate(Layer):
    """ 
    Keras Layer: negative of the input.
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

    If you find this class useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148
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
        return tf.map_fn(self._single_resize, vol)

    def compute_output_shape(self, input_shape):

        output_shape = [input_shape[0]]
        output_shape += [int(input_shape[1:-1][f] * self.zoom_factor[f]) for f in range(self.ndims)]
        output_shape += [input_shape[-1]]
        return tuple(output_shape)

    def _single_resize(self, inputs):
        return utils.resize(inputs, self.zoom_factor, interp_method=self.interp_method)


# Zoom naming of resize, to match scipy's naming
Zoom = Resize


class SoftQuantize(Layer):
    """ 
    Keras Layer: soft quantization of intentity input

    If you find this class useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    def __init__(self,
                 alpha=1,
                 bin_centers=None,
                 nb_bins=16,
                 min_clip=-np.inf,
                 max_clip=np.inf,
                 return_log=False,
                 **kwargs):

        self.alpha = alpha
        self.bin_centers = bin_centers
        self.nb_bins = nb_bins
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.return_log = return_log
        super(SoftQuantize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SoftQuantize, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -utils.soft_quantize(x,
                                    alpha=self.alpha,
                                    bin_centers=self.bin_centers,
                                    nb_bins=self.nb_bins,
                                    min_clip=self.min_clip,
                                    max_clip=self.max_clip,
                                    return_log=False)              # [bs, ..., B]

    def compute_output_shape(self, input_shape):
        output_shape_lst = list(input_shape) + [self.nb_bins]
        return tuple(output_shape_lst)


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
    Blur a tensor by convolving it with a Gaussian kernel. The layer supports isotropic and
    anisotropic blurring, randomized or not.

    If you find this layer useful, please cite:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    def __init__(self,
                 sigma=None,
                 level=None,
                 random=False,
                 min_sigma=0,
                 isotropic=False,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            sigma: SD of the blurring kernel, as a scalar or length-N iterable. While specifying
                a single SD will result in isotropic blurring, you can pass a separate SD for each
                axis of space. For `random=True`, this argument defines the upper bounds on the
                blurring SDs, and the layer will sample the SD for axis i between the i-th
                elements of `min_sigma` and `sigma`, respectively. Careful: random blur is
                anisotropic unless you pass `isotropic=True`!
            level: Deprecated in favor of `sigma`.
            random: Sample the blurring SDs uniformly from the interval [`min_sigma`, `sigma`).
            min_sigma: Lower bound on the blurring SDs, only used for random sampling. See `sigma`.
            isotropic: For non-random blur, isotropy is implicitly controlled by the length of
                `sigma`. For random isotropic blur, set `isotropic=True` and pass a single SD.
            seed: Integer for reproducible randomization. Only considered for random sampling.
        """
        assert sigma is not None or level is not None, 'sigma or level must be provided'
        assert not (sigma is not None and level is not None), 'only sigma or level must be provided'

        if level is not None:
            warnings.warn('The `level` argument to ne.layers.GaussianBlur is deprecated and will '
                          'be removed in a future version. Please use `sigma` instead.')
            if level < 1:
                raise ValueError('Gaussian blur level must not be less than 1')
            if random:
                raise ValueError('level argument incompatible with random blurring')

            self.sigma = (level - 1) ** 2

        if isotropic and not random:
            raise ValueError('For non-random blurring, isotropy is implicitly controlled by the '
                             'number of sigmas provided. Set `isotropic` only for random blur.')

        self.sigma = sigma
        self.random = random
        self.min_sigma = min_sigma
        self.isotropic = isotropic
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sigma': self.sigma,
            'random': self.random,
            'min_sigma': self.min_sigma,
            'isotropic': self.isotropic,
            'seed': self.seed,
        })
        return config

    def _normalize_sigma(self, sigma, ndims):
        sigma = np.ravel(sigma)
        sigma = sigma.tolist()
        if len(sigma) not in (1, ndims):
            raise ValueError(f'1 or {ndims} sigmas expected in {ndims}D space, got {len(sigma)}')

        if any(s < 0 for s in sigma):
            raise ValueError('Gaussian blur sigma must not be less than 0')

        if len(sigma) > 1 and self.isotropic:
            raise ValueError(f'random isotropic blur requires a single sigma, got {len(sigma)}')

        if len(sigma) == 1:
            sigma = sigma * ndims
        return sigma

    def build(self, input_shape):
        # Convert to N-element lists.
        ndims = len(input_shape) - 2
        self.sigma = self._normalize_sigma(self.sigma, ndims)
        self.min_sigma = self._normalize_sigma(self.min_sigma, ndims)

        # Have `separable_conv` apply the same random kernel along all axes.
        if self.isotropic and self.random:
            self.sigma = self.sigma[:1]
            self.min_sigma = self.min_sigma[:1]

        super().build(input_shape)

    def call(self, x):
        """
        Parameters:
            x: Tensor that will be blurred.
        """
        if not any(s > 0 for s in self.sigma):
            return x

        kernel = utils.gaussian_kernel(sigma=self.sigma,
                                       random=self.random,
                                       min_sigma=self.min_sigma,
                                       separate=True,
                                       dtype=x.dtype)

        return utils.separable_conv(x, kernel, batched=True)


class Subsample(Layer):
    """
    Symmetrically subsample a tensor by a factor f (stride) along a single spatial axis using
    nearest-neighbor interpolation and optionally upsample again, to reduce its resolution. Both
    f and the subsampling axis can be randomly drawn.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 stride_min=1,
                 stride_max=8,
                 axes=None,
                 prob=1,
                 upsample=True,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            stride_min: Lower bound on the subsampling factor.
            stride_max: Upper bound on the subsampling factor.
            axes: Spatial axes to draw the subsampling axis from. None means all.
            prob: Subsampling probability. A value of 1 means always, 0 never.
            upsample: Upsample the tensor to restore its original shape.
            seed: Integer for reproducible randomization.
        """
        self.stride_min = stride_min
        self.stride_max = stride_max
        self.axes = axes
        self.prob = prob
        self.upsample = upsample
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'stride_min': self.stride_min,
            'stride_max': self.stride_max,
            'axes': self.axes,
            'prob': self.prob,
            'upsample': self.upsample,
            'seed': self.seed,
        })
        return config

    def build(self, input_shape):
        ndims = len(input_shape) - 2
        assert ndims in (1, 2, 3), 'only 1D, 2D, or 3D supported'

        allowed = range(1, ndims + 1)
        self.axes = py.utils.normalize_axes(self.axes, input_shape, allowed, none_means_all=True)
        super().build(input_shape)

    def call(self, x):
        """
        Parameters:
            x: Input tensor to subsample.
        """
        if self.prob == 0 or self.stride_max == 1:
            return x

        shape = tf.shape(x)
        x = utils.subsample_axis(x,
                                 stride_min=self.stride_min,
                                 stride_max=self.stride_max,
                                 axes=self.axes,
                                 prob=self.prob,
                                 upsample=self.upsample,
                                 seed=self.seed)

        # Avoid unnecessary dynamic shapes (showing up as `None`).
        return tf.reshape(x, shape) if self.upsample else x


class RandomCrop(Layer):
    """
    Randomly crop the content of a tensor by multiplying with a binary mask.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 crop_min=0,
                 crop_max=0.5,
                 axis=None,
                 prob=1,
                 bilateral=False,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            crop_min: Minimum proportion of voxels to remove, in [0, `crop_max`].
            crop_max: Maximum proportion of voxels to remove, in [`crop_min`, 1].
            axis: Spatial axis along which to crop, where None means any spatial axis. With more
                than one axis specified, a single axis will be drawn at each execution.
            prob: Cropping probability, where 1 means always and 0 never.
            bilateral: Randomly distribute the cropping proportion between top/bottom end.
            seed: Integer for reproducible randomization.
        """
        self.crop_min = crop_min
        self.crop_max = crop_max
        self.axis = axis
        self.prob = prob
        self.bilateral = bilateral
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'crop_min': self.crop_min,
            'crop_max': self.crop_max,
            'axis': self.axis,
            'prob': self.prob,
            'bilateral': self.bilateral,
            'seed': self.seed,
        })
        return config

    def build(self, input_shape):
        ndims = len(input_shape) - 2
        allowed = range(1, ndims + 1)
        self.axis = py.utils.normalize_axes(self.axis, input_shape, allowed, none_means_all=True)
        super().build(input_shape)

    def call(self, x):
        """
        Parameters:
            x: Input tensor whose content to crop.
        """
        if self.prob == 0:
            return x

        return x * utils.augment.draw_crop_mask(x,
                                                crop_min=self.crop_min,
                                                crop_max=self.crop_max,
                                                axis=self.axis,
                                                prob=self.prob,
                                                bilateral=self.bilateral,
                                                seed=self.seed)


class RandomClip(Layer):
    """
    Clip the contents of a tensor in a randomized fashion.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 clip_min=None,
                 clip_max=None,
                 prob_min=1,
                 prob_max=1,
                 axes=0,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            clip_min: Minimum value to clip to, as a scalar. Any value below this value will be
                set to `clip_min` when clipping is performed. Pass a two-element iterable to
                uniformly sample this threshold from the interval [`clip_min[0], `clip_min[1]`).
                None means no clipping at the lower end.
            clip_max: Maximum value to clip to. See `clip_min`.
            prob_min: Clipping probability at the lower end, where 1 means always and 0 never.
            prob_max: Clipping probability at the upper end. See `prob_min`.
            axes: Axes along which clipping will vary independently. None means the the input
                tensor will be clipped as a whole. Pass `axes=(0, -1)` to randomize clipping
                independently for each channel of each batch.
            seed: Integer for reproducible randomization.
        """
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.prob_min = prob_min
        self.prob_max = prob_max
        self.axes = axes
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
            'prob_min': self.prob_min,
            'prob_max': self.prob_max,
            'axes': self.axes,
            'seed': self.seed,
        })
        return config

    def build(self, input_shape):
        self.ndims = len(input_shape)
        self.axes = py.utils.normalize_axes(self.axes, input_shape, none_means_all=False)
        self.rand = np.random.default_rng(self.seed)
        super().build(input_shape)

    def draw_thresh(self, bounds, no_clip_tensor, prob, shape):
        assert 0 <= prob <= 1, f'{prob} is not a probability'
        assert tf.is_tensor(no_clip_tensor), 'not a tensor'

        def seed():
            return self.rand.integers(np.iinfo(int).max)

        # Threshold.
        if bounds is None or prob == 0:
            return no_clip_tensor

        if np.isscalar(bounds):
            clip_at = tf.constant(bounds, no_clip_tensor.dtype)

        else:
            clip_at = tf.random.uniform(shape, minval=bounds[0], maxval=bounds[1], seed=seed())
            if clip_at.dtype != no_clip_tensor.dtype:
                clip_at = tf.cast(clip_at, no_clip_tensor.dtype)

        # Randomization.
        if prob < 1:
            rand_bit = tf.less(tf.random.uniform(shape, seed=seed()), prob)
            rand_bit = tf.cast(rand_bit, no_clip_tensor.dtype)
            clip_at = rand_bit * clip_at + (1 - rand_bit) * no_clip_tensor

        return clip_at

    def call(self, x):
        """
        Parameters:
            x: Input tensor containing values to clip.
        """
        if self.prob_min == self.prob_max == 0:
            return x

        # No-clip values.
        x_min = tf.reduce_min(x)
        x_max = tf.reduce_max(x)

        # Shape. Defines along which axes clipping varies independently.
        shape = [(tf.shape(x)[i] if i in self.axes else 1,) for i in range(self.ndims)]
        shape = tf.concat(shape, axis=0)

        # Randomized thresholds.
        low = self.draw_thresh(self.clip_min, x_min, self.prob_min, shape)
        upp = self.draw_thresh(self.clip_max, x_max, self.prob_max, shape)

        return tf.clip_by_value(x, clip_value_min=low, clip_value_max=upp)


class RandomGamma(Layer):
    """Exponentiate the voxel intensities of a tensor by a random parameter.

    The layer draws gamma parameters from a uniform distribution.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 low=0.5,
                 high=2,
                 shared=False,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            low: Lower bound on the sampled exponents, as a scalar.
            high: Upper bound on the sampled exponents, as a scalar.
            shared: Apply the same gamma exponentiation across all channels.
            seed: Integer for reproducible randomization.
        """
        self.low = low
        self.high = high
        self.shared = shared
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'low': self.low,
            'high': self.high,
            'shared': self.shared,
            'seed': self.seed,
        })
        return config

    def call(self, x):
        """
        Parameters:
            x: Input tensor.
        """
        # Dimensions.
        num_dim = len(x.shape) - 2
        num_batch = tf.shape(x)[0]
        num_chan = 1 if self.shared else tf.shape(x)[-1]
        shape = (num_batch, *[1] * num_dim, num_chan)

        gamma = tf.random.uniform(
            shape,
            minval=self.low,
            maxval=self.high,
            dtype=x.dtype,
            seed=self.seed,
        )
        return tf.pow(x, gamma)


class RandomIntensityLookup(Layer):
    """Augment the contrast of a grayscale image using random intensity lookup tables.

    At each invocation, this layer will synthesize a smoothly varying lookup table (LUT),
    associating a new output intensity to each intensity value in the input image. We will apply
    the LUT to the input image to synthesize a new image contrast. Each batch will undergo an
    independent lookup.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 levels=256,
                 blur_min=32,
                 blur_max=64,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            levels: Number of grayscale levels to look up and re-assign.
            blur_min: Lower bound on the smoothing SD for random-contrast lookup.
            blur_max: Upper bound on the smoothing SD for random-contrast lookup.
            seed: Integer for reproducible randomization.
        """
        self.levels = levels
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'levels': self.levels,
            'blur_min': self.blur_min,
            'blur_max': self.blur_max,
            'seed': self.seed,
        })
        return config

    def call(self, x):
        """
        Parameters:
            x: Input image.
        """
        max_val = self.levels - 1
        if not x.dtype.is_floating:
            x = tf.cast(x, self.dtype)

        # Oversample LUT. Convolution requires trailing singleton dimension.
        num_draw = 5 * self.levels
        lut = tf.random.uniform(
            shape=(tf.shape(x)[0], num_draw, 1),
            minval=0,
            maxval=max_val,
            dtype=x.dtype,
            seed=self.seed,
        )

        # Smoothing. Filter shape: space, in, out.
        kernel = utils.gaussian_kernel(
            sigma=self.blur_max,
            min_sigma=self.blur_min,
            random=self.blur_min != self.blur_max,
            dtype=x.dtype,
            seed=self.seed,
        )
        kernel = tf.reshape(kernel, shape=(-1, 1, 1))
        lut = tf.nn.convolution(lut, kernel, padding='SAME')[..., 0]

        # Remove tapered edges from zero-padding.
        keep = np.arange(self.levels) + (num_draw - self.levels) // 2
        lut = tf.gather(lut, indices=keep, axis=1)

        # Normalize batches independently.
        space = range(1, len(x.shape) - 1)
        x = max_val * utils.minmax_norm(x, axis=space)
        lut = max_val * utils.minmax_norm(lut, axis=1)

        # Lookup.
        indices = tf.cast(x, tf.int32)
        return tf.map_fn(
            fn=lambda x: tf.gather(*x, axis=0),
            elems=(lut, indices),
            fn_output_signature=lut.dtype,
        )


class RandomClearLabel(Layer):
    """Randomly clear image regions corresponding to specific labels.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 prob,
                 clear=0,
                 shared=False,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            prob: Probability that we clear image regions corresponding to the label.
            shared: Synchronize the random clearing across all channels.
            clear: Integer labels to clear, as a scalar or iterable. When passing several values,
                the layer will combine and treat them as a single structure.
            seed: Integer for reproducible randomization.
        """
        self.prob = prob
        self.clear = clear
        self.shared = shared
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'prob': self.prob,
            'clear': self.clear,
            'shared': self.shared,
            'seed': self.seed,
        })
        return config

    def call(self, inputs):
        """
        Parameters:
            inputs: Input image and corresponding label map as an iterable.
        """
        image, labels = inputs
        if self.prob == 0:
            return image

        # Labels to clear.
        bg = [self.clear] if np.isscalar(self.clear) else np.unique(self.clear)
        bg = tf.convert_to_tensor(bg, labels.dtype)

        # Dimensions.
        num_dim = len(image.shape) - 2
        num_batch = tf.shape(image)[0]
        num_chan = 1 if self.shared else tf.shape(image)[-1]
        shape = (num_batch, *[1] * num_dim, num_chan)

        # Randomization.
        rand = tf.random.uniform(shape, dtype=self.dtype, seed=self.seed)
        rand = tf.less(rand, self.prob)

        # Mask.
        mask = tf.reduce_any(tf.equal(labels[..., None], bg), axis=-1)
        mask = tf.math.logical_and(mask, rand)
        mask = tf.math.logical_xor(True, mask)

        return image * tf.cast(mask, image.dtype)


class DrawImage(Layer):
    """ Generate an image from a label map by uniformly sampling a random intensity for each label.

    This layer assumes that input label maps have a single channel and integer values in the
    interval [0, N), where N corresponds to `max_label`. If you pass a list defining intensity
    bounds, the (zero-based) i-th element of the list applies to label value i.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 max_label,
                 low=0,
                 high=1,
                 channels=1,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            max_label: Highest integer value the input index label maps will ever take.
            low: Lower bounds on the intensities drawn for each label, as a scalar or list.
            high: Upper bounds on the intensities drawn for each label, as a scalar or list.
            channels: Number of generated output channels.
            seed: Integer for reproducible randomization.
        """
        self.max_label = max_label
        self.channels = channels
        self.low = low
        self.high = high
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_label': self.max_label,
            'channels': self.channels,
            'low': self.low,
            'high': self.high,
            'seed': self.seed,
        })
        return config

    def call(self, x):
        """
        Parameters:
            x: Input label map with non-negative, zero-based index label values.
        """
        if x.dtype not in (tf.int32, tf.int64):
            x = tf.cast(x, tf.int32)

        num_dim = len(x.shape) - 2
        num_batch = tf.shape(x)[0]
        num_label = self.max_label + 1

        # Random means.
        means = tf.random.uniform(
            shape=(num_batch, self.channels, num_label),
            minval=self.low,
            maxval=self.high,
            dtype=self.dtype,
            seed=self.seed,
        )
        means = tf.reshape(means, shape=(-1,))

        # Label intensities.
        offset_chan = tf.range(self.channels) * num_label
        offset_batch = tf.range(num_batch) * self.channels * num_label
        offset_batch = tf.reshape(offset_batch, shape=(-1, *[1] * num_dim, 1))
        x += offset_batch + offset_chan
        return tf.gather(means, indices=x)


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
        mt = K.transpose(M)  # d x D
        mtm_inv = tf.matrix_inverse(K.dot(mt, M))  # d x d
        self.W = K.dot(mtm_inv, mt)  # d x D

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
            W = self.W  # d x D

            w_tmp = K.expand_dims(W, 0)  # 1 x d x D
            Wo = K.permute_dimensions(w_tmp, [0, 2, 1]) * \
                K.expand_dims(y_mask_flat, -1)  # N x D x d
            WoT = K.permute_dimensions(Wo, [0, 2, 1])    # N x d x D
            WotWo_inv = tf.matrix_inverse(K.batch_dot(WoT, Wo))  # N x d x d
            pre = K.batch_dot(WotWo_inv, WoT)  # N x d x D
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

    If you find this class useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148
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

    If you find this class useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148
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
                             '(only "valid" is supported if implementation is 1): ' + padding)
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
            output = \
                LocallyConnected3D.local_conv(inputs,
                                              self.kernel,
                                              self.kernel_size, self.strides,
                                              (self.output_row, self.output_col, self.output_z),
                                              self.data_format)

        elif self.implementation == 2:
            output = LocallyConnected3D.local_conv_matmul(inputs, self.kernel,
                                                          self.kernel_mask,
                                                          self.compute_output_shape(inputs.shape))

        elif self.implementation == 3:
            output = \
                LocallyConnected3D.local_conv_sparse_matmul(inputs,
                                                            self.kernel,
                                                            self.kernel_idxs,
                                                            self.kernel_shape,
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
                           [K.shape(output_flat)[0], ] + output_shape.as_list()[1:])
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
        output_flat = gen_sparse_ops.sparse_tensor_dense_mat_mul(
            kernel_idxs, kernel, kernel_shape, inputs_flat, adjoint_b=True)
        output_flat_transpose = K.transpose(output_flat)

        output_reshaped = K.reshape(
            output_flat_transpose,
            [K.shape(output_flat_transpose)[0], ] + output_shape.as_list()[1:]
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
                                                                        kernel_shape,
                                                                        output_position,
                                                                        strides,
                                                                        padding)
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
            mean = 1 / input_shape[-1]
            stddev = 0.01
            self.mult_initializer = tf.keras.initializers.RandomNormal(mean=mean, stddev=stddev)

        self.mult = self.add_weight(name='mult-kernel',
                                    shape=mult_shape,
                                    initializer=self.mult_initializer,
                                    regularizer=self.mult_regularizer,
                                    trainable=True)

        if self.use_bias:
            if self.bias_initializer is None:
                mean = 1 / input_shape[-1]
                stddev = 0.01
                self.bias_initializer = tf.keras.initializers.RandomNormal(mean=mean, stddev=stddev)

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
        y = tf.matmul(x, mult)[..., 0, :]
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
            mean = 1 / input_shape[-1]
            stddev = 0.01
            self.mult_initializer = tf.keras.initializers.RandomNormal(mean=mean, stddev=stddev)

        self.mult = self.add_weight(name='mult-kernel',
                                    shape=mult_shape,
                                    initializer=self.mult_initializer,
                                    regularizer=self.mult_regularizer,
                                    trainable=True)

        self.trf = self.add_weight(name='def-kernel',
                                   shape=mult_shape + [ndims],
                                   initializer=tf.keras.initializers.RandomNormal(
                                       mean=0, stddev=0.001),
                                   trainable=True)

        if self.use_bias:
            if self.bias_initializer is None:
                mean = 1 / input_shape[-1]
                stddev = 0.01
                self.bias_initializer = tf.keras.initializers.RandomNormal(mean=mean, stddev=stddev)

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
                trf_vol = transform(vol[..., i], self.trf[..., i, j, :] *
                                    self.trf_mult, interp_method=self.interp_method)
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

    If you find this class useful, please cite the original paper this was written for:
        A.V. Dalca, M. Rakic, J. Guttag, M.R. Sabuncu.
        Learning Conditional Deformable Templates with Convolutional Networks 
        NeurIPS: Advances in Neural Information Processing Systems. pp 804-816, 2019. 

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
            name = prefix + '_' + str(K.get_uid(prefix))

        if not dtype:
            dtype = K.floatx()

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
        output_tensor._keras_history = tf.python.keras.engine.base_layer.KerasHistory(self, 0, 0)
        output_tensor._batch_input_shape = self.shape

        self.trainable = True
        self.built = True
        self.is_placeholder = False

        # create new node
        tf.python.keras.engine.base_layer.node_module.Node(self,
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

    The neuron.layers.LocalParam has an issue where _keras_shape 
        gets lost upon calling get_output :(

    tried using call() but this requires an input (or i don't know how to fix it)
    the fix was that after the return, for every time that tensor would be used i would 
        need to do something like
        new_vec._keras_shape = old_vec._keras_shape
    which messed up the code. 
        Instead, we'll do this quick version where we need an input, but we'll ignore it.

    this doesn't have the _keras_shape issue since we built on the input and use call()

    If you find this class useful, please cite the original paper this was written for:
        A.V. Dalca, M. Rakic, J. Guttag, M.R. Sabuncu.
        Learning Conditional Deformable Templates with Convolutional Networks 
        NeurIPS: Advances in Neural Information Processing Systems. pp 804-816, 2019. 
    """

    def __init__(self, shape, initializer='RandomNormal', mult=1.0, **kwargs):
        self.shape = shape
        self.initializer = initializer
        self.biasmult = mult
        print('LocalParamWithInput: Consider using neuron.layers.LocalParam()')
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shape': self.shape,
        })
        return config

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=self.shape,  # input_shape[1:]
                                      initializer=self.initializer,
                                      trainable=True)
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # want the x variable for it's keras properties and the batch.
        xslice = K.batch_flatten(x)[:, 0:1]
        b = xslice * tf.zeros((1,), dtype=x.dtype) + tf.ones((1,), dtype=x.dtype)
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
# Stream layers
##########################################


class MeanStream(Layer):
    """ 
    Maintain stream of data mean. 

    cap refers to mainting an approximation of up to that number of subjects -- that is,
    any incoming datapoint will have at least 1/cap weight.

    If you find this class useful, please cite the original paper this was written for:
        A.V. Dalca, M. Rakic, J. Guttag, M.R. Sabuncu.
        Learning Conditional Deformable Templates with Convolutional Networks 
        NeurIPS: Advances in Neural Information Processing Systems. pp 804-816, 2019. 
    """

    def __init__(self, cap=100, **kwargs):
        self.cap = float(cap)
        super(MeanStream, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create mean and count
        # These are weights because just maintaining variables don't get saved with the model,
        # and we'd like to have these numbers saved when we save the model.
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

    def call(self, x, training=None):
        training = _get_training_value(training, self.trainable)

        # get batch shape:
        this_bs_int = K.shape(x)[0]

        # prep for broadcasting :(
        p = tf.concat((K.reshape(this_bs_int, (1,)), K.shape(self.mean)), 0)
        z = tf.ones(p)

        # If calling in inference mode, use moving stats
        if training is False:
            return K.minimum(1., self.count / self.cap) * (z * K.expand_dims(self.mean, 0))

        # get new mean and count
        new_mean, new_count = _mean_update(self.mean, self.count, x, self.cap)

        # update op
        self.count.assign(new_count)
        self.mean.assign(new_mean)

        # the first few 1000 should not matter that much towards this cost
        return K.minimum(1., new_count / self.cap) * (z * K.expand_dims(new_mean, 0))

    def compute_output_shape(self, input_shape):
        return input_shape


class CovStream(Layer):
    """ 
    Maintain stream of data covariance. 

    cap refers to mainting an approximation of up to that number of subjects -- that is,
    any incoming datapoint will have at least 1/cap weight.

    If you find this class useful, please cite the original paper this was written for:
        A.V. Dalca, M. Rakic, J. Guttag, M.R. Sabuncu.
        Learning Conditional Deformable Templates with Convolutional Networks 
        NeurIPS: Advances in Neural Information Processing Systems. pp 804-816, 2019. 
    """

    def __init__(self, cap=100, **kwargs):
        self.cap = float(cap)
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

    def call(self, x, training=None):
        training = _get_training_value(training, self.trainable)

        # get batch shape:
        this_bs_int = K.shape(x)[0]

        # prep for broadcasting :(
        p = tf.concat((K.reshape(this_bs_int, (1,)), K.shape(self.cov)), 0)
        z = tf.ones(p)

        # If calling in inference mode, use moving stats
        if training is False:
            return K.minimum(1., self.count / self.cap) * (z * K.expand_dims(self.cov, 0))

        x_orig = x

        # update mean
        new_mean, new_count = _mean_update(self.mean, self.count, x, self.cap)

        # x reshape
        this_bs = tf.cast(this_bs_int, 'float32')  # this batch size
        prev_count = self.count
        x = K.batch_flatten(x)  # B x N

        # new C update. Should be B x N x N
        x = K.expand_dims(x, -1)
        C_delta = K.batch_dot(x, K.permute_dimensions(x, [0, 2, 1]))

        # update cov
        prev_cap = K.minimum(prev_count, self.cap)
        C = self.cov * (prev_cap - 1) + K.sum(C_delta, 0)
        new_cov = C / (prev_cap + this_bs - 1)

        # updates
        self.count.assign(new_count)
        self.mean.assign(new_mean)
        self.cov.assign(new_cov)

        return K.minimum(1., new_count / self.cap) * (z * K.expand_dims(new_cov, 0))

    def compute_output_shape(self, input_shape):
        v = np.prod(input_shape[1:])
        return (input_shape[0], v, v)


def _mean_update(pre_mean, pre_count, x, pre_cap=None):

    # compute this batch stats
    this_sum = tf.reduce_sum(x, 0)
    this_bs = tf.cast(K.shape(x)[0], 'float32')  # this batch size

    # increase count and compute weights
    new_count = pre_count + this_bs
    alpha = this_bs / K.minimum(new_count, pre_cap)

    # compute new mean. Note that once we reach self.cap (e.g. 1000),
    # the 'previous mean' matters less
    new_mean = pre_mean * (1 - alpha) + (this_sum / this_bs) * alpha

    return (new_mean, new_count)


def _get_training_value(training, trainable_flag):
    """
    Return a flag indicating whether a layer should be called in training
    or inference mode.

    Modified from https://git.io/JUGHX

    training: the setting used when layer is called for inference.
    trainable: flag indicating whether the layer is trainable.
    """
    if training is None:
        training = K.learning_phase()

    if isinstance(training, int):
        training = bool(training)

    # If layer not trainable, override value passed from model.
    if trainable_flag is False:
        training = False

    return training


##########################################
# FFT Layers
##########################################

class FFT(Layer):
    """
    Apply the fast Fourier transform (FFT) to a tensor. Supports forward and backward
    (inverse) transforms, and the transformed axes can be specified. The first and last
    dimensions of the input tensor are supposed to indicate batches and features,
    respectively. The output tensor will be complex.

    If you find this class useful, please cite the original paper this was written for:
        Deep-learning-based Optimization of the Under-sampling Pattern in MRI
        C. Bahadir, A.Q. Wang, A.V. Dalca, M.R. Sabuncu.
        IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020.
    """

    def __init__(self, axes=None, inverse=False, **kwargs):
        """
        Parameters:
            axes: Spatial axes along which to take the FFT. None means all axes.
            inverse: Whether to perform a backward (inverse) transform.
        """
        self.axes = axes
        self.inverse = inverse
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axes': self.axes,
            'inverse': self.inverse,
        })
        return config

    def build(self, input_shape):
        ndims = len(input_shape) - 2
        assert ndims in (1, 2, 3), 'only 1D, 2D, or 3D supported'
        self.axes = py.utils.normalize_axes(self.axes,
                                            input_shape,
                                            allowed=range(1, ndims + 1),
                                            none_means_all=True)
        super().build(input_shape)

    def call(self, x):
        return utils.fftn(x, axes=self.axes, inverse=self.inverse)


class IFFT(FFT):
    """
    Apply the inverse fast Fourier transform (iFFT) to a tensor. The transformed axes can be
    specified. The first and last dimensions of the input tensor are supposed to indicate
    batches and features, respectively. The output tensor will be complex. For more information
    see ne.layers.FFT.

    If you find this class useful, please cite the original paper this was written for:
        Deep-learning-based Optimization of the Under-sampling Pattern in MRI
        C. Bahadir, A.Q. Wang, A.V. Dalca, M.R. Sabuncu.
        IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, inverse=True, **kwargs)


class FFTShift(Layer):
    """
    Shift the zero-frequency component to the center of the tensor.
    """

    def __init__(self, axes=None, inverse=False, **kwargs):
        """
        Parameters:
            axes: Spatial axes along which to shift the spectrum. None means all axes.
            inverse: Undo the shift operation.
        """
        self.axes = axes
        self.inverse = inverse
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axes': self.axes,
            'inverse': self.inverse,
        })
        return config

    def build(self, input_shape):
        ndims = len(input_shape) - 2
        assert ndims in (1, 2, 3), 'only 1D, 2D, or 3D supported'
        self.axes = py.utils.normalize_axes(self.axes,
                                            input_shape,
                                            allowed=range(1, ndims + 1),
                                            none_means_all=True)
        super().build(input_shape)

    def call(self, x):
        f = tf.signal.ifftshift if self.inverse else tf.signal.fftshift
        return f(x, axes=self.axes)


class IFFTShift(FFTShift):
    """
    Undo the effect of applying FFTShift. While FFTShift and IFFTShift are identical for
    even-size tensor dimensions, their effect differs by one voxel for dimensions of odd
    size. For more information, see ne.layers.FFTShift.

    If you find this class useful, please cite the original paper this was written for:
        Deep-learning-based Optimization of the Under-sampling Pattern in MRI
        C. Bahadir, A.Q. Wang, A.V. Dalca, M.R. Sabuncu.
        IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, inverse=True, **kwargs)


class ComplexToChannels(Layer):
    """
    Split a complex tensor into a real tensor with features corresponding to the
    real and imaginary components.

    If you find this class useful, please cite the original paper this was written for:
        Deep-learning-based Optimization of the Under-sampling Pattern in MRI 
        C. Bahadir, A.Q. Wang, A.V. Dalca, M.R. Sabuncu.
        IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020.
    """

    def call(self, x):
        return utils.complex_to_channels(x)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] *= 2
        return tuple(shape)


class ChannelsToComplex(Layer):
    """
    Convert a real tensor with an even number N of features into a complex N/2-feature tensor.
    The first N/2 features will be taken as real, the last N/2 features as imaginary components.

    If you find this class useful, please cite the original paper this was written for:
        Deep-learning-based Optimization of the Under-sampling Pattern in MRI 
        C. Bahadir, A.Q. Wang, A.V. Dalca, M.R. Sabuncu.
        IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020.
    """

    def call(self, x):
        return utils.channels_to_complex(x)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = shape[-1] // 2
        return tuple(shape)


##########################################
# Stochastic Sampling layers
##########################################

class SampleNormalLogVar(Layer):
    """ 
    Keras Layer: Gaussian sample given mean and log_variance

    If you find this class useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148

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
        noise = tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # make it a sample from N(mu, sigma^2)
        z = mu + tf.exp(log_var / 2.0) * noise
        return z


class GaussianNoise(Layer):
    """
    Sample and add or return Gaussian noise.

    The layer first draws a standard deviation (SD) from a uniform distribution, then samples
    noise from a normal distribution using the drawn SD. It adds the noise to the input tensor
    or returns only the noise tensor, which will have the same shape as the input tensor.

    If you find this layer useful, please cite:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    def __init__(self,
                 noise_min=0.01,
                 noise_max=0.10,
                 noise_only=False,
                 absolute=False,
                 axes=(0, -1),
                 seed=None,
                 **kwargs):
        """
        Parameters:
            noise_min: Minimum noise SD. Shape must be broadcastable with the output shape.
            noise_max: Maximum noise SD. Shape must be broadcastable with the output shape.
            noise_only: Return the noise tensor rather than adding it to the input.
            absolute: Instead of interpreting the SD bounds relative to the absolute maximum
                value of the input tensor, treat them as absolute values.
            axes: Input axes along which noise will be sampled with a separate SD.
            seed: Optional seed for initializing the random number generation.
        """
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.noise_only = noise_only
        self.absolute = absolute
        self.axes = axes
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'noise_min': self.noise_min,
            'noise_max': self.noise_max,
            'noise_only': self.noise_only,
            'absolute': self.absolute,
            'axes': self.axes,
            'seed': self.seed,
        })
        return config

    def build(self, in_shape):
        num_dim = len(in_shape)
        self.axes = np.ravel(self.axes)
        self.axes = [ax + num_dim if ax < 0 else ax for ax in self.axes]
        assert all(0 <= ax < num_dim for ax in self.axes), 'invalid axes'

        self.rand = tf.random.Generator.from_non_deterministic_state()
        if self.seed is not None:
            self.rand.reset_from_seed(self.seed)

    def call(self, x):
        """
        Parameters:
            x: Input tensor to add noise to.
        """
        if self.noise_max == 0 and not self.noise_only:
            return x

        # Types.
        assert x.dtype.is_floating or x.dtype.is_complex, 'non-FP output type'
        real_type = x.dtype.real_dtype

        # Shapes.
        shape_out = tf.shape(x)
        shape_sd = []
        for i, _ in enumerate(x.shape):
            shape_sd.append(shape_out[i] if i in self.axes else 1)

        # Standard deviation.
        sd = self.rand.uniform(
            shape_sd, minval=self.noise_min, maxval=self.noise_max, dtype=real_type,
        )
        if not self.absolute:
            sd *= tf.reduce_max(tf.abs(x))

        # Direct sampling of complex numbers not supported.
        if x.dtype.is_complex:
            noise = tf.complex(
                self.rand.normal(shape_out, stddev=sd, dtype=real_type),
                self.rand.normal(shape_out, stddev=sd, dtype=real_type),
            )

        else:
            noise = self.rand.normal(shape_out, stddev=sd, dtype=real_type)

        return noise if self.noise_only else x + noise


class PerlinNoise(tf.keras.layers.Layer):
    """
    Sample Perlin Noise.

    The function combines noise tensors at different scales by drawing them at full resolution and
    smoothing randomly. Users control the number of levels via the length of the smoothing bounds.

    At each scale, we uniformly draw a standard deviation (SD). Second, we sample noise from a
    normal distribution with that SD. Before averaging over scales, we blur each tensor over its
    spatial dimensions, keeping constant a global statistic of choice, such as the SD.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 shape=None,
                 noise_min=0.01,
                 noise_max=1,
                 fwhm_min=4,
                 fwhm_max=32,
                 isotropic=False,
                 reduce=tf.math.reduce_std,
                 out_type=tf.float32,
                 axes=None,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            shape: Output shape including feature but excluding batch dimension. None means the
                shape of the input tensor (from which the layer will infer the batch size).
            noise_min: Lower bound on the sampled noise SDs, as a scalar.
            noise_max: Upper bound on the sampled noise SDs, as a scalar.
            fwhm_min: Lower bounds on the smoothing FWHMs. Scalar or iterable. One element/level.
            fwhm_max: Upper bounds on the smoothing FWHMs. Same length as `fwhm_min`.
            isotropic: Whether smoothing should be isotropic.
            reduce: TensorFlow function returning a global statistic to keep constant.
            out_type: Floating-point data type of the output tensor.
            axes: Axes along which the noise has a separate SD. As the layer always varies the SD
                along the batch dimension, passing `0` is not allowed. With `axes=-1`, each channel
                will use a separate sampling SD. With `None`, the same SD will be used for all axes.
            seed: Integer for reproducible randomization.
        """
        self.shape = shape
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.fwhm_min = fwhm_min
        self.fwhm_max = fwhm_max
        self.isotropic = isotropic
        self.reduce = reduce
        self.out_type = tf.dtypes.as_dtype(out_type)
        self.axes = axes
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shape': self.shape,
            'noise_min': self.noise_min,
            'noise_max': self.noise_max,
            'fwhm_min': self.fwhm_min,
            'fwhm_max': self.fwhm_max,
            'isotropic': self.isotropic,
            'reduce': self.reduce,
            'out_type': self.out_type,
            'axes': self.axes,
            'seed': self.seed,
        })
        return config

    def build(self, input_shape):
        self.rand = np.random.default_rng(self.seed)

        allowed = range(1, len(input_shape))
        self.axes = py.utils.normalize_axes(self.axes, input_shape, allowed, none_means_all=False)
        super().build(input_shape)

    def call(self, x):
        """
        Parameters:
            x: Input tensor defining the batch size (and potentially shape).
        """
        shape = x.shape[1:] if self.shape is None else self.shape
        dtype = self.out_type
        return tf.map_fn(lambda x: self._single_batch(x, shape), x, fn_output_signature=dtype)

    def _single_batch(self, _, shape):
        return utils.augment.draw_perlin_full(shape,
                                              noise_min=self.noise_min,
                                              noise_max=self.noise_max,
                                              isotropic=self.isotropic,
                                              fwhm_min=self.fwhm_min,
                                              fwhm_max=self.fwhm_max,
                                              batched=False,
                                              featured=True,
                                              dtype=self.out_type,
                                              seed=self.rand.integers(np.iinfo(int).max),
                                              axes=[ax - 1 for ax in self.axes],
                                              reduce=self.reduce)


class Constant(Layer):
    """Convert an input value to a constant Keras layer with a batch dimension.

    The layer will derive the batch size from an input tensor. Without an input tensor or if you
    pass an empty list, the batch size will be one.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self, value, **kwargs):
        """
        Parameters:
            value: Constant value. Must be convertible to a tensor and exclude the batch dimension.
            out_type: Data type of the constant. Has to be convertible to a TensorFlow type.
                Defaults to None, meaning the type will be inferred from the passed value.
        """
        self.value = value
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(value=self.value)
        return config

    def build(self, _):
        self.const = tf.cast(self.value, self.dtype)
        if tf.rank(self.const) == 0:
            self.const = tf.reshape(self.const, shape=(1,))

    def call(self, x=[]):
        """
        Parameters:
            x: Any input tensor that we can derive a batch size from. Without an input tensor or an
                empty list, the batch size will be one. Some Keras versions may be incompatible
                with the default argument and require that you explicitly pass an empty list.
        """
        batch = tf.maximum(tf.shape(x)[0], 1)
        shape = tf.concat(([batch], tf.shape(self.const)), axis=0)
        return tf.broadcast_to(self.const, shape)


##########################################
# HyperMorph Layers
##########################################

class HyperConv(Layer):
    """
    Private, abstract N-D hyper-convolution layer for use in hypernetworks.
    This layer has no trainable weights, as it performs a convolution
    using externel kernel (and bias) weights that are provided as
    input tensors. The expected layer input is a tensor list:

        [input_features, kernel_weights, bias_weights]

    Parameters:
        rank: Rank of the convolution.
        filters: The dimensionality of the output space.
        kernel_size: An int or int list specifying the convolution window size.
        strides: An int or int list specifying the stride of the convolution. Default is 1.
        padding: One of 'valid' or 'same' (case-insensitive). Default is 'valid'.
        dilation_rate: Dilation rate to use for dilated convolution. Default is 1.
        activation: Activation function. Default is None.
        use_bias: Whether the layer applies a bias. Default is True.
        name: Layer name.
    """

    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 name=None,
                 **kwargs):

        super().__init__(name=name, **kwargs)

        # TODO: filters doesn't actually need to be specified as it can be
        # determined by the input kernel size
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)

        if self.padding == 'causal':
            raise ValueError('Causal padding is not supported for HyperConv')

        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self._build_conv_op(tf.TensorShape(input_shape[0]))
        self.built = True

    def _build_conv_op(self, input_shape):
        """
        Configures the convolutional op for the input tensors, given the input shape.
        """
        kernel_shape = tf.TensorShape(self.kernel_size + (int(input_shape[-1]), self.filters))
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=kernel_shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format('channels_last', self.rank + 2))

    def call(self, inputs):
        """
        Runs per-batch convolution on the inputs, consisting of input features, kernels weights,
        and optional bias weights (when use_bias is True).
        """
        outputs = tf.map_fn(self._convolve_batch, inputs, fn_output_signature=inputs[0].dtype)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def _convolve_batch(self, inputs):
        """
        Performs convolution on a single batch of input features, kernels weights,
        and optional bias weights.
        """

        # add batch axis for input layer
        features_input = tf.expand_dims(inputs[0], axis=0)
        kernel_weights = inputs[1]

        # convolve
        outputs = self._convolution_op(features_input, kernel_weights)

        # add bias weights
        if self.use_bias:
            bias_weights = inputs[2]
            outputs = tf.nn.bias_add(outputs, bias_weights, data_format='NHWC')

        # remove added batch axis
        outputs = outputs[0]
        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes output tensor shape.
        """
        input_shape = input_shape[0]  # grab features input tensor
        input_shape = tf.TensorShape(input_shape).as_list()
        space = input_shape[1:-1]
        new_space = []

        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i]
            )
            new_space.append(new_dim)

        return tf.TensorShape([input_shape[0]] + new_space + [self.filters])

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HyperConv2D(HyperConv):
    """
    2D hyper-convolution layer for use in hypernetworks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class HyperConv3D(HyperConv):
    """
    3D hyper-convolution layer for use in hypernetworks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


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
        hyperkernel_activation: Activation for the hyper-kernel mapping. Default is None.
        hyperbias_activation: Activation for the hyper-bias mapping. Default is None.
        name: Layer name.
        kwargs: Forwarded to the HyperConv constructor.
    """

    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 hyperkernel_use_bias=True,
                 hyperbias_use_bias=True,
                 hyperkernel_activation=None,
                 hyperbias_activation=None,
                 name=None,
                 **kwargs):

        super().__init__(rank, filters, kernel_size, name=name, **kwargs)
        self.hyperkernel_use_bias = hyperkernel_use_bias
        self.hyperbias_use_bias = hyperbias_use_bias
        self.hyperkernel_activation = tf.keras.activations.get(hyperkernel_activation)
        self.hyperbias_activation = tf.keras.activations.get(hyperbias_activation)

    def build(self, input_shape):
        """
        Builds a hyper-conv layer from a tensor with two internal dense operations,
        'pseudo dense layers', that predict convolutional kernel and optional bias weights,
        if use_bias is True.
        """
        last_dim = int(input_shape[1][-1])
        kernel_shape = tf.TensorShape(self.kernel_size + (int(input_shape[0][-1]), self.filters))

        # builds the internal dense layer (kernel and bias weights that
        # create the hyper-conv kernel weights)
        self.hyperkernel = self._build_dense_pseudo_layer(
            name='hyperkernel',
            last_dim=last_dim,
            target_shape=kernel_shape,
            use_bias=self.hyperkernel_use_bias,
            activation=self.hyperkernel_activation)

        # builds the internal dense layer (kernel and bias weights that
        # create the hyper-conv bias weights)
        if self.use_bias:
            self.hyperbias = self._build_dense_pseudo_layer(
                name='hyperbias',
                last_dim=last_dim,
                target_shape=[self.filters],
                use_bias=self.hyperbias_use_bias,
                activation=self.hyperbias_activation)

        # build the convolutional op (see HyperConv)
        self._build_conv_op(tf.TensorShape(input_shape[0]))
        self.built = True

    def call(self, inputs):
        """
        Calls the internal dense layers that compute the hyper kernel and bias
        weights, then convolves the input features with those computed weights.
        """
        kernel = self._call_dense_pseudo_layer(inputs[1], self.hyperkernel)

        if self.use_bias:
            bias = self._call_dense_pseudo_layer(inputs[1], self.hyperbias)
            return super().call([inputs[0], kernel, bias])

        return super().call([inputs[0], kernel])

    def _build_dense_pseudo_layer(self, name, last_dim, target_shape, use_bias, activation):
        """
        Creates weights for an internal dense 'pseudo-layer' described
        in the build() documentation.
        """
        target_shape = tf.TensorShape(target_shape)
        units = np.prod(target_shape.as_list())

        # create dense kernel weights
        kernel = self.add_weight(
            name='%s_kernel' % name,
            shape=[last_dim, units],
            trainable=True)

        # create dense bias weights
        if use_bias:
            bias = self.add_weight(
                name='%s_bias' % name,
                shape=[units],
                trainable=True)
        else:
            bias = None

        return (kernel, bias, activation, target_shape)

    def _call_dense_pseudo_layer(self, inputs, params):
        """
        Calls an internal dense 'pseudo-layer' described in the build() documentation.
        """
        kernel, bias, activation, target_shape = params
        inputs = tf.cast(inputs, self.compute_dtype)

        if K.is_sparse(inputs):
            outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, kernel)
        else:
            outputs = gen_math_ops.mat_mul(inputs, kernel)

        if bias is not None:
            outputs = tf.nn.bias_add(outputs, bias)

        if activation is not None:
            outputs = activation(outputs)

        return tf.reshape(outputs, (-1, *target_shape))

    def get_config(self):
        config = {
            'hyperkernel_use_bias': self.hyperkernel_use_bias,
            'hyperbias_use_bias': self.hyperbias_use_bias,
            'hyperkernel_activation': tf.keras.activations.serialize(self.hyperkernel_activation),
            'hyperbias_activation': tf.keras.activations.serialize(self.hyperbias_activation)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class HyperConv2DFromDense(HyperConvFromDense):
    """
    2D hyper-convolution dense wrapping layer for use in hypernetworks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class HyperConv3DFromDense(HyperConvFromDense):
    """
    3D hyper-convolution dense wrapping layer for use in hypernetworks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


class HyperDense(Layer):
    """
    Hyper-dense layer for use in hypernetworks. This layer has no
    trainable weights, as it performs a dense operation using externel kernel
    (and bias) weights that are provided as input tensors. The expected layer
    input is a tensor list:

        [input, kernel_weights, bias_weights]

    Parameters:
        units: Dimensionality of the output space.
        activation: Activation function. Default is None.
        use_bias: Whether the layer applies a bias. Default is True.
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 **kwargs):

        super().__init__(**kwargs)

        # TODO: units doesn't actually need to be specified as it can be
        # determined by the input kernel size
        self.units = int(units) if not isinstance(units, int) else units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.supports_masking = True

    def call(self, inputs):
        """
        Runs per-batch dense operation on the inputs, consisting of input features,
        kernels weights, and optional bias weights (when use_bias is True).
        """
        outputs = tf.map_fn(self._call_batch, inputs, inputs[0].dtype)

        # apply activation to all batches
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def _call_batch(self, inputs):
        """
        Performs dense operation on a single batch of input features,
        kernels weights, and optional bias weights.
        """
        x = tf.expand_dims(inputs[0], axis=0)  # add batch axis for input layer
        x = tf.cast(x, self.compute_dtype)
        kernel = inputs[1]

        if K.is_sparse(x):
            outputs = sparse_ops.sparse_tensor_dense_matmul(x, kernel)
        else:
            outputs = gen_math_ops.mat_mul(x, kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, inputs[2])

        outputs = outputs[0]  # remove added batch axis
        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes output tensor shape.
        """
        input_shape = input_shape[0]  # grab 'true' input tensor
        input_shape = tf.TensorShape(input_shape).with_rank_at_least(2)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        })
        return config


class HyperDenseFromDense(HyperDense):
    """
    Hyper-dense wrapping layer that includes the dense mapping from a
    final hypernetwork layer to the internal kernel/bias weights. The
    expected layer input is a tensor list:

        [input_features, last_hypernetwork_output]

    Parameters:
        units: Dimensionality of the output space.
        hyperkernel_use_bias: Enable bias in hyper-kernel mapping. Default is True.
        hyperbias_use_bias: Enable bias in hyper-bias mapping. Default is True.
        hyperkernel_activation: Activation for the hyper-kernel mapping. Default is None.
        hyperbias_activation: Activation for the hyper-bias mapping. Default is None.
        kwargs: Forwarded to the HyperDense constructor.
    """

    def __init__(self,
                 units,
                 hyperkernel_use_bias=True,
                 hyperbias_use_bias=True,
                 hyperkernel_activation=None,
                 hyperbias_activation=None,
                 **kwargs):

        super().__init__(units, **kwargs)
        self.hyperkernel_use_bias = hyperkernel_use_bias
        self.hyperbias_use_bias = hyperbias_use_bias
        self.hyperkernel_activation = tf.keras.activations.get(hyperkernel_activation)
        self.hyperbias_activation = tf.keras.activations.get(hyperbias_activation)

    def build(self, input_shape):
        """
        Builds a hyper-dense layer from a tensor with two internal dense operations,
        'pseudo dense layers', that predict hyper-dense kernel and optional bias weights,
        if use_bias is True.
        """
        last_dim = int(input_shape[1][-1])

        # builds the internal dense layer (kernel and bias weights that
        # create the hyper-dense kernel weights)
        self.hyperkernel = self._build_dense_pseudo_layer(
            name='hyperkernel',
            last_dim=last_dim,
            target_shape=[int(input_shape[0][-1]), self.units],
            use_bias=self.hyperkernel_use_bias,
            activation=self.hyperkernel_activation)

        # builds the internal dense layer (kernel and bias weights that
        # create the hyper-dense bias weights)
        if self.use_bias:
            self.hyperbias = self._build_dense_pseudo_layer(
                name='hyperbias',
                last_dim=last_dim,
                target_shape=[self.units],
                use_bias=self.hyperbias_use_bias,
                activation=self.hyperbias_activation)

        self.built = True

    def call(self, inputs):
        """
        Calls the internal dense layers that compute the hyper kernel and bias
        weights, then calls the real dense operation on the input layer with
        those computed weights.
        """
        kernel = self._call_dense_pseudo_layer(inputs[1], self.hyperkernel)

        if self.use_bias:
            bias = self._call_dense_pseudo_layer(inputs[1], self.hyperbias)
            return super().call([inputs[0], kernel, bias])

        return super().call([inputs[0], kernel])

    def _build_dense_pseudo_layer(self, name, last_dim, target_shape, use_bias, activation):
        """
        Creates weights for an internal dense 'pseudo-layer' described
        in the build() documentation.
        """
        target_shape = tf.TensorShape(target_shape)
        units = np.prod(target_shape.as_list())

        # create dense kernel weights
        kernel = self.add_weight(
            name='%s_kernel' % name,
            shape=[last_dim, units],
            trainable=True)

        # create dense bias weights
        if use_bias:
            bias = self.add_weight(
                name='%s_bias' % name,
                shape=[units],
                trainable=True)
        else:
            bias = None

        return (kernel, bias, activation, target_shape)

    def _call_dense_pseudo_layer(self, inputs, params):
        """
        Calls an internal dense 'pseudo-layer' described in the build() documentation.
        """
        kernel, bias, activation, target_shape = params
        inputs = tf.cast(inputs, self.compute_dtype)

        if K.is_sparse(inputs):
            outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, kernel)
        else:
            outputs = gen_math_ops.mat_mul(inputs, kernel)

        if bias is not None:
            outputs = tf.nn.bias_add(outputs, bias)

        if activation is not None:
            outputs = activation(outputs)

        return tf.reshape(outputs, (-1, *target_shape))

    def get_config(self):
        config = {
            'hyperkernel_use_bias': self.hyperkernel_use_bias,
            'hyperbias_use_bias': self.hyperbias_use_bias,
            'hyperkernel_activation': tf.keras.activations.serialize(self.hyperkernel_activation),
            'hyperbias_activation': tf.keras.activations.serialize(self.hyperbias_activation)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
