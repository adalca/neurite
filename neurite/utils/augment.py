import warnings
import numpy as np
import tensorflow as tf
import neurite as ne


def draw_perlin(out_shape,
                scales,
                min_std=0,
                max_std=1,
                dtype=tf.float32,
                seed=None):
    '''
    Generate Perlin noise by drawing from Gaussian distributions at different
    resolutions, upsampling and summing. There are a couple of key differences
    between this function and the Neurite equivalent ne.utils.perlin_vol, which
    are not straightforwardly consolidated.

    Neurite function:
        (1) Iterates over scales in range(a, b) where a, b are input arguments.
        (2) Noise volumes are sampled at resolutions vol_shape / 2**scale.
        (3) Noise volumes are sampled uniformly in the interval [0, 1].
        (4) Volume weights are {1, 2, ...N} (normalized) where N is the number
            of scales, or sampled uniformly from [0, 1].

    This function:
        (1) Specific scales are passed as a list.
        (2) Noise volumes are sampled at resolutions vol_shape / scale.
        (3) Noise volumes are sampled normally, with SDs drawn uniformly from
            [min_std, max_std].

    Parameters:
        out_shape: List defining the output shape. In N-dimensional space, it
            should have N+1 elements, the last one being the feature dimension.
        scales: List of relative resolutions at which noise is sampled normally.
            A scale of 2 means half resolution relative to the output shape.
        min_std: Minimum standard deviation (SD) for drawing noise volumes.
        max_std: Maximum SD for drawing noise volumes.
        dtype: Output data type.
        seed: Integer for reproducible randomization. This may only have an
            effect if the function is wrapped in a Lambda layer.
    '''
    out_shape = np.asarray(out_shape, dtype=np.int32)
    if np.isscalar(scales):
        scales = [scales]

    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(int).max)
    out = tf.zeros(out_shape, dtype=dtype)
    for scale in scales:
        sample_shape = np.ceil(out_shape[:-1] / scale)
        sample_shape = np.int32((*sample_shape, out_shape[-1]))

        std = tf.random.uniform(
            shape=[], minval=min_std, maxval=max_std, dtype=dtype, seed=seed(),
        )
        gauss = tf.random.normal(sample_shape, stddev=std, dtype=dtype, seed=seed())

        zoom = [o / s for o, s in zip(out_shape, sample_shape)]
        out += gauss if scale == 1 else ne.utils.resize(gauss, zoom[:-1])

    return out


def random_blur_rescale(x,
                        std_min=8 / 2.355,
                        std_max=32 / 2.355,
                        isotropic=False,
                        seed=None,
                        reduce=tf.math.reduce_std,
                        batched=False):
    """Randomly smooth and rescale a tensor.

    Smooth a tensor by convolving its spatial dimensions with a random Gaussian kernel and rescale
    such that a global statistic of choice computed over all tensor elements remains unchanged.

    Parameters:
        x: Input tensor with a feature dimension. See `batched` for the batch dimension.
        std_min: Lower bound on the smoothing SD.
        std_max: Upper bound on the smoothing SD.
        isotropic: Whether the smoothing kernels are isotropic.
        seed: Integer for reproducible randomization.
        reduce: TensorFlow function returning a scalar that defines the global statistic.
        batched: Whether the input tensor has a leading batch dimension.

    Returns:
        Smoothed tensor with the same shape and data type as the input.

    See also:
        ne.layers.GaussianBlur

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    # Kernel.
    n_dim = len(x.shape[int(batched):-1])
    prop = dict(sigma=std_max, separate=True, random=True, min_sigma=std_min, dtype=x.dtype)

    rand = np.random.default_rng(seed)
    seeds = rand.integers(np.iinfo(int).max, size=n_dim)
    kernel = [ne.utils.gaussian_kernel(**prop, seed=s) for s in seeds]
    if isotropic:
        kernel = kernel[:1] * n_dim

    # Rescaling.
    before = reduce(x)
    x = ne.utils.separable_conv(x, kernel, batched=batched)
    after = reduce(x)
    return x * tf.math.divide_no_nan(before, after)


def draw_perlin_full(shape,
                     noise_min=0.01,
                     noise_max=1,
                     fwhm_min=4,
                     fwhm_max=32,
                     isotropic=False,
                     batched=False,
                     featured=False,
                     reduce=tf.math.reduce_std,
                     dtype=tf.float32,
                     axes=None,
                     seed=None):
    """Draw Perlin-noise images without interpolating.

    The function combines noise tensors at different scales by drawing them at full resolution and
    smoothing randomly. This results in more variable and natural looking noise while increasing
    efficiency over previous interpolation-based implementations (for reasonable kernels). The
    user controls the number of levels via the length of the smoothing bounds.

    At each scale, we uniformly draw a standard deviation (SD). Second, we sample noise from a
    normal distribution with that SD. Before averaging over scales, we blur each tensor, keeping
    constant a global statistic of choice, for example the SD of the noise tensor.

    Parameters:
        shape: Shape of the output tensor. See `batched` and `featured`.
        noise_min: Lower bound on the sampled noise SD.
        noise_max: Upper bound on the sampled noise SD.
        fwhm_min: Lower bounds on the blurring FWHM in voxels. Scalar or iterable, 1 element/level.
        fwhm_max: Upper bounds on the blurring FWHM in voxels. Same length as `fwhm_min`.
        isotropic: Blur isotropically.
        batched: Indicate that `shape` includes a leading batch dimension.
        featured: Indicate that `shape` includes a trailing feature dimension.
        reduce: TensorFlow function returning a global statistic to keep constant while smoothing.
        dtype: Floating-point data type of the output tensor.
        axes: Axes indexing into shape and along which noise will be sampled with a separate SD.
            A value of None means noise will be sampled using a single global SD at each execution.
        seed: Integer for reproducible randomization.

    Returns:
        Perlin noise tensor of the specified shape and type.

    See also:
        ne.layers.PerlinNoise

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    # Settings.
    assert 0 < noise_min <= noise_max, f'invalid noise-SD bounds {(noise_min, noise_max)}'
    rand = np.random.default_rng(seed)
    dtype = tf.dtypes.as_dtype(dtype)

    def seed():
        return rand.integers(np.iinfo(int).max)

    # Dimensions. Increment axes if we prepend a batch dimension.
    axes = ne.py.utils.normalize_axes(axes, shape, none_means_all=False)
    if not batched:
        shape = tf.concat(([1], shape), axis=0)
        axes = [ax + 1 for ax in axes]
    if not featured:
        shape = tf.concat((shape, [1]), axis=0)

    # SD shape.
    indices = tf.range(tf.size(shape))
    in_axes = tf.reduce_any(tf.equal(indices[..., None], axes), axis=-1)
    in_axes = tf.cast(in_axes, shape.dtype)
    shape_sd = shape * in_axes + tf.ones_like(shape) * (1 - in_axes)

    if not hasattr(fwhm_min, '__iter__'):
        fwhm_min = [fwhm_min]
    if not hasattr(fwhm_max, '__iter__'):
        fwhm_max = [fwhm_max]
    assert len(fwhm_min) == len(fwhm_max), 'different number of lower and upper bounds'

    # Levels.
    out = []
    for low, upp in zip(fwhm_min, fwhm_max):
        noise = tf.random.uniform(
            shape=shape_sd,
            minval=noise_min,
            maxval=noise_max,
            dtype=dtype,
            seed=seed(),
        )
        noise = tf.random.normal(shape, stddev=noise, dtype=dtype, seed=seed())
        noise = random_blur_rescale(
            noise,
            std_min=low / 2.355,
            std_max=upp / 2.355,
            batched=True,
            isotropic=isotropic,
            seed=seed(),
            reduce=reduce,
        )
        out.append(noise)

    # Output. Compute mean to maintain the noise level when adding scales.
    out = tf.reduce_mean(out, axis=0)
    if not batched:
        out = out[0, ...]
    if not featured:
        out = out[..., 0]
    return out


def draw_crop_mask(x, crop_min=0, crop_max=0.5, axis=None, prob=1, bilateral=False, seed=None):
    """Draw a mask to multiplicatively crop the field of view of an ND tensor along an axis.

    Parameters:
        x: Input tensor or NumPy array defining the shape and data type of the mask.
        crop_min: Minimum proportion of voxels to remove, in [0, `crop_max`].
        crop_max: Maximum proportion of voxels to remove, in [`crop_min`, 1].
        axis: Axis along which to crop, where None means any axis. With more than one
            axis specified, a single axis will be drawn at each execution.
        prob: Cropping probability, where 1 means always and 0 never.
        bilateral: Randomly distribute the cropping proportion between top/bottom end.
        seed: Integer for reproducible randomization.

    Returns:
        ND tensor of the input data type, with singleton dimensions where needed.

    See also:
        ne.layers.RandomCrop

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    rand = np.random.default_rng(seed)

    def seed():
        return rand.integers(np.iinfo(int).max)

    # Normalize inputs.
    x = tf.concat(x, axis=0)
    axis = ne.py.utils.normalize_axes(axis, x.shape, none_means_all=True)
    assert 0 <= crop_min <= crop_max <= 1, f'invalid proportions {crop_min}, {crop_max}'

    # Decide how much to crop, making sure maxval is >0 to avoid errors.
    prop_cut = tf.constant(crop_max, tf.float32)
    if crop_min < crop_max:
        prop_cut = tf.random.uniform(shape=[], minval=crop_min, maxval=crop_max, seed=seed())

    # Decide whether to crop.
    assert 0 <= prob <= 1, f'{prob} not a probability'
    if prob < 1:
        rand_bit = tf.less(tf.random.uniform(shape=[], seed=seed()), prob)
        prop_cut *= tf.cast(rand_bit, prop_cut.dtype)

    # Split cropping proportion for top and bottom.
    rand_prop = tf.random.uniform(shape=[], seed=seed())
    if not bilateral:
        rand_prop = tf.cast(tf.less(rand_prop, 0.5), prop_cut.dtype)
    prop_low = prop_cut * rand_prop
    prop_cen = 1 - prop_cut

    # Draw cropping axis.
    ind = tf.random.uniform(shape=[], maxval=len(axis), dtype=tf.int32, seed=seed())
    crop = tf.gather(axis, ind)

    # One-dimensional masks for each axis.
    masks = []
    for ax in axis:
        width = tf.shape(x)[ax]
        prop = tf.range(1, delta=1 / tf.cast(width, prop_cut.dtype))
        mask = tf.logical_and(
            tf.greater_equal(prop, prop_low),
            tf.less(prop, prop_low + prop_cen),
        )
        mask = tf.cast(mask, x.dtype)
        is_axis = tf.cast(tf.equal(crop, ax), x.dtype)
        mask = mask * is_axis + tf.ones_like(mask) * (1 - is_axis)
        shape = tf.roll(input=(width, *[1] * len(x.shape[1:])), shift=ax, axis=0)
        mask = tf.reshape(mask, shape)
        mask = tf.cast(mask, x.dtype)
        masks.append(mask)

    # Broadcast-muliply together.
    out = masks.pop()
    for mask in masks:
        out *= mask

    return out
