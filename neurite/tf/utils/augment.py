import numpy as np
import tensorflow as tf
import neurite as ne


def draw_perlin(out_shape, scales, max_std=1, modulate=True, dtype=tf.float32, seed=None):
    '''Generate Perlin noise by drawing from Gaussian distributions at different
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
        (3) Noise volumes are sampled normally, using SD 1.
        (4) Volume weights are all max_sigma, or sampled uniformly from
            [0, max_sigma], where max_sigma is an input argument.

    Parameters:
        out_shape: List defining the output shape. In N-dimensional space, it
            should have N+1 elements, the last one being the feature dimension.
        scales: List of relative resolutions at which noise is sampled normally.
            A scale 2 means half resolution relative to the output shape.
        max_std: Maximum standard deviation (SD) for drawing noise volumes.
        modulate: Whether the SD for each scale is drawn from [0, max_std].
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
        sample_shape = np.ceil(out_shape[:-1] / scale).astype(int)
        sample_shape = (*sample_shape, out_shape[-1])

        std = max_std
        if modulate:
            std = tf.random.uniform((1,), maxval=max_std, dtype=dtype, seed=seed())

        gauss = tf.random.normal(sample_shape, stddev=std, dtype=dtype, seed=seed())
        zoom = [o / s for o, s in zip(out_shape, sample_shape)]
        out += gauss if scale == 1 else ne.utils.resize(gauss, zoom[:-1])

    return out
