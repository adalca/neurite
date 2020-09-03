"""
tensorflow/keras utilities for the neurite project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

or for the transformation/interpolation related functions:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

# python imports
import itertools

# third party imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

# local imports
import pystrum.pynd.ndutils as nd
import neurite as ne
import neurite.py.utils


def interpn(vol, loc, interp_method='linear', fill_value=None):
    """
    N-D gridded interpolation in tensorflow

    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice 
    for the first dimensions

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'
        fill_value: value to use for points outside the domain. If None, the nearest
            neighbors will be used (default).

    Returns:
        new interpolated volume of the same size as the entries in loc

    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """
    
    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)
    nb_dims = loc.shape[-1]

    if len(vol.shape) not in [nb_dims, nb_dims+1]:
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    # flatten and float location Tensors
    if loc.dtype != tf.float32:
        loc = tf.cast(loc, 'float32')
    

    if isinstance(vol.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = vol.shape.as_list()
    else:
        volshape = vol.shape

    max_loc = [d - 1 for d in vol.get_shape().as_list()]

    # interpolate
    if interp_method == 'linear':
        # get floor. 
        # This has to remain a tf.float32 since we will be using loc1 in a float32 op
        loc0 = tf.floor(loc)

        # clip values
        clipped_loc = [tf.clip_by_value(loc[...,d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [tf.clip_by_value(loc0[...,d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0] # note reverse ordering since weights are inverse of diff.

        # go through all the cube corners, indexed by a ND binary vector 
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0
        
        for c in cube_pts:
            
            # get nd values
            # note re: indices above volumes via https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's too
            #   expensive. Instead we fill the output with zero for the corresponding value. The CPU
            #   version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because the latter needs tf.stack which is slow :(
            idx = sub2ind2d(vol.shape[:-1], subs)
            vol_val = tf.gather(tf.reshape(vol, [-1, volshape[-1]]), idx)

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            # tf stacking is slow, we will use prod_n()
            # wlm = tf.stack(wts_lst, axis=0)
            # wt = tf.reduce_prod(wlm, axis=0)
            wt = prod_n(wts_lst)
            wt = K.expand_dims(wt, -1)
            
            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val
        
    else:
        assert interp_method == 'nearest', 'method should be linear or nearest, got: %s' % interp_method
        roundloc = tf.cast(tf.round(loc), 'int32')
        roundloc = [tf.clip_by_value(roundloc[...,d], 0, max_loc[d]) for d in range(nb_dims)]

        # get values
        # tf stacking is slow. replace with gather
        # roundloc = tf.stack(roundloc, axis=-1)
        # interp_vol = tf.gather_nd(vol, roundloc)
        idx = sub2ind2d(vol.shape[:-1], roundloc)
        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx) 

    if fill_value is not None:
        out_type = interp_vol.dtype
        fill_value = tf.constant(fill_value, dtype=out_type)
        below = [tf.less(loc[...,d], 0) for d in range(nb_dims)]
        above = [tf.greater(loc[...,d], max_loc[d]) for d in range(nb_dims)]
        out_of_bounds = tf.reduce_any(tf.stack(below + above, axis=-1), axis=-1, keepdims=True)
        interp_vol *= tf.cast(tf.logical_not(out_of_bounds), dtype=out_type)
        interp_vol += tf.cast(out_of_bounds, dtype=out_type) * fill_value

    return interp_vol


def resize(vol, zoom_factor, interp_method='linear'):
    """
    if zoom_factor is a list, it will determine the ndims, in which case vol has to be of length ndims of ndims + 1

    if zoom_factor is an integer, then vol must be of length ndims + 1

    """

    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]
        
        assert len(vol_shape) in (ndims, ndims+1), \
            "zoom_factor length %d does not match ndims %d" % (len(vol_shape), ndims)

    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims
    if not isinstance(vol_shape[0], int):
        vol_shape = vol_shape.as_list()

    new_shape = [vol_shape[f] * zoom_factor[f] for f in range(ndims)]
    new_shape = [int(f) for f in new_shape]

    lin = [tf.linspace(0., vol_shape[d]-1., new_shape[d]) for d in range(ndims)]
    grid = ne.utils.ndgrid(*lin)

    return ne.utils.interpn(vol, grid, interp_method=interp_method)


zoom = resize


###############################################################################
# volumetric / axis operations
###############################################################################

def tf_map_fn_axis(fn, elems, axis, **kwargs):
    """
    apply tf.map_fn along a specific axis
    
    Parameters:
        fn: function to apply
        elems:
            if elems is a Tensor, axis is an int
            if elems is a list, axis is a list of same length
        axis: axis to apply along
        kwargs: other arguments for tf.map_fn

    """
    
    # determine lists
    islist = isinstance(elems, (tuple, list))
    if not islist:
        elems = [elems]
        assert not isinstance(axis, (tuple, list)), 'axis cannot be list if elements are not list'
        axis = [axis]
        
        
    elems_perm = []
    for xi, x in enumerate(elems):
        a = axis[xi]
        s = len(x.get_shape().as_list())
        if a == -1: a = s - 1

        # move channels to front, so x will be [axis, ...]
        perm = [a] + list(range(0, a)) + list(range(a + 1, s))
        elems_perm.append(K.permute_dimensions(x, perm))

    # compute sptial deformation regularization for this channel
    if not islist:
        elems_perm = elems_perm[0]
        
    x_perm_trf = tf.map_fn(fn, elems_perm, **kwargs)
    if not islist:
        x_perm_trf = [x_perm_trf]
        

    # move in_channels back to end
    elems_trf = []
    for xi, x in enumerate(x_perm_trf):
        a = axis[xi]
        s = len(x.get_shape().as_list())
        if a == -1: a = s - 1
            
        perm = list(range(1, a + 1)) + [0] + list(range(a + 1, s))
        elems_trf.append(K.permute_dimensions(x, perm))
        
    if not islist:
        elems_trf = elems_trf[0]
    
    return elems_trf


def volshape_to_ndgrid(volshape, **kwargs):
    """
    compute Tensor ndgrid from a volume size

    Parameters:
        volshape: the volume size
        **args: "name" (optional)

    Returns:
        A list of Tensors

    See Also:
        ndgrid
    """
    
    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return ndgrid(*linvec, **kwargs)


def volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size

    Warning: this uses the tf.meshgrid convention, of 'xy' indexing.
    to use `ij` indexing, use the ndgrid equivalent

    Parameters:
        volshape: the volume size
        **args: "name" (optional)

    Returns:
        A list of Tensors

    See Also:
        tf.meshgrid, meshgrid, ndgrid, volshape_to_ndgrid
    """
    
    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)


def ndgrid(*args, **kwargs):
    """
    broadcast Tensors on an N-D grid with ij indexing
    uses meshgrid with ij indexing

    Parameters:
        *args: Tensors with rank 1
        **args: "name" (optional)

    Returns:
        A list of Tensors
    
    """
    return meshgrid(*args, indexing='ij', **kwargs)


def meshgrid(*args, **kwargs):
    """
    
    meshgrid code that builds on (copies) tensorflow's meshgrid but dramatically
    improves runtime by changing the last step to tiling instead of multiplication.
    https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/python/ops/array_ops.py#L1921
    
    Broadcasts parameters for evaluation on an N-D grid.
    Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
    of N-D coordinate arrays for evaluating expressions on an N-D grid.
    Notes:
    `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
    When the `indexing` argument is set to 'xy' (the default), the broadcasting
    instructions for the first two dimensions are swapped.
    Examples:
    Calling `X, Y = meshgrid(x, y)` with the tensors
    ```python
    x = [1, 2, 3]
    y = [4, 5, 6]
    X, Y = meshgrid(x, y)
    # X = [[1, 2, 3],
    #      [1, 2, 3],
    #      [1, 2, 3]]
    # Y = [[4, 4, 4],
    #      [5, 5, 5],
    #      [6, 6, 6]]
    ```
    Args:
    *args: `Tensor`s with rank 1.
    **kwargs:
      - indexing: Either 'xy' or 'ij' (optional, default: 'xy').
      - name: A name for the operation (optional).
    Returns:
    outputs: A list of N `Tensor`s with rank N.
    Raises:
    TypeError: When no keyword arguments (kwargs) are passed.
    ValueError: When indexing keyword argument is not one of `xy` or `ij`.
    """

    indexing = kwargs.pop("indexing", "xy")
    name = kwargs.pop("name", "meshgrid")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError("'{}' is an invalid keyword argument "
                    "for this function".format(key))

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")

    # with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]

    # output_dtype = tf.convert_to_tensor(args[0]).dtype.base_dtype
    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    # This is the part of the implementation from tf that is slow. 
    # We replace it below to get a ~6x speedup (essentially using tile instead of * tf.ones())
    # TODO(nolivia): improve performance with a broadcast  
    # mult_fact = tf.ones(shapes, output_dtype)
    # return [x * mult_fact for x in output]
    for i in range(len(output)):       
        stack_sz = [*sz[:i], 1, *sz[(i+1):]]
        if indexing == 'xy' and ndim > 1 and i < 2:
            stack_sz[0], stack_sz[1] = stack_sz[1], stack_sz[0]
        output[i] = tf.tile(output[i], tf.stack(stack_sz))
    return output


def flatten(v):
    """
    flatten Tensor v
    
    Parameters:
        v: Tensor to be flattened
    
    Returns:
        flat Tensor
    """

    return tf.reshape(v, [-1])
   
    
###############################################################################
# simple math functions, often used as activations 
###############################################################################

def softmax(x, axis=-1, alpha=1):
    """
    building on keras implementation, with additional alpha parameter

    Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
        alpha: a value to multiply all x
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    x = alpha * x
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


def logtanh(x, a=1):
    """
    log * tanh

    See Also: arcsinh
    """
    return K.tanh(x) *  K.log(2 + a * abs(x))


def arcsinh(x, alpha=1):
    """
    asignh

    See Also: logtanh
    """
    return tf.asinh(x * alpha) / alpha


def logistic(x, x0=0., alpha=1., L=1.):
    """
    returns L/(1+exp(-alpha * (x-x0)))
    """
    assert L > 0, 'L (height of logistic) should be > 0'
    assert alpha > 0, 'alpha (slope) of logistic should be > 0'
    
    return L / (1 + tf.exp(-alpha * (x-x0)))


def sigmoid(x):
    return logistic(x, x0=0., alpha=1., L=1.)


def logistic_fixed_ends(x, start=-1., end=1., L=1., **kwargs):
    """
    f is logistic with fixed ends, so that f(start) = 0, and f(end) = L.
    this is currently done a bit heuristically: it's a sigmoid, with a linear function added to correct the ends.
    """
    assert end > start, 'End of fixed points should be greater than start'
    # tf.assert_greater(end, start, message='assert')
    
    # clip to start and end
    x = tf.clip_by_value(x, start, end)
    
    # logistic function
    xv = logistic(x, L=L, **kwargs)
    
    # ends of linear corrective function
    sv = logistic(start, L=L, **kwargs)
    ev = logistic(end, L=L, **kwargs)
    
    # corrective function
    df = end - start
    linear_corr = (end-x)/df * (- sv) + (x-start)/df * (-ev + L)
    
    # return fixed logistic
    return xv + linear_corr


def sigmoid_fixed_ends(x, start=-1., end=1., L=1., **kwargs):
    return logistic_fixed_ends(x, start=-1., end=1., L=1., x0=0., alpha=1.)


def soft_round(x, alpha=25):
    fx = tf.floor(x)
    xd = x - fx
    return fx + logistic_fixed_ends(xd, start=0., end=1., x0=0.5, alpha=alpha)


def soft_delta(x, x0=0., alpha=100, reg='l1'):
    """
    recommended defaults:
    alpha = 100 for l1
    alpha = 1000 for l2
    """
    if reg == 'l1':
        xa = tf.abs(x - x0)
    else:
        assert reg == 'l2'
        xa = tf.square(x - x0)
    return (1 - logistic(xa, alpha=alpha)) * 2


def odd_shifted_relu(x, shift=-0.5, scale=2.0):
    """
    Odd shifted ReLu
    Essentially in x > 0, it is a shifted ReLu, and in x < 0 it's a negative mirror. 
    """

    shift = float(shift)
    scale = float(scale)
    return scale * K.relu(x - shift)  - scale * K.relu(- x - shift)


###############################################################################
# other
###############################################################################

def perlin_vol(vol_shape, min_scale=0, max_scale=None, interp_method='linear', wt_type='monotonic'):
    """
    generate perlin noise ND volume 

    rough algorithm:
    
    vol = zeros
    for scale in scales:
        rand = generate random uniform noise at given scale
        vol += wt * upsampled rand to vol_shape 
        

    Parameters
    ----------
    vol_shape: list indicating input shape.
    min_scale: higher min_scale = less high frequency noise
      the minimum rescale vol_shape/(2**min_scale), min_scale of 0 (default) 
      means start by not rescaling, and go down.
    max_scale: maximum scale, if None computes such that smallest volume shape is [1]
    interp_order: interpolation (upscale) order, as used in ne.utils.zoom
    wt_type: the weight type between volumes. default: monotonically decreasing with image size.
      options: 'monotonic', 'random'
    
    https://github.com/adalca/matlib/blob/master/matlib/visual/perlin.m
    loosely inspired from http://nullprogram.com/blog/2007/11/20
    """

    # input handling
    assert wt_type in ['monotonic', 'random'], \
        "wt_type should be in 'monotonic', 'random', got: %s"  % wt_type

    if max_scale is None:
        max_width = np.max(vol_shape)
        max_scale = np.ceil(np.log2(max_width)).astype('int')

    # decide on scales:
    scale_shapes = []
    wts = []
    for i in range(min_scale, max_scale + 1):
        scale_shapes.append(np.ceil([f / (2**i) for f in vol_shape]).astype('int'))
    
        # determine weight
        if wt_type == 'monotonic':
            wts.append(i + 1)  # larger images (so more high frequencies) get lower weight
        else:
            wts.append(K.random_uniform([1])[0])

    wts = K.stack(wts)/K.sum(wts)
    wts = tf.cast(wts, tf.float32)


    # get perlin volume
    vol = K.zeros(vol_shape)
    for sci, sc in enumerate(scale_shapes):

        # get a small random volume
        rand_vol = K.random_uniform(sc)
        
        # interpolated rand volume to upper side
        reshape_factor = [vol_shape[d]/sc[d] for d in range(len(vol_shape))]
        interp_vol = zoom(rand_vol, reshape_factor, interp_method=interp_method)[..., 0]

        # add to existing volume
        vol = vol + wts[sci] * interp_vol
        
    return vol


def gaussian_kernel(sigma, windowsize=None, indexing='ij'):
    """
    sigma will be a number of a list of numbers.

    # some guidance from my MATLAB file 
    https://github.com/adalca/mivt/blob/master/src/gaussFilt.m

    Parameters:
        sigma: scalar or list of scalars
        windowsize (optional): scalar or list of scalars indicating the shape of the kernel
    
    Returns:
        ND kernel the same dimensiosn as the number of sigmas.

    Todo: could use MultivariateNormalDiag
    """

    if not isinstance(sigma, (list, tuple)):
        sigma = [sigma]
    sigma = [np.maximum(f, np.finfo(float).eps) for f in sigma]

    nb_dims = len(sigma)

    # compute windowsize
    if windowsize is None:
        windowsize = [np.round(f * 3) * 2 + 1 for f in sigma]

    if len(sigma) != len(windowsize):
        raise ValueError('sigma and windowsize should have the same length.'
                         'Got vectors: ' + str(sigma) + 'and' + str(windowsize))

    # ok, let's get to work.
    mid = [(w - 1)/2 for w in windowsize]

    # list of volume ndgrid
    # N-long list, each entry of shape volshape
    mesh = volshape_to_meshgrid(windowsize, indexing=indexing)  
    mesh = [tf.cast(f, 'float32') for f in mesh]

    # compute independent gaussians
    diff = [mesh[f] - mid[f] for f in range(len(windowsize))]
    exp_term = [- K.square(diff[f])/(2 * (sigma[f]**2)) for f in range(nb_dims)]
    norms = [exp_term[f] - np.log(sigma[f] * np.sqrt(2 * np.pi)) for f in range(nb_dims)]

    # add an all-ones entry and transform into a large matrix
    norms_matrix = tf.stack(norms, axis=-1)  # *volshape x N
    g = K.sum(norms_matrix, -1)  # volshape
    g = tf.exp(g)
    g /= tf.reduce_sum(g)

    return g


def sub2ind2d(siz, subs, **kwargs):
    """
    assumes column-order major
    """
    # subs is a list
    assert len(siz) == len(subs), \
        'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])

    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx


def prod_n(lst):
    """
    Alternative to tf.stacking and prod, since tf.stacking can be slow
    """
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod


###############################################################################
# functions from external source
###############################################################################

def batch_gather(reference, indices):
    """
    C+P From Keras pull request https://github.com/keras-team/keras/pull/6377/files
    
    Batchwise gathering of row indices.

    The numpy equivalent is `reference[np.arange(batch_size), indices]`, where
    `batch_size` is the first dimension of the reference tensor.

    # Arguments
        reference: A tensor with ndim >= 2 of shape.
          (batch_size, dim1, dim2, ..., dimN)
        indices: A 1d integer tensor of shape (batch_size) satisfying
          0 <= i < dim2 for each element i.

    # Returns
        The selected tensor with shape (batch_size, dim2, ..., dimN).

    # Examples
        1. If reference is `[[3, 5, 7], [11, 13, 17]]` and indices is `[2, 1]`
        then the result is `[7, 13]`.

        2. If reference is
        ```
          [[[2, 3], [4, 5], [6, 7]],
           [[10, 11], [12, 13], [16, 17]]]
        ```
        and indices is `[2, 1]` then the result is `[[6, 7], [12, 13]]`.
    """
    batch_size = K.shape(reference)[0]
    indices = tf.stack([tf.range(batch_size), indices], axis=1)
    return tf.gather_nd(reference, indices)
