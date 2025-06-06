"""
utilities for the neuron project

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

# python imports
import itertools

# third party imports
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

# local imports
import pystrum.pynd.ndutils as nd
import neurite as ne


def setup_device(gpuid=None):
    """
    Configures the appropriate TF device from a cuda device string.
    Returns the device id and total number of devices.
    """

    if gpuid is not None and not isinstance(gpuid, str):
        gpuid = str(gpuid)

    if gpuid is not None:
        nb_devices = len(gpuid.split(','))
    else:
        nb_devices = 1

    if gpuid is not None and (gpuid != '-1'):
        device = '/gpu:' + gpuid
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

        # GPU memory configuration differs between TF 1 and 2
        if hasattr(tf, 'ConfigProto'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            tf.keras.backend.set_session(tf.Session(config=config))
        else:
            tf.config.set_soft_device_placement(True)
            for pd in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(pd, True)
    else:
        device = '/cpu:0'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    return device, nb_devices


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

    If you find this function useful, please cite the original paper this was written for:
        VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
        G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
        IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

        Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
        A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
        MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """

    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)
    nb_dims = loc.shape[-1]
    input_vol_shape = vol.shape

    if len(vol.shape) not in [nb_dims, nb_dims + 1]:
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = K.expand_dims(vol, -1)

    # flatten and float location Tensors
    if not loc.dtype.is_floating:
        target_loc_dtype = vol.dtype if vol.dtype.is_floating else 'float32'
        loc = tf.cast(loc, target_loc_dtype)
    elif vol.dtype.is_floating and vol.dtype != loc.dtype:
        loc = tf.cast(loc, vol.dtype)

    if isinstance(vol.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = vol.shape.as_list()
    else:
        volshape = vol.shape

    max_loc = [d - 1 for d in vol.get_shape().as_list()]

    # interpolate
    if interp_method == 'linear':
        # floor has to remain floating-point since we will use it in such operation
        loc0 = tf.floor(loc)

        # clip values
        clipped_loc = [tf.clip_by_value(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [tf.clip_by_value(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        # note reverse ordering since weights are inverse of diff.
        weights_loc = [diff_loc1, diff_loc0]

        # go through all the cube corners, indexed by a ND binary vector
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0

        for c in cube_pts:

            # get nd values
            # note re: indices above volumes via
            #   https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's
            #   too expensive. Instead we fill the output with zero for the corresponding value.
            #   The CPU version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because gather_nd needs tf.stack which is slow :(
            idx = sub2ind2d(vol.shape[:-1], subs)
            vol_reshape = tf.reshape(vol, [-1, volshape[-1]])
            vol_val = tf.gather(vol_reshape, idx)

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
        assert interp_method == 'nearest', \
            'method should be linear or nearest, got: %s' % interp_method
        roundloc = tf.cast(tf.round(loc), 'int32')
        roundloc = [tf.clip_by_value(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        # get values
        # tf stacking is slow. replace with gather
        # roundloc = tf.stack(roundloc, axis=-1)
        # interp_vol = tf.gather_nd(vol, roundloc)
        idx = sub2ind2d(vol.shape[:-1], roundloc)
        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx)

    if fill_value is not None:
        out_type = interp_vol.dtype
        fill_value = tf.constant(fill_value, dtype=out_type)
        below = [tf.less(loc[..., d], 0) for d in range(nb_dims)]
        above = [tf.greater(loc[..., d], max_loc[d]) for d in range(nb_dims)]
        out_of_bounds = tf.reduce_any(tf.stack(below + above, axis=-1), axis=-1, keepdims=True)
        interp_vol *= tf.cast(tf.logical_not(out_of_bounds), out_type)
        interp_vol += tf.cast(out_of_bounds, out_type) * fill_value

    # if only inputted volume without channels C, then return only that channel
    if len(input_vol_shape) == nb_dims:
        assert interp_vol.shape[-1] == 1, 'Something went wrong with interpn channels'
        interp_vol = interp_vol[..., 0]

    return interp_vol


def resize(vol, zoom_factor, interp_method='linear'):
    """
    if zoom_factor is a list, it will determine the ndims, in which case vol has to be of 
        length ndims of ndims + 1

    if zoom_factor is an integer, then vol must be of length ndims + 1

    If you find this function useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148

    """

    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]

        assert len(vol_shape) in (ndims, ndims + 1), \
            "zoom_factor length %d does not match ndims %d" % (len(vol_shape), ndims)

    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims

    # Avoid unnecessary work.
    if all(z == 1 for z in zoom_factor):
        return vol

    if not isinstance(vol_shape[0], int):
        vol_shape = vol_shape.as_list()

    new_shape = [vol_shape[f] * zoom_factor[f] for f in range(ndims)]
    new_shape = [int(f) for f in new_shape]

    lin = [tf.linspace(0., vol_shape[d] - 1., new_shape[d]) for d in range(ndims)]
    grid = ne.utils.ndgrid(*lin)

    return ne.utils.interpn(vol, grid, interp_method=interp_method)


zoom = resize


###############################################################################
# volumetric / axis operations
###############################################################################

def map_fn_axis(fn, elems, axis, **kwargs):
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
    else:
        if not isinstance(axis, (tuple, list)):
            axis = [axis] * len(elems)

    elems_perm = []
    for xi, x in enumerate(elems):
        a = axis[xi]
        s = len(x.get_shape().as_list())
        if a == -1:
            a = s - 1

        # move channels to front, so x will be [axis, ...]
        perm = [a] + list(range(0, a)) + list(range(a + 1, s))
        elems_perm.append(K.permute_dimensions(x, perm))

    # compute sptial deformation regularization for this channel
    if not islist:
        elems_perm = elems_perm[0]

    x_perm_trf = tf.map_fn(fn, elems_perm, **kwargs)
    output_is_list = isinstance(x_perm_trf, (tuple, list))
    if not output_is_list:
        x_perm_trf = [x_perm_trf]

    # move in_channels back to end
    elems_trf = []
    for xi, x in enumerate(x_perm_trf):
        a = axis[xi]
        s = len(x.get_shape().as_list())
        if a == -1:
            a = s - 1

        perm = list(range(1, a + 1)) + [0] + list(range(a + 1, s))
        elems_trf.append(K.permute_dimensions(x, perm))

    if not output_is_list:
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
    # name = kwargs.pop("name", "meshgrid")
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
        stack_sz = [*sz[:i], 1, *sz[(i + 1):]]
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


def take(x, indices, axis):
    """
    Take elements from an array along axis. Similar to np.take.
    This just wraps tf.gather, but gather can be overwhelming at times :)

    Args:
        x (Tensor): the ND volume to be indexed into
        indices (Tensor, int, or list): indexes along axis.
            If given int or Tensor of shape (), then returned volume will be one lower dim.
            If given list of Tensor of shape (?, ), then returned volume will be same dim, 
            even if list or Tensor have only one element.
        axis (int): the axis to index into

    Returns:
        Tensor: with only given indices along specified axis, ND or (N-1)D
    """
    return tf.gather(x, indices, axis=axis)


def barycenter(x, axes=None, normalize=False, shift_center=False, dtype=tf.float32):
    """
    Compute barycenter along specified axes.

    Arguments:
        x:
            Input tensor of any type. Will be cast to FP32 if needed.
        axes:
            Axes along which to compute the barycenter. None means all axes.
        normalize:
            Normalize grid dimensions to unit length.
        shift_center:
            Shift grid to image center.
        dtype:
            Output data type. The computation always uses single precision.

    Returns:
        Center of mass of the specified data type.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    dtype = tf.dtypes.as_dtype(dtype)
    compute_type = tf.float32
    if not tf.is_tensor(x) or x.dtype != compute_type:
        x = tf.cast(x, compute_type)

    # Move reduction axes to end.
    axes_all = range(len(x.shape))
    if axes is None:
        axes = axes_all
    axes_sub = tuple(ax for ax in axes_all if ax not in axes)
    if axes_sub:
        x = tf.transpose(x, perm=(*axes_sub, *axes))

    num_dim = len(axes)
    vol_shape = x.shape[-num_dim:]

    # Coordinate grid.
    grid = (np.arange(f, dtype=x.dtype.as_numpy_dtype) for f in vol_shape)
    if shift_center:
        grid = (g - (v - 1) / 2 for g, v in zip(grid, vol_shape))
    if normalize:
        grid = (g / v for g, v in zip(grid, vol_shape))
    grid = np.meshgrid(*grid, indexing='ij')
    grid = np.stack(grid, axis=-1)

    # Reduction.
    axes_red = axes_all[-num_dim:]
    x = tf.expand_dims(x, axis=-1)
    x = tf.math.divide_no_nan(
        tf.reduce_sum(grid * x, axis=axes_red),
        tf.reduce_sum(x, axis=axes_red),
    )

    return tf.cast(x, dtype) if dtype != compute_type else x


###############################################################################
# filtering
###############################################################################


def gaussian_kernel(sigma,
                    windowsize=None,
                    indexing='ij',
                    separate=False,
                    random=False,
                    min_sigma=0,
                    dtype=tf.float32,
                    seed=None):
    '''
    Construct an N-dimensional Gaussian kernel.

    Parameters:
        sigma: Standard deviations, scalar or list of N scalars.
        windowsize: Extent of the kernel in each dimension, scalar or list.
        indexing: Whether the grid is constructed with 'ij' or 'xy' indexing.
            Ignored if the kernel is separated.
        separate: Whether the kernel is returned as N separate 1D filters.
        random: Whether each standard deviation is uniformily sampled from the
            interval [min_sigma, sigma).
        min_sigma: Lower bound of the standard deviation, only considered for
            random sampling.
        dtype: Data type of the output. Should be floating-point.
        seed: Integer for reproducible randomization. It is possible that this parameter only
            has an effect if the function is wrapped in a Lambda layer.

    Returns:
        ND Gaussian kernel where N is the number of input sigmas. If separated,
        a list of 1D kernels will be returned.

    For more information see:
        https://github.com/adalca/mivt/blob/master/src/gaussFilt.m

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    '''
    # Data type.
    dtype = tf.dtypes.as_dtype(dtype)
    assert dtype.is_floating, f'{dtype.name} is not a real floating-point type'

    # Kernel width.
    if not isinstance(sigma, (list, tuple)):
        sigma = [sigma]
    if not isinstance(min_sigma, (list, tuple)):
        min_sigma = [min_sigma] * len(sigma)
    sigma = [max(f, np.finfo(dtype.as_numpy_dtype).eps) for f in sigma]
    min_sigma = [max(f, np.finfo(dtype.as_numpy_dtype).eps) for f in min_sigma]

    # Kernel size.
    if windowsize is None:
        windowsize = [np.round(f * 3) * 2 + 1 for f in sigma]
    if not isinstance(windowsize, (list, tuple)):
        windowsize = [windowsize]
    if len(sigma) != len(windowsize):
        raise ValueError(f'sigma {sigma} and width {windowsize} differ in length')

    # Precompute grid.
    center = [(w - 1) / 2 for w in windowsize]
    mesh = [np.arange(w) - c for w, c in zip(windowsize, center)]
    mesh = [-0.5 * x**2 for x in mesh]
    if not separate:
        mesh = np.meshgrid(*mesh, indexing=indexing)
    mesh = [tf.constant(m, dtype=dtype) for m in mesh]

    # Exponents.
    if random:
        seeds = np.random.default_rng(seed).integers(np.iinfo(int).max, size=len(sigma))
        max_sigma = sigma
        sigma = []
        for a, b, s in zip(min_sigma, max_sigma, seeds):
            sigma.append(tf.random.uniform(shape=(1,), minval=a, maxval=b, seed=s, dtype=dtype))
    exponent = [m / s**2 for m, s in zip(mesh, sigma)]

    # Kernel.
    if not separate:
        exponent = [tf.reduce_sum(tf.stack(exponent), axis=0)]
    kernel = [tf.exp(x) for x in exponent]
    kernel = [x / tf.reduce_sum(x) for x in kernel]

    return kernel if len(kernel) > 1 else kernel[0]


def separable_conv(x,
                   kernels,
                   axis=None,
                   batched=False,
                   padding='SAME',
                   strides=None,
                   dilations=None):
    """
    Efficiently apply 1D kernels along axes of a tensor with a trailing feature
    dimension. The same filters will be applied across features.

    Inputs:
        x: Input tensor with trailing feature dimension.
        kernels: A single kernel or a list of kernels, as tensors or NumPy arrays.
            If a single kernel is passed, it will be applied along all specified axes.
        axis: Spatial axes along which to apply the kernels, starting from zero.
            A value of None means all spatial axes.
        padding: Whether padding is to be used, either "VALID" or "SAME".
        strides: Optional output stride as a scalar, list or NumPy array. If several
            values are passed, these will be applied to the specified axes, in order.
        dilations: Optional filter dilation rate as a scalar, list or NumPy array. If several
            values are passed, these will be applied to the specified axes, in order.

    Returns:
        Tensor with the same type as the input.

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    # Shape.
    if not batched:
        x = tf.expand_dims(x, axis=0)
    shape_space = tf.shape(x)[1:-1]
    num_dim = len(x.shape[1:-1])

    # Axes.
    if np.isscalar(axis):
        axis = [axis]
    axes_space = range(num_dim)
    if axis is None:
        axis = axes_space
    assert all(ax in axes_space for ax in axis), 'non-spatial axis passed'

    # Conform strides and dilations.
    ones = np.ones(num_dim, np.int32)
    f = map(lambda x: 1 if x is None else x, (strides, dilations))
    f = map(np.ravel, f)
    f = map(np.ndarray.tolist, f)
    f = map(lambda x: x * len(axis) if len(x) == 1 else x, f)
    f = map(lambda x: [(*ones[:ax], x[i], *ones[ax + 1:]) for i, ax in enumerate(axis)], f)
    strides, dilations = f
    assert len(strides) == len(axis), 'number of strides and axes differ'
    assert len(dilations) == len(axis), 'number of dilations and axes differ'

    # Conform kernels.
    if not isinstance(kernels, (tuple, list)):
        kernels = [kernels]
    if len(kernels) == 1:
        kernels = kernels.copy() * len(axis)
    assert len(kernels) == len(axis), 'number of kernels and axes differ'

    # Merge features and batches.
    ind = np.arange(num_dim + 2)
    forward = (0, ind[-1], *ind[1:-1])
    backward = (0, *ind[2:], 1)
    x = tf.transpose(x, forward)
    shape_bc = tf.shape(x)[:2]
    x = tf.reshape(x, shape=tf.concat((
        tf.reduce_prod(shape_bc, keepdims=True),
        shape_space,
        [1],
    ), axis=0))

    # Convolve.
    for ax, k, s, d in zip(axis, kernels, strides, dilations):
        width = np.prod(k.shape)
        k = tf.reshape(k, shape=(*ones[:ax], width, *ones[ax + 1:], 1, 1))
        x = tf.nn.convolution(x, k, padding=padding, strides=s, dilations=d)

    # Restore dimensions.
    x = tf.reshape(x, shape=tf.concat((shape_bc, tf.shape(x)[1:-1]), axis=0))
    x = tf.transpose(x, backward)

    return x if batched else x[0, ...]


def subsample_axis(x,
                   stride_min=1,
                   stride_max=8,
                   axes=None,
                   prob=1,
                   upsample=True,
                   seed=None):
    """
    Symmetrically subsample a tensor by a factor f (stride) along a single axis
    using nearest-neighbor interpolation and optionally upsample again, to reduce
    its resolution. Both f and the subsampling axis can be randomly drawn.

    Parameters:
        x: Input tensor or NumPy array of any type.
        stride_min: Minimum subsampling factor.
        stride_max: Maximum subsampling factor.
        axes: Tensor axes to draw the subsampling axis from. None means all axes.
        prob: Subsampling probability. A value of 1 means always, 0 never.
        upsample: Upsample the tensor to restore its original shape.
        seed: Integer for reproducible randomization.

    Returns:
        Tensor with randomly thick slices along a random axis.

    See also:
        ne.layers.Subsample

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    import warnings
    warnings.warn('`utils.subsample_axis` and `layers.Subsample` are deprecated and will be '
                  'removed in the future. Please use vxm.augment.down_up_sample` and '
                  '`vxm.layers.DownUpSample` instead.')

    # Validate inputs.
    if not tf.is_tensor(x):
        x = tf.constant(x)
    axes = ne.py.utils.normalize_axes(axes, shape=x.shape, none_means_all=True)
    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(int).max)

    # Draw thickness and subsampling axis.
    assert 0 < stride_min and stride_min <= stride_max, 'invalid strides'
    thick = tf.random.uniform(shape=[], minval=stride_min, maxval=stride_max, seed=seed())
    ind = tf.random.uniform(shape=[], minval=0, maxval=len(axes), dtype=tf.int32, seed=seed())
    subsample = tf.gather(axes, ind)

    # Decide whether to downsample.
    assert 0 <= prob <= 1, f'{prob} not a probability'
    if prob < 1:
        rand_bit = tf.less(tf.random.uniform(shape=[], seed=seed()), prob)
        rand_not = tf.logical_not(rand_bit)
        thick = thick * tf.cast(rand_bit, thick.dtype) + tf.cast(rand_not, thick.dtype)

    # Resample.
    if not upsample:
        num_slice = tf.cast(width, thick.dtype) / thick
        num_slice = tf.cast(num_slice + 0.5, width.dtype)
        ind = tf.linspace(start=0, stop=width - 1, num=num_slice)
        ind = tf.cast(ind + 0.5, width.dtype)
        return tf.gather(x, ind, axis=subsample)

    # Resample with constant tensor sizes at compile time, for XLA. Make sure
    # the operation is symmetric on average.
    for ax in axes:
        width = tf.shape(x)[ax]
        ind = tf.range(width, dtype=tf.float32) // thick
        ind = tf.cast(ind * thick + 0.5, tf.int32)

        bit = tf.less(tf.random.uniform(shape=[], seed=seed()), 0.5)
        bit = tf.cast(bit, ind.dtype)
        ind = bit * ind + (1 - bit) * (width - 1 - ind)[::-1]

        is_axis = tf.cast(tf.equal(subsample, ax), x.dtype)
        x = is_axis * tf.gather(x, ind, axis=ax) + (1 - is_axis) * x

    return x


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
    return K.tanh(x) * K.log(2 + a * abs(x))


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

    return L / (1 + tf.exp(-alpha * (x - x0)))


def sigmoid(x):
    return logistic(x, x0=0., alpha=1., L=1.)


def logistic_fixed_ends(x, start=-1., end=1., L=1., **kwargs):
    """
    f is logistic with fixed ends, so that f(start) = 0, and f(end) = L.
    this is currently done a bit heuristically: it's a sigmoid, with a linear function added to 
    correct the ends.
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
    linear_corr = (end - x) / df * (- sv) + (x - start) / df * (-ev + L)

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
    return scale * K.relu(x - shift) - scale * K.relu(- x - shift)


def minmax_norm(x, axis=None):
    """
    Min-max normalize tensor using a safe division.

    Arguments:
        x: Tensor to be normalized.
        axis: Dimensions to reduce during normalization. If None, all axes will be considered,
            treating the input as a single image. To normalize batches or features independently,
            exclude the respective dimensions.

    Returns:
        Normalized tensor.
    """
    x_min = tf.reduce_min(x, axis=axis, keepdims=True)
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.compat.v1.div_no_nan(x - x_min, x_max - x_min)


def whiten(x, mean=0., std=1.):
    """
    whiteninng, with optional mean and std modification

    Args:
        x (Tensor): data to be whitenened
        mean (float, optional): output mean. Defaults to 0..
        std (float, optional): output standard deviation. Defaults to 1..

    Returns:
        Tensor: whitened Tensor
    """
    x = x - tf.reduce_mean(x)
    return x / tf.math.reduce_std(x) * std + mean


###############################################################################
# other
###############################################################################

def perlin_vol(vol_shape,
               min_scale=0,
               max_scale=None,
               interp_method='linear',
               wt_type='monotonic'):
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

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    # input handling
    assert wt_type in ['monotonic', 'random'], \
        "wt_type should be in 'monotonic', 'random', got: %s" % wt_type

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
            wts.append(K.random_uniform([1])[0])  # note this gets executed once I think

    wts = K.stack(wts) / K.sum(wts)
    wts = tf.cast(wts, tf.float32)

    # get perlin volume
    vol = 0
    for sci, sc in enumerate(scale_shapes):

        # get a small random volume
        rand_vol = K.random_uniform(sc)

        # interpolated rand volume to upper side
        reshape_factor = [vol_shape[d] / sc[d] for d in range(len(vol_shape))]
        interp_vol = zoom(rand_vol, reshape_factor, interp_method=interp_method)

        # add to existing volume
        vol = vol + wts[sci] * interp_vol

    return vol


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


def soft_digitize(*args, **kwargs):
    return soft_quantize(*args, **kwargs)


def soft_quantize(x,
                  bin_centers=None,
                  nb_bins=16,
                  alpha=1,
                  min_clip=-np.inf,
                  max_clip=np.inf,
                  return_log=False):
    """
    (Softly) quantize intensities (values) in a given volume, based on RBFs. 
    In numpy this (hard quantization) is called "digitize".

    Specify bin_centers OR number of bins 
        (which will estimate bin centers based on a heuristic using the min/max of the image)

    Algorithm: 
    - create (or obtain) a set of bins
    - for each array element, that value v gets assigned to all bins with 
        a weight of exp(-alpha * (v - c)), where c is the bin center
    - return volume x nb_bins

    Parameters:
        x [bs, ...]: intensity image. 
        bin_centers (np.float32 or list, optional): bin centers for soft histogram.
            Defaults to None.
        nb_bins (int, optional): number of bins, if bin_centers is not specified. 
            Defaults to 16.
        alpha (int, optional): alpha in RBF.
            Defaults to 1.
        min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
        max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
        return_log (bool, optional): [description]. Defaults to False.

    Returns:
        tf.float32: volume with one more dimension [bs, ..., B]

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    if bin_centers is not None:
        bin_centers = tf.convert_to_tensor(bin_centers, tf.float32)
        assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
        nb_bins = bin_centers.shape[0]
    else:
        if nb_bins is None:
            nb_bins = 16
        # get bin centers dynamically
        # TODO: perhaps consider an option to quantize by percentiles:
        #   minval = tfp.stats.percentile(x, 1)
        #   maxval = tfp.stats.percentile(x, 99)
        minval = K.min(x)
        maxval = K.max(x)
        bin_centers = tf.linspace(minval, maxval, nb_bins)

    # clipping at bin values
    x = x[..., tf.newaxis]                                                # [..., 1]
    x = tf.clip_by_value(x, min_clip, max_clip)

    # reshape bin centers to be (1, 1, .., B)
    new_shape = [1] * (len(x.shape) - 1) + [nb_bins]
    bin_centers = K.reshape(bin_centers, new_shape)                       # [1, 1, ..., B]

    # compute image terms
    # TODO: need to go to log space? not sure
    bin_diff = K.square(x - bin_centers)                                  # [..., B]
    log = -alpha * bin_diff                                               # [..., B]

    if return_log:
        return log                                                        # [..., B]
    else:
        return K.exp(log)                                                 # [..., B]


def batch_channel_flatten(x):
    """
    flatten volume elements outside of batch and channel

    using naming based on keras backend

    Args:
        x: a Tensor of N dims, size [batch_size, ..., channel_size]

    Returns:
        a Tensor of size [batch_size, V, channel_size] where V is the number of elements in the 
          middle N-2 dimensions
    """
    return flatten_axes(x, range(1, K.ndim(x) - 1))


# have both namings around
flatten_batch_channel = batch_channel_flatten


def flatten_axes(x, axes):
    """
    Flatten the axes[0], ..., axes[-1] of a Tensor x

    Example:
        v = tf.random.uniform((3, 4, 5, 6))
        ne.utils.flatten_axes(v, [1, 2]).shape # returns TensorShape([3, 20, 6])

    Args:
        x (Tensor): Tensor to flatten axes
        axes: list of axes to flatten

    Returns:
        Tensor: with flattened axes

    See Also:
        batch_channel_flatten
        tensorflow.keras.backend.batch_flatten
    """
    assert isinstance(axes, (list, tuple, range)), \
        'axes must be list or tuple of axes to be flattened'
    assert np.all(np.diff(axes) == 1), 'axes need to be contiguous'
    if axes[0] < 0:
        assert axes[-1] < 0, 'if one axis is negative, all have to be negative'
    assert axes[-1] < K.ndim(x), 'axis %d outside max axis %d' % (axes[-1], K.ndim(x) - 1)

    shp = K.shape(x)
    lst = [shp[:axes[0]], - tf.ones((1,), dtype=tf.int32)]
    if axes[-1] < len(x.shape) - 1 and not (axes[-1] == -1):
        lst.append(shp[axes[-1] + 1:])
    reshape = tf.concat(lst, 0)
    return K.reshape(x, reshape)


def fftn(x, axes=None, inverse=False):
    """
    Conveniently apply the fast Fourier transform (FFT) along any axes of a
    tensor. Supports forward and backward (inverse) transforms. Non-complex
    inputs will first be converted to tf.complex64.

    Arguments:
        x: Input tensor or NumPy array of any type.
        axes: Tensor axes along which to compute the FFT. None means all axes.
        inverse: Whether to compute the inverse FFT.

    Returns:
        Transformed tensor with complex floating-point type of the input.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    # Validate axes, make them unique and sort in descending order.
    ax_all = range(len(x.shape))
    axes = ne.py.utils.normalize_axes(axes, x.shape, none_means_all=True)[::-1]

    # Support NumPy arrays and real inputs.
    if not tf.is_tensor(x):
        x = tf.constant(x)
    if not x.dtype.is_complex:
        x = tf.cast(x, tf.complex64)

    # Compute the FFT.
    fft = getattr(tf.signal, 'ifft' if inverse else 'fft')
    for i in axes:
        x = tf.transpose(x, perm=(*ax_all[:i], *ax_all[i + 1:], i))
        x = fft(x)

    # Restore original axis order.
    ax_now = (*(i for i in ax_all if i not in axes), *axes)
    return tf.transpose(x, perm=np.argsort(ax_now))


def ifftn(x, axes=None):
    """
    Apply the inverse fast Fourier transform (FFT) along any tensor axes.
    """
    return fftn(x, axes, inverse=True)


# Provide interface to TensorFlow's fftshift functions.
fftshift = tf.signal.fftshift
ifftshift = tf.signal.ifftshift


def complex_to_channels(x):
    """
    Split a complex N-feature tensor into a real tensor with 2N features
    that are the real followed by the imaginary components of the input
    features. Considers the rightmost dimension as the feature dimension.

    Arguments:
        x: Complex tensor or NumPy array of any type.

    Returns:
        Real tensor with 2N channels.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    assert x.dtype.is_complex, 'non-complex input passed'
    return tf.concat((tf.math.real(x), tf.math.imag(x)), axis=-1)


def channels_to_complex(x):
    """
    Split a real tensor with an even number N of features into a complex
    N/2-feature tensor. The first N/2 features will be taken as the real, the
    last N/2 features as the imaginary components. Considers the rightmost
    dimension as the feature dimension.

    Arguments:
        x: Real input tensor or NumPy array of any type.

    Returns:
        Complex tensor with N/2 channels.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    axis = -1
    num_chan = x.shape[axis]
    assert x.dtype not in (tf.complex64, tf.complex128), 'complex input passed'

    # Type tf.float16 not supported by tf.complex.
    if x.dtype not in (tf.float32, tf.float64):
        x = tf.cast(x, tf.float32)

    real, imag = tf.split(x, num_or_size_splits=2, axis=axis)
    return tf.complex(real, imag)


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
