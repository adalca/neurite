"""
segmentation tools for neurite
"""

# python imports
import itertools

# third party imports
import numpy as np
from tqdm import tqdm_notebook as tqdm
from pprint import pformat
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

# local imports
import neurite as ne
import neurite.py.utils
import pystrum.pynd.ndutils as nd
import pystrum.pytools.patchlib as pl
import pystrum.pytools.timer as timer


def predict_volumes(models,
                    data_generator,
                    batch_size,
                    patch_size,
                    patch_stride,
                    grid_size,
                    nan_func=np.nanmedian,
                    do_extra_vol=False,  # should compute vols beyond label
                    do_prob_of_true=False,  # should compute prob_of_true vols
                    verbose=False):
    """
    Note: we allow models to be a list or a single model.
    Normally, if you'd like to run a function over a list for some param,
    you can simply loop outside of the function. here, however, we are dealing with a generator,
    and want the output of that generator to be consistent for each model.

    Returns:
    if models isa list of more than one model:
        a tuple of model entried, each entry is a tuple of:
        true_label, pred_label, <vol>, <prior_label>, <pred_prob_of_true>, <prior_prob_of_true>
    if models is just one model:
        a tuple of
        (true_label, pred_label, <vol>, <prior_label>, <pred_prob_of_true>, <prior_prob_of_true>)

    TODO: could add prior
    """

    if not isinstance(models, (list, tuple)):
        models = (models,)

    # get the input and prediction stack
    with timer.Timer('predict_volume_stack', verbose):
        vol_stack = predict_volume_stack(models,
                                         data_generator,
                                         batch_size,
                                         grid_size,
                                         verbose)
    if len(models) == 1:
        do_prior = len(vol_stack) == 4
    else:
        do_prior = len(vol_stack[0]) == 4

    # go through models and volumes
    ret = ()
    for midx, _ in enumerate(models):

        stack = vol_stack if len(models) == 1 else vol_stack[midx]

        if do_prior:
            all_true, all_pred, all_vol, all_prior = stack
        else:
            all_true, all_pred, all_vol = stack

        # get max labels
        all_true_label, all_pred_label = pred_to_label(all_true, all_pred)

        # quilt volumes and aggregate overlapping patches, if any
        args = [patch_size, grid_size, patch_stride]
        label_kwargs = {'nan_func_layers':nan_func, 'nan_func_K':nan_func, 'verbose':verbose}
        vol_true_label = _quilt(all_true_label, *args, **label_kwargs).astype('int')
        vol_pred_label = _quilt(all_pred_label, *args, **label_kwargs).astype('int')

        ret_set = (vol_true_label, vol_pred_label)

        if do_extra_vol:
            vol_input = _quilt(all_vol, *args)
            ret_set += (vol_input, )

            if do_prior:
                all_prior_label, = pred_to_label(all_prior)
                vol_prior_label = _quilt(all_prior_label, *args, **label_kwargs).astype('int')
                ret_set += (vol_prior_label, )

        # compute the probability of prediction and prior
        # instead of quilting the probabilistic volumes and then computing the probability
        # of true label, which takes a long time, we'll first compute the probability of label,
        # and then quilt. This is faster, but we'll need to take median votes
        if do_extra_vol and do_prob_of_true:
            all_pp = prob_of_label(all_pred, all_true_label)
            pred_prob_of_true = _quilt(all_pp, *args, **label_kwargs)
            ret_set += (pred_prob_of_true, )

            if do_prior:
                all_pp = prob_of_label(all_prior, all_true_label)
                prior_prob_of_true = _quilt(all_pp, *args, **label_kwargs)

                ret_set += (prior_prob_of_true, )

        ret += (ret_set, )

    if len(models) == 1:
        ret = ret[0]

    # return
    return ret


def predict_volume_stack(models,
                         data_generator,
                         batch_size,
                         grid_size,
                         verbose=False):
    """
    predict all the patches in a volume

    requires batch_size to be a divisor of the number of patches (prod(grid_size))

    Note: we allow models to be a list or a single model.
    Normally, if you'd like to run a function over a list for some param,
    you can simply loop outside of the function. here, however, we are dealing with a generator,
    and want the output of that generator to be consistent for each model.

    Returns:
    if models isa list of more than one model:
        a tuple of model entried, each entry is a tuple of:
        all_true, all_pred, all_vol, <all_prior>
    if models is just one model:
        a tuple of
        all_true, all_pred, all_vol, <all_prior>
    """

    if not isinstance(models, (list, tuple)):
        models = (models,)

    # compute the number of batches we need for one volume
    # we need the batch_size to be a divisor of nb_patches,
    # in order to loop through batches and form full volumes
    nb_patches = np.prod(grid_size)
    # assert np.mod(nb_patches, batch_size) == 0, \
        # "batch_size %d should be a divisor of nb_patches %d" %(batch_size, nb_patches)
    nb_batches = ((nb_patches - 1) // batch_size) + 1

    # go through the patches
    batch_gen = tqdm(range(nb_batches)) if verbose else range(nb_batches)
    for batch_idx in batch_gen:
        sample = next(data_generator)
        nb_vox = np.prod(sample[1].shape[1:-1])
        do_prior = isinstance(sample[0], (list, tuple))

        # pre-allocate all the data
        if batch_idx == 0:
            nb_labels = sample[1].shape[-1]
            all_vol = [np.zeros((nb_patches, nb_vox)) for f in models]
            all_true = [np.zeros((nb_patches, nb_vox * nb_labels)) for f in models]
            all_pred = [np.zeros((nb_patches, nb_vox * nb_labels)) for f in models]
            all_prior = [np.zeros((nb_patches, nb_vox * nb_labels)) for f in models]

        # get in_vol, y_true, y_pred
        for idx, model in enumerate(models):
            # with timer.Timer('prediction', verbose):
            pred = model.predict(sample[0])
            assert pred.shape[0] == batch_size, \
                "batch size mismatch. sample has batch size %d, given batch size is %d" %(pred.shape[0], batch_size)
            input_batch = sample[0] if not do_prior else sample[0][0]

            # compute batch range
            batch_start = batch_idx * batch_size
            batch_end = np.minimum(batch_start + batch_size, nb_patches)
            batch_range = np.arange(batch_start, batch_end)
            batch_vox_idx = batch_end-batch_start

            # update stacks
            all_vol[idx][batch_range, :] = K.batch_flatten(input_batch)[0:batch_vox_idx, :]
            all_true[idx][batch_range, :] = K.batch_flatten(sample[1])[0:batch_vox_idx, :]
            all_pred[idx][batch_range, :] = K._batch_flatten(pred)[0:batch_vox_idx, :]
            if do_prior:
                all_prior[idx][batch_range, :] = K.batch_flatten(sample[0][1])[0:batch_vox_idx, :]

    # reshape probabilistic answers
    for idx, _ in enumerate(models):
        all_true[idx] = np.reshape(all_true[idx], [nb_patches, nb_vox, nb_labels])
        all_pred[idx] = np.reshape(all_pred[idx], [nb_patches, nb_vox, nb_labels])
        if do_prior:
            all_prior[idx] = np.reshape(all_prior[idx], [nb_patches, nb_vox, nb_labels])

    # prepare output tuple
    ret = ()
    for midx, _ in enumerate(models):
        if do_prior:
            ret += ((all_true[midx], all_pred[midx], all_vol[midx], all_prior[midx]), )
        else:
            ret += ((all_true[midx], all_pred[midx], all_vol[midx]), )

    if len(models) == 1:
        ret = ret[0]
    return ret


def prob_of_label(vol, labelvol):
    """
    compute the probability of the labels in labelvol in each of the volumes in vols

    Parameters:
        vol (float numpy array of dim (nd + 1): volume with a prob dist at each voxel in a nd vols
        labelvol (int numpy array of dim nd): nd volume of labels

    Returns:
        nd volume of probabilities
    """

    # check dimensions
    nb_dims = np.ndim(labelvol)
    assert np.ndim(vol) == nb_dims + 1, "vol dimensions do not match [%d] vs [%d]" % (np.ndim(vol)-1, nb_dims)
    shp = vol.shape
    nb_voxels = np.prod(shp[0:nb_dims])
    nb_labels = shp[-1]

    # reshape volume to be [nb_voxels, nb_labels]
    flat_vol = np.reshape(vol, (nb_voxels, nb_labels))

    # normalize accross second dimension
    rows_sums = flat_vol.sum(axis=1)
    flat_vol_norm = flat_vol / rows_sums[:, np.newaxis]

    # index into the flattened volume
    idx = list(range(nb_voxels))
    v = flat_vol_norm[idx, labelvol.flat]
    return np.reshape(v, labelvol.shape)


def next_pred_label(model, data_generator, verbose=False):
    """
    predict the next sample batch from the generator, and compute max labels
    return sample, prediction, max_labels
    """
    sample = next(data_generator)
    with timer.Timer('prediction', verbose):
        pred = model.predict(sample[0])
    sample_input = sample[0] if not isinstance(sample[0], (list, tuple)) else sample[0][0]
    max_labels = pred_to_label(sample_input, pred)
    return (sample, pred) + max_labels


def next_label(model, data_generator):
    """
    predict the next sample batch from the generator, and compute max labels
    return max_labels
    """
    batch_proc = next_pred_label(model, data_generator)
    return (batch_proc[2], batch_proc[3])


def sample_to_label(model, sample):
    """
    redict a sample batch and compute max labels
    return max_labels
    """
    # predict output for a new sample
    res = model.predict(sample[0])
    # return
    return pred_to_label(sample[1], res)


def pred_to_label(*y):
    """
    return the true and predicted labels given true and predicted nD+1 volumes
    """
    # compute resulting volume(s)
    return tuple(np.argmax(f, -1).astype(int) for f in y)


def next_vol_pred(model, data_generator, verbose=False):
    """
    get the next batch, predict model output

    returns (input_vol, y_true, y_pred, <prior>)
    """

    # batch to input, output and prediction
    sample = next(data_generator)
    with timer.Timer('prediction', verbose):
        pred = model.predict(sample[0])
    data = (sample[0], sample[1], pred)
    if isinstance(sample[0], (list, tuple)):  # if given prior, might be a list
        data = (sample[0][0], sample[1], pred, sample[0][1])

    return data


###############################################################################
# helper functions
###############################################################################

def _quilt(patches, patch_size, grid_size, patch_stride, verbose=False, **kwargs):
    assert len(patches.shape) >= 2, "patches has bad shape %s" % pformat(patches.shape)

    # reshape to be [nb_patches x nb_vox]
    patches = np.reshape(patches, (patches.shape[0], -1, 1))

    # quilt
    quilted_vol = pl.quilt(patches, patch_size, grid_size, patch_stride=patch_stride, **kwargs)
    assert quilted_vol.ndim == len(patch_size), "problem with dimensions after quilt"

    # return
    return quilted_vol
