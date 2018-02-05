""" various utilities for the neuron project """


# third party imports
import numpy as np
from tqdm import tqdm_notebook as tqdm
from pprint import pformat

import pytools.patchlib as pl
import pytools.timer as timer

# local imports
import pynd.ndutils as nd


# often changed file
from imp import reload
import keras
import keras.backend as K
import tensorflow as tf
reload(pl)


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
            all_vol[idx][batch_range, :] = _batch_flatten(input_batch)[0:batch_vox_idx, :]
            all_true[idx][batch_range, :] = _batch_flatten(sample[1])[0:batch_vox_idx, :]
            all_pred[idx][batch_range, :] = _batch_flatten(pred)[0:batch_vox_idx, :]
            if do_prior:
                all_prior[idx][batch_range, :] = _batch_flatten(sample[0][1])[0:batch_vox_idx, :]

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
    ndims = np.ndim(labelvol)
    assert np.ndim(vol) == ndims + 1, "vol dimensions do not match [%d] vs [%d]" % (np.ndim(vol)-1, ndims)
    shp = vol.shape
    nb_voxels = np.prod(shp[0:ndims])
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

def softmax(x, axis):
    """
    softmax of a numpy array along a given dimension
    """

    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)



def copy_model_weights(src_model, dst_model):
    """ copy weights from the src model to the dst model """

    for idx in range(len(dst_model.layers)):
        layer = dst_model.layers[idx]
        wts = src_model.layers[idx].get_weights()
        layer.set_weights(wts)

    # for layer in dst_model.layers:
    #     layer.set_weights(src_model.get_layer(layer.name).get_weights())

    return dst_model

    # seg_model_load = keras.models.load_model('/data/vision/polina/users/adalca/fsCNN/output/unet-prior-v3/hdf5/run_5/model.88-0.00.hdf5', custom_objects={'loss': wcce46})
    # wts46 = seg_model_load.get_layer("likelihood").get_weights()
    # print(wts46[0].shape, wts46[1].shape)

    # for layer in seg_model.layers:
    #     if layer.name == "likelihood":
    #         nwts0, nwts1 = seg_model.get_layer("likelihood").get_weights()
    #         nwts0[:,:,0:19] = wts46[0][:,:,0:19]
    #         nwts0[:,:,20:] = wts46[0][:,:,19:]
    #         nwts1[0:19] = wts46[1][0:19]
    #         nwts1[20:] = wts46[1][19:]
    #         seg_model.get_layer("likelihood").set_weights((nwts0,nwts1))
    #     else:
    #         layer.set_weights(seg_model_load.get_layer(layer.name).get_weights())



def stack_models(models, connecting_node_ids=None):
    """
    stacks models sequentially without nesting the models into layers
        (the nominal behaviour in keras as of 1/13/2018 is to nest models)
    This preserves the layers (i.e. does not copy layers). This means that if you modify the
    original layer weights, you are automatically affecting the new stacked model.

    models is a list of models, in order of: [input_model, second_model, ..., final_output_model]

    1/13/2018:
    currently using extract_submodel subfunction which is a bit finicky.    
    """

    output_tensors = models[0].outputs
    stacked_inputs = [*models[0].inputs]

    # go through models 1 onwards and stack with current graph
    for mi in range(1, len(models)):
        
        # prepare input nodes - a combination of 
        new_input_nodes = list(models[mi].inputs)
        stacked_inputs_contrib = list(models[mi].inputs)

        if connecting_node_ids is None: 
            conn_id = list(range(len(new_input_nodes)))
            assert len(new_input_nodes) == len(models[mi-1].outputs), \
                'argument count does not match'
        else:
            conn_id = connecting_node_ids[mi-1]

        for out_idx, ii in enumerate(conn_id):
            new_input_nodes[ii] = output_tensors[out_idx]
            stacked_inputs_contrib[ii] = None
        
        output_tensors = mod_submodel(models[mi], new_input_nodes=new_input_nodes)
        stacked_inputs = stacked_inputs + stacked_inputs_contrib

    stacked_inputs = [i for i in stacked_inputs if i is not None]
    new_model = keras.models.Model(stacked_inputs, output_tensors)
    return new_model



def mod_submodel(orig_model,
                 new_input_nodes=None,
                 input_layers=None):
    """
    cut and/or stitch submodel

    given an original model:
        model stitching: given new input node(s), get output tensors of having pushed these 
        nodes through the model
        
        model cutting: given input layer (pointers) inside the model, the new input nodes will match the new input
        layers, hence allowing cutting the model
    """
    
    def _get_new_layer_output(layer, layers, new_layer_outputs, inp_layers):
        """
        (recursive) given a layer, get new output based on new inputs 

        new_layer_outputs is a (reference) dictionary that we will be adding
        to within the recursion stack.
        """

        if layer not in new_layer_outputs:

            # for all input layers to this layer, gather their output (our input)
            input_nodes = [None] * len(inp_layers[layer])
            for li, inp_layer in enumerate(inp_layers[layer]):
                if inp_layer in new_layer_outputs:
                    input_nodes[li] = new_layer_outputs[inp_layer]
                else: # recursive call
                    input_nodes[li] = _get_new_layer_output(inp_layer, layers, new_layer_outputs, inp_layers)
            
            # layer call
            if len(input_nodes) == 1:
                new_layer_outputs[layer] = layer(*input_nodes)
            else:
                new_layer_outputs[layer] = layer(input_nodes)

        return new_layer_outputs[layer]

    # for each layer create list of input layers
    inp_layers = {}
    for layer in orig_model.layers:
        if hasattr(layer, '_inbound_nodes') and len(layer._inbound_nodes) > 0:
            # Get the first input node, and if it's in the dictionary of output_node:[layers],
            # that means that this layer's can be connected to another layer through this node
            # We only use the first inbound node, it is sufficient for layer connectivity
            layer_inp_layers = []
            for input_node in layer._inbound_nodes:
                if len(input_node.inbound_layers) > 0:
                    layer_inp_layers += input_node.inbound_layers
            if len(layer_inp_layers) > 0:
                inp_layers[layer] = list(set(layer_inp_layers))            

    # get input layers
    #   These layers will be 'ignored' in that they will not be called!
    #   instead, the outbound nodes of the layers will be the input nodes
    #   computed below or passed in
    if input_layers is None: # if none provided, search for them
        InputLayerClass = keras.engine.topology.InputLayer
        input_layers = [l for l in orig_model.layers if isinstance(l, InputLayerClass)]

    else:
        if not isinstance(input_layers, (tuple, list)):
            input_layers = [input_layers]
        for idx, input_layer in enumerate(input_layers):
            # if it's a string, assume it's layer name, and get the layer pointer
            if isinstance(input_layer, str):
                input_layers[idx] = orig_model.get_layer(input_layer)

    # process new input nodes
    if new_input_nodes is None:
        input_nodes = list(orig_model.inputs)
    else:
        input_nodes = new_input_nodes
    assert len(input_nodes) == len(input_layers)

    # initialize dictionary of layer:new_output_node
    #   note: the input layers are not called, instead their outbound nodes
    #   are assumed to be the given input nodes. If we call the nodes, we can run
    #   into multiple-inbound-nodes problems, or if we completely skip the layers altogether
    #   we have problems with multiple inbound input layers into subsequent layers
    new_layer_outputs = {}
    for i, input_layer in enumerate(input_layers):
        new_layer_outputs[input_layer] = input_nodes[i]

    # recursively go back from output layers and request new input nodes
    output_layers = []
    for layer in orig_model.layers:
        if hasattr(layer, '_inbound_nodes'):
            for i in range(len(layer._inbound_nodes)):
                if layer.get_output_at(i) in orig_model.outputs:
                    output_layers.append(layer)
                    break
    assert len(output_layers) == len(orig_model.outputs), "Number of output layers don't match"

    outputs = [None] * len(output_layers)
    for li, output_layer in enumerate(output_layers):
        outputs[li] = _get_new_layer_output(layer, orig_model.layers, new_layer_outputs, inp_layers)

    return outputs


###############################################################################
# simple functions
###############################################################################

def crop3d(kvec, start, end):
    """ crop a 3D volume of shape (None, dim_1, dim_2, dim_3) """
    ndims = len(kvec.get_shape())
    if ndims == 5:
        return kvec[:, start[0]:end[0], start[1]:end[1], start[2]:end[1], :]
    if ndims == 4:
        return kvec[:, start[0]:end[0], start[1]:end[1], start[2]:end[1]]

def mid_cc_3d(x, y, start, end):
    xnew = crop3d(x, start, end)
    ynew = crop3d(y, start, end)
    return keras.losses.categorical_crossentropy(xnew, ynew)

def mid_mse_3d(x, y, start, end):
    xnew = crop3d(x, start, end)
    ynew = crop3d(y, start, end)
    return keras.losses.mean_squared_error(xnew, ynew)

# AE lambda layers
def longtanh(x, a=1):
    return K.tanh(x) *  K.log(2 + a * abs(x))

def arcsinh(x):
    return tf.asinh(x)

###############################################################################
# helper functions
###############################################################################

def _concat(lists, dim):
    if lists[0].size == 0:
        lists = lists[1:]

    return np.concatenate(lists, dim)

def _quilt(patches, patch_size, grid_size, patch_stride, verbose=False, **kwargs):
    assert len(patches.shape) >= 2, "patches has bad shape %s" % pformat(patches.shape)

    # reshape to be [nb_patches x nb_vox]
    patches = np.reshape(patches, (patches.shape[0], -1, 1))

    # quilt
    quilted_vol = pl.quilt(patches, patch_size, grid_size, patch_stride=patch_stride, **kwargs)
    assert quilted_vol.ndim == len(patch_size), "problem with dimensions after quilt"

    # return
    return quilted_vol

def _batch_flatten(x):
    return np.reshape(x, (x.shape[0], -1))
