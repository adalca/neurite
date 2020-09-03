"""
utilities for model management in tf/keras
"""

# python imports
import itertools
from tempfile import NamedTemporaryFile

# third party imports
import numpy as np
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.utils


def stack_models(models, connecting_node_ids=None):
    """
    stacks keras models sequentially without nesting the models into layers
        (the nominal behaviour in keras as of 1/13/2018 is to nest models)
    This preserves the layers (i.e. does not copy layers). This means that if you modify the
    original layer weights, you are automatically affecting the new stacked model.

    Parameters:
        models: a list of models, in order of: [input_model, second_model, ..., final_output_model]
        connecting_node_ids (optional): a list of connecting node pointers from Nth model to N+1th model

    Returns:
        new stacked model pointer
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

    stacked_inputs_ = [i for i in stacked_inputs if i is not None]
    # check for unique, but keep order:
    stacked_inputs = []
    for inp in stacked_inputs_:
        if inp not in stacked_inputs:
            stacked_inputs.append(inp)
    new_model = keras.models.Model(stacked_inputs, output_tensors)
    return new_model


def mod_submodel(orig_model,
                 new_input_nodes=None,
                 input_layers=None):
    """
    modify (cut and/or stitch) keras submodel

    layer objects themselved will be untouched - the new model, even if it includes, 
    say, a subset of the previous layers, those layer objects will be shared with
    the original model

    given an original model:
        model stitching: given new input node(s), get output tensors of having pushed these 
        nodes through the model
        
        model cutting: given input layer (pointers) inside the model, the new input nodes
        will match the new input layers, hence allowing cutting the model

    Parameters:
        orig_model: original keras model pointer
        new_input_nodes: a pointer to a new input node replacement
        input_layers: the name of the layer in the original model to replace input nodes
    
    Returns:
        pointer to modified model
    """

    def _layer_dependency_dict(orig_model):
        """
        output: a dictionary of all layers in the orig_model
        for each layer:
            dct[layer] is a list of lists of layers.
        """

        if hasattr(orig_model, 'output_layers'):
            out_layers = orig_model.output_layers
            out_node_idx = orig_model.output_layers_node_indices
            node_list = [ol._inbound_nodes[out_node_idx[i]] for i, ol in enumerate(out_layers)]

        else:
            out_layers = orig_model._output_layers
            
            node_list = []
            for i, ol in enumerate(orig_model._output_layers):
                node_list += ol._inbound_nodes
            node_list  = list(set(node_list ))
            
        dct = {}
        dct_node_idx = {}
        while len(node_list) > 0:
            node = node_list.pop(0)
            node_input_layers = node.inbound_layers
            node_indices = node.node_indices
            if not isinstance(node_input_layers, (list, tuple)):
                node_input_layers = [node_input_layers]
                node_indices = [node_indices]
                
            add = True
            # if not empty. we need to check that we're not adding the same layers through the same node.
            if len(dct.setdefault(node.outbound_layer, [])) > 0:
                for li, layers in enumerate(dct[node.outbound_layer]):
                    if layers == node.inbound_layers and \
                        dct_node_idx[node.outbound_layer][li] == node_indices:
                        add = False
                        break
            if add:
                dct[node.outbound_layer].append(node_input_layers)
                dct_node_idx.setdefault(node.outbound_layer, []).append(node_indices)
            # append is in place

            # add new node
            
            for li, layer in enumerate(node_input_layers):
                if hasattr(layer, '_inbound_nodes'):
                    node_list.append(layer._inbound_nodes[node_indices[li]])
            
        return dct

    def _get_new_layer_output(layer, new_layer_outputs, inp_layers):
        """
        (recursive) given a layer, get new outbound_nodes based on new inbound_nodes

        new_layer_outputs is a (reference) dictionary that we will be adding
        to within the recursion stack.
        """

        if layer not in new_layer_outputs:

            if layer not in inp_layers:
                raise Exception('layer %s is not in inp_layers' % layer.name)

            # for all input layers to this layer, gather their output (our input)
            for group in inp_layers[layer]:
                input_nodes = [None] * len(group)
                for li, inp_layer in enumerate(group):
                    if inp_layer in new_layer_outputs:
                        input_nodes[li] = new_layer_outputs[inp_layer]
                    else: # recursive call
                        input_nodes[li] = _get_new_layer_output(inp_layer, new_layer_outputs, inp_layers)

                # layer call
                if len(input_nodes) == 1:
                    new_layer_outputs[layer] = layer(*input_nodes)
                else:
                    new_layer_outputs[layer] = layer(input_nodes)

        return new_layer_outputs[layer]



    # for each layer create list of input layers
    inp_layers = _layer_dependency_dict(orig_model)

    # get input layers
    #   These layers will be 'ignored' in that they will not be called!
    #   instead, the outbound nodes of the layers will be the input nodes
    #   computed below or passed in
    if input_layers is None: # if none provided, search for them
        # InputLayerClass = keras.engine.topology.InputLayer
        InputLayerClass = type(tf.keras.layers.InputLayer())
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
    assert len(input_nodes) == len(input_layers), \
        'input_nodes (%d) and input_layers (%d) have to match' % (len(input_nodes), len(input_layers))

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
        outputs[li] = _get_new_layer_output(output_layer, new_layer_outputs, inp_layers)

    return outputs


def reset_weights(model, session=None):
    """
    reset weights of model with the appropriate initializer.
    Note: only uses "kernel_initializer" and "bias_initializer"
    does not close session.

    Reference:
    https://www.codementor.io/nitinsurya/how-to-re-initialize-keras-model-weights-et41zre2g

    Parameters:
        model: keras model to reset
        session (optional): the current session
    """

    if session is None:
        session = K.get_session()

    for layer in model.layers: 
        reset = False
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
            reset = True
        
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)
            reset = True
        
        if not reset:
            print('Could not find initializer for layer %s. skipping', layer.name)


def copy_weights(src_model, dst_model):
    """
    copy weights from the src keras model to the dst keras model via layer names

    Parameters:
        src_model: source keras model to copy from
        dst_model: destination keras model to copy to
    """

    for layer in tqdm(dst_model.layers):
        try:
            wts = src_model.get_layer(layer.name).get_weights()
            layer.set_weights(wts)
        except:
            print('Could not copy weights of %s' % layer.name)
            continue


def robust_multi_gpu(model, gpus, verbose=True):
    """
    re-work keras model for multi-gpus if number of gpus is > 1

    Parameters:
        model: keras Model
        gpus: list of gpus to split to (e.g. [1, 4, 6]), or count of gpus available (e.g. 3)
            Note: if given int, assume that is the count of gpus, 
            so if you want a single specific gpu, this function will not do that.
        verbose: whether to display what happened (default: True)
    
    Returns:
        keras model
    """

    islist = isinstance(gpus, (list, tuple))
    if (islist and len(gpus) > 1) or (not islist and gpus > 1):
        count = gpus if not islist else len(gpus)
        print("Returning multi-gpu (%d) model" % count)
        return keras.utils.multi_gpu_model(model, count)

    else:
        print("Returning keras model back (single gpu found)")
        return model


def diagram(model):
    outfile = NamedTemporaryFile().name + '.png'
    tf.keras.utils.plot_model(model, to_file=outfile, show_shapes=True)

    from IPython.display import Image
    Image(outfile, width=100)
