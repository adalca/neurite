"""
Keras CNN models

Tested on keras 2.0
"""

# third party
import numpy as np
import keras.layers as KL
from keras.models import Model, Sequential
import keras.backend as K


def design_unet(nb_features, patch_size, nb_levels, conv_size, nb_labels,
                feat_mult=1, pool_size=(2, 2, 2),
                padding='same', activation='relu',
                nb_conv_per_level=2, add_prior_layer=False):
    """
    unet-style model

    for U-net like architecture, we need to use Deconvolution3D.
    However, this is not yet available (maybe soon, it's on a dev branch in github I believe)
    Until then, we'll upsample and convolve.
    TODO: Need to check that UpSampling3D actually does NN-upsampling!
    """

    # kwargs for the convolution layer
    conv_kwargs = {'padding': padding, 'activation': activation}

    # initialize a dictionary
    layers_dict = {}

    # first layer: input
    name = 'input'
    layers_dict[name] = KL.Input(shape=patch_size + (1,), name=name)
    last_layer = layers_dict[name]

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    for level in range(nb_levels):
        for conv in range(nb_conv_per_level):
            name = 'conv_downarm_%d_%d' % (level, conv)
            nb_local_features = nb_features*(feat_mult**level)
            layers_dict[name] = KL.Conv3D(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
            last_layer = layers_dict[name]

        # max pool if we're not at the last level
        if level < (nb_levels - 1):
            name = 'maxpool_%d' % level
            layers_dict[name] = KL.MaxPooling3D(pool_size=pool_size)(last_layer)
            last_layer = layers_dict[name]

    # up arm:
    # nb_levels - 1 layers of Deconvolution3D
    #    (approx via up + conv + ReLu) + merge + conv + ReLu + conv + ReLu
    for level in range(nb_levels - 1):
        # upsample matching the max pooling layers
        name = 'up_%d' % (nb_levels + level)
        layers_dict[name] = KL.UpSampling3D(size=pool_size, name=name)(last_layer)
        last_layer = layers_dict[name]

        # upsample matching the max pooling layers
        name = 'upconv_%d' % (nb_levels + level)
        nb_local_features = nb_features*(feat_mult**(nb_levels-2-level))
        layers_dict[name] = KL.Convolution3D(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
        last_layer = layers_dict[name]

        # merge layers combining previous layer
        conv_name = 'conv_downarm_%d_%d' % (nb_levels - 2 - level, nb_conv_per_level - 1)
        name = 'merge_%d' % (nb_levels + level)
        layers_dict[name] = KL.concatenate([layers_dict[conv_name], last_layer], axis=4, name=name)
        last_layer = layers_dict[name]

        for conv in range(nb_conv_per_level):
            name = 'conv_uparm_%d_%d' % (nb_levels + level, conv)
            layers_dict[name] = KL.Conv3D(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
            last_layer = layers_dict[name]

    # reshape last layer for prediction
    name = 'conv_uparm_%d_%d_reshape' % (2 * nb_levels - 2, nb_conv_per_level - 1)
    vol_numel = np.prod(patch_size)
    layers_dict[name] = KL.Reshape((vol_numel, nb_features), name=name)(last_layer)
    last_layer = layers_dict[name]

    if add_prior_layer:
        # likelihood layer
        name = 'likelihood'
        layers_dict[name] = KL.Conv1D(nb_labels, 1, activation='softmax', name=name)(last_layer)
        last_layer = layers_dict[name]

        # prior input layer
        name = 'prior-input'
        layers_dict[name] = KL.Input(shape=patch_size + (nb_labels,), name=name)
        name = 'prior-input-reshape'
        layers_dict[name] = KL.Reshape((vol_numel, nb_labels), name=name)(layers_dict['prior-input'])

        # final prediction
        name = 'prediction'
        layers_dict[name] = KL.multiply([layers_dict['prior-input-reshape'], layers_dict['likelihood']])

        model_inputs = [layers_dict['input'], layers_dict['prior-input']]

    else:
        # output (liklihood) prediction layer
        name = 'prediction'
        layers_dict[name] = KL.Conv1D(nb_labels, 1, activation='softmax', name=name)(last_layer)

        model_inputs = [layers_dict['input']]

    # create the model
    model = Model(inputs=model_inputs, outputs=[layers_dict['prediction']])

    # compile
    return model


def design_dnn(nb_features, patch_size, nb_levels, conv_size, nb_labels,
               feat_mult=1, pool_size=(2, 2, 2),
               padding='same', activation='relu',
               final_layer='dense',
               nb_conv_per_level=2):
    """
    "deep" cnn with dense or global max pooling layer @ end...

    Could use sequential...
    """

    # kwargs for the convolution layer
    conv_kwargs = {'padding': padding, 'activation': activation}

    # initialize a dictionary
    layers_dict = {}

    # first layer: input
    name = 'input'
    layers_dict[name] = KL.Input(shape=patch_size + (1,), name=name)
    last_layer = layers_dict[name]

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    for level in range(nb_levels):
        for conv in range(nb_conv_per_level):
            name = 'conv_%d_%d' % (level, conv)
            nb_local_features = nb_features*(feat_mult**level)
            layers_dict[name] = KL.Conv3D(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
            last_layer = layers_dict[name]

        # max pool
        name = 'maxpool_%d' % level
        layers_dict[name] = KL.MaxPooling3D(pool_size=pool_size)(last_layer)
        last_layer = layers_dict[name]

    # dense layer
    if final_layer == 'dense':
        name = 'dense'
        print("pre-dense size", last_layer)
        layers_dict[name] = KL.Dense(nb_labels, name=name)(last_layer)
        print("dense size", layers_dict[name].get_shape())

    # global max pooling layer
    elif final_layer == 'globalmaxpooling':
        # name = 'squeeze'
        # target_shape = (np.prod(last_layer.get_shape()[1:4]), nb_features)
        # layers_dict[name] = KL.Reshape(target_shape, name=name)(last_layer)
        # last_layer = layers_dict[name]

        print("pre-gmp size", last_layer)
        name = 'global_max_pool'
        # layers_dict[name] = KL.GlobalMaxPooling1D(name=name, activation="sigmoid")(last_layer)
        layers_dict[name] = KL.Lambda(_global_max_nd, name=name, activation="sigmoid")(last_layer)
        print("gmp size", layers_dict[name].get_shape())

    last_layer = layers_dict[name]

    # create the model
    model = Model(inputs=[layers_dict['input']], outputs=[last_layer])
    return model

def _global_max_nd(x):
    y = K.batch_flatten(x)
    return K.max(y, 1, keepdims=True)