"""
Keras CNN models

Tested on keras 2.0
"""

# third party
import numpy as np
import keras.layers as KL
from keras.models import Model, Sequential
import keras.backend as K
from keras.constraints import maxnorm



def design_unet(nb_features, patch_size, nb_levels, conv_size, nb_labels,
                feat_mult=1, pool_size=(2, 2, 2),
                use_logp=False, nb_input_features=1,
                padding='same', activation='relu', use_residuals=False,
                nb_conv_per_level=2, add_prior_layer=False):
    """
    unet-style model

    for U-net like architecture, we need to use Deconvolution3D.
    However, this is not yet available (maybe soon, it's on a dev branch in github I believe)
    Until then, we'll upsample and convolve.
    TODO: Need to check that UpSampling3D actually does NN-upsampling!
    """

    ndims = len(patch_size)
    convL = KL.Conv3D if ndims == 3 else KL.Conv2D
    maxpool = KL.MaxPooling3D if ndims == 3 else KL.MaxPooling2D
    upsample = KL.UpSampling3D if ndims == 3 else KL.UpSampling2D
    vol_numel = np.prod(patch_size)

    
    # kwargs for the convolution layer
    conv_kwargs = {'padding': padding, 'activation': activation}

    # initialize a dictionary
    layers_dict = {}

    # first layer: input
    name = 'input'
    layers_dict[name] = KL.Input(shape=patch_size + (nb_input_features,), name=name)
    last_layer = layers_dict[name]

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    for level in range(nb_levels):
        level_init_layer = last_layer
        nb_local_features = nb_features*(feat_mult**level)

        for conv in range(nb_conv_per_level):
            name = 'conv_downarm_%d_%d' % (level, conv)
            layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
            last_layer = layers_dict[name]

        if use_residuals:
            convarm_layer = last_layer
            name = 'expand_down_merge_%d' % (level)
            layers_dict[name] = convL(nb_local_features, [1, 1, 1], **conv_kwargs, name=name)(level_init_layer)
            last_layer = layers_dict[name]
            name = 'res_down_merge_%d' % (level)
            layers_dict[name] = KL.add([last_layer, convarm_layer], name=name)
            last_layer = layers_dict[name]

        # max pool if we're not at the last level
        if level < (nb_levels - 1):
            name = 'maxpool_%d' % level
            layers_dict[name] = maxpool(pool_size=pool_size)(last_layer)
            last_layer = layers_dict[name]

    # up arm:
    # nb_levels - 1 layers of Deconvolution3D
    #    (approx via up + conv + ReLu) + merge + conv + ReLu + conv + ReLu
    for level in range(nb_levels - 1):

        # upsample matching the max pooling layers
        name = 'up_%d' % (nb_levels + level)
        layers_dict[name] = upsample(size=pool_size, name=name)(last_layer)
        last_layer = layers_dict[name]

        # upsample matching the max pooling layers
        name = 'upconv_%d' % (nb_levels + level)
        nb_local_features = nb_features*(feat_mult**(nb_levels-2-level))
        layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
        last_layer = layers_dict[name]

        # merge layers combining previous layer
        conv_name = 'conv_downarm_%d_%d' % (nb_levels - 2 - level, nb_conv_per_level - 1)
        name = 'merge_%d' % (nb_levels + level)
        layers_dict[name] = KL.concatenate([layers_dict[conv_name], last_layer], axis=ndims+1, name=name)
        last_layer = layers_dict[name]

        for conv in range(nb_conv_per_level):
            name = 'conv_uparm_%d_%d' % (nb_levels + level, conv)
            layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
            last_layer = layers_dict[name]

        if use_residuals:
            conv_name = 'upconv_%d' % (nb_levels + level)
            convarm_layer = last_layer

            name = 'expand_up_merge_%d' % (level)
            layers_dict[name] = convL(nb_local_features, [1, 1, 1], **conv_kwargs, name=name)(layers_dict[conv_name])
            last_layer = layers_dict[name]

            name = 'res_up_merge_%d' % (level)
            layers_dict[name] = KL.add([convarm_layer,last_layer ], name=name)
            last_layer = layers_dict[name]

    # reshape last layer for prediction
    name = 'conv_uparm_%d_%d_reshape' % (2 * nb_levels - 2, nb_conv_per_level - 1)
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
        if use_logp:
            assert False, 'UNFINISHED'
            name = 'log-prediction'
            layers_dict[name] = KL.add([layers_dict['prior-input-reshape'], layers_dict['likelihood']])
            name = 'prediction'
            # layers_dict[name] = ...

        else:
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
               conv_dropout=0,
               conv_maxnorm=0,
               nb_conv_per_level=2):
    """
    "deep" cnn with dense or global max pooling layer @ end...

    Could use sequential...
    """

    ndims = len(patch_size)
    convL = KL.Conv3D if len(patch_size) == 3 else KL.Conv2D
    maxpool = KL.MaxPooling3D if len(patch_size) == 3 else KL.MaxPooling2D

    # kwargs for the convolution layer
    conv_kwargs = {'padding': padding, 'activation': activation}
    if conv_maxnorm > 0:
        conv_kwargs['kernel_constraint'] = maxnorm(conv_maxnorm)

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
            
            if conv_dropout > 0:
                name = 'dropout_%d_%d' % (level, conv)
                layers_dict[name] = KL.Dropout(conv_dropout)
                last_layer = layers_dict[name]

            name = 'conv_%d_%d' % (level, conv)
            nb_local_features = nb_features*(feat_mult**level)
            layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
            last_layer = layers_dict[name]

        # max pool
        name = 'maxpool_%d' % level
        layers_dict[name] = maxpool(pool_size=pool_size)(last_layer)
        last_layer = layers_dict[name]

    # dense layer
    if final_layer == 'dense':

        name = "flatten"
        layers_dict[name] = KL.Flatten(name=name)(last_layer)
        last_layer = layers_dict[name]

        name = 'dense'
        layers_dict[name] = KL.Dense(nb_labels, name=name)(last_layer)

    # global max pooling layer
    elif final_layer == 'myglobalmaxpooling':

        name = 'batch_norm'
        layers_dict[name] = KL.BatchNormalization(name=name)(last_layer)
        last_layer = layers_dict[name]

        name = 'global_max_pool'
        layers_dict[name] = KL.Lambda(_global_max_nd, name=name)(last_layer)
        last_layer = layers_dict[name]

        name = 'global_max_pool_reshape'
        layers_dict[name] = KL.Reshape((1, 1), name=name)(last_layer)
        last_layer = layers_dict[name]

        # cannot do activation in lambda layer. Could code inside, but will do extra lyaer
        name = 'global_max_pool_sigmoid'
        layers_dict[name] = KL.Conv1D(1, 1, name=name, activation="sigmoid", use_bias=True)(last_layer)

    elif final_layer == 'globalmaxpooling':

        name = 'conv_to_featmaps'
        layers_dict[name] = KL.Conv3D(2, 1, name=name, activation="relu")(last_layer)
        last_layer = layers_dict[name]

        name = 'global_max_pool'
        layers_dict[name] = KL.GlobalMaxPooling3D(name=name)(last_layer)
        last_layer = layers_dict[name]

        # cannot do activation in lambda layer. Could code inside, but will do extra lyaer
        name = 'global_max_pool_softmax'
        layers_dict[name] = KL.Activation('softmax', name=name)(last_layer)

    last_layer = layers_dict[name]

    # create the model
    model = Model(inputs=[layers_dict['input']], outputs=[last_layer])
    return model




def copy_weights(src_model, dst_model):
    """ copy weights from the src model to the dst model """

    for idx in range(len(dst_model.layers)):
        layer = dst_model.layers[idx]
        print("dst layer", layer)
        wts = src_model.layers[idx].get_weights()
        print("src layer", src_model.layers[idx])
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

    # seg_model.save('/data/vision/polina/users/adalca/fsCNN/output/unet-prior-v3-patches/init-model.hdf5')


def _global_max_nd(x):
    y = K.batch_flatten(x)
    return K.max(y, 1, keepdims=True)


def _log_layer(x):
    return K.log(x + K.epsilon)

def _global_max_nd(x):
    return K.exp(x)
