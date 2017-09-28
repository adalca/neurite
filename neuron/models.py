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



def design_unet(nb_features,
                patch_size,
                nb_levels,
                conv_size,
                nb_labels,
                name=None,
                prefix=None,
                feat_mult=1,
                pool_size=None,
                use_logp=False,
                nb_input_features=1,
                use_skip_connections=True,
                padding='same',
                activation='relu',
                use_residuals=False,
                final_pred_activation='softmax',
                nb_conv_per_level=2,
                add_prior_layer=False,
                nb_mid_level_dense=0):
    """
    unet-style model with lots of parametrization

    for U-net like architecture, we need to use Deconvolution3D.
    However, this is not yet available (maybe soon, it's on a dev branch in github I believe)
    Until then, we'll upsample and convolve.
    TODO: Need to check that UpSampling3D actually does NN-upsampling!
    """

    model_name = name
    if model_name is None:
        model_name = 'model_1'
    if prefix is None:
        prefix = model_name

    ndims = len(patch_size)
    patch_size = tuple(patch_size)
    convL = KL.Conv3D if ndims == 3 else KL.Conv2D
    maxpool = KL.MaxPooling3D if ndims == 3 else KL.MaxPooling2D
    upsample = KL.UpSampling3D if ndims == 3 else KL.UpSampling2D
    vol_numel = np.prod(patch_size)
    if pool_size is None:
        pool_size = (2, 2, 2) if len(patch_size) == 3 else (2, 2)

    # kwargs for the convolution layer
    conv_kwargs = {'padding': padding, 'activation': activation}

    # initialize a dictionary
    layers_dict = {}

    # first layer: input
    name = '%s_input' % prefix
    input_name = name
    layers_dict[name] = KL.Input(shape=patch_size + (nb_input_features,), name=name)
    last_layer = layers_dict[name]

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    for level in range(nb_levels):
        level_init_layer = last_layer
        nb_local_features = nb_features*(feat_mult**level)

        for conv in range(nb_conv_per_level):
            name = '%s_conv_downarm_%d_%d' % (prefix, level, conv)
            layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
            last_layer = layers_dict[name]

        if use_residuals:
            convarm_layer = last_layer
            name = '%s_expand_down_merge_%d' % (prefix, level)
            layers_dict[name] = convL(nb_local_features, [1, 1, 1], **conv_kwargs, name=name)(level_init_layer)
            last_layer = layers_dict[name]
            name = '%s_res_down_merge_%d' % (prefix, level)
            layers_dict[name] = KL.add([last_layer, convarm_layer], name=name)
            last_layer = layers_dict[name]

        # max pool if we're not at the last level
        if level < (nb_levels - 1):
            name = '%s_maxpool_%d' % (prefix, level)
            layers_dict[name] = maxpool(pool_size=pool_size)(last_layer)
            last_layer = layers_dict[name]

    # if want to go through a dense layer in the middle of the U, need to:
    # - flatten last layer
    # - do dense encoding and decoding
    # - unflatten (rehsape spatially)
    if nb_mid_level_dense > 0:
        save_shape = last_layer.get_shape().as_list()[1:]

        name = '%s_mid_dense_down_flat' % prefix
        layers_dict[name] = KL.Flatten(name=name)(last_layer)
        last_layer = layers_dict[name]

        name = '%s_mid_dense_enc' % prefix
        layers_dict[name] = KL.Dense(nb_mid_level_dense, activation=activation, name=name)(last_layer)
        last_layer = layers_dict[name]

        name = '%s_mid_dense_dec_flat' % prefix
        layers_dict[name] = KL.Dense(np.prod(save_shape), activation=activation, name=name)(last_layer)
        last_layer = layers_dict[name]

        name = '%s_mid_dense_dec' % prefix
        layers_dict[name] = KL.Reshape(save_shape, name=name)(last_layer)
        last_layer = layers_dict[name]

    # up arm:
    # nb_levels - 1 layers of Deconvolution3D
    #    (approx via up + conv + ReLu) + merge + conv + ReLu + conv + ReLu
    for level in range(nb_levels - 1):

        # upsample matching the max pooling layers
        name = '%s_up_%d' % (prefix, nb_levels + level)
        layers_dict[name] = upsample(size=pool_size, name=name)(last_layer)
        last_layer = layers_dict[name]

        # upsample matching the max pooling layers
        name = '%s_upconv_%d' % (prefix, nb_levels + level)
        nb_local_features = nb_features*(feat_mult**(nb_levels-2-level))
        layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
        last_layer = layers_dict[name]

        # merge layers combining previous layer
        if use_skip_connections:
            conv_name = '%s_conv_downarm_%d_%d' % (prefix, nb_levels - 2 - level, nb_conv_per_level - 1)
            name = '%s_merge_%d' % (prefix, nb_levels + level)
            layers_dict[name] = KL.concatenate([layers_dict[conv_name], last_layer], axis=ndims+1, name=name)
            last_layer = layers_dict[name]

        # convolution layers
        for conv in range(nb_conv_per_level):
            name = '%s_conv_uparm_%d_%d' % (prefix, nb_levels + level, conv)
            layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
            last_layer = layers_dict[name]

        if use_residuals:
            conv_name = '%s_upconv_%d' % (prefix, nb_levels + level)
            convarm_layer = last_layer

            name = '%s_expand_up_merge_%d' % (prefix, level)
            layers_dict[name] = convL(nb_local_features, [1, 1, 1], **conv_kwargs, name=name)(layers_dict[conv_name])
            last_layer = layers_dict[name]

            name = '%s_res_up_merge_%d' % (prefix, level)
            layers_dict[name] = KL.add([convarm_layer,last_layer ], name=name)
            last_layer = layers_dict[name]

    # reshape last layer for prediction
    name = '%s_conv_uparm_%d_%d_reshape' % (prefix, 2 * nb_levels - 2, nb_conv_per_level - 1)
    layers_dict[name] = KL.Reshape((vol_numel, nb_features), name=name)(last_layer)
    last_layer = layers_dict[name]

    if add_prior_layer:

        # likelihood layer
        name = '%s_likelihood' % prefix
        act = activation if use_logp else final_pred_activation
        layers_dict[name] = KL.Conv1D(nb_labels, 1, activation=act, name=name)(last_layer)
        last_layer = layers_dict[name]

        # prior input layer
        name = '%s_prior-input' % prefix
        prior_input_name = name
        layers_dict[name] = KL.Input(shape=patch_size + (nb_labels,), name=name)
        name = '%s_prior-input-reshape' % prefix
        layers_dict[name] = KL.Reshape((vol_numel, nb_labels), name=name)(layers_dict[prior_input_name])

        # final prediction
        if use_logp:
            # log of prior
            name = '%s_prior-log' % prefix
            layers_dict[name] = KL.Lambda(_log_layer, name=name)(layers_dict['%s_prior-input-reshape' % prefix])
            last_layer = layers_dict[name]

            # compute log prediction
            name = '%s_log-prediction' % prefix
            layers_dict[name] = KL.add([layers_dict['%s_prior-log' % prefix], layers_dict['%s_likelihood' % prefix]])
            last_layer = layers_dict[name]

            name = '%s_prediction' % prefix
            layers_dict[name] = KL.Activation(final_pred_activation, name=name)(last_layer)
            last_layer = layers_dict[name]

        else:
            name = '%s_prediction' % prefix
            layers_dict[name] = KL.multiply([layers_dict['%s_prior-input-reshape' % prefix], layers_dict['%s_likelihood' % prefix]])

        model_inputs = [layers_dict[input_name], layers_dict[prior_input_name]]

    else:

        # output (liklihood) prediction layer
        name = '%s_prediction' % prefix
        layers_dict[name] = KL.Conv1D(nb_labels, 1, activation=final_pred_activation, name=name)(last_layer)

        model_inputs = [layers_dict[input_name]]

    # create the model
    model = Model(inputs=model_inputs, outputs=[layers_dict['%s_prediction' % prefix]], name=model_name)
    
    # compile
    return model


def design_dnn(nb_features, patch_size, nb_levels, conv_size, nb_labels,
               feat_mult=1,
               pool_size=None,
               padding='same',
               activation='relu',
               final_layer='dense-sigmoid',
               conv_dropout=0,
               conv_maxnorm=0,
               nb_input_features=1,
               name=None,
               prefix=None,
               use_strided_convolution_maxpool=True,
               nb_conv_per_level=2):
    """
    "deep" cnn with dense or global max pooling layer @ end...

    Could use sequential...
    """

    model_name = name
    if model_name is None:
        model_name = 'model_1'
    if prefix is None:
        prefix = model_name

    ndims = len(patch_size)
    patch_size = tuple(patch_size)

    convL = KL.Conv3D if len(patch_size) == 3 else KL.Conv2D
    maxpool = KL.MaxPooling3D if len(patch_size) == 3 else KL.MaxPooling2D
    if pool_size is None:
        pool_size = (2, 2, 2) if len(patch_size) == 3 else (2, 2)

    # kwargs for the convolution layer
    conv_kwargs = {'padding': padding, 'activation': activation}
    if conv_maxnorm > 0:
        conv_kwargs['kernel_constraint'] = maxnorm(conv_maxnorm)

    # initialize a dictionary
    layers_dict = {}

    # first layer: input
    name = '%s_input' % prefix
    layers_dict[name] = KL.Input(shape=patch_size + (nb_input_features,), name=name)
    last_layer = layers_dict[name]

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    for level in range(nb_levels):
        for conv in range(nb_conv_per_level):
            if conv_dropout > 0:
                name = '%s_dropout_%d_%d' % (prefix, level, conv)
                layers_dict[name] = KL.Dropout(conv_dropout)(last_layer)
                last_layer = layers_dict[name]

            name = '%s_conv_%d_%d' % (prefix, level, conv)
            nb_local_features = nb_features*(feat_mult**level)
            layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
            last_layer = layers_dict[name]

        # max pool
        if use_strided_convolution_maxpool:
            name = '%s_strided_conv_%d' % (prefix, level)
            layers_dict[name] = convL(nb_local_features, pool_size, **conv_kwargs, name=name)(last_layer)
            last_layer = layers_dict[name]
        else:
            name = '%s_maxpool_%d' % (prefix, level)
            layers_dict[name] = maxpool(pool_size=pool_size)(last_layer)
            last_layer = layers_dict[name]

    # dense layer
    if final_layer == 'dense-sigmoid':

        name = "%s_flatten" % prefix
        layers_dict[name] = KL.Flatten(name=name)(last_layer)
        last_layer = layers_dict[name]

        name = '%s_dense' % prefix
        layers_dict[name] = KL.Dense(1, name=name, activation="sigmoid")(last_layer)

    elif final_layer == 'dense-tanh':

        name = "%s_flatten" % prefix
        layers_dict[name] = KL.Flatten(name=name)(last_layer)
        last_layer = layers_dict[name]

        name = '%s_dense' % prefix
        layers_dict[name] = KL.Dense(1, name=name)(last_layer)
        last_layer = layers_dict[name]

        # Omittting BatchNorm for now, it seems to have a cpu vs gpu problem
        # https://github.com/tensorflow/tensorflow/pull/8906
        # https://github.com/fchollet/keras/issues/5802
        # name = '%s_dense_bn' % prefix
        # layers_dict[name] = KL.BatchNormalization(name=name)(last_layer)
        # last_layer = layers_dict[name]

        name = '%s_dense_tanh' % prefix
        layers_dict[name] = KL.Activation(activation="tanh", name=name)(last_layer)

    elif final_layer == 'dense-softmax':

        name = "%s_flatten" % prefix
        layers_dict[name] = KL.Flatten(name=name)(last_layer)
        last_layer = layers_dict[name]

        name = '%s_dense' % prefix
        layers_dict[name] = KL.Dense(nb_labels, name=name, activation="softmax")(last_layer)

    # global max pooling layer
    elif final_layer == 'myglobalmaxpooling':

        name = '%s_batch_norm' % prefix
        layers_dict[name] = KL.BatchNormalization(name=name)(last_layer)
        last_layer = layers_dict[name]

        name = '%s_global_max_pool' % prefix
        layers_dict[name] = KL.Lambda(_global_max_nd, name=name)(last_layer)
        last_layer = layers_dict[name]

        name = '%s_global_max_pool_reshape' % prefix
        layers_dict[name] = KL.Reshape((1, 1), name=name)(last_layer)
        last_layer = layers_dict[name]

        # cannot do activation in lambda layer. Could code inside, but will do extra lyaer
        name = '%s_global_max_pool_sigmoid' % prefix
        layers_dict[name] = KL.Conv1D(1, 1, name=name, activation="sigmoid", use_bias=True)(last_layer)

    elif final_layer == 'globalmaxpooling':

        name = '%s_conv_to_featmaps' % prefix
        layers_dict[name] = KL.Conv3D(2, 1, name=name, activation="relu")(last_layer)
        last_layer = layers_dict[name]

        name = '%s_global_max_pool' % prefix
        layers_dict[name] = KL.GlobalMaxPooling3D(name=name)(last_layer)
        last_layer = layers_dict[name]

        # cannot do activation in lambda layer. Could code inside, but will do extra lyaer
        name = '%s_global_max_pool_softmax' % prefix
        layers_dict[name] = KL.Activation('softmax', name=name)(last_layer)

    last_layer = layers_dict[name]

    # create the model
    model = Model(inputs=[layers_dict['%s_input' % prefix]], outputs=[last_layer], name=model_name)
    return model




def copy_weights(src_model, dst_model):
    """ copy weights from the src model to the dst model """

    for idx in range(len(dst_model.layers)):
        layer = dst_model.layers[idx]
        wts = src_model.layers[idx].get_weights()
        print(len(wts), len(layer.get_weights()))
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
    return K.log(x + K.epsilon())

def _global_max_nd(x):
    return K.exp(x)
