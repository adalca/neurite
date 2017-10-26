"""
Keras CNN models

Tested on keras 2.0
"""

# third party
import numpy as np
import keras
import keras.layers as KL
from keras.models import Model, Sequential
import keras.backend as K
from keras.constraints import maxnorm



def design_unet_prior():
    # design unet
    # design prior
    pass

def design_prior(input_model,
                 patch_size,
                 nb_labels,
                 name=None,
                 prefix=None,
                 use_logp=False,
                 final_pred_activation='softmax',
                 add_prior_layer=False):
    """
    Add prior layers to existing model

    # TODO: consider prior layer at beginning, and backwards model. Should do this test. Ps->S->I (can do inference backwards ?) and I -> S <- Ps
    """

    # prepare model name
    model_name = name
    if model_name is None:
        model_name = 'model_1'
    if prefix is None:
        prefix = model_name

    ndims = len(patch_size)
    patch_size = tuple(patch_size)
    convL = KL.Conv3D if ndims == 3 else KL.Conv2D

    # get likelihood layer prediction (no activation yet)
    layers_dict = {}
    name = '%s_likelihood' % prefix
    layers_dict[name] = input_model.get_layer(name).output
    like_layer = layers_dict[name]
    last_layer = layers_dict[name]

    nb_labels = layers_dict[name].get_shape()[-1]

    # add optional prior
    model_inputs = input_model.inputs
    if add_prior_layer:

        # prior input layer
        prior_input_name = '%s_prior-input' % prefix
        layers_dict[prior_input_name] = KL.Input(shape=patch_size + (nb_labels,), name=prior_input_name)
        prior_layer = layers_dict[prior_input_name]
        
        # operation varies depending on whether we log() prior or not.
        merge_op = KL.multiply
        if use_logp:
            name = '%s_prior-log' % prefix
            prior_input = layers_dict['%s_prior-input' % prefix]
            layers_dict[name] = KL.Lambda(_log_layer, name=name)(prior_input)
            prior_layer = layers_dict[name]

            merge_op = KL.add
        else:
            # using sigmoid to get the likelihood values between 0 and 1
            name = '%s_likelihood_sigmoid' % prefix
            layers_dict[name] = KL.Activation('sigmoid', name=name)(like_layer)
            like_layer = layers_dict[name]
            
        # merge the likelihood and prior layers into posterior layer
        name = '%s_posterior' % prefix
        layers_dict[name] = merge_op([prior_layer, like_layer])
        last_layer = layers_dict[name]

        # update model inputs
        model_inputs = [input_model.inputs, layers_dict[prior_input_name]]

    # output prediction layer
    # we use a softmax to compute P(L_x|I) where x is each location. 
    print(final_pred_activation, "final_pred_activation");
    if final_pred_activation == 'softmax':
        print('softmaxing', add_prior_layer, use_logp);
        assert (not add_prior_layer) or use_logp

        name = '%s_prediction' % prefix
        softmax_lambda_fcn = lambda x: keras.activations.softmax(x, axis=-1)
        layers_dict[name] = KL.Lambda(softmax_lambda_fcn, name=name)(like_layer)

    else:
        name = '%s_prediction' % prefix
        layers_dict[name] = KL.Activation('linear', name=name)(like_layer)

    # create the model
    model = Model(inputs=model_inputs, outputs=[layers_dict['%s_prediction' % prefix]], name=model_name)
    
    # compile
    return model


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
                nb_mid_level_dense=0,
                do_vae=False):
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
    # vol_numel = np.prod(patch_size)
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
            if conv < (nb_conv_per_level-1) or (not use_residuals):
              layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
            else:  # no activation
              layers_dict[name] = convL(nb_local_features, conv_size, padding=padding, name=name)(last_layer)
              
            last_layer = layers_dict[name]

        if use_residuals:
            convarm_layer = last_layer
            
            # the "add" layer is the original input. However, it may not have the right number of features to be added
            nb_feats_in = level_init_layer.get_shape()[-1]
            nb_feats_out = convarm_layer.get_shape()[-1]
            add_layer = level_init_layer
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
              name = '%s_expand_down_merge_%d' % (prefix, level)
              layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(level_init_layer)
              last_layer = layers_dict[name]
              add_layer = last_layer
            
            name = '%s_res_down_merge_%d' % (prefix, level)
            
            layers_dict[name] = KL.add([add_layer, convarm_layer], name=name)
            last_layer = layers_dict[name]
            
            name = '%s_res_down_merge_act_%d' % (prefix, level)
            layers_dict[name] = KL.Activation(activation, name=name)(last_layer)
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

        if do_vae: # variational auto-encoder
            name = '%s_mid_mu_enc_%d' % (prefix, nb_mid_level_dense)
            layers_dict[name] = KL.Dense(nb_mid_level_dense, name=name)(last_layer)

            name = '%s_mid_sigma_enc_%d' % (prefix, nb_mid_level_dense)
            layers_dict[name] = KL.Dense(nb_mid_level_dense, name=name)(last_layer)
            last_layer = layers_dict[name]

            sampler = _VAESample(nb_mid_level_dense).sample_z

            name = '%s_mid_dense_dec_sample' % prefix
            pname = '%s_mid_mu_enc_%d' % (prefix, nb_mid_level_dense)
            layers_dict[name] = KL.Lambda(sampler, name=name)([layers_dict[pname], last_layer])
            last_layer = layers_dict[name]

            name = '%s_mid_dense_dec_flat' % prefix
            layers_dict[name] = KL.Dense(np.prod(save_shape), name=name)(last_layer)
            last_layer = layers_dict[name]

        
        else: # normal
            name = '%s_mid_dense_enc_%d' % (prefix, nb_mid_level_dense)
            layers_dict[name] = KL.Dense(nb_mid_level_dense, name=name)(last_layer)
            last_layer = layers_dict[name]

            name = '%s_mid_dense_dec_flat_%d' % (prefix, nb_mid_level_dense)
            layers_dict[name] = KL.Dense(np.prod(save_shape), name=name)(last_layer)
            last_layer = layers_dict[name]

        name = '%s_mid_dense_dec' % prefix
        layers_dict[name] = KL.Reshape(save_shape, name=name)(last_layer)
        last_layer = layers_dict[name]

    # up arm:
    # nb_levels - 1 layers of Deconvolution3D
    #    (approx via up + conv + ReLu) + merge + conv + ReLu + conv + ReLu
    for level in range(nb_levels - 1):
        nb_local_features = nb_features*(feat_mult**(nb_levels-2-level))

        # upsample matching the max pooling layers
        name = '%s_up_%d' % (prefix, nb_levels + level)
        layers_dict[name] = upsample(size=pool_size, name=name)(last_layer)
        last_layer = layers_dict[name]

        # upsample matching the max pooling layers
        #name = '%s_upconv_%d' % (prefix, nb_levels + level)
        #layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
        #last_layer = layers_dict[name]

        # merge layers combining previous layer
        # TODO: add Cropping3D or Cropping2D if 'valid' padding
        if use_skip_connections:
            conv_name = '%s_conv_downarm_%d_%d' % (prefix, nb_levels - 2 - level, nb_conv_per_level - 1)
            name = '%s_merge_%d' % (prefix, nb_levels + level)
            layers_dict[name] = KL.concatenate([layers_dict[conv_name], last_layer], axis=ndims+1, name=name)
            last_layer = layers_dict[name]

        # convolution layers
        for conv in range(nb_conv_per_level):
            name = '%s_conv_uparm_%d_%d' % (prefix, nb_levels + level, conv)
            if conv < (nb_conv_per_level-1) or (not use_residuals):
              layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(last_layer)
            else:
              layers_dict[name] = convL(nb_local_features, conv_size, padding=padding, name=name)(last_layer)
            last_layer = layers_dict[name]


        if use_residuals:
            conv_name = '%s_up_%d' % (prefix, nb_levels + level)
            convarm_layer = last_layer

            # name = '%s_expand_up_merge_%d' % (prefix, level)
            # layers_dict[name] = convL(nb_local_features, conv_size, **conv_kwargs, name=name)(layers_dict[conv_name])
            # last_layer = layers_dict[name]

            name = '%s_res_up_merge_%d' % (prefix, level)
            layers_dict[name] = KL.add([convarm_layer, layers_dict[conv_name] ], name=name)
            last_layer = layers_dict[name]
            
            name = '%s_res_up_merge_act_%d' % (prefix, level)
            layers_dict[name] = KL.Activation(activation, name=name)(last_layer)
            last_layer = layers_dict[name]


    # Compute likelyhood prediction (no activation yet)
    name = '%s_likelihood' % prefix
    layers_dict[name] = convL(nb_labels, 1, activation=None, name=name)(last_layer)
    like_layer = layers_dict[name]
    last_layer = layers_dict[name]

    # add optional prior
    model_inputs = [layers_dict[input_name]]
    if add_prior_layer:

        # prior input layer
        prior_input_name = '%s_prior-input' % prefix
        layers_dict[prior_input_name] = KL.Input(shape=patch_size + (nb_labels,), name=prior_input_name)
        prior_layer = layers_dict[prior_input_name]
        
        # operation varies depending on whether we log() prior or not.
        merge_op = KL.multiply
        if use_logp:
            name = '%s_prior-log' % prefix
            prior_input = layers_dict['%s_prior-input' % prefix]
            layers_dict[name] = KL.Lambda(_log_layer, name=name)(prior_input)
            prior_layer = layers_dict[name]

            merge_op = KL.add
        else:
            # using sigmoid to get the likelihood values between 0 and 1
            # note: they won't add up to 1.
            name = '%s_likelihood_sigmoid' % prefix
            layers_dict[name] = KL.Activation('sigmoid', name=name)(like_layer)
            like_layer = layers_dict[name]
            
        # merge the likelihood and prior layers into posterior layer
        name = '%s_posterior' % prefix
        layers_dict[name] = merge_op([prior_layer, like_layer], name=name)
        last_layer = layers_dict[name]

        # update model inputs
        model_inputs = [layers_dict[input_name], layers_dict[prior_input_name]]

    # output prediction layer
    # we use a softmax to compute P(L_x|I) where x is each location. 
    if final_pred_activation == 'softmax':
        assert (not add_prior_layer) or use_logp, 'softmaxing cannot be done when adding prior in P() form' 
        
        name = '%s_prediction' % prefix
        softmax_lambda_fcn = lambda x: keras.activations.softmax(x, axis=ndims + 1)
        layers_dict[name] = KL.Lambda(softmax_lambda_fcn, name=name)(last_layer)

    else:
        name = '%s_prediction' % prefix
        layers_dict[name] = KL.Activation('linear', name=name)(like_layer)

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

# def _global_max_nd(x):
    # return K.exp(x)


class _VAESample():
    def __init__(self, nb_z):
        self.nb_z = nb_z

    def sample_z(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=(K.shape(mu)[0], self.nb_z), mean=0., stddev=1.)
        return mu + K.exp(log_sigma / 2) * eps
