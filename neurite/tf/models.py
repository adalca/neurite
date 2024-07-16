"""
models (networks) for the neurite project

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

# core python
import sys
import warnings

# third party
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.python.keras.constraints import maxnorm

# local
from . import layers
from . import utils
from . import modelio

###############################################################################
# Roughly volume preserving (e.g. high dim to high dim) models
###############################################################################


def dilation_net(nb_features,
                 input_shape,  # input layer shape, vector of size ndims + 1(nb_channels)
                 nb_levels,
                 conv_size,
                 nb_labels,
                 name='dilation_net',
                 prefix=None,
                 feat_mult=1,
                 pool_size=2,
                 use_logp=True,
                 padding='same',
                 dilation_rate_mult=1,
                 activation='elu',
                 use_residuals=False,
                 final_pred_activation='softmax',
                 nb_conv_per_level=1,
                 add_prior_layer=False,
                 add_prior_layer_reg=0,
                 layer_nb_feats=None,
                 batch_norm=None):

    return unet(nb_features,
                input_shape,  # input layer shape, vector of size ndims + 1(nb_channels)
                nb_levels,
                conv_size,
                nb_labels,
                name='unet',
                prefix=None,
                feat_mult=1,
                pool_size=2,
                use_logp=True,
                padding='same',
                activation='elu',
                use_residuals=False,
                dilation_rate_mult=dilation_rate_mult,
                final_pred_activation='softmax',
                nb_conv_per_level=1,
                add_prior_layer=False,
                add_prior_layer_reg=0,
                layer_nb_feats=None,
                batch_norm=None)


def unet(nb_features,
         input_shape,
         nb_levels,
         conv_size,
         nb_labels,
         name='unet',
         prefix=None,
         feat_mult=1,
         pool_size=2,
         use_logp=True,
         padding='same',
         dilation_rate_mult=1,
         activation='elu',
         use_residuals=False,
         final_pred_activation='softmax',
         nb_conv_per_level=1,
         add_prior_layer=False,
         add_prior_layer_reg=0,
         layer_nb_feats=None,
         conv_dropout=0,
         batch_norm=None,
         convL=None):  # conv layer function
    """
    unet-style keras model with an overdose of parametrization.

    downsampling: 

    for U-net like architecture, we need to use Deconvolution3D.
    However, this is not yet available (maybe soon, it's on a dev branch in github I believe)
    Until then, we'll upsample and convolve.
    TODO: Need to check that UpSampling3D actually does NN-upsampling!

    Parameters:
        nb_features: the number of features at each convolutional level
            see below for `feat_mult` and `layer_nb_feats` for modifiers to this number
            if nb_features is a list of lists, they will be taken to specify
            the number of filters and layers at each level
            overriding other options such as nb_levels and feat_mult
        input_shape: input layer shape, vector of size ndims + 1 (nb_channels)
        conv_size: the convolution kernel size
        nb_levels: the number of Unet levels (number of downsamples) in the "encoder" 
            (e.g. 4 would give you 4 levels in encoder, 4 in decoder)
        nb_labels: number of output channels
        name (default: 'unet'): the name of the network
        prefix (default: `name` value): prefix to be added to layer names
        feat_mult (default: 1) multiple for `nb_features` as we go down the encoder levels.
            e.g. feat_mult of 2 and nb_features of 16 would yield 32 features in the 
            second layer, 64 features in the third layer, etc
        pool_size (default: 2): max pooling size (integer or list if specifying per dimension)
        use_logp:
        padding:
        dilation_rate_mult:
        activation:
        use_residuals:
        final_pred_activation:
        nb_conv_per_level:
        add_prior_layer:
        add_prior_layer_reg:
        layer_nb_feats:
        conv_dropout:
        batch_norm:
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    if isinstance(input_shape[0], (tuple, list, np.ndarray)):
        # input_shape is a list of lists, and so we need to configure
        # multiple inputs here, concatenate, and pass them to the encoder
        src_input = []
        for i, shape in enumerate(input_shape):
            if not np.array_equal(shape[:-1], input_shape[0][:-1]):
                raise ValueError('spatial dimensions must match if multiple input shapes '
                                 'are provided, but got shapes '
                                 f'{input_shape[0][:-1]} and {shape[:-1]}')
            src_input.append(KL.Input(shape=shape, name=f'{prefix}_input_{i}'))
        src = KL.concatenate(src_input, axis=-1, name=f'{prefix}_input_concat')
        # also we should set input_shape to the first value for further calls
        input_shape = input_shape[0]
    else:
        src = None
        src_input = None

    # volume size data
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # if nb_features is a list of lists use it to specify the # of levels
    # and the number of filters at each level (in each sublist)
    if isinstance(nb_features, list):
        if nb_levels is not None:
            warnings.warn('nb_levels is not None while ' +
                          'nb_features list of lists specified - overriding')

        if feat_mult is not None:
            warnings.warn('feat_mult is not None while ' +
                          'nb_features list of lists specified - overriding')

        nb_levels = len(nb_features)
        assert isinstance(nb_features[0], list), \
            'nb_features must be a scalar or a list of lists (not a list of scalars)'

    # get encoding model
    enc_model = conv_enc(nb_features,
                         input_shape,
                         nb_levels,
                         conv_size,
                         name=model_name,
                         prefix=prefix,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         padding=padding,
                         dilation_rate_mult=dilation_rate_mult,
                         activation=activation,
                         use_residuals=use_residuals,
                         nb_conv_per_level=nb_conv_per_level,
                         layer_nb_feats=layer_nb_feats,
                         conv_dropout=conv_dropout,
                         batch_norm=batch_norm,
                         src=src,
                         src_input=src_input,
                         convL=convL)

    # get decoder
    # use_skip_connections=1 makes it a u-net
    lnf = layer_nb_feats[(nb_levels * nb_conv_per_level):] if layer_nb_feats is not None else None
    dec_model = \
        conv_dec(nb_features,
                 None,
                 nb_levels,
                 conv_size,
                 nb_labels,
                 name=model_name,
                 prefix=prefix,
                 feat_mult=feat_mult,
                 pool_size=pool_size,
                 use_skip_connections=1,
                 padding=padding,
                 dilation_rate_mult=dilation_rate_mult,
                 activation=activation,
                 use_residuals=use_residuals,
                 final_pred_activation='linear' if add_prior_layer else final_pred_activation,
                 nb_conv_per_level=nb_conv_per_level,
                 batch_norm=batch_norm,
                 layer_nb_feats=lnf,
                 conv_dropout=conv_dropout,
                 input_model=enc_model,
                 convL=convL)

    final_model = dec_model
    if add_prior_layer:
        final_model = add_prior(dec_model,
                                [*input_shape[:-1], nb_labels],
                                name=model_name + '_prior',
                                use_logp=use_logp,
                                final_pred_activation=final_pred_activation,
                                add_prior_layer_reg=add_prior_layer_reg)

    return final_model


def ae(nb_features,
       input_shape,
       nb_levels,
       conv_size,
       nb_labels,
       enc_size,
       name='ae',
       prefix=None,
       feat_mult=1,
       pool_size=2,
       padding='same',
       activation='elu',
       use_residuals=False,
       nb_conv_per_level=1,
       batch_norm=None,
       enc_batch_norm=None,
       ae_type='conv',  # 'dense', or 'conv'
       enc_lambda_layers=None,
       add_prior_layer=False,
       add_prior_layer_reg=0,
       use_logp=True,
       conv_dropout=0,
       include_mu_shift_layer=False,
       # whether to return a single model, or a tuple of models that can be stacked.
       single_model=False,
       final_pred_activation='softmax',
       src=None,
       src_input=None,
       do_vae=False,
       convL=None):  # conv layer function
    """
    Convolutional Auto-Encoder.
    Optionally Variational.
    Optionally Dense middle layer

    "Mostly" in that the inner encoding can be (optionally) constructed via dense features.

    Parameters:
        do_vae (bool): whether to do a variational auto-encoder or not.

    enc_lambda_layers functions to try:
        K.softsign

        a = 1
        longtanh = lambda x: K.tanh(x) *  K.log(2 + a * abs(x))
    """

    # naming
    model_name = name

    # volume size data
    ndims = len(input_shape) - 1
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # get encoding model
    enc_model = conv_enc(nb_features,
                         input_shape,
                         nb_levels,
                         conv_size,
                         name=model_name,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         padding=padding,
                         activation=activation,
                         use_residuals=use_residuals,
                         nb_conv_per_level=nb_conv_per_level,
                         conv_dropout=conv_dropout,
                         batch_norm=batch_norm,
                         src=src,
                         src_input=src_input,
                         convL=convL)

    # middle AE structure
    if single_model:
        in_input_shape = None
        in_model = enc_model
    else:
        in_input_shape = enc_model.output.shape.as_list()[1:]
        in_model = None
    mid_ae_model = single_ae(enc_size,
                             in_input_shape,
                             conv_size=conv_size,
                             name=model_name,
                             ae_type=ae_type,
                             input_model=in_model,
                             batch_norm=enc_batch_norm,
                             enc_lambda_layers=enc_lambda_layers,
                             include_mu_shift_layer=include_mu_shift_layer,
                             do_vae=do_vae)

    # decoder
    if single_model:
        in_input_shape = None
        in_model = mid_ae_model
    else:
        in_input_shape = mid_ae_model.output.shape.as_list()[1:]
        in_model = None
    dec_model = conv_dec(nb_features,
                         in_input_shape,
                         nb_levels,
                         conv_size,
                         nb_labels,
                         name=model_name,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         use_skip_connections=False,
                         padding=padding,
                         activation=activation,
                         use_residuals=use_residuals,
                         final_pred_activation=final_pred_activation,
                         nb_conv_per_level=nb_conv_per_level,
                         batch_norm=batch_norm,
                         conv_dropout=conv_dropout,
                         input_model=in_model,
                         convL=convL)

    if add_prior_layer:
        dec_model = add_prior(dec_model,
                              [*input_shape[:-1], nb_labels],
                              name=model_name,
                              prefix=model_name + '_prior',
                              use_logp=use_logp,
                              final_pred_activation=final_pred_activation,
                              add_prior_layer_reg=add_prior_layer_reg)

    if single_model:
        return dec_model
    else:
        return (dec_model, mid_ae_model, enc_model)


def add_prior(input_model,
              prior_shape,
              name='prior_model',
              prefix=None,
              use_logp=True,
              final_pred_activation='softmax',
              add_prior_layer_reg=0):
    """
    Append post-prior layer to a given model
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # prior input layer
    prior_input_name = '%s-input' % prefix
    prior_tensor = KL.Input(shape=prior_shape, name=prior_input_name)
    prior_tensor_input = prior_tensor
    like_tensor = input_model.output

    # operation varies depending on whether we log() prior or not.
    if use_logp:
        # name = '%s-log' % prefix
        # prior_tensor = KL.Lambda(_log_layer_wrap(add_prior_layer_reg), name=name)(prior_tensor)
        print("Breaking change: use_logp option now requires log input!", file=sys.stderr)
        merge_op = KL.add

    else:
        # using sigmoid to get the likelihood values between 0 and 1
        # note: they won't add up to 1.
        name = '%s_likelihood_sigmoid' % prefix
        like_tensor = KL.Activation('sigmoid', name=name)(like_tensor)
        merge_op = KL.multiply

    # merge the likelihood and prior layers into posterior layer
    name = '%s_posterior' % prefix
    post_tensor = merge_op([prior_tensor, like_tensor], name=name)

    # output prediction layer
    # we use a softmax to compute P(L_x|I) where x is each location
    pred_name = '%s_prediction' % prefix
    if final_pred_activation == 'softmax':
        assert use_logp, 'cannot do softmax when adding prior via P()'
        print("using final_pred_activation %s for %s" % (final_pred_activation, model_name))
        softmax_lambda_fcn = lambda x: tf.keras.activations.softmax(x, axis=-1)
        pred_tensor = KL.Lambda(softmax_lambda_fcn, name=pred_name)(post_tensor)

    else:
        pred_tensor = KL.Activation('linear', name=pred_name)(post_tensor)

    # create the model
    model_inputs = [*input_model.inputs, prior_tensor_input]
    model = Model(inputs=model_inputs, outputs=[pred_tensor], name=model_name)

    # compile
    return model


def single_ae(enc_size,
              input_shape,
              name='single_ae',
              prefix=None,
              ae_type='dense',  # 'dense', or 'conv'
              conv_size=None,
              input_model=None,
              enc_lambda_layers=None,
              batch_norm=True,
              padding='same',
              activation=None,
              include_mu_shift_layer=False,
              do_vae=False):
    """
    single-layer Autoencoder (i.e. input - encoding - output)
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    if enc_lambda_layers is None:
        enc_lambda_layers = []

    # prepare input
    input_name = '%s_input' % prefix
    if input_model is None:
        assert input_shape is not None, 'input_shape of input_model is necessary'
        input_tensor = KL.Input(shape=input_shape, name=input_name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.input
        last_tensor = input_model.output
        input_shape = last_tensor.shape.as_list()[1:]
    input_nb_feats = last_tensor.shape.as_list()[-1]

    # prepare conv type based on input
    if ae_type == 'conv':
        ndims = len(input_shape) - 1
        convL = getattr(KL, 'Conv%dD' % ndims)
        assert conv_size is not None, 'with conv ae, need conv_size'
    conv_kwargs = {'padding': padding, 'activation': activation}

    # if want to go through a dense layer in the middle of the U, need to:
    # - flatten last layer if not flat
    # - do dense encoding and decoding
    # - unflatten (rehsape spatially) at end
    if ae_type == 'dense' and len(input_shape) > 1:
        name = '%s_ae_%s_down_flat' % (prefix, ae_type)
        last_tensor = KL.Flatten(name=name)(last_tensor)

    # recall this layer
    pre_enc_layer = last_tensor

    # encoding layer
    if ae_type == 'dense':
        assert len(enc_size) == 1, "enc_size should be of length 1 for dense layer"

        enc_size_str = ''.join(['%d_' % d for d in enc_size])[:-1]
        name = '%s_ae_mu_enc_dense_%s' % (prefix, enc_size_str)
        last_tensor = KL.Dense(enc_size[0], name=name)(pre_enc_layer)

    else:  # convolution
        # convolve then resize. enc_size should be [nb_dim1, nb_dim2, ..., nb_feats]
        assert len(enc_size) == len(input_shape), \
            "encoding size does not match input shape %d %d" % (len(enc_size), len(input_shape))

        if list(enc_size)[:-1] != list(input_shape)[:-1] and \
                all([f is not None for f in input_shape[:-1]]) and \
                all([f is not None for f in enc_size[:-1]]):

            # assert len(enc_size) - 1 == 2, "Sorry, I have not yet implemented non-2D
            #   resizing -- need to check out interpn!"
            name = '%s_ae_mu_enc_conv' % (prefix)
            last_tensor = convL(enc_size[-1], conv_size, name=name, **conv_kwargs)(pre_enc_layer)

            name = '%s_ae_mu_enc' % (prefix)
            zf = [enc_size[:-1][f] / last_tensor.shape.as_list()[1:-1][f]
                  for f in range(len(enc_size) - 1)]
            last_tensor = layers.Resize(zoom_factor=zf, name=name)(last_tensor)
            # resize_fn = lambda x: tf.image.resize_bilinear(x, enc_size[:-1])
            # last_tensor = KL.Lambda(resize_fn, name=name)(last_tensor)

        elif enc_size[-1] is None:  # convolutional, but won't tell us bottleneck
            name = '%s_ae_mu_enc' % (prefix)
            last_tensor = KL.Lambda(lambda x: x, name=name)(pre_enc_layer)

        else:
            name = '%s_ae_mu_enc' % (prefix)
            last_tensor = convL(enc_size[-1], conv_size, name=name, **conv_kwargs)(pre_enc_layer)

    if include_mu_shift_layer:
        # shift
        name = '%s_ae_mu_shift' % (prefix)
        last_tensor = layers.LocalBias(name=name)(last_tensor)

    # encoding clean-up layers
    for layer_fcn in enc_lambda_layers:
        lambda_name = layer_fcn.__name__
        name = '%s_ae_mu_%s' % (prefix, lambda_name)
        last_tensor = KL.Lambda(layer_fcn, name=name)(last_tensor)

    if batch_norm is not None:
        name = '%s_ae_mu_bn' % (prefix)
        last_tensor = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

    # have a simple layer that does nothing to have a clear name before sampling
    name = '%s_ae_mu' % (prefix)
    last_tensor = KL.Lambda(lambda x: x, name=name)(last_tensor)

    # if doing variational AE, will need the sigma layer as well.
    if do_vae:
        mu_tensor = last_tensor

        # encoding layer
        if ae_type == 'dense':
            name = '%s_ae_sigma_enc_dense_%s' % (prefix, enc_size_str)
            #    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5),
            #    bias_initializer=tf.keras.initializers.RandomNormal(mean=-5.0, stddev=1e-5)
            last_tensor = KL.Dense(enc_size[0], name=name)(pre_enc_layer)

        else:
            if list(enc_size)[:-1] != list(input_shape)[:-1] and \
                    all([f is not None for f in input_shape[:-1]]) and \
                    all([f is not None for f in enc_size[:-1]]):

                # assert len(enc_size) - 1 == 2,
                # "Sorry, I have not yet implemented non-2D resizing..."
                name = '%s_ae_sigma_enc_conv' % (prefix)
                last_tensor = convL(enc_size[-1], conv_size, name=name,
                                    **conv_kwargs)(pre_enc_layer)

                name = '%s_ae_sigma_enc' % (prefix)
                zf = [enc_size[:-1][f] / last_tensor.shape.as_list()[1:-1][f]
                      for f in range(len(enc_size) - 1)]
                last_tensor = layers.Resize(zoom_factor=zf, name=name)(last_tensor)
                # resize_fn = lambda x: tf.image.resize_bilinear(x, enc_size[:-1])
                # last_tensor = KL.Lambda(resize_fn, name=name)(last_tensor)

            elif enc_size[-1] is None:  # convolutional, but won't tell us bottleneck
                name = '%s_ae_sigma_enc' % (prefix)
                last_tensor = convL(pre_enc_layer.shape.as_list()
                                    [-1], conv_size, name=name, **conv_kwargs)(pre_enc_layer)
                # cannot use lambda, then mu and sigma will be same layer.
                # last_tensor = KL.Lambda(lambda x: x, name=name)(pre_enc_layer)

            else:
                name = '%s_ae_sigma_enc' % (prefix)
                last_tensor = convL(enc_size[-1], conv_size, name=name,
                                    **conv_kwargs)(pre_enc_layer)

        # encoding clean-up layers
        for layer_fcn in enc_lambda_layers:
            lambda_name = layer_fcn.__name__
            name = '%s_ae_sigma_%s' % (prefix, lambda_name)
            last_tensor = KL.Lambda(layer_fcn, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_ae_sigma_bn' % (prefix)
            last_tensor = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

        # have a simple layer that does nothing to have a clear name before sampling
        name = '%s_ae_sigma' % (prefix)
        last_tensor = KL.Lambda(lambda x: x, name=name)(last_tensor)

        logvar_tensor = last_tensor

        # VAE sampling
        name = '%s_ae_sample' % (prefix)
        last_tensor = layers.SampleNormalLogVar(name=name)([mu_tensor, logvar_tensor])

    if include_mu_shift_layer:
        # shift
        name = '%s_ae_sample_shift' % (prefix)
        last_tensor = layers.LocalBias(name=name)(last_tensor)

    # decoding layer
    if ae_type == 'dense':
        name = '%s_ae_%s_dec_flat_%s' % (prefix, ae_type, enc_size_str)
        last_tensor = KL.Dense(np.prod(input_shape), name=name)(last_tensor)

        # unflatten if dense method
        if len(input_shape) > 1:
            name = '%s_ae_%s_dec' % (prefix, ae_type)
            last_tensor = KL.Reshape(input_shape, name=name)(last_tensor)

    else:

        if list(enc_size)[:-1] != list(input_shape)[:-1] and \
                all([f is not None for f in input_shape[:-1]]) and \
                all([f is not None for f in enc_size[:-1]]):

            name = '%s_ae_mu_dec' % (prefix)
            zf = [input_shape[:-1][f] / enc_size[:-1][f] for f in range(len(enc_size) - 1)]
            last_tensor = layers.Resize(zoom_factor=zf, name=name)(last_tensor)
            # resize_fn = lambda x: tf.image.resize_bilinear(x, input_shape[:-1])
            # last_tensor = KL.Lambda(resize_fn, name=name)(last_tensor)

        name = '%s_ae_%s_dec' % (prefix, ae_type)
        last_tensor = convL(input_nb_feats, conv_size, name=name, **conv_kwargs)(last_tensor)

    if batch_norm is not None:
        name = '%s_bn_ae_%s_dec' % (prefix, ae_type)
        last_tensor = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

    # create the model and retun
    model = Model(inputs=input_tensor, outputs=[last_tensor], name=model_name)
    return model


def labels_to_image_old(
    in_shape,
    in_label_list,
    out_label_list=None,
    out_shape=None,
    num_chan=1,
    input_model=None,
    mean_min=None,
    mean_max=None,
    std_min=None,
    std_max=None,
    zero_background=0.2,
    warp_res=[16],
    warp_std=0.5,
    warp_modulate=True,
    bias_res=40,
    bias_std=0.3,
    bias_modulate=True,
    blur_std=1,
    blur_modulate=True,
    normalize=True,
    gamma_std=0.25,
    dc_offset=0,
    one_hot=True,
    seeds={},
    return_vel=False,
    return_def=False,
    id=0,
):
    """
    Generative model for augmenting label maps and synthesizing images from them.

    Parameters:
        in_shape: List of the spatial dimensions of the input label maps.
        in_label_list: List of all possible input labels.
        out_label_list (optional): List of labels in the output label maps. If
            a dictionary is passed, it will be used to convert labels, e.g. to
            GM, WM and CSF. All labels not included will be converted to
            background with value 0. If 0 is among the output labels, it will be
            one-hot encoded. Defaults to the input labels.
        out_shape (optional): List of the spatial dimensions of the outputs.
            Inputs will be symmetrically cropped or zero-padded to fit.
            Defaults to the input shape.
        num_chan (optional): Number of image channels to be synthesized.
            Defaults to 1.
        input_model (optional): Optional input model that feeds directly into
            this model.
        mean_min (optional): List of lower bounds on the means drawn to generate
            the intensities for each label. Defaults to 0 for the background and
            25 for all other labels.
        mean_max (optional): List of upper bounds on the means drawn to generate
            the intensities for each label. Defaults to 225 for each label.
        std_min (optional): List of lower bounds on the SDs drawn to generate
            the intensities for each label. Defaults to 0 for the background and
            5 for all other labels.
        std_max (optional): List of upper bounds on the SDs drawn to generate
            the intensities for each label. Defaults to 25 for each label.
            25 for all other labels.
        zero_background (float, optional): Probability that the background is set
            to zero. Defaults to 0.2.
        warp_res (optional): List of factors N determining the
            resultion 1/N relative to the inputs at which the SVF is drawn.
            Defaults to 16.
        warp_std (float, optional): Upper bound on the SDs used when drawing
            the SVF. Defaults to 0.5.
        warp_modulate (bool, optional): Whether to draw the SVF with random SDs.
            If disabled, each batch will use the maximum SD. Defaults to True.
        bias_res (optional): List of factors N determining the
            resultion 1/N relative to the inputs at which the bias field is
            drawn. Defaults to 40.
        bias_std (float, optional): Upper bound on the SDs used when drawing
            the bias field. Defaults to 0.3.
        bias_modulate (bool, optional): Whether to draw the bias field with
            random SDs. If disabled, each batch will use the maximum SD.
            Defaults to True.
        blur_std (float, optional): Upper bound on the SD of the kernel used
            for Gaussian image blurring. Defaults to 1.
        blur_modulate (bool, optional): Whether to draw random blurring SDs.
            If disabled, each batch will use the maximum SD. Defaults to True.
        normalize (bool, optional): Whether the image is min-max normalized.
            Defaults to True.
        gamma_std (float, optional): SD of random global intensity
            exponentiation, i.e. gamma augmentation. Defaults to 0.25.
        dc_offset (float, optional): Upper bound on global DC offset drawn and
            added to the image after normalization. Defaults to 0.
        one_hot (bool, optional): Whether output label maps are one-hot encoded.
            Only the specified output labels will be included. Defaults to True.
        seeds (dictionary, optional): Integers for reproducible randomization.
        return_vel (bool, optional): Whether to append the half-resolution SVF
            to the model outputs. Defaults to False.
        return_def (bool, optional): Whether to append the combined displacement
            field to the model outputs. Defaults to False.
        id (int, optional): Model identifier used to create unique layer names
            for including this model multiple times. Defaults to 0.

    Returns:
        Label-augmentation and image-synthesis model.

    If you find this model useful, please cite:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879

        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    warnings.warn('model `labels_to_image_old` is deprecated in favor `labels_to_image`')

    import voxelmorph as vxm

    if out_shape is None:
        out_shape = in_shape
    in_shape, out_shape = map(np.asarray, (in_shape, out_shape))
    num_dim = len(in_shape)

    # Inputs.
    if input_model is None:
        labels_input = KL.Input(shape=(*in_shape, 1), name=f'labels_input_{id}')
        labels = labels_input
    else:
        assert len(input_model.outputs) == 1
        labels = input_model.outputs[0]
        labels_input = input_model.inputs

    if not labels.dtype.is_integer:
        labels = tf.cast(labels, tf.int32)
    batch_size = tf.shape(labels)[0]

    # Transform labels into [0, 1, ..., N-1].
    in_label_list = np.int32(np.unique(in_label_list))
    num_in_labels = len(in_label_list)
    new_in_label_list = np.arange(num_in_labels)
    in_lut = np.zeros(np.max(in_label_list) + 1, dtype=np.float32)
    for i, lab in enumerate(in_label_list):
        in_lut[lab] = i
    labels = tf.gather(in_lut, indices=labels)

    if warp_std > 0:
        # Velocity field.
        vel_shape = (*out_shape // 2, num_dim)
        vel_scale = np.asarray(warp_res) / 2
        vel_draw = lambda x: utils.augment.draw_perlin(
            vel_shape, scales=vel_scale,
            min_std=0 if warp_modulate else warp_std, max_std=warp_std,
            seed=seeds.get('warp')
        )
        # One per batch.
        vel_field = KL.Lambda(lambda x: tf.map_fn(
            vel_draw, x, fn_output_signature='float32'), name=f'vel_{id}')(labels)
        # Deformation field.
        def_field = vxm.layers.VecInt(int_steps=5, name=f'vec_int_{id}')(vel_field)
        def_field = layers.RescaleValues(2)(def_field)
        def_field = layers.Resize(2, interp_method='linear', name=f'def_{id}')(def_field)
        # Resampling.
        labels = vxm.layers.SpatialTransformer(
            interp_method='nearest', fill_value=0, name=f'trans_{id}')([labels, def_field])

    labels = tf.cast(labels, tf.int32)

    # Intensity means and standard deviations.
    if mean_min is None:
        mean_min = [0] + [25] * (num_in_labels - 1)
    if mean_max is None:
        mean_max = [225] * num_in_labels
    if std_min is None:
        std_min = [0] + [5] * (num_in_labels - 1)
    if std_max is None:
        std_max = [25] * num_in_labels
    m0, m1, s0, s1 = map(np.asarray, (mean_min, mean_max, std_min, std_max))
    mean = tf.random.uniform(
        shape=(batch_size, num_chan, num_in_labels),
        minval=m0, maxval=m1,
        seed=seeds.get('mean'),
    )
    std = tf.random.uniform(
        shape=(batch_size, num_chan, num_in_labels),
        minval=s0, maxval=s1,
        seed=seeds.get('std'),
    )

    # Synthetic image.
    image = tf.random.normal(tf.shape(labels), seed=seeds.get('noise'))
    indices = tf.concat([labels + i * num_in_labels for i in range(num_chan)], axis=-1)
    gather = lambda x: tf.gather(tf.reshape(x[0], (-1,)), x[1])
    mean = KL.Lambda(lambda x: tf.map_fn(gather, x, fn_output_signature='float32'))([mean, indices])
    std = KL.Lambda(lambda x: tf.map_fn(gather, x, fn_output_signature='float32'))([std, indices])
    image = image * std + mean

    # Zero background.
    if zero_background > 0:
        rand_flip = tf.random.uniform(
            shape=(batch_size, *[1] * num_dim, num_chan), seed=seeds.get('background'),
        )
        rand_flip = tf.less(rand_flip, zero_background)
        image *= 1. - tf.cast(tf.logical_and(labels == 0, rand_flip), image.dtype)

    # Blur.
    if blur_std > 0:
        kernels = utils.gaussian_kernel(
            [blur_std] * num_dim, separate=True, random=blur_modulate,
            dtype=image.dtype, seed=seeds.get('blur'),
        )
        image = utils.separable_conv(image, kernels, batched=True)

    # Bias field.
    if bias_std > 0:
        bias_shape = (*out_shape, 1)
        bias_draw = lambda x: utils.augment.draw_perlin(
            bias_shape, scales=bias_res, seed=seeds.get('bias'),
            min_std=0 if bias_modulate else bias_std, max_std=bias_std,
        )
        bias_field = KL.Lambda(lambda x: tf.map_fn(
            bias_draw, x, fn_output_signature='float32'))(labels)
        image *= tf.exp(bias_field, name=f'bias_{id}')

    # Intensity manipulations.
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255, name=f'clip_{id}')
    if normalize:
        image = KL.Lambda(lambda x: tf.map_fn(utils.minmax_norm, x))(image)
    if gamma_std > 0:
        gamma = tf.random.normal(
            shape=(batch_size, *[1] * num_dim, num_chan), stddev=gamma_std, seed=seeds.get('gamma'),
        )
        image = tf.pow(image, tf.exp(gamma), name=f'gamma_{id}')
    if dc_offset > 0:
        image += tf.random.uniform(
            shape=(batch_size, *[1] * num_dim, num_chan),
            maxval=dc_offset,
            seed=seeds.get('dc_offset'),
        )

    # Lookup table for converting the index labels back to the original values,
    # setting unwanted labels to background. If the output labels are provided
    # as a dictionary, it can be used e.g. to convert labels to GM, WM, CSF.
    if out_label_list is None:
        out_label_list = in_label_list
    if isinstance(out_label_list, (tuple, list, np.ndarray)):
        out_label_list = {lab: lab for lab in out_label_list}
    out_lut = np.zeros(num_in_labels, dtype='int32')
    for i, lab in enumerate(in_label_list):
        if lab in out_label_list:
            out_lut[i] = out_label_list[lab]

    # For one-hot encoding, update the lookup table such that the M desired
    # output labels are rebased into the interval [0, M-1[. If the background
    # with value 0 is not part of the output labels, set it to -1 to remove it
    # from the one-hot maps.
    if one_hot:
        hot_label_list = np.unique(list(out_label_list.values()))  # Sorted.
        hot_lut = np.full(hot_label_list[-1] + 1, fill_value=-1, dtype='int32')
        for i, lab in enumerate(hot_label_list):
            hot_lut[lab] = i
        out_lut = hot_lut[out_lut]

    # Convert indices to output labels only once.
    labels = tf.gather(out_lut, labels, name=f'labels_back_{id}')
    if one_hot:
        labels = tf.one_hot(labels[..., 0], depth=len(hot_label_list), name=f'one_hot_{id}')

    outputs = [image, labels]
    if return_vel:
        outputs.append(vel_field)
    if return_def:
        outputs.append(def_field)

    return tf.keras.Model(labels_input, outputs, name=f'synth_{id}')


def labels_to_image_new(*args, **kwargs):
    warnings.warn('model `labels_to_image_new` has been renamed to `labels_to_image` and will be '
                  'removed in the future')
    return labels_to_image(*args, **kwargs)


def labels_to_image(
    labels_in,
    labels_out=None,
    in_shape=None,
    out_shape=None,
    input_model=None,
    num_chan=1,
    aff_shift=0,
    aff_rotate=0,
    aff_scale=0,
    aff_shear=0,
    aff_normal_shift=False,
    aff_normal_rotate=False,
    aff_normal_scale=False,
    aff_normal_shear=False,
    axes_flip=False,
    axes_swap=False,
    warp_min=0.01,
    warp_max=2,
    warp_blur_min=(8, 8),
    warp_blur_max=(32, 32),
    warp_zero_mean=False,
    crop_min=0,
    crop_max=0.2,
    crop_prob=0,
    crop_axes=None,
    mean_min=None,
    mean_max=None,
    noise_min=0.1,
    noise_max=0.2,
    zero_background=0,
    blur_min=0,
    blur_max=1,
    bias_min=0.01,
    bias_max=0.1,
    bias_blur_min=32,
    bias_blur_max=64,
    bias_func=tf.exp,
    slice_stride_min=1,
    slice_stride_max=8,
    slice_prob=0,
    slice_axes=None,
    normalize=True,
    gamma=0.5,
    one_hot=True,
    half_res=False,
    seeds={},
    return_im=True,
    return_map=True,
    return_vel=False,
    return_def=False,
    return_aff=False,
    return_mean=False,
    return_bias=False,
    id=0,
):
    """Build model that augments label maps and synthesizes images from them.

    Parameters:
        labels_in: All possible input label values as an iterable. Passing a dictionary will remap
            input labels to the generation labels defined by the dictionary values, for example to
            draw the same intensity for left and right cortex. Generation labels must be hashable
            but not necessarily numeric. Remapping has no effect on the output labels.
        labels_out: Subset of the input labels to include in the output label maps, as an iterable.
            Passing a dictionary will remap the included input labels, for example to GM, WM, and
            CSF. Any input label missing in `labels_out` will be set to 0 (background) or excluded
            if one-hot encoding. None means the output labels will be the same as the input labels.
            Output labels must be numeric.
        in_shape: Spatial dimensions of the input label maps as an iterable.
        out_shape: Spatial dimensions of the outputs as an iterable. Inputs will be symmetrically
            cropped or zero-padded to fit. None means `in_shape`.
        input_model: Model whose outputs will be used as data inputs, and whose inputs will be used
            as inputs to the returned model.
        num_chan: Number of image channels to synthesize.
        aff_shift: Upper bound on the magnitude of translations, drawn uniformly in voxels.
        aff_rotate: Upper bound on the magnitude of rotations, drawn uniformly in degrees.
        aff_scale: Upper bound on the difference of the scaling magnitude from 1, drawn uniformly.
        aff_shear: Upper bound on the shearing magnitude, drawn uniformly.
        aff_normal_shift: Sample translations normally, with `aff_shift` as SD.
        aff_normal_rotate: Sample rotation angles normally, with `aff_rotate` as SD.
        aff_normal_scale: Sample scaling normally, with `aff_scale` as SD, truncating beyond 2 SDs.
        aff_normal_shear: Sample shearing normally, with `aff_shear` as SD.
        axes_flip: Randomly flip axes of the label map before generating the image. Should not be
            used with lateralized labels.
        axes_swap: Randomly permute axes of the label map before generating the image. Should not
            be used with lateralized labels. Requires an isotropic output shape.
        warp_min: Lower bound on the SDs used when drawing the SVF.
        warp_max: Upper bound on the SDs used when drawing the SVF.
        warp_zero_mean: Ensure that the SVF components have zero mean.
        crop_min: Lower bound on the proportion of the FOV to crop.
        crop_max: Upper bound on the proportion of the FOV to crop.
        crop_prob: Probability that we crop the FOV along an axis.
        crop_axes: Axes from which to draw for FOV cropping. None means all spatial axes.
        mean_min: List of lower bounds on the intensities drawn for each label. None means 0.
        mean_max: List of upper bounds on the intensities drawn for each label. None means 1.
        noise_min: Lower bound on the noise SD relative to the max image intensity, 0.1 means 10%.
        noise_max: Upper bound on the noise SD relative to the max image intensity, 0.1 means 10%.
        zero_background: Probability that we set the background to zero.
        blur_min: Lower bound on the smoothing SDs. Can be a scalar or list.
        blur_max: Upper bound on the smoothing SDs. Can be a scalar or list.
        bias_min: Lower bound on the bias sampling SDs.
        bias_max: Upper bound on the bias sampling SDs.
        bias_blur_min: Lower bound on the bias smoothing FWHM.
        bias_blur_max: Upper bound on the bias smoothing FWHM.
        bias_func: Function applied voxel-wise to condition the bias field.
        slice_stride_min: Lower bound on slice thickness in original voxel units.
        slice_stride_max: Upper bound on slice thickness in original voxel units.
        slice_prob: Probability that we subsample to create thick slices.
        slice_axes: Axes from which to draw slice normal direction. None means all spatial axes.
        normalize: Min-max normalize the image.
        gamma: Maximum deviation of the gamma augmentation exponent from 1. Careful: without
            normalization, you may attempt to compute random exponents of negative numbers.
        one_hot: One-hot encode the output label map.
        seeds: Seeds for reproducible randomization or synchronization of components of this model
            across multiple instances. Pass a dictionary linking specific component names to integer
            seeds. If you pass an iterable of component names, seeds will be auto-generated. You
            will have to check the source code to identify component names.
        return_vel: Return the half-resolution SVF.
        return_def: Return the combined displacement field.
        return_aff: Return the (N+1)-by-(N+1) affine transformation matrix.
        return_mean: Return an uncorrupted copy of the image.
        return_bias: Return the applied bias field.
        id: Model identifier used to create unique layer names.

    Returns:
        Label-augmentation and image-synthesis model.

    If you find this model useful, please cite:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879

        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    import voxelmorph as vxm

    # Compute type.
    compute_type = tf.keras.mixed_precision.global_policy().compute_dtype
    compute_type = tf.dtypes.as_dtype(compute_type)
    integer_type = tf.int32

    # Seeds.
    if isinstance(seeds, str):
        seeds = [seeds]
    if isinstance(seeds, dict):
        seeds = seeds.copy()
    if not isinstance(seeds, dict):
        seeds = {f: hash(f) for f in seeds}

    # Input model.
    if input_model is None:
        labels = KL.Input(shape=(*in_shape, 1), name=f'input_{id}', dtype=compute_type)
        input_model = tf.keras.Model(*[labels] * 2)
    labels = input_model.output
    if labels.dtype != compute_type:
        labels = tf.cast(labels, compute_type)

    # Dimensions.
    in_shape = np.asarray(labels.shape[1:-1])
    if out_shape is None:
        out_shape = in_shape
    out_shape = np.array(out_shape) // (2 if half_res else 1)
    num_dim = len(in_shape)
    batch_size = tf.shape(labels)[0]

    # Affine transform.
    parameters = vxm.layers.DrawAffineParams(
        shift=aff_shift,
        rot=aff_rotate,
        scale=aff_scale,
        shear=aff_shear,
        normal_shift=aff_normal_shift,
        normal_rot=aff_normal_rotate,
        normal_scale=aff_normal_scale,
        normal_shear=aff_normal_shear,
        ndims=num_dim,
        out_type=compute_type,
        seeds={t: seeds.pop(t, None) for t in ('shift', 'rot', 'scale', 'shear')},
    )(labels)
    affine = vxm.layers.ParamsToAffineMatrix(
        ndims=num_dim, deg=True, shift_scale=True, last_row=True,
    )(parameters)

    # Shift origin to rotate about image center.
    origin = np.eye(num_dim + 1)
    origin[:num_dim, -1] = -0.5 * (in_shape - 1)

    # Symmetric padding: center output volume within input volume.
    center = np.eye(num_dim + 1)
    center[:num_dim, -1] = np.round(0.5 * (in_shape - (2 if half_res else 1) * out_shape))

    # Apply transform in input space to avoid rescaling translations at half resolution.
    scale = np.diag((*[2 if half_res else 1] * num_dim, 1))
    trans = np.linalg.inv(origin) @ affine @ origin @ center @ scale

    # Axis randomization.
    if axes_flip:
        trans = KL.Lambda(lambda x: x @ vxm.utils.draw_flip_matrix(
            out_shape, shift_center=False, dtype=compute_type, seed=seeds.pop('flip', None),
        ))(trans)
    if axes_swap:
        assert all(x == out_shape[0] for x in out_shape), 'non-isotropic output shape'
        trans = KL.Lambda(lambda x: x @ vxm.utils.draw_swap_matrix(
            num_dim, dtype=compute_type, seed=seeds.pop('swap', None),
        ))(trans)

    # Convert to dense transform.
    trans = vxm.layers.AffineToDenseShift(out_shape, shift_center=False)(trans[:, :num_dim, :])

    # Diffeomorphic deformation.
    if warp_max > 0:
        vel_field = layers.PerlinNoise(
            shape=(*out_shape // (1 if half_res else 2), num_dim),
            noise_min=warp_min,
            noise_max=warp_max,
            isotropic=False,
            fwhm_min=np.asarray(warp_blur_min) / 2,
            fwhm_max=np.asarray(warp_blur_max) / 2,
            reduce=tf.math.reduce_max,
            axes=-1,
            out_type=compute_type,
            seed=seeds.pop('warp', None),
        )(labels)
        if warp_zero_mean:
            vel_field -= tf.reduce_mean(vel_field, axis=range(1, num_dim + 1))
        def_field = vxm.layers.VecInt(int_steps=5, name=f'vec_int_{id}')(vel_field)
        if not half_res:
            def_field = vxm.layers.RescaleTransform(zoom_factor=2, name=f'def_{id}')(def_field)

        # Compose with affine.
        trans = vxm.layers.ComposeTransform()([trans, def_field])

    # Apply transform.
    labels = vxm.layers.SpatialTransformer(
        interp_method='nearest', fill_value=0, name=f'trans_{id}',
    )([labels, trans])
    labels = tf.cast(labels, integer_type)

    # Cropping.
    labels = layers.RandomCrop(
        crop_min=crop_min,
        crop_max=crop_max,
        prob=crop_prob,
        axis=crop_axes,
        seed=seeds.pop('crop', None),
    )(labels)

    # Generation labels. Default to all input labels.
    if not isinstance(labels_in, dict):
        labels_in = {i: i for i in labels_in}
    labels_gen = set(labels_in.values())

    # Rebase into [0, N): LUT from input to generation label to index.
    ind = {gen: i for i, gen in enumerate(labels_gen)}
    lut = [ind.get(labels_in.get(i), 0) for i in range(max(labels_in) + 1)]
    lut = tf.cast(lut, integer_type)
    indices = tf.gather(lut, indices=labels)

    # Intensity means and standard deviations.
    num_label = len(labels_gen)
    if mean_min is None:
        mean_min = [0] * num_label
    if mean_max is None:
        mean_max = [1] * num_label
    mean = tf.random.uniform(
        shape=(batch_size, num_chan, num_label),
        minval=mean_min,
        maxval=mean_max,
        dtype=compute_type,
        seed=seeds.pop('mean', None),
    )

    # Label intensities.
    off_chan = tf.range(num_chan) * num_label
    off_batch = tf.range(batch_size) * num_chan * num_label
    indices += tf.reshape(off_batch, shape=(-1, *[1] * num_dim, 1)) + off_chan
    mean = tf.gather(tf.reshape(mean, shape=(-1,)), indices=indices)
    image = mean

    # Bias field.
    if bias_max > 0:
        bias_field = layers.PerlinNoise(
            noise_min=bias_min,
            noise_max=bias_max,
            isotropic=False,
            fwhm_min=bias_blur_min / (2 if half_res else 1),
            fwhm_max=bias_blur_max / (2 if half_res else 1),
            reduce=tf.math.reduce_max,
            out_type=compute_type,
            seed=seeds.pop('bias', None),
        )(image)
        bias_field = KL.Lambda(bias_func, name=f'bias_func_{id}')(bias_field)
        image *= bias_field

    # Noise.
    image = layers.GaussianNoise(noise_min, noise_max, seed=seeds.pop('noise', None))(image)

    # Background clearing.
    if zero_background > 0:
        bg_rand = tf.random.uniform(
            shape=(batch_size, *[1] * num_dim, 1),
            dtype=compute_type,
            seed=seeds.pop('background', None),
        )
        bg_zero = tf.math.less(bg_rand, zero_background)
        bg_zero = tf.math.logical_and(tf.equal(labels, 0), bg_zero)
        bg_zero = tf.math.logical_xor(True, bg_zero)
        image *= tf.cast(bg_zero, compute_type)

    # Blur.
    image = layers.GaussianBlur(
        sigma=blur_max,
        min_sigma=blur_min,
        random=True,
        seed=seeds.pop('blur', None),
    )(image)

    # Create thick slices.
    image = layers.Subsample(
        prob=slice_prob,
        stride_min=max(1, slice_stride_min / (2 if half_res else 1)),
        stride_max=max(1, slice_stride_max / (2 if half_res else 1)),
        axes=slice_axes,
        seed=seeds.pop('slice', None),
    )(image)

    # Intensity manipulations.
    if normalize:
        image = KL.Lambda(lambda x: tf.map_fn(utils.minmax_norm, x))(image)
    if gamma > 0:
        assert 0 < gamma < 1, f'gamma value {gamma} outside interval [0, 1)'
        gamma = tf.random.uniform(
            shape=(batch_size, *[1] * num_dim, num_chan),
            minval=1 - gamma,
            maxval=1 + gamma,
            dtype=image.dtype,
            seed=seeds.pop('gamma', None),
        )
        image = tf.pow(image, gamma)

    # Output labels. Recode the original input labels if desired.
    lut = list(labels_in) if labels_out is None else labels_out
    if not isinstance(lut, dict):
        lut = {i: i for i in lut}
    labels_out = set(lut.values())

    # Rebase the M desired output labels into [0, M): LUT from input to index.
    if one_hot:
        ind = {out: i for i, out in enumerate(labels_out)}
        lut = {inp: ind[out] for inp, out in lut.items()}

    # Conversion. Set labels to -1 to exclude them from the one-hot maps.
    if any(k != lut[k] for k in lut) or set(labels_in) - set(lut):
        lut = [lut.get(i, -1 if one_hot else 0) for i in range(max(labels_in) + 1)]
        lut = tf.cast(lut, integer_type)
        labels = tf.gather(lut, indices=labels)

    if one_hot:
        labels = tf.one_hot(labels[..., 0], depth=len(labels_out), dtype=compute_type)

    outputs = []
    if return_im:
        outputs.append(image)
    if return_map:
        outputs.append(labels)
    if return_vel:
        outputs.append(vel_field)
    if return_def:
        outputs.append(def_field)
    if return_aff:
        outputs.append(affine)
    if return_mean:
        outputs.append(mean)
    if return_bias:
        outputs.append(bias_field)

    assert not seeds, f'unknown seeds {seeds}'
    return tf.keras.Model(input_model.inputs, outputs)


###############################################################################
# Encoders, decoders, etc.
###############################################################################


def conv_enc(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             dilation_rate_mult=1,
             padding='same',
             activation='elu',
             layer_nb_feats=None,
             use_residuals=False,
             nb_conv_per_level=2,
             conv_dropout=0,
             batch_norm=None,
             convL=None,  # conv layer function
             src=None,
             src_input=None):
    """
    Fully Convolutional Encoder
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # volume size data
    ndims = len(input_shape) - 1
    input_shape = tuple(input_shape)
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # prepare layers
    if convL is None:
        convL = getattr(KL, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation}
    maxpool = getattr(KL, 'MaxPooling%dD' % ndims)

    # first layer: input
    if src is None:
        name = '%s_input' % prefix
        last_tensor = KL.Input(shape=input_shape, name=name)
        input_tensor = last_tensor
    else:
        assert src_input is not None, 'need to provide src_input if given src'
        input_tensor = src_input
        last_tensor = src

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    lfidx = 0
    for level in range(nb_levels):
        lvl_first_tensor = last_tensor

        # if nb_features is a list of lists use it to specify the # of levels
        # and the number of filters at each level (in each sublist)
        if isinstance(nb_features, list):
            lfidx = 0
            if isinstance(nb_features[level], list):
                layer_nb_feats = nb_features[level]
                nb_conv_per_level = len(layer_nb_feats)
            else:  # not supported yet
                nb_lvl_feats = nb_conv_per_level * [nb_features[level]]
        else:
            nb_lvl_feats = np.round(nb_features * feat_mult**level).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult**level

        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_downarm_%d_%d' % (prefix, level, conv)
            if conv < (nb_conv_per_level - 1) or (not use_residuals):
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:  # no activation
                last_tensor = convL(nb_lvl_feats, conv_size,
                                    padding=padding, name=name)(last_tensor)

            if conv_dropout > 0:
                # conv dropout along feature space only
                name = '%s_dropout_downarm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                versions = tf.__version__.split('.')
                ver = int(versions[0])
                rev = int(versions[1])
                if ver < 2 or (ver == 2 and rev < 2):  # < 2.2
                    noise_shape = None
                last_tensor = KL.Dropout(conv_dropout, noise_shape=noise_shape)(last_tensor)

        if use_residuals:
            convarm_layer = last_tensor

            # the "add" layer is the original input
            # However, it may not have the right number of features to be added
            nb_feats_in = lvl_first_tensor.get_shape()[-1]
            nb_feats_out = convarm_layer.get_shape()[-1]
            add_layer = lvl_first_tensor
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_down_merge_%d' % (prefix, level)
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs,
                                    name=name)(lvl_first_tensor)
                add_layer = last_tensor

                if conv_dropout > 0:
                    name = '%s_dropout_down_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                    versions = tf.__version__.split('.')
                    ver = int(versions[0])
                    rev = int(versions[1])
                    if ver < 2 or (ver == 2 and rev < 2):  # < 2.2
                        noise_shape = None
                    last_tensor = KL.Dropout(conv_dropout, noise_shape=noise_shape)(last_tensor)

            name = '%s_res_down_merge_%d' % (prefix, level)
            last_tensor = KL.add([add_layer, convarm_layer], name=name)

            name = '%s_res_down_merge_act_%d' % (prefix, level)
            last_tensor = KL.Activation(activation, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_bn_down_%d' % (prefix, level)
            last_tensor = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

        # max pool if we're not at the last level
        if level < (nb_levels - 1):
            name = '%s_maxpool_%d' % (prefix, level)
            last_tensor = maxpool(pool_size=pool_size, name=name, padding=padding)(last_tensor)

    # create the model and return
    model = Model(inputs=input_tensor, outputs=[last_tensor], name=model_name)
    return model


def conv_dec(nb_features,
             input_shape,
             nb_levels,
             conv_size,
             nb_labels,
             name=None,
             prefix=None,
             feat_mult=1,
             pool_size=2,
             use_skip_connections=False,
             padding='same',
             dilation_rate_mult=1,
             activation='elu',
             use_residuals=False,
             final_pred_activation='softmax',
             nb_conv_per_level=2,
             layer_nb_feats=None,
             batch_norm=None,
             conv_dropout=0,
             convL=None,
             input_model=None):
    """
    Fully Convolutional Decoder

    Parameters:
        ...
        use_skip_connections (bool): if true, turns an Enc-Dec to a U-Net.
            If true, input_tensor and tensors are required.
            It assumes a particular naming of layers. conv_enc...
    """

    # naming
    model_name = name
    if prefix is None:
        prefix = model_name

    # if using skip connections, make sure need to use them.
    if use_skip_connections:
        assert input_model is not None, "is using skip connections, tensors dictionary is required"

    # first layer: input
    input_name = '%s_input' % prefix
    if input_model is None:
        input_tensor = KL.Input(shape=input_shape, name=input_name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.input
        last_tensor = input_model.output
        input_shape = last_tensor.shape.as_list()[1:]

    # vol size info
    ndims = len(input_shape) - 1
    input_shape = tuple(input_shape)
    if isinstance(pool_size, int):
        if ndims > 1:
            pool_size = (pool_size,) * ndims
    if ndims == 1 and isinstance(pool_size, tuple):
        pool_size = pool_size[0]  # 1D upsampling takes int not tuple

    # prepare layers
    if convL is None:
        convL = getattr(KL, 'Conv%dD' % ndims)
    conv_kwargs = {'padding': padding, 'activation': activation}
    upsample = getattr(KL, 'UpSampling%dD' % ndims)

    # up arm:
    # nb_levels - 1 layers of Deconvolution3D
    #    (approx via up + conv + ReLu) + merge + conv + ReLu + conv + ReLu
    lfidx = 0
    for level in range(nb_levels - 1):
        # if nb_features is a list of lists use it to specify the # of levels
        # and the number of filters at each level (in each sublist)
        if isinstance(nb_features, list):
            lfidx = 0  # this will index into list of num filters
            lindex = nb_levels - level - 2
            if isinstance(nb_features[lindex], list):
                layer_nb_feats = nb_features[lindex]
                nb_conv_per_level = len(layer_nb_feats)
            else:  # not supported yet
                nb_lvl_feats = nb_features[lindex]
        else:
            nb_lvl_feats = np.round(nb_features * feat_mult**(nb_levels - 2 - level)).astype(int)
        conv_kwargs['dilation_rate'] = dilation_rate_mult**(nb_levels - 2 - level)

        # upsample matching the max pooling layers size
        name = '%s_up_%d' % (prefix, nb_levels + level)
        last_tensor = upsample(size=pool_size, name=name)(last_tensor)
        up_tensor = last_tensor

        # merge layers combining previous layer
        # TODO: add Cropping3D or Cropping2D if 'valid' padding
        if use_skip_connections:
            conv_name = '%s_conv_downarm_%d_%d' % (
                prefix, nb_levels - 2 - level, nb_conv_per_level - 1)
            cat_tensor = input_model.get_layer(conv_name).output

            name = '%s_merge_%d' % (prefix, nb_levels + level)
            last_tensor = KL.concatenate([cat_tensor, last_tensor], axis=ndims + 1, name=name)

        # convolution layers
        for conv in range(nb_conv_per_level):
            if layer_nb_feats is not None:
                nb_lvl_feats = layer_nb_feats[lfidx]
                lfidx += 1

            name = '%s_conv_uparm_%d_%d' % (prefix, nb_levels + level, conv)
            if conv < (nb_conv_per_level - 1) or (not use_residuals):
                last_tensor = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(last_tensor)
            else:
                last_tensor = convL(nb_lvl_feats, conv_size,
                                    padding=padding, name=name)(last_tensor)

            if conv_dropout > 0:
                name = '%s_dropout_uparm_%d_%d' % (prefix, level, conv)
                noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                versions = tf.__version__.split('.')
                ver = int(versions[0])
                rev = int(versions[1])
                if ver < 2 or (ver == 2 and rev < 2):  # < 2.2
                    noise_shape = None
                last_tensor = KL.Dropout(conv_dropout, noise_shape=noise_shape)(last_tensor)

        # residual block
        if use_residuals:

            # the "add" layer is the original input
            # However, it may not have the right number of features to be added
            add_layer = up_tensor
            nb_feats_in = add_layer.get_shape()[-1]
            nb_feats_out = last_tensor.get_shape()[-1]
            if nb_feats_in > 1 and nb_feats_out > 1 and (nb_feats_in != nb_feats_out):
                name = '%s_expand_up_merge_%d' % (prefix, level)
                add_layer = convL(nb_lvl_feats, conv_size, **conv_kwargs, name=name)(add_layer)

                if conv_dropout > 0:
                    name = '%s_dropout_up_merge_%d_%d' % (prefix, level, conv)
                    noise_shape = [None, *[1] * ndims, nb_lvl_feats]
                    last_tensor = KL.Dropout(conv_dropout, noise_shape=noise_shape)(last_tensor)

            name = '%s_res_up_merge_%d' % (prefix, level)
            last_tensor = KL.add([last_tensor, add_layer], name=name)

            name = '%s_res_up_merge_act_%d' % (prefix, level)
            last_tensor = KL.Activation(activation, name=name)(last_tensor)

        if batch_norm is not None:
            name = '%s_bn_up_%d' % (prefix, level)
            last_tensor = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)

    # Compute likelyhood prediction (no activation yet)
    name = '%s_likelihood' % prefix
    last_tensor = convL(nb_labels, 1, activation=None, name=name)(last_tensor)
    like_tensor = last_tensor

    # output prediction layer
    # we use a softmax to compute P(L_x|I) where x is each location
    if final_pred_activation == 'softmax':
        print("using final_pred_activation %s for %s" % (final_pred_activation, model_name))
        name = '%s_prediction' % prefix
        softmax_lambda_fcn = lambda x: tf.keras.activations.softmax(x, axis=ndims + 1)
        pred_tensor = KL.Lambda(softmax_lambda_fcn, name=name)(last_tensor)

    # otherwise create a layer that does nothing.
    else:
        name = '%s_prediction' % prefix
        if final_pred_activation is None:
            pred_tensor = KL.Activation('linear', name=name)(like_tensor)
        else:
            pred_tensor = KL.Activation(final_pred_activation, name=name)(like_tensor)

    # create the model and retun
    model = Model(inputs=input_tensor, outputs=pred_tensor, name=model_name)
    return model


def design_dnn(nb_features, input_shape, nb_levels, conv_size, nb_labels,
               feat_mult=1,
               pool_size=2,
               padding='same',
               activation='elu',
               final_layer='dense-sigmoid',
               conv_dropout=0,
               conv_maxnorm=0,
               nb_input_features=1,
               batch_norm=False,
               name=None,
               prefix=None,
               use_strided_convolution_maxpool=True,
               nb_conv_per_level=2):
    """
    "deep" cnn with dense or global max pooling layer @ end...

    Could use sequential...
    """

    def _global_max_nd(xtens):
        ytens = K.batch_flatten(xtens)
        return K.max(ytens, 1, keepdims=True)

    model_name = name
    if model_name is None:
        model_name = 'model_1'
    if prefix is None:
        prefix = model_name

    ndims = len(input_shape)
    input_shape = tuple(input_shape)

    convL = getattr(KL, 'Conv%dD' % ndims)
    maxpool = KL.MaxPooling3D if len(input_shape) == 3 else KL.MaxPooling2D
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * ndims

    # kwargs for the convolution layer
    conv_kwargs = {'padding': padding, 'activation': activation}
    if conv_maxnorm > 0:
        conv_kwargs['kernel_constraint'] = maxnorm(conv_maxnorm)

    # initialize a dictionary
    enc_tensors = {}

    # first layer: input
    name = '%s_input' % prefix
    enc_tensors[name] = KL.Input(shape=input_shape + (nb_input_features,), name=name)
    last_tensor = enc_tensors[name]

    # down arm:
    # add nb_levels of conv + ReLu + conv + ReLu. Pool after each of first nb_levels - 1 layers
    for level in range(nb_levels):
        for conv in range(nb_conv_per_level):
            if conv_dropout > 0:
                name = '%s_dropout_%d_%d' % (prefix, level, conv)
                enc_tensors[name] = KL.Dropout(conv_dropout)(last_tensor)
                last_tensor = enc_tensors[name]

            name = '%s_conv_%d_%d' % (prefix, level, conv)
            nb_lvl_feats = np.round(nb_features * feat_mult**level).astype(int)
            enc_tensors[name] = convL(nb_lvl_feats, conv_size, **
                                      conv_kwargs, name=name)(last_tensor)
            last_tensor = enc_tensors[name]

        # max pool
        if use_strided_convolution_maxpool:
            name = '%s_strided_conv_%d' % (prefix, level)
            enc_tensors[name] = convL(nb_lvl_feats, pool_size, **
                                      conv_kwargs, name=name)(last_tensor)
            last_tensor = enc_tensors[name]
        else:
            name = '%s_maxpool_%d' % (prefix, level)
            enc_tensors[name] = maxpool(pool_size=pool_size, name=name,
                                        padding=padding)(last_tensor)
            last_tensor = enc_tensors[name]

    # dense layer
    if final_layer == 'dense-sigmoid':

        name = "%s_flatten" % prefix
        enc_tensors[name] = KL.Flatten(name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_dense' % prefix
        enc_tensors[name] = KL.Dense(1, name=name, activation="sigmoid")(last_tensor)

    elif final_layer == 'dense-tanh':

        name = "%s_flatten" % prefix
        enc_tensors[name] = KL.Flatten(name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_dense' % prefix
        enc_tensors[name] = KL.Dense(1, name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        # Omittting BatchNorm for now, it seems to have a cpu vs gpu problem
        # https://github.com/tensorflow/tensorflow/pull/8906
        # https://github.com/fchollet/keras/issues/5802
        # name = '%s_%s_bn' % prefix
        # enc_tensors[name] = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)
        # last_tensor = enc_tensors[name]

        name = '%s_%s_tanh' % prefix
        enc_tensors[name] = KL.Activation(activation="tanh", name=name)(last_tensor)

    elif final_layer == 'dense-softmax':

        name = "%s_flatten" % prefix
        enc_tensors[name] = KL.Flatten(name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_dense' % prefix
        enc_tensors[name] = KL.Dense(nb_labels, name=name, activation="softmax")(last_tensor)

    # global max pooling layer
    elif final_layer == 'myglobalmaxpooling':

        name = '%s_batch_norm' % prefix
        enc_tensors[name] = KL.BatchNormalization(axis=batch_norm, name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_global_max_pool' % prefix
        enc_tensors[name] = KL.Lambda(_global_max_nd, name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_global_max_pool_reshape' % prefix
        enc_tensors[name] = KL.Reshape((1, 1), name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        # cannot do activation in lambda layer. Could code inside, but will do extra lyaer
        name = '%s_global_max_pool_sigmoid' % prefix
        enc_tensors[name] = KL.Conv1D(
            1, 1, name=name, activation="sigmoid", use_bias=True)(last_tensor)

    elif final_layer == 'globalmaxpooling':

        name = '%s_conv_to_featmaps' % prefix
        enc_tensors[name] = KL.Conv3D(2, 1, name=name, activation="relu")(last_tensor)
        last_tensor = enc_tensors[name]

        name = '%s_global_max_pool' % prefix
        enc_tensors[name] = KL.GlobalMaxPooling3D(name=name)(last_tensor)
        last_tensor = enc_tensors[name]

        # cannot do activation in lambda layer. Could code inside, but will do extra lyaer
        name = '%s_global_max_pool_softmax' % prefix
        enc_tensors[name] = KL.Activation('softmax', name=name)(last_tensor)

    last_tensor = enc_tensors[name]

    # create the model
    model = Model(inputs=[enc_tensors['%s_input' % prefix]], outputs=[last_tensor], name=model_name)
    return model


###############################################################################
# Experimental, might be deprecated
###############################################################################

def EncoderNet(nb_features,
               input_shape,
               nb_levels,
               conv_size,
               name=None,
               prefix=None,
               feat_mult=1,
               pool_size=2,
               dilation_rate_mult=1,
               padding='same',
               activation='elu',
               layer_nb_feats=None,
               use_residuals=False,
               nb_conv_per_level=2,
               conv_dropout=0,
               dense_size=256,
               nb_labels=2,
               final_activation=None,
               rescale=None,
               dropout=None,
               batch_norm=None):
    """
    Fully Convolutional Encoder-based classifer
    if nb_labels is 0 assume it is a regression net and use linear activation
    (if None specified)
    The end of the encoders/downsampling is flattened and attached to a dense
    layer with dense_size units, then to the nb_labels output nodes. For other
    parameters see conv_env documentation
    """

    # allocate the encoder arm
    enc_model = conv_enc(nb_features,
                         input_shape,
                         nb_levels,
                         conv_size,
                         name=name,
                         feat_mult=feat_mult,
                         pool_size=pool_size,
                         padding=padding,
                         activation=activation,
                         use_residuals=use_residuals,
                         nb_conv_per_level=nb_conv_per_level,
                         conv_dropout=conv_dropout,
                         batch_norm=batch_norm)

    # run the encoder outputs through a dense layer
    flat = KL.Flatten()(enc_model.outputs[0])
    if dropout is not None and dropout > 0:
        flat = KL.Dropout(dropout, name='dropout_flat')(flat)
    dense = KL.Dense(dense_size, name='dense')(flat)
    if dropout is not None and dropout > 0:
        dense = KL.Dropout(dropout, name='dropout_dense')(dense)

    if nb_labels <= 0:  # if labels <=0 assume a regression net
        nb_labels = 1
        if (final_activation is None):
            final_activation = 'linear'
    else:  # if labels>=1 assume a classification net
        if (final_activation is None):
            final_activation = 'softmax'

    if (rescale is not None):
        dense = layers.RescaleValues(rescale)(dense)
    out = KL.Dense(nb_labels, name='output_dense', activation=final_activation)(dense)
    model = tf.keras.models.Model(inputs=enc_model.inputs, outputs=out)

    return model


def DenseLayerNet(inshape, layer_sizes, nb_labels=2, activation='relu',
                  final_activation='softmax', dropout=None, batch_norm=None):
    """
    A densenet that connects a set of dense layers to  a classification
    output. 
    if nb_labels is 0 assume it is a regression net and use linear activation
    (if None specified)
    """
    warnings.warn('incorrect PEP8 naming. Will be deprecated')

    inputs = KL.Input(shape=inshape, name='input')
    prev_layer = KL.Flatten(name='flat_inputs')(inputs)
    # to prevent overfitting include some kernel and bias regularization
    kreg = tf.kerasregularizers.l1_l2(l1=1e-5, l2=1e-4)
    breg = tf.kerasregularizers.l2(1e-4)

    # connect the list of dense layers to each other
    for lno, layer_size in enumerate(layer_sizes):
        prev_layer = KL.Dense(layer_size, name='dense%d' % lno, activation=activation,
                              kernel_regularizer=kreg, bias_regularizer=breg)(prev_layer)
        if dropout is not None:
            prev_layer = KL.Dropout(dropout, name='dropout%d' % lno)(prev_layer)
        if batch_norm is not None:
            prev_layer = KL.BatchNormalization(name='BatchNorm%d' % lno)(prev_layer)

    # tie the previous dense layer to a onehot encoded output layer
    last_layer = KL.Dense(nb_labels, name='last_dense', activation=final_activation)(prev_layer)

    model = tf.kerasmodels.Model(inputs=inputs, outputs=last_layer)
    return model


###############################################################################
# Synthstrip networks
###############################################################################


class SynthStrip(modelio.LoadableModel):
    """
    SynthStrip model for learning a mask with arbitrary contrasts synthesized from label maps.
    """

    @modelio.store_config_args
    def __init__(self,
                 inshape,
                 labels_in,
                 labels_out,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 src_feats=1,
                 gen_args={}):
        """
        Parameters:
            inshape: Input image shape, e.g. (160, 160, 192).
            labels_in: List of all labels included in the training segmentations.
            labels_out: List of labels to encode in the output maps.
            nb_unet_features: U-Net features.
            nb_unet_levels: Number of U-Net levels.
            unet_feat_mult: U-Net feature multiplier.
            nb_unet_conv_per_level: Number of convolutions per U-Net level.
            gen_args: Keyword arguments passed to the internal generative model.
        """

        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        inshape = tuple(inshape)

        # take input label map and synthesize an image from it
        gen_model = labels_to_image(
            inshape,
            labels_in,
            labels_out,
            id=0,
            return_def=False,
            one_hot=False,
            **gen_args
        )

        synth_image, synth_labels = gen_model.outputs

        # build a unet to apply to the synthetic image and strip it
        unet_model = unet(
            nb_unet_features,
            (*inshape, 1),
            nb_unet_levels,
            ndims * (3,),
            1,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            final_pred_activation='linear'
        )

        unet_outputs = unet_model(synth_image)

        # output the prob of brain and the warped label map
        # so that the loss function can compute brain/nonbrain
        stacked_output = KL.Concatenate(axis=-1, name='image_label')(
            [unet_outputs, tf.cast(synth_labels, tf.float32)])

        super().__init__(inputs=gen_model.inputs, outputs=stacked_output)

        # cache pointers to important layers and tensors for future reference
        self.references = modelio.LoadableModel.ReferenceContainer()
        self.references.unet = unet_model     # this is the stripping net
        self.references.gen_model = gen_model
        self.references.synth_image = synth_image

    def get_strip_model(self):
        """
        Return the stripping model (just the U-Net).
        """
        return self.references.unet
