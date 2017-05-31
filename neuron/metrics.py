"""
custom metrics for the neuron project 
This module is currently very experimental...

contact: adalca@csail.mit.edu
"""
import sys
import numpy as np
import keras.backend as K
from keras import losses
import tensorflow as tf

# local packages
import medipy.metrics
from . import metrics as nrn_metrics
import neuron.generators as nrn_gen

from imp import reload
reload(nrn_gen)


class CategoricalCrossentropy(object):
    """
    Categorical crossentropy with optional categorical weights and spatial prior

    Adapted from weighted categorical crossentropy via wassname:
    https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
        prior: path to numpy file with variable 'prior' in it,
            or the ready prior, reshaped for keras model

    Usage:
        loss = CategoricalCrossentropy().loss # or
        loss = CategoricalCrossentropy(weights=weights).loss # or
        loss = CategoricalCrossentropy(..., prior=prior).loss
        model.compile(loss=loss, optimizer='adam')
    """

    def __init__(self, weights=None, prior=None, use_float16=False,
                 crop=None,
                 use_sep_prior=False, patch_size=None, patch_stride=1,
                 batch_size=1):
        self.use_float16 = use_float16
        self.weights = weights if (weights is not None) else None
        self.use_sep_prior = use_sep_prior

        if use_sep_prior:
            assert prior is None, "cannot use both prior and separate prior"

        # process prior
        if prior is not None:
            if isinstance(prior, str):  # assuming file
                loc_vol = np.load(prior)['prior']
                if self.use_float16:
                    loc_vol = loc_vol.astype('float16')
                prior=loc_vol

            if patch_size is None:
                patch_size = prior.shape[0:3]
            nb_channels = prior.shape[3]

            prior_gen = nrn_gen.patch(loc_vol, patch_size + (nb_channels,),
                                      patch_stride=patch_stride, batch_size=batch_size,
                                      infinite=True)
            # prior = np.expand_dims(loc_vol, axis=0)  # reshape for keras model

            self.log_prior = prior_gen
        else:
            self.log_prior = None

        self.crop = crop

    def loss(self, y_true, y_pred):
        """ categorical crossentropy loss """

        if self.use_sep_prior:
            self.log_prior = K.log(y_true[1])
            y_true = y_true[0]

        if self.use_float16:
            y_true = K.cast(y_true, 'float16')
            y_pred = K.cast(y_pred, 'float16')

        # scale and clip probabilities
        # this should not be necessary for softmax output.
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute log probability
        log_post = K.log(y_pred)  # likelihood

        # add prior to form posterior
        if self.log_prior is not None:
            prior = next(self.log_prior)
            ps = prior.shape
            prior = np.reshape(prior, (ps[0], np.prod(ps[1:4]), ps[4]))

            # prior to keras
            if self.use_float16:
                k_prior = K.variable(prior, dtype='float16')
            else:
                k_prior = K.variable(prior)
            log_post += K.log(K.clip(k_prior, K.epsilon(), 1))

        # loss
        loss = - y_true * log_post

        # weighted loss
        if self.weights is not None:
            loss *= self.weights

        # take the mean loss
        return K.mean(K.sum(K.cast(loss, 'float32'), -1))



class Dice(object):
    """ UNTESTED
    compute the Dice of two Tensors.

    Tensors should either be:
    - probabilitic for each label
        i.e. [batch_size, *vol_size, nb_labels], where vol_size is the size of the volume (n-dims)
        e.g. for a 2D vol, y has 4 dimensions, where each entry is a prob for that voxel
    - max_label
        i.e. [batch_size, *vol_size], where vol_size is the size of the volume (n-dims).
        e.g. for a 2D vol, y has 3 dimensions, where each entry is the max label of that voxel

    Variables:
        nb_labels: optional numpy array of shape (L,) where L is the number of labels
            if not provided, all non-background (0) labels are computed and averaged
        weights: optional numpy array of shape (L,) giving relative weights of each label
        input_type is 'prob', or 'max_label'
        dice_type is hard or soft

    Usage:
        diceloss = metrics.dice(weights=[1, 2, 3])
        model.compile(diceloss, ...)

    Test:
        import keras.utils as nd_utils
        reload(nrn_metrics)
        weights = [0.1, 0.2, 0.3, 0.4, 0.5]
        nb_labels = len(weights)
        vol_size = [10, 20]
        batch_size = 7

        dice_loss = metrics.Dice(nb_labels=nb_labels).loss
        dice = metrics.Dice(nb_labels=nb_labels).dice
        dice_wloss = metrics.Dice(nb_labels=nb_labels, weights=weights).loss

        # vectors
        lab_size = [batch_size, *vol_size]
        r = nd_utils.to_categorical(np.random.randint(0, nb_labels, lab_size), nb_labels)
        vec_1 = np.reshape(r, [*lab_size, nb_labels])
        r = nd_utils.to_categorical(np.random.randint(0, nb_labels, lab_size), nb_labels)
        vec_2 = np.reshape(r, [*lab_size, nb_labels])

        # get some standard vectors
        tf_vec_1 = tf.constant(vec_1, dtype=tf.float32)
        tf_vec_2 = tf.constant(vec_2, dtype=tf.float32)

        # compute some metrics
        res = [f(tf_vec_1, tf_vec_2) for f in [dice, dice_loss, dice_wloss]]
        res_same = [f(tf_vec_1, tf_vec_1) for f in [dice, dice_loss, dice_wloss]]

        # tf run
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(res)
            sess.run(res_same)
            print(res[2].eval())
            print(res_same[2].eval())
    """


    def __init__(self, nb_labels, weights=None, input_type='prob', dice_type='hard'):
        """
        input_type is 'prob', or 'max_label'
        dice_type is hard or soft
        """

        self.nb_labels = nb_labels
        self.weights = None if weights is None else K.variable(weights)
        self.input_type = input_type
        self.dice_type = dice_type

    def dice(self, y_true, y_pred):
        """ compute dice for given Tensors """

        # if given predicted probability and we're doing hard Dice, compute max_label
        get_max_label = self.input_type == 'prob' and self.dice_type == 'hard'
        if get_max_label:
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1)
            # We assume that y_true is probabilistic, but just in case:
            y_true /= K.sum(y_true, axis=-1, keepdims=True)
            y_true = K.clip(y_true, K.epsilon(), 1)

            y_pred = K.argmax(y_pred, axis=-1)
            y_true = K.argmax(y_true, axis=-1)

        # if computed or given max label, apply one-hot expansion of batch_flattened data
        # y_true and y_pred will now be [batch_size, nb_voxels, nb_labels]
        if get_max_label or self.input_type == 'max_label':
            y_pred_flatten = K.batch_flatten(y_pred)
            y_true_flatten = K.batch_flatten(y_true)
            y_pred_onehot = K.one_hot(y_pred_flatten, self.nb_labels)
            y_true_onehot = K.one_hot(y_true_flatten, self.nb_labels)

        # compute dice for each entry in batch.
        # dice will now be [batch_size, nb_labels]
        top = 2 * K.sum(y_true_onehot * y_pred_onehot, 1)
        bottom = (K.sum(y_true_onehot, 1) + K.sum(y_pred_onehot, 1))
        bottom = K.maximum(bottom, K.epsilon())  # make sure we have no 0s on the bottom.
        return top / bottom


    def loss(self, y_true, y_pred):
        """ the loss. Assumes y_pred is prob (in [0,1] and sum_row = 1) """

        # compute dice, which will now be [batch_size, nb_labels]
        dice_metric = self.dice(y_true, y_pred)

        # weigh the entries in the dice matrix:
        if self.weights is not None:
            dice_metric *= self.weights

        # return one minus mean dice as loss
        return 1 - K.mean(dice_metric)


class WGAN_GP(object):
    """
    based on https://github.com/rarilurelo/keras_improved_wgan/blob/master/wgan_gp.py
    """

    def __init__(self, disc, lambda_gp=10):
        self.disc = disc
        self.lambda_gp = lambda_gp

    def loss(self, y_true, y_pred):
        # get the value for the true and fake images
        disc_true = self.disc(y_true)
        disc_pred = self.disc(y_pred)

        # sample a x_hat by sampling along the line between true and pred
        # z = tf.placeholder(tf.float32, shape=[None, 1])
        alpha = K.random_uniform(shape=[None, 1, 1, 1])
        diff = y_pred - y_true
        interp = y_true + alpha * diff

        # take gradient of D(x_hat)
        gradients = K.gradients(self.disc(interp), [interp])[0]
        grad_pen = K.mean(K.square(K.sqrt(K.sum(K.square(gradients), axis=1))-1))

        # compute loss
        return (K.mean(disc_pred) - K.mean(disc_true)) + self.lambda_gp * grad_pen





def l1(y_true, y_pred):
    return losses.mean_absolute_error(y_true, y_pred)

def l2(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)


class Nonbg(object):
    """ UNTESTED
    class to modify output on operating only on the non-bg class

    All data is aggregated and the (passed) metric is called on flattened true and
    predicted outputs in all (true) non-bg regions

    Usage:
        loss = metrics.dice
        nonbgloss = nonbg(loss).loss
    """

    def __init__(self, metric):
        self.metric = metric

    def loss(self, y_true, y_pred):
        """ prepare a loss of the given metric/loss operating on non-bg data """
        yt = y_true #.eval()
        ytbg = np.where(yt == 0)
        y_true_fix = K.variable(yt.flat(ytbg))
        y_pred_fix = K.variable(y_pred.flat(ytbg))
        return self.metric(y_true_fix, y_pred_fix)
