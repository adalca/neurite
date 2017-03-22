'''
custom metrics for the neuron project 
This module is currently very experimental...

contact: adalca@csail.mit.edu
'''
import sys
import numpy as np
import keras.backend as K

# local packages
import medipy.metrics
import neuron.src.tftools.metrics



class CategoricalCrossentropy(object):
    '''
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
    '''

    def __init__(self, weights=None, prior=None):
        self.weights = weights if (weights is not None) else K.variable(weights)

        # process prior
        if prior is not None:
            if isinstance(prior, str):  # assuming file
                loc_vol = np.load(prior)['prior']
                prior = np.expand_dims(loc_vol, axis=0)  # reshape for keras model
            self.log_prior = K.log(K.clip(K.variable(prior), K.epsilon(), 1))
        else:
            self.log_prior = None

    def loss(self, y_true, y_pred):
        ''' categorical crossentropy loss '''
        # scale and clip probabilities
        # this should not be necessary for softmax output.
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute log probability
        log_post = K.log(y_pred)  # likelihood

        # add prior to form posterior
        if self.log_prior is not None:
            log_post += self.log_prior

        # loss
        loss = - y_true * log_post

        # weighted loss
        if self.weights is not None:
            loss *= self.weights

        # take the mean loss
        return K.mean(K.sum(loss, -1))



class Dice(object):
    ''' UNTESTED
    Currently there is a problem with tf being able to compute derivates for this implementation

    Dice-based metric(s)

    Variables:
        labels: optional numpy array of shape (L,) where L is the number of labels to be evaluated.
            if not provided, all non-background (0) labels are computed and averaged
        weights: optional numpy array of shape (L,) giving relative weights of each label
        prior: filename of spatial priors to be added to y_pred before Dice
            TODO: maybe move this to a 'Prior' layer with a set weight in architecture ?

    Usage
        diceloss = metrics.dice([1, 2, 3])
        model.compile(diceloss, ...)
    '''

    def __init__(self, labels=[1], weights=None, prior=None):

        self.labels = labels
        if weights is None:
            weights = np.ones(len(labels))
        self.weights = K.variable(weights.flatten())

        # process prior
        if prior is not None:
            data = np.load(prior)
            loc_vol = data['prior']
            loc_vol = np.expand_dims(loc_vol, axis=0) # reshape for model
            loc_vol /= np.sum(loc_vol, axis=-1, keepdims=True)
            self.log_prior = K.log(K.clip(K.variable(loc_vol), K.epsilon(), 1))
        else:
            self.log_prior = None

    def loss(self, y_true, y_pred):
        ''' the loss. Assumes y_pred is prob (in [0,1] and sum_row = 1) '''

        y_pred_np = K.log(K.clip(y_pred, K.epsilon(), 1))
        if self.log_prior is not None:
            y_pred_np = y_pred_np + self.log_prior
        lab_pred = K.argmax(y_pred_np, axis=2)
        lab_true = K.argmax(y_true, axis=2)

        # compute dice measure
        dicem = tftools.metrics.dice(lab_true, lab_pred, self.labels)
        dicem = K.variable(dicem)

        # weight the labels
        if self.weights is not None:
            print(dicem)
            print(self.weights)
            dicem *= self.weights

        # return negative mean dice as loss
        return K.mean(-dicem)



class Nonbg(object):
    ''' UNTESTED
    class to modify output on operating only on the non-bg class

    All data is aggregated and the (passed) metric is called on flattened true and
    predicted outputs in all (true) non-bg regions

    Usage:
        loss = metrics.dice
        nonbgloss = nonbg(loss).loss
    '''

    def __init__(self, metric):
        self.metric = metric

    def loss(self, y_true, y_pred):
        ''' prepare a loss of the given metric/loss operating on non-bg data '''
        yt = y_true #.eval()
        ytbg = np.where(yt == 0)
        y_true_fix = K.variable(yt.flat(ytbg))
        y_pred_fix = K.variable(y_pred.flat(ytbg))
        return self.metric(y_true_fix, y_pred_fix)
