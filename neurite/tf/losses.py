"""
tensorflow/keras utilities for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""


import sys

# third party
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses

# local
from . import utils
from . import metrics


class Dice(metrics.Dice):
    """
    inherits ne.metrics.Dice
    """
    def __init__(self, *args, **kwargs):
        """
        inherits ne.metrics.Dice
        """
        super().__init__(*args, **kwargs)

    def loss(self, y_true, y_pred):
        """ 
        dice loss (negative Dice score)

        Args:
            y_pred, y_true: Tensors
                - if prob/onehot, then shape [batch_size, ..., nb_labels]
                - if max_label (label at each location), then shape [batch_size, ...]

        Returns: 
            Tensor of size [batch_size, nb_labels]
        """
        return  - self.dice(y_true, y_pred)

    def mean_loss(self, y_true, y_pred):
        """ 
        mean dice loss (negative Dice score) across all patches and labels 
        optionally weighted

        Args:
            y_pred, y_true: Tensors
                - if prob/onehot, then shape [batch_size, ..., nb_labels]
                - if max_label (label at each location), then shape [batch_size, ...]

        Returns: 
            negative mean dice (Tensor of size 1, tf.float32)
        """
        return  - self.mean_dice(y_true, y_pred)


class SoftDice(metrics.SoftDice):
    """
    inherits ne.metrics.Dice
    """
    def __init__(self, *args, **kwargs):
        """
        inherits ne.metrics.Dice
        """
        super().__init__(*args, **kwargs)

    def loss(self, y_true, y_pred):
        """ 
        dice loss (negative Dice score)

        Args:
            y_pred, y_true: Tensors
                - if prob/onehot, then shape [batch_size, ..., nb_labels]
                - if max_label (label at each location), then shape [batch_size, ...]

        Returns: 
            Tensor of size [batch_size, nb_labels]
        """
        return  - self.dice(y_true, y_pred)

    def mean_loss(self, y_true, y_pred):
        """ 
        mean dice loss (negative Dice score) across all patches and labels 
        optionally weighted

        Args:
            y_pred, y_true: Tensors
                - if prob/onehot, then shape [batch_size, ..., nb_labels]
                - if max_label (label at each location), then shape [batch_size, ...]

        Returns: 
            negative mean dice (Tensor of size 1, tf.float32)
        """
        return  - self.mean_dice(y_true, y_pred)


class HardDice(metrics.HardDice):
    """
    inherits ne.metrics.Dice
    """
    def __init__(self, *args, **kwargs):
        """
        inherits ne.metrics.Dice
        """
        super().__init__(*args, **kwargs)

    def loss(self, y_true, y_pred):
        """ 
        dice loss (negative Dice score)

        Args:
            y_pred, y_true: Tensors
                - if prob/onehot, then shape [batch_size, ..., nb_labels]
                - if max_label (label at each location), then shape [batch_size, ...]

        Returns: 
            Tensor of size [batch_size, nb_labels]
        """
        return  - self.dice(y_true, y_pred)

    def mean_loss(self, y_true, y_pred):
        """ 
        mean dice loss (negative Dice score) across all patches and labels 
        optionally weighted

        Args:
            y_pred, y_true: Tensors
                - if prob/onehot, then shape [batch_size, ..., nb_labels]
                - if max_label (label at each location), then shape [batch_size, ...]

        Returns: 
            negative mean dice (Tensor of size 1, tf.float32)
        """
        return  - self.mean_dice(y_true, y_pred)


class MultipleLosses():
    """ a mix of several losses for the same output """

    def __init__(self, losses, loss_weights=None):
        self.losses = losses
        self.loss_weights = loss_weights

        if loss_weights is None:
            self.loss_weights = np.ones(len(loss_weights))

    def loss(self, y_true, y_pred):
        total_loss = 0
        for idx, loss in enumerate(self.losses):
            total_loss += self.loss_weights[idx] * loss(y_true, y_pred)

        return total_loss
