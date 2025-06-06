"""
losses for the neuron project

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

# core imports
import sys

# third party
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses
# simple metrics renamed mae -> l1, mse -> l2
from tensorflow.keras.losses import MAE as l1
from tensorflow.keras.losses import MSE as l2

# local
from . import utils
from . import metrics


# explicit import of MutualInformation class -- although there is not direct loss member class,
# since there's many ways to use MutualInformation as a loss for many settings
# (maximizing, minimze, volumes, segmentation maps, etc)
from .metrics import MutualInformation


class Dice(metrics.Dice):
    """
    inherits ne.metrics.Dice


    Dice of two Tensors. 
    Enables both 'soft' and 'hard' Dice, and weighting per label (or per batch entry)

    More information/citations:
    - Dice. Measures of the amount of ecologic association between species. Ecology. 1945
        [original paper describing metric]
    - Dalca AV, Guttag J, Sabuncu MR Anatomical Priors in Convolutional Networks for 
      Unsupervised Biomedical Segmentation. CVPR 2018. https://arxiv.org/abs/1903.03148
        [paper for which we developed this method]    
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
        return - self.dice(y_true, y_pred)

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
        return - self.mean_dice(y_true, y_pred)


class SoftDice(metrics.SoftDice):
    """
    inherits ne.metrics.Dice

    More information/citations:
    - Dalca AV, Guttag J, Sabuncu MR Anatomical Priors in Convolutional Networks for 
      Unsupervised Biomedical Segmentation. CVPR 2018. https://arxiv.org/abs/1903.03148
    - Milletari et al, V-net: Fully convolutional neural networks for volumetric medical image 
      segmentation. 3DV 2016.
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
        return - self.dice(y_true, y_pred)

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
        return - self.mean_dice(y_true, y_pred)


class HardDice(metrics.HardDice):
    """
    inherits ne.metrics.Dice

    More information/citations:
    - Dice. Measures of the amount of ecologic association between species. Ecology. 1945
        [original paper describing metric]
    - Dalca AV, Guttag J, Sabuncu MR Anatomical Priors in Convolutional Networks for 
      Unsupervised Biomedical Segmentation. CVPR 2018. https://arxiv.org/abs/1903.03148
        [paper for which we developed this method]
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
        return - self.dice(y_true, y_pred)

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
        return - self.mean_dice(y_true, y_pred)


class CategoricalCrossentropy(metrics.CategoricalCrossentropy):
    def __init__(self, *args, **kwargs):
        """
        wraps tf.keras.losses.CategoricalCrossentropy, but enables label_weights as an 
        explicit parameter (which is also possible in the tf version, but a bit more cumbersome)

        see metrics.CategoricalCrossentropy
        """

        super().__init__(*args, **kwargs)

    def loss(self, *args, **kwargs):
        return self.cce(*args, **kwargs)


class MeanSquaredErrorProb(metrics.MeanSquaredErrorProb):
    def __init__(self, *args, **kwargs):
        """
        wraps tf.keras.losses.MeanSquaredError, but specifically assumes the last dimension of 
        the Tensors is the log-probability of labels, and allows for label weights along those 
        labels. (this is also possible in the tf version, but a bit more cumbersome)

        see doc for metrics.MeanSquaredErrorProb
        """
        super().__init__(*args, **kwargs)

    def loss(self, *args, **kwargs):
        return self.mse(*args, **kwargs)


###############################################################################
# decorators
###############################################################################

def multiple_losses_decorator(losses, weights=None):
    """
    Applies multiple losses to a given output

    Args:
        losses (list): list of losses, each taking in two Tensors
        weights (list or np.array, optional): weight for each metric.
            Defaults to None.
    """

    if weights is None:
        weights = np.ones(len(losses))

    def loss(y_true, y_pred):
        total_val = 0
        for idx, los in enumerate(losses):
            total_val += weights[idx] * los(y_true, y_pred)
        return total_val

    return loss
