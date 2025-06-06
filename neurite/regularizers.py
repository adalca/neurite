"""
tensorflow/keras regularizers for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

or for the transformation/interpolation related functions:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import tensorflow as tf
import tensorflow.keras.backend as K

from .utils import soft_delta


def soft_l0_wrap(wt=1.):

    def soft_l0(x):
        """
        maximize the number of 0 weights
        """
        nb_weights = tf.cast(tf.size(x), tf.float32)
        nb_zero_wts = tf.reduce_sum(soft_delta(K.flatten(x)))
        return wt * (nb_weights - nb_zero_wts) / nb_weights

    return soft_l0
