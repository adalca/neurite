"""
metrics for the neuron project

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
import tensorflow.keras.backend as K
from tensorflow.keras import losses
# simple metrics renamed mae -> l1, mse -> l2
from tensorflow.keras.losses import MAE as l1
from tensorflow.keras.losses import MSE as l2

# local
import neurite as ne
from . import utils


class MutualInformation:
    """
    Soft Mutual Information approximation for intensity volumes and probabilistic volumes 
      (e.g. probabilistic segmentaitons)

    More information/citation:
    - Courtney K Guo. 
      Multi-modal image registration with unsupervised deep learning. 
      PhD thesis, Massachusetts Institute of Technology, 2019.
    - M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
      SynthMorph: learning contrast-invariant registration without acquired images
      IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
      https://doi.org/10.1109/TMI.2021.3116879

    # TODO: add local MI by using patches. This is quite memory consuming, though.

    Includes functions that can compute mutual information between volumes, 
      between segmentations, or between a volume and a segmentation map

    mi = MutualInformation()
    mi.volumes      
    mi.segs         
    mi.volume_seg
    mi.channelwise
    mi.maps
    """

    def __init__(self,
                 bin_centers=None,
                 nb_bins=None,
                 soft_bin_alpha=None,
                 min_clip=None,
                 max_clip=None):
        """
        Initialize the mutual information class

        Arguments below are related to soft quantizing of volumes, which is done automatically 
        in functions that comptue MI over volumes (e.g. volumes(), volume_seg(), channelwise()) 
        using these parameters

        Args:
            bin_centers (np.float32, optional): Array or list of bin centers. Defaults to None.
            nb_bins (int, optional):  Number of bins. Defaults to 16 if bin_centers
                is not specified.
            soft_bin_alpha (int, optional): Alpha in RBF of soft quantization. Defaults
                to `1 / 2 * square(sigma)`.
            min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
            max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
        """

        self.bin_centers = None
        if bin_centers is not None:
            self.bin_centers = tf.convert_to_tensor(bin_centers, dtype=tf.float32)
            assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]

        self.nb_bins = nb_bins
        if bin_centers is None and nb_bins is None:
            self.nb_bins = 16

        self.min_clip = min_clip
        if self.min_clip is None:
            self.min_clip = -np.inf

        self.max_clip = max_clip
        if self.max_clip is None:
            self.max_clip = np.inf

        self.soft_bin_alpha = soft_bin_alpha
        if self.soft_bin_alpha is None:
            sigma_ratio = 0.5
            if self.bin_centers is None:
                sigma = sigma_ratio / (self.nb_bins - 1)
            else:
                sigma = sigma_ratio * tf.reduce_mean(tf.experimental.numpy.diff(bin_centers))
            self.soft_bin_alpha = 1 / (2 * tf.square(sigma))
            print(self.soft_bin_alpha)

    def volumes(self, x, y):
        """
        Mutual information for each item in a batch of volumes. 

        Algorithm: 
        - use neurite.utils.soft_quantize() to create a soft quantization (binning) of 
          intensities in each channel
        - channelwise()

        Parameters:
            x and y:  [bs, ..., 1]

        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = K.shape(x)[-1]
        tensor_channels_y = K.shape(y)[-1]
        msg = 'volume_mi requires two single-channel volumes. See channelwise().'
        tf.debugging.assert_equal(tensor_channels_x, 1, msg)
        tf.debugging.assert_equal(tensor_channels_y, 1, msg)

        # volume mi
        return K.flatten(self.channelwise(x, y))

    def segs(self, x, y):
        """
        Mutual information between two probabilistic segmentation maps. 
        Wraps maps()        

        Parameters:
            x and y:  [bs, ..., nb_labels]

        Returns:
            Tensor of size [bs]
        """
        # volume mi
        return self.maps(x, y)

    def volume_seg(self, x, y):
        """
        Mutual information between a volume and a probabilistic segmentation maps. 
        Wraps maps()        

        Parameters:
            x and y: a volume and a probabilistic (soft) segmentation. Either:
              - x: [bs, ..., 1] and y: [bs, ..., nb_labels], Or:
              - x: [bs, ..., nb_labels] and y: [bs, ..., 1]

        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = K.shape(x)[-1]
        tensor_channels_y = K.shape(y)[-1]
        msg = 'volume_seg_mi requires one single-channel volume.'
        tf.debugging.assert_equal(tf.minimum(tensor_channels_x, tensor_channels_y), 1, msg)
        # otherwise we don't know which one is which
        msg = 'volume_seg_mi requires one multi-channel segmentation.'
        tf.debugging.assert_greater(tf.maximum(tensor_channels_x, tensor_channels_y), 1, msg)

        # transform volume to soft-quantized volume
        if tensor_channels_x == 1:
            x = self._soft_sim_map(x[..., 0])                       # [bs, ..., B]
        else:
            y = self._soft_sim_map(y[..., 0])                       # [bs, ..., B]

        return self.maps(x, y)  # [bs]

    def channelwise(self, x, y):
        """
        Mutual information for each channel in x and y. Thus for each item and channel this 
        returns retuns MI(x[...,i], x[...,i]). To do this, we use neurite.utils.soft_quantize() to 
        create a soft quantization (binning) of the intensities in each channel

        Parameters:
            x and y:  [bs, ..., C]

        Returns:
            Tensor of size [bs, C]
        """
        # check shapes
        tensor_shape_x = K.shape(x)
        tensor_shape_y = K.shape(y)
        tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y, 'volume shapes do not match')

        # reshape to [bs, V, C]
        if tensor_shape_x.shape[0] != 3:
            new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
            x = tf.reshape(x, new_shape)                             # [bs, V, C]
            y = tf.reshape(y, new_shape)                             # [bs, V, C]

        # move channels to first dimension
        ndims_k = len(x.shape)
        permute = [ndims_k - 1] + list(range(ndims_k - 1))
        cx = tf.transpose(x, permute)                                # [C, bs, V]
        cy = tf.transpose(y, permute)                                # [C, bs, V]

        # soft quantize
        cxq = self._soft_sim_map(cx)                                  # [C, bs, V, B]
        cyq = self._soft_sim_map(cy)                                  # [C, bs, V, B]

        # get mi
        map_fn = lambda x: self.maps(*x)
        cout = tf.map_fn(map_fn, [cxq, cyq], dtype=tf.float32)       # [C, bs]

        # permute back
        return tf.transpose(cout, [1, 0])                            # [bs, C]

    def maps(self, x, y):
        """
        Computes mutual information for each entry in batch, assuming each item contains 
        probability or similarity maps *at each voxel*. These could be e.g. from a softmax output 
        (e.g. when performing segmentaiton) or from soft_quantization of intensity image.

        Note: the MI is computed separate for each itemin the batch, so the joint probabilities 
        might be  different across inputs. In some cases, computing MI actoss the whole batch 
        might be desireable (TODO).

        Parameters:
            x and y are probability maps of size [bs, ..., B], where B is the size of the 
              discrete probability domain grid (e.g. bins/labels). B can be different for x and y.

        Returns:
            Tensor of size [bs]
        """

        # check shapes
        tensor_shape_x = K.shape(x)
        tensor_shape_y = K.shape(y)
        tf.debugging.assert_equal(tensor_shape_x, tensor_shape_y)
        tf.debugging.assert_non_negative(x)
        tf.debugging.assert_non_negative(y)

        eps = K.epsilon()

        # reshape to [bs, V, B]
        if tensor_shape_x.shape[0] != 3:
            new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
            x = tf.reshape(x, new_shape)                             # [bs, V, B1]
            y = tf.reshape(y, new_shape)                             # [bs, V, B2]

        # joint probability for each batch entry
        x_trans = tf.transpose(x, (0, 2, 1))                         # [bs, B1, V]
        pxy = K.batch_dot(x_trans, y)                                # [bs, B1, B2]
        pxy = pxy / (K.sum(pxy, axis=[1, 2], keepdims=True) + eps)   # [bs, B1, B2]

        # x probability for each batch entry
        px = K.sum(x, 1, keepdims=True)                              # [bs, 1, B1]
        px = px / (K.sum(px, 2, keepdims=True) + eps)                # [bs, 1, B1]

        # y probability for each batch entry
        py = K.sum(y, 1, keepdims=True)                              # [bs, 1, B2]
        py = py / (K.sum(py, 2, keepdims=True) + eps)                # [bs, 1, B2]

        # independent xy probability
        px_trans = K.permute_dimensions(px, (0, 2, 1))               # [bs, B1, 1]
        pxpy = K.batch_dot(px_trans, py)                             # [bs, B1, B2]
        pxpy_eps = pxpy + eps

        # mutual information
        log_term = K.log(pxy / pxpy_eps + eps)                       # [bs, B1, B2]
        mi = K.sum(pxy * log_term, axis=[1, 2])                      # [bs]
        return mi

    def _soft_log_sim_map(self, x):
        """
        soft quantization of intensities (values) in a given volume

        See neurite.utils.soft_quantize

        Parameters:
            x [bs, ...]: intensity image. 

        Returns:
            volume with one more dimension [bs, ..., B]
        """

        return ne.utils.soft_quantize(x,
                                      alpha=self.soft_bin_alpha,
                                      bin_centers=self.bin_centers,
                                      nb_bins=self.nb_bins,
                                      min_clip=self.min_clip,
                                      max_clip=self.max_clip,
                                      return_log=True)               # [bs, ..., B]

    def _soft_sim_map(self, x):
        """
        See neurite.utils.soft_quantize

        Parameters:
            x [bs, ...]: intensity image. 

        Returns:
            volume with one more dimension [bs, ..., B]
        """
        return ne.utils.soft_quantize(x,
                                      alpha=self.soft_bin_alpha,
                                      bin_centers=self.bin_centers,
                                      nb_bins=self.nb_bins,
                                      min_clip=self.min_clip,
                                      max_clip=self.max_clip,
                                      return_log=False)              # [bs, ..., B]

    def _soft_prob_map(self, x, **kwargs):
        """
        normalize a soft_quantized volume at each voxel, so that each voxel now holds a prob. map

        Parameters:
            x [bs, ..., B]: soft quantized volume

        Returns:
            x [bs, ..., B]: renormalized so that each voxel adds to 1 across last dimension
        """
        x_hist = self._soft_sim_map(x, **kwargs)                      # [bs, ..., B]
        x_hist_sum = K.sum(x_hist, -1, keepdims=True), K.epsilon()   # [bs, ..., B]
        x_prob = x_hist / (x_hist_sum)                               # [bs, ..., B]
        return x_prob


class Dice:
    """
    Dice of two Tensors. 
    Enables both 'soft' and 'hard' Dice, and weighting per label (or per batch entry)

    More information/citations:
    - Dice. Measures of the amount of ecologic association between species. Ecology. 1945
        [original paper describing metric]
    - Dalca AV, Guttag J, Sabuncu MR Anatomical Priors in Convolutional Networks for 
      Unsupervised Biomedical Segmentation. CVPR 2018. https://arxiv.org/abs/1903.03148
        [paper for which we developed this method]
    """

    def __init__(self,
                 dice_type='soft',
                 input_type='prob',
                 nb_labels=None,
                 weights=None,
                 check_input_limits=True,
                 laplace_smoothing=0.,
                 normalize=False):  # regularization for bottom of Dice coeff
        """
        Dice of two Tensors. 

        If Tensors are probablistic/one-hot, should be size 
            [batch_size, *vol_size, nb_labels], where vol_size is the size of the volume (n-dims)
            e.g. for a 2D vol, y has 4 dimensions, where each entry is a prob for that voxel
        If Tensors contain the label id at each location, size should be
            i.e. [batch_size, *vol_size], where vol_size is the size of the volume (n-dims).
            e.g. for a 2D vol, y has 3 dimensions, where each entry is the max label of that voxel
            If you provide [batch_size, *vol_size, 1], everything will still work since that just
            assumes a volume with an extra dimension, but the Dice score would be the same.

        Args:
            dice_type (str, optional): 'soft' or 'hard'. Defaults to 'soft'.
                hard dice will not provide gradients (and hence should not be used with backprop)
            input_type (str, optional): 'prob', 'one_hot', or 'max_label'
                'prob' (or 'one_hot' which will be treated the same) means we assume prob label maps
                'max_label' means we assume each volume location entry has the id of the seg label
                Defaults to 'prob'.
            nb_labels (int, optional): number of labels (maximum label + 1) 
                *Required* if using hard dice with max_label data. Defaults to None.
            weights (np.array or tf.Tensor, optional): weights matrix, broadcastable to 
                [batch_size, nb_labels]. most often, would want to weight the labels, so would be 
                an array of size [1, nb_labels]. 
                Defaults to None.
            check_input_limits (bool, optional): whether to check that input Tensors are in [0, 1].
                using tf debugging asserts. Defaults to True.
            laplace_smoothing (float, optional): amount of laplace smoothing 
                (adding to the numerator and denominator), 
                use 0 for no smoothing (in which case we employ div_no_nan)
                Default to 0.
            normalize (bool, optional): whether to renormalize probabilistic Tensors.
                Defaults to False.
        """
        # input_type is 'prob', or 'max_label'
        # dice_type is hard or soft

        self.dice_type = dice_type
        self.input_type = input_type
        self.nb_labels = nb_labels
        self.weights = weights
        self.normalize = normalize
        self.check_input_limits = check_input_limits
        self.laplace_smoothing = laplace_smoothing

        # checks
        assert self.input_type in ['prob', 'max_label']

        if self.dice_type == 'hard' and self.input_type == 'max_label':
            assert self.nb_labels is not None, 'If doing hard Dice need nb_labels'

        if self.dice_type == 'soft':
            assert self.input_type in ['prob', 'one_hot'], \
                'if doing soft Dice, must use probabilistic (one_hot)encoding'

    def dice(self, y_true, y_pred):
        """
        compute dice between two Tensors

        Args:
            y_pred, y_true: Tensors
                - if prob/onehot, then shape [batch_size, ..., nb_labels]
                - if max_label (label at each location), then shape [batch_size, ...]

        Returns: 
            Tensor of size [batch_size, nb_labels]
        """

        # input checks
        if self.input_type in ['prob', 'one_hot']:

            # Optionally re-normalize.
            # Note that in some cases you explicitly don't wnat to, e.g. if you only return a
            # subset of the labels
            if self.normalize:
                y_true = tf.math.divide_no_nan(y_true, K.sum(y_true, axis=-1, keepdims=True))
                y_pred = tf.math.divide_no_nan(y_pred, K.sum(y_pred, axis=-1, keepdims=True))

            # some value checking
            if self.check_input_limits:
                msg = 'value outside range'
                tf.debugging.assert_greater_equal(y_true, 0., msg)
                tf.debugging.assert_greater_equal(y_pred, 0., msg)
                tf.debugging.assert_less_equal(y_true, 1., msg)
                tf.debugging.assert_less_equal(y_pred, 1., msg)

        # Prepare the volumes to operate on
        # If we're doing 'hard' Dice, then we will prepare one_hot-based matrices of size
        # [batch_size, nb_voxels, nb_labels], where for each voxel in each batch entry,
        # the entries are either 0 or 1
        if self.dice_type == 'hard':

            # if given predicted probability, transform to "hard max""
            if self.input_type == 'prob':
                # this breaks differentiability, since argmax is not differentiable.
                warnings.warn('You are using ne.metrics.Dice with probabilistic inputs'
                              'and computing *hard* dice. \n For this, we use argmax to'
                              'get the optimal label at each location, which is not'
                              'differentiable. Do not use expecting gradients.')

                if self.nb_labels is None:
                    self.nb_labels = y_pred.shape.as_list()[-1]

                y_pred = K.argmax(y_pred, axis=-1)
                y_true = K.argmax(y_true, axis=-1)

            # transform to one hot notation
            y_pred = K.one_hot(y_pred, self.nb_labels)
            y_true = K.one_hot(y_true, self.nb_labels)

        # reshape to [batch_size, nb_voxels, nb_labels]
        y_true = ne.utils.batch_channel_flatten(y_true)
        y_pred = ne.utils.batch_channel_flatten(y_pred)

        # compute dice for each entry in batch.
        # dice will now be [batch_size, nb_labels]
        top = 2 * K.sum(y_true * y_pred, 1)
        bottom = K.sum(K.square(y_true), 1) + K.sum(K.square(y_pred), 1)
        if self.laplace_smoothing > 0:
            eps = self.laplace_smoothing
            return (top + eps) / (bottom + eps)
        else:
            return tf.math.divide_no_nan(top, bottom)

    def mean_dice(self, y_true, y_pred):
        """ 
        mean dice across all patches and labels 
        optionally weighted

        Args:
            y_pred, y_true: Tensors
                - if prob/onehot, then shape [batch_size, ..., nb_labels]
                - if max_label (label at each location), then shape [batch_size, ...]

        Returns: 
            dice (Tensor of size 1, tf.float32)
        """

        # compute dice, which will now be [batch_size, nb_labels]
        dice_metric = self.dice(y_true, y_pred)

        # weigh the entries in the dice matrix:
        if self.weights is not None:
            assert len(self.weights.shape) == 2, \
                'weights should be a matrix broadcastable to [batch_size, nb_labels]'
            dice_metric *= self.weights

        # return one minus mean dice as loss
        mean_dice_metric = K.mean(dice_metric)
        tf.debugging.assert_all_finite(mean_dice_metric, 'metric not finite')
        return mean_dice_metric

    def loss(self, y_true, y_pred):
        """
        Deprecate anytime after 12/01/2021
        """
        warnings.warn('ne.metrics.*.loss functions are deprecated.'
                      'Please use the ne.losses.*.loss functions.')

        return - self.mean_dice(y_true, y_pred)


class SoftDice(Dice):
    """
    Soft Dice of two Tensors. 

    More information/citations:
    - Dalca AV, Guttag J, Sabuncu MR Anatomical Priors in Convolutional Networks for 
      Unsupervised Biomedical Segmentation. CVPR 2018. https://arxiv.org/abs/1903.03148
    - Milletari et al, V-net: Fully convolutional neural networks for volumetric medical image 
      segmentation. 3DV 2016.
    """

    def __init__(self,
                 weights=None,
                 check_input_limits=True,
                 laplace_smoothing=0.,
                 normalize=False):
        """
        soft Dice score, inherits from Dice() class

        Args:
            weights (np.array or tf.Tensor, optional): weights matrix, broadcastable to 
                [batch_size, nb_labels]. most often, would want to weight the labels, so would be 
                an array of size [1, nb_labels]. 
                Defaults to None.
            check_input_limits (bool, optional): whether to check that input Tensors are in [0, 1].
                using tf debugging asserts. Defaults to True.
            laplace_smoothing (float, optional): amount of laplace smoothing
                (adding to the numerator and denominator),
                use 0 for no smoothing (in which case we employ div_no_nan)
                Default to 0.
            normalize (bool, optional): whether to renormalize probabilistic Tensors.
                Defaults to False.
        """
        super().__init__(dice_type='soft',
                         input_type='prob',
                         weights=weights,
                         check_input_limits=check_input_limits,
                         laplace_smoothing=laplace_smoothing,
                         normalize=normalize)


class HardDice(Dice):
    """
    "Hard" Dice of two Tensors. 

    More information/citations:
    - Dice. Measures of the amount of ecologic association between species. Ecology. 1945
        [original paper describing metric]
    - Dalca AV, Guttag J, Sabuncu MR Anatomical Priors in Convolutional Networks for 
      Unsupervised Biomedical Segmentation. CVPR 2018. https://arxiv.org/abs/1903.03148
        [paper for which we developed this method]
    """

    def __init__(self,
                 nb_labels,
                 input_type='max_label',
                 weights=None,
                 check_input_limits=True,
                 laplace_smoothing=0.,
                 normalize=False):
        """
        hard Dice score, inherits from Dice() class

        Tensors are assumed to be optimal label ids at each location. 
        If you wish to compute Hard Dice with Tensors 

        Args:
            nb_labels: the number of labels (maximum label + 1) in the Tensors
            input_type (str, optional): 'max_label', 'prob', 'one_hot'
                'max_label' means we assume each volume location entry has the id of the seg label.
                or 
                'prob' (or 'one_hot' which will be treated the same) means we assume prob label maps
                we will take tf.argmax() along the last dimension before running hard dice. 
                There will be no gradient.
                Defaults to 'max_label'.
            weights (np.array or tf.Tensor, optional): weights matrix, broadcastable to 
                [batch_size, nb_labels]. most often, would want to weight the labels, so would be 
                an array of size [1, nb_labels]. 
                Defaults to None.
            check_input_limits (bool, optional): whether to check that input Tensors are in [0, 1].
                using tf debugging asserts. Defaults to True.
            laplace_smoothing (float, optional): amount of laplace smoothing
                (adding to the numerator and denominator),
                use 0 for no smoothing (in which case we employ div_no_nan)
                Default to 0.
            normalize (bool, optional): whether to renormalize probabilistic Tensors.
                Defaults to False.
        """
        super().__init__(dice_type='hard',
                         input_type=input_type,
                         nb_labels=nb_labels,
                         weights=weights,
                         check_input_limits=check_input_limits,
                         laplace_smoothing=laplace_smoothing,
                         normalize=normalize)


class CategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):

    def __init__(self, label_weights=None, **kwargs):
        """
        wraps tf.keras.losses.CategoricalCrossentropy, but enables label_weights as an 
        explicit parameter (which is also possible in the tf version, but a bit more cumbersome)

        Args:
            label_weights: list, numpy array or Tensor with the same length as the number of 
                labels in the probabilistic maps
            other tf.keras.losses.CategoricalCrossentropy kwargs
        """
        self.label_weights = None
        if label_weights is not None:
            self.label_weights = tf.convert_to_tensor(label_weights)

        super().__init__(**kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.cce(y_true, y_pred, sample_weight=sample_weight)

    def cce(self, y_true, y_pred, sample_weight=None):
        if self.label_weights is not None:
            yf = y_pred.shape[-1]
            lf = self.label_weights.shape[-1]
            if yf != lf:
                raise ValueError(f'Label weights must be of len {yf}, but got {lf}.')

            # keras's CCE reduces axis=-1 before returning, so multiply the label weights now
            y_true = tf.cast(self.label_weights, y_true.dtype) * y_true

        return super().__call__(y_true, y_pred, sample_weight=sample_weight)


class MeanSquaredErrorProb(tf.keras.losses.MeanSquaredError):

    def __init__(self, label_weights=None, **kwargs):
        """
        wraps tf.keras.losses.MeanSquaredError, but specifically assumes the last dimension of 
        the Tensors is the log-probability of labels, and allows for label weights along those 
        labels. (this is also possible in the tf version, but a bit more cumbersome)

        Args:
            label_weights: list, numpy array or Tensor with the same length as the number of 
                labels in the probabilistic maps
            other tf.keras.losses.CategoricalCrossentropy kwargs
        """
        self.label_weights = None
        if label_weights is not None:
            self.label_weights = tf.convert_to_tensor(label_weights)

        super().__init__(**kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.mse(y_true, y_pred, sample_weight=sample_weight)

    def mse(self, y_true, y_pred, sample_weight=None):
        if self.label_weights is not None:
            yf = y_pred.shape[-1]
            lf = self.label_weights.shape[0]
            if yf != lf:
                raise ValueError(f'Label weights must be of len {yf}, but got {lf}.')

            # add dimension since Keras MSE reduces axis -1 before
            # scaling by the sample weight
            y_true = y_true[..., None]
            y_pred = y_pred[..., None]

            if sample_weight is not None:
                sample_weight = sample_weight * self.label_weights
            else:
                sample_weight = self.label_weights

        return super().__call__(y_true, y_pred, sample_weight=sample_weight)


###############################################################################
# decorators
###############################################################################

def multiple_metrics_decorator(metrics, weights=None):
    """
    Applies multiple metrics to a given output

    Args:
        metrics (list): list of metrics, each taking in two Tensors
        weights (list or np.array, optional): weight for each metric.
            Defaults to None.
    """

    if weights is None:
        weights = np.ones(len(metrics))

    def metric(y_true, y_pred):
        total_val = 0
        for idx, met in enumerate(metrics):
            total_val += weights[idx] * met(y_true, y_pred)
        return total_val

    return metric
