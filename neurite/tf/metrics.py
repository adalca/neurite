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


# TODO: 
#  - separate metrics from losses
#  - clean up categoricalcrossentropy to use the tf built-in weights. 
#    Maybe wrap with 'label weights', 'voxel_weights', or 'weights'?
#  - Have wrapper classes for Dice: SoftDice, HardDice.
#  - metric wrapper/decorator? for cropping. 

class MutualInformation:
    """
    Soft Mutual Information approximation for intensity volumes and probabilistic volumes 
      (e.g. probabilistic segmentaitons)

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

    def __init__(self, bin_centers=None, nb_bins=None, min_clip=None, max_clip=None, soft_bin_alpha=1):
        """
        Initialize the mutual information class
        
        Arguments below are related to soft quantizing of volumes, which is done automatically 
        in functions that comptue MI over volumes (e.g. volumes(), volume_seg(), channelwise()) 
        using these parameters

        Args:
            bin_centers (np.float32, optional): array or list of bin centers. 
                Defaults to None.
            nb_bins (int, optional):  number of bins, if bin_centers is not specified. 
                Defaults to 16.
            min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
            max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
            soft_bin_alpha (int, optional): alpha in RBF of soft quantization. Defaults to 1.
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
        msg = 'volume_seg_mi requires one multi-channel segmentation.' # otherwise we don't know which one is which
        tf.debugging.assert_greater(tf.maximum(tensor_channels_x, tensor_channels_y), 1, msg)

        # transform volume to soft-quantized volume
        if tensor_channels_x == 1:
            x = self._soft_sim_map(x[..., 0])                       # [bs, ..., B]
        else:
            y = self._soft_sim_map(y[..., 0])                       # [bs, ..., B]

        return self.maps(x, y) # [bs]


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
        if len(tensor_shape_x) != 3:
            new_shape = K.stack([tensor_shape_x[0], -1, tensor_shape_x[-1]])
            x = tf.reshape(x, new_shape)                             # [bs, V, C]
            y = tf.reshape(y, new_shape)                             # [bs, V, C]

        # move channels to first dimension
        ndims_k = len(x.shape)
        permute = [ndims_k-1] + list(range(ndims_k-1))
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
        if len(tensor_shape_x) != 3:
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


class Dice(object):
    """
    Dice of two Tensors.

    # TODO in cleanup: remove hard max, remove nb_labels unless it's "max_label" type and 'hard' dice

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

    def __init__(self,
                 nb_labels, # should only be necessary if doing hard/one-hot. Could be optional for 
                 weights=None,
                 input_type='prob',
                 dice_type='soft',
                 approx_hard_max=True,
                 vox_weights=None,
                 crop_indices=None,
                 re_norm=False,
                 area_reg=0.1):  # regularization for bottom of Dice coeff
        """
        input_type is 'prob', or 'max_label'
        dice_type is hard or soft
        approx_hard_max - see note below

        Note: for hard dice, we grab the most likely label and then compute a
        one-hot encoding for each voxel with respect to possible labels. To grab the most
        likely labels, argmax() can be used, but only when Dice is used as a metric
        For a Dice *loss*, argmax is not differentiable, and so we can't use it
        Instead, we approximate the prob->one_hot translation when approx_hard_max is True.
        """

        self.nb_labels = nb_labels
        self.weights = None if weights is None else K.variable(weights)
        self.vox_weights = None if vox_weights is None else K.variable(vox_weights)
        self.input_type = input_type
        self.dice_type = dice_type
        self.approx_hard_max = approx_hard_max
        self.area_reg = area_reg
        self.crop_indices = crop_indices
        self.re_norm = re_norm

        if self.crop_indices is not None and vox_weights is not None:
            self.vox_weights = utils.batch_gather(self.vox_weights, self.crop_indices)

    def dice(self, y_true, y_pred):
        """
        compute dice for given Tensors

        """
        if self.crop_indices is not None:
            y_true = utils.batch_gather(y_true, self.crop_indices)
            y_pred = utils.batch_gather(y_pred, self.crop_indices)

        if self.input_type in ['prob', 'one-hot']:
            # We assume that y_true is probabilistic, but just in case:
            if self.re_norm:
                y_true = tf.div_no_nan(y_true, K.sum(y_true, axis=-1, keepdims=True))
            y_true = K.clip(y_true, K.epsilon(), 1)

            # make sure pred is a probability
            if self.re_norm:
                y_pred = tf.div_no_nan(y_pred, K.sum(y_pred, axis=-1, keepdims=True))
            y_pred = K.clip(y_pred, K.epsilon(), 1)

        # Prepare the volumes to operate on
        # If we're doing 'hard' Dice, then we will prepare one-hot-based matrices of size
        # [batch_size, nb_voxels, nb_labels], where for each voxel in each batch entry,
        # the entries are either 0 or 1
        if self.dice_type == 'hard':

            # if given predicted probability, transform to "hard max""
            if self.input_type == 'prob':
                if self.approx_hard_max:
                    y_pred_op = _hard_max(y_pred, axis=-1)
                    y_true_op = _hard_max(y_true, axis=-1)
                else:
                    # TODO: this is a *huge* amount of memory, probably should not be computing this... 
                    # but might actually be the same thing as looping over voxels and getting the maps
                    y_pred_op = _label_to_one_hot(K.argmax(y_pred, axis=-1), self.nb_labels)  
                    y_true_op = _label_to_one_hot(K.argmax(y_true, axis=-1), self.nb_labels)

            # if given predicted label, transform to one hot notation
            else:
                assert self.input_type == 'max_label'
                y_pred_op = _label_to_one_hot(y_pred, self.nb_labels)
                y_true_op = _label_to_one_hot(y_true, self.nb_labels)

        # If we're doing soft Dice, require prob output, and the data already is as we need it
        # [batch_size, nb_voxels, nb_labels]
        else:
            assert self.input_type == 'prob', "cannot do soft dice with max_label input"
            y_pred_op = y_pred
            y_true_op = y_true

        # reshape to [batch_size, nb_voxels, nb_labels]
        batch_size = K.shape(y_true)[0]
        y_pred_op = K.reshape(y_pred_op, [batch_size, -1, K.shape(y_true)[-1]])
        y_true_op = K.reshape(y_true_op, [batch_size, -1, K.shape(y_true)[-1]])

        # compute dice for each entry in batch.
        # dice will now be [batch_size, nb_labels]
        top = 2 * K.sum(y_true_op * y_pred_op, 1)
        bottom = K.sum(K.square(y_true_op), 1) + K.sum(K.square(y_pred_op), 1)
        # make sure we have no 0s on the bottom. K.epsilon()
        bottom = K.maximum(bottom, self.area_reg)
        return top / bottom

    def mean_dice(self, y_true, y_pred):
        """ weighted mean dice across all patches and labels """

        # compute dice, which will now be [batch_size, nb_labels]
        dice_metric = self.dice(y_true, y_pred)

        # weigh the entries in the dice matrix:
        if self.weights is not None:
            dice_metric *= self.weights
        if self.vox_weights is not None:
            dice_metric *= self.vox_weights

        # return one minus mean dice as loss
        mean_dice_metric = K.mean(dice_metric)
        tf.compat.v1.verify_tensor_all_finite(mean_dice_metric, 'metric not finite')
        return mean_dice_metric


    def loss(self, y_true, y_pred):
        """ the loss. Assumes y_pred is prob (in [0,1] and sum_row = 1) """

        # compute dice, which will now be [batch_size, nb_labels]
        dice_metric = self.dice(y_true, y_pred)

        # loss
        dice_loss = 1 - dice_metric

        # weigh the entries in the dice matrix:
        if self.weights is not None:
            dice_loss *= self.weights

        # return one minus mean dice as loss
        mean_dice_loss = K.mean(dice_loss)
        tf.compat.v1.verify_tensor_all_finite(mean_dice_loss, 'Loss not finite')
        return mean_dice_loss


    def _label_to_one_hot(tens, nb_labels):
        """
        Transform a label nD Tensor to a one-hot 3D Tensor. The input tensor is first
        batch-flattened, and then each batch and each voxel gets a one-hot representation
        """
        y = K.batch_flatten(tens)
        return K.one_hot(y, nb_labels)


    def _hard_max(tens, axis):
        """
        we can't use the argmax function in a loss, as it's not differentiable
        We can use it in a metric, but not in a loss function
        therefore, we replace the 'hard max' operation (i.e. argmax + onehot)
        with this approximation
        """
        assert False, 'deprecated, you should really just use a soft dice.'
        tensmax = K.max(tens, axis=axis, keepdims=True)
        eps_hot = K.maximum(tens - tensmax + K.epsilon(), 0)
        one_hot = eps_hot / K.epsilon()
        return one_hot



class CategoricalCrossentropy(object):
    """
    Categorical crossentropy with optional categorical weights and spatial prior

    Adapted from weighted categorical crossentropy via wassname:
    https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        loss = CategoricalCrossentropy().loss # or
        loss = CategoricalCrossentropy(weights=weights).loss # or
        loss = CategoricalCrossentropy(..., prior=prior).loss
        model.compile(loss=loss, optimizer='adam')
    """

    def __init__(self, weights=None, use_float16=False, vox_weights=None, crop_indices=None):
        """
        Parameters:
            vox_weights is either a numpy array the same size as y_true,
                or a string: 'y_true' or 'expy_true'
            crop_indices: indices to crop each element of the batch
                if each element is N-D (so y_true is N+1 dimensional)
                then crop_indices is a Tensor of crop ranges (indices)
                of size <= N-D. If it's < N-D, then it acts as a slice
                for the last few dimensions.
                See Also: tf.gather_nd
        """

        self.weights = weights if (weights is not None) else None
        self.use_float16 = use_float16
        self.vox_weights = vox_weights
        self.crop_indices = crop_indices

        if self.crop_indices is not None and vox_weights is not None:
            self.vox_weights = utils.batch_gather(self.vox_weights, self.crop_indices)

    def loss(self, y_true, y_pred):
        """ categorical crossentropy loss """

        if self.crop_indices is not None:
            y_true = utils.batch_gather(y_true, self.crop_indices)
            y_pred = utils.batch_gather(y_pred, self.crop_indices)

        if self.use_float16:
            y_true = K.cast(y_true, 'float16')
            y_pred = K.cast(y_pred, 'float16')

        # scale and clip probabilities
        # this should not be necessary for softmax output.
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        # compute log probability
        log_post = K.log(y_pred)  # likelihood

        # loss
        loss = - y_true * log_post

        # weighted loss
        if self.weights is not None:
            loss *= self.weights

        if self.vox_weights is not None:
            loss *= self.vox_weights

        # take the total loss
        # loss = K.batch_flatten(loss)
        mloss = K.mean(K.sum(K.cast(loss, 'float32'), -1))
        tf.compat.v1.verify_tensor_all_finite(mloss, 'Loss not finite')
        return mloss




class MeanSquaredError():
    """
    MSE with several weighting options
    """


    def __init__(self, weights=None, vox_weights=None, crop_indices=None):
        """
        Parameters:
            vox_weights is either a numpy array the same size as y_true,
                or a string: 'y_true' or 'expy_true'
            crop_indices: indices to crop each element of the batch
                if each element is N-D (so y_true is N+1 dimensional)
                then crop_indices is a Tensor of crop ranges (indices)
                of size <= N-D. If it's < N-D, then it acts as a slice
                for the last few dimensions.
                See Also: tf.gather_nd
        """
        self.weights = weights
        self.vox_weights = vox_weights
        self.crop_indices = crop_indices

        if self.crop_indices is not None and vox_weights is not None:
            self.vox_weights = utils.batch_gather(self.vox_weights, self.crop_indices)
        
    def loss(self, y_true, y_pred):

        if self.crop_indices is not None:
            y_true = utils.batch_gather(y_true, self.crop_indices)
            y_pred = utils.batch_gather(y_pred, self.crop_indices)

        ksq = K.square(y_pred - y_true)

        if self.vox_weights is not None:
            if self.vox_weights == 'y_true':
                ksq *= y_true
            elif self.vox_weights == 'expy_true':
                ksq *= tf.exp(y_true)
            else:
                ksq *= self.vox_weights

        if self.weights is not None:
            ksq *= self.weights

        return K.mean(ksq)


class MultipleMetrics():
    """ a mix of several losses for the same output """

    def __init__(self, losses, loss_weights=None):
        self.losses = losses
        self.loss_wts = loss_wts

        if loss_wts is None:
            self.loss_wts = np.ones(len(loss_wts))

    def loss(self, y_true, y_pred):
        total_loss = 0
        for idx, loss in enumerate(self.losses):
            total_loss += self.loss_weights[idx] * loss(y_true, y_pred)

        return total_loss



###############################################################################
# simple function losses
###############################################################################


def l1(y_true, y_pred):
    """ L1 metric (MAE) """
    return losses.mean_absolute_error(y_true, y_pred)


def l2(y_true, y_pred):
    """ L2 metric (MSE) """
    return losses.mean_squared_error(y_true, y_pred)
