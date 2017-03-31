'''
tensorflow metrics

Contact: adalca@csail.mit.edu
'''

#  imports
import numpy as np
import tensorflow as tf


def dice(vol1, vol2, labels=None, nargout=1):
    '''
    THERE SEEMS TO BE A GRADIENT ISSUE SOMEWHERE HERE...

    Dice [1] volume overlap metric.
    Note that labels behaves somewhat differently than our numpy-based implementation

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : Tensor. The first volume (e.g. predicted volume)
    vol2 : Tensor. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, we assume a bianry image and compute dice for any voxels > 0
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''

    # make sure we're working with ints
    vol1 = tf.cast(vol1, tf.int32)
    vol2 = tf.cast(vol2, tf.int32)

    if labels is None:
        labels = np.array([1])
        vol1 = vol1 > 0
        vol2 = vol2 > 0

    # initialize dice variables
    # dicem = tf.Variable(0, dtype='float32')
    for idx, lab in enumerate(labels):
        vol1lab = tf.equal(vol1, lab)
        vol2lab = tf.equal(vol2, lab)
        volslab = tf.logical_and(vol1lab, vol2lab)
        top = 2 * tf.reduce_sum(tf.cast(volslab, tf.float32))
        bottom = tf.reduce_sum(tf.cast(vol1lab, tf.float32)) + tf.reduce_sum(tf.cast(vol2lab, tf.float32))
        q = tf.reshape(tf.cast(top/bottom, tf.float32), (1,1))
        if idx == 0:
            dicem = q
        else:
            dicem = tf.concat([dicem, q], 0)
        # dicem = tf.add(dicem, top/bottom)

    # dicem = tf.reduce_sum(dicem)
    # dicem = tf.divide(dicem, len(labels))
    return dicem




''' test

zero = tf.constant(0, dtype=tf.int32)
a = tf.constant([5,0,4,4,5], dtype=tf.int32)
where = tf.not_equal(a, zero)
b = tf.boolean_mask(a, where)

d = tf.constant([5, 0, 0,4,5], dtype=tf.int32)
z = dice(a, d, [4, 5])
init_op = tf.global_variables_initializer()

with tf.Session() as sess:   
    sess.run(init_op)
    sess.run(z)
    print(where.eval())
    print(a.eval())
    print(b.eval())
    print(z.eval())
'''
