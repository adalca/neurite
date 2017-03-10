'''
custom metrics for the fs cnn project 
This module is currently very experimental...
'''

import numpy as np



# using dice loss functions inspired from
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
smooth = 1
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     print(y_true, y_true_f)
#     z = tf.to_float(tf.argmax(y_pred, axis=4)); # argmax seems to cause some massive error in keras/tf :S. Not sure why.
#     y_pred_f = K.flatten(z)
#     z2 = tf.reduce_max(y_pred, axis=4);
# #     y_pred_f = K.flatten(z2)
#     print("z:", z, z2)
#     print(y_pred, y_pred_f)
#     #     intersection = K.sum(y_true_f * y_pred_f)
#     #     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     intersection = tf.to_float(K.sum(tf.to_float(tf.equal(y_true_f, y_pred_f))))
#     print(intersection)
#     res = K.sum(1. * intersection / tf.to_float(tf.size(y_true_f)))
#     res2 = K.sum(K.categorical_crossentropy(y_true_f, y_pred_f))
#     print(res) 
#     print(res2)
#     return res2



# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     #     intersection = K.sum(y_true_f * y_pred_f)
#     #     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# #     intersection = K.sum(tf.to_float(tf.equal(y_true_f, y_pred_f)))
# #     res = intersection
#     n = y_true.get_shape()
#     print(n)
#     y_true_r = tf.reshape(y_true, [n/nb_classes, nb_classes])
#     y_pred_r = tf.reshape(y_pred, [n/nb_classes, nb_classes])
#     res2 = -K.sum(K.binary_crossentropy(y_true_r, y_pred_r))
#     return res2

# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)



# https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
class weighted_categorical_crossentropy(object):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        loss = weighted_categorical_crossentropy(weights).loss
        model.compile(loss=loss,optimizer='adam')
    """
    
    def __init__(self,weights):
        self.weights = K.variable(weights)
        
    def loss(self,y_true, y_pred):
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        # calc        
        p = y_true*K.log(y_pred)
        loss = p*self.weights
        loss =-K.sum(loss,-1)
#         return loss
        return K.mean(loss)

# def nonzero_acc(y_true, y_pred):
#     lab_true = K.argmax(y_true, axis=-1)
#     lab_pred = K.argmax(y_pred, axis=-1)
#     non_zero = ~K.equal(lab_true, 0)
#     print(lab_pred.get_shape())
#     print(non_zero.get_shape())
#     print(K.equal(lab_true, lab_pred).get_shape())
#     K.
#     return K.sum(K.equal(non_zero, K.equal(lab_true, lab_pred)))


# samples=3
# maxlen=4
# vocab=5

# # test
# y_pred_n = np.random.random((samples,maxlen,vocab))
# y_pred = tf.Variable(y_pred_n)
# y_true_n = np.random.random((samples,maxlen,vocab))
# y_true = tf.Variable(y_true_n)
# init_op = tf.global_variables_initializer()
# r = nonzero_acc(y_true,y_pred)
# sess.run(init_op)
# sess.run(r)


# import numpy as np
# from keras.activations import softmax
# from keras.objectives import categorical_crossentropy

# samples=3
# maxlen=4
# vocab=5

# sess = tf.Session()

# y_pred_n = np.random.random((samples,maxlen,vocab))
# y_pred = tf.Variable(y_pred_n)

# v2 = np.random.random((samples,maxlen,vocab))
# y_true = tf.nn.softmax(v2) # this isn't binary
# y_true = tf.cast(y_true, tf.float64)

# weights = np.array([0, 0.2, 0.4, 0.6, 0.8])
# # weights = tf.cast(tf.Variable(weights / np.sum(weights)), tf.float64)
# print('y_true:', y_true)
# print('y_pred:', y_pred)
# print('weights:', weights)

# init_op = tf.global_variables_initializer()
# r=weighted_categorical_crossentropy(weights).loss(y_true,y_pred)
# sess.run(init_op)
# sess.run(r)
# #     rr=categorical_crossentropy(y_true_n,y_pred_n).eval()
# #     np.testing.assert_almost_equal(r,rr)
# #     print('OK')
