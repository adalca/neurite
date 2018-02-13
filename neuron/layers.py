# third party
import numpy as np
from keras import backend as K
import keras
from keras.layers import Layer

class LocalBiasLayer(Layer):
    """ 
    local bias layer
    """

    def __init__(self, my_initializer='RandomNormal', **kwargs):
        self.initializer = my_initializer
        super(LocalBiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalBiasLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x + self.kernel  # weights are difference from input

    def compute_output_shape(self, input_shape):
        return input_shape


class LocalLinearLayer(Layer):
    """ 
    local linear bias layer
    """

    def __init__(self, my_initializer='RandomNormal', **kwargs):
        self.initializer = my_initializer
        super(LocalLinearLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.mult = self.add_weight(name='mult-kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias-kernel', 
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalLinearLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.mult + self.bias 

    def compute_output_shape(self, input_shape):
        return input_shape
 
