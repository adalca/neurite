"""
python utilities for neuron
"""

# internal python imports
import os

# third party imports
import numpy as np

# local (our) imports


def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    NEURITE_BACKEND environment variable is set to 'pytorch'.
    """
    return 'pytorch' if os.environ.get('NEURITE_BACKEND') == 'pytorch' else 'tensorflow'


def softmax(x, axis):
    """
    softmax of a numpy array along a given dimension
    """

    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
