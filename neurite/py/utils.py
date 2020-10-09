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


def rebase_lab(labels):
    """
    Rebase labels and return lookup table (LUT) to convert to new labels in
    interval [0, N[ as: LUT[label_map]. Be sure to pass all possible labels.
    """
    labels = np.unique(labels) # Sorted.
    assert np.issubdtype(labels.dtype, np.integer), 'non-integer data'

    lab_to_ind = np.zeros(np.max(labels) + 1, dtype='int_')
    for i, lab in enumerate(labels):
        lab_to_ind[ lab ] = i
    ind_to_lab = labels

    return lab_to_ind, ind_to_lab

