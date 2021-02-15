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
    labels = np.unique(labels)  # Sorted.
    assert np.issubdtype(labels.dtype, np.integer), 'non-integer data'

    lab_to_ind = np.zeros(np.max(labels) + 1, dtype='int_')
    for i, lab in enumerate(labels):
        lab_to_ind[lab] = i
    ind_to_lab = labels

    return lab_to_ind, ind_to_lab


def load_label_table(filename):
    """
    Reads a label lookup-table from file. File is expected to
    define the anatomical name and color for each label ID.
    Each line in the file should have the format:

    ```
    ID    AnatomicalName    R G B
    ```

    Parameters:
        filename (str): File to load.
    Returns:
        dict: Label lookup dictionary.
    """
    label_table = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.rstrip()
            if not line or line[0] == '#':
                continue
            tokens = line.split()
            sid = int(tokens[0])
            name = tokens[1]
            label_table[sid] = {'name': name}
            if len(tokens) > 2:
                label_table[sid]['color'] = [int(c) for c in tokens[2:5]]
    return label_table


def color_seg(seg, label_table):
    """
    Converts a hard segmentation into an RGB color image given a
    label lookup-table dictionary.

    Parameters:
        seg (ndarray): Hard segmentation array.
        label_table (dict): Label lookup.
    Returns:
        ndarray: RGB (3-frame) image with shape of input seg.
    """
    unique = np.unique(seg)
    color_seg = np.zeros((*seg.shape, 3), dtype='uint8')
    for sid in unique:
        label = label_table.get(sid)
        if label is not None:
            color_seg[seg == sid] = label['color']
    return color_seg
