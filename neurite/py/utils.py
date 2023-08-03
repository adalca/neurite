"""
python utilities for neuron
"""

# internal python imports
import os

# third party imports
import numpy as np
import matplotlib

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
    # Numerically stable softmax computation
    max_x = np.amax(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def rebase_lab(labels):
    """
    Rebase labels and return lookup table (LUT) to convert to new labels in
    interval [0, N[ as: LUT[label_map]. Be sure to pass all possible labels.
    """
    labels, counts = np.unique(labels, return_counts=True)
    assert np.issubdtype(labels.dtype, np.integer), 'non-integer data'
    lab_to_ind = np.zeros(np.max(labels) + 1, dtype='int_')
    lab_to_ind[labels] = np.cumsum(counts)
    lab_to_ind[0] = 0
    ind_to_lab = labels
    return lab_to_ind, ind_to_lab


def load_fs_lut(filename):
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


def seg_to_rgb_fs_lut(seg, label_table):
    """
    Converts a hard segmentation into an RGB color image given a
    freesurfer-style label lookup-table dictionary.

    Parameters:
        seg (ndarray): Hard segmentation array.
        label_table (dict): Label lookup.
    Returns:
        ndarray: RGB (3-frame) image with shape of input seg.
    """
    unique, inv_idx = np.unique(seg, return_inverse=True)
    has_color = np.array([sid in label_table for sid in unique])
    color_table = np.zeros((len(unique), 3), dtype=np.uint8)
    color_table[has_color] = [label_table[sid]['color'] for sid in unique[has_color]]
    return color_table[inv_idx].reshape((*seg.shape, 3))


def fs_lut_to_cmap(lut):
    """ 
    convert a freesurfer LUT to a matplotlib colormap.

    example
    lut = ne.py.utils.load_fs_lut('/path/to/seg32_labels.lut')
    fs_cmap = ne.py.utils.fs_lut_to_cmap(lut)

    Args:
        lut (dict/str): string (path to LUT file) or 
            dict with keys being integers (label ids), and each value should be a 
            dictionary with the key 'color' which is a list with 3 elements, 
            the RGB colors (0 to 255)

    Returns:
        matplotlib ListedColormap: [description]
    """
    if isinstance(lut, str):
        lut = load_fs_lut(lut)
    keys = list(lut.keys())
    rgb = np.zeros((np.array(keys).max() + 1, 3), dtype='float')
    has_color = np.zeros((np.array(keys).max() + 1,), dtype=np.bool)
    for key in keys:
        has_color[key] = 'color' in lut[key]
        rgb[key] = lut[key]['color'] if has_color[key] else 0
    cmap = matplotlib.colors.ListedColormap(rgb / 255)
    for i, key in enumerate(keys):
        if has_color[key]:
            cmap.set_bad(color=rgb[key] / 255, alpha=0)
            cmap.set_over(color=rgb[key] / 255, alpha=0)
    return cmap
