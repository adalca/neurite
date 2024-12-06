"""
python utilities for neuron
"""

# internal python imports
import os
import warnings

# third party imports
import numpy as np
import matplotlib

# local (our) imports


def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    NEURITE_BACKEND environment variable is set to 'pytorch'.
    """
    backend = os.environ.get('NEURITE_BACKEND')
    # Determine if backend has been defined
    if backend not in {'tensorflow', 'pytorch'}:
        # If not, set backend to the current default (TensorFlow)
        backend = 'tensorflow'
        # Issue warning about the default change to pytorch in the future
        warnings.warn(
            "The default backend will soon be changing to 'pytorch'. If you prefer to use "
            "TensorFlow, please set the NEURITE_BACKEND environment variable to 'tensorflow'.",
            FutureWarning,
            stacklevel=2
        )
    return backend


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
    unique = np.unique(seg)
    color_seg = np.zeros((*seg.shape, 3), dtype='uint8')
    for sid in unique:
        label = label_table.get(sid)
        if label is not None:
            color_seg[seg == sid] = label['color']
    return color_seg


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
    for key in keys:
        rgb[key] = lut[key]['color']
    return matplotlib.colors.ListedColormap(rgb / 255)


def normalize_axes(axes, shape, allowed=None, none_means_all=False):
    """
    Normalize and validate axes indexing into an N-dimensional (ND) tensor shape. Specifically, the
    function sorts and deduplicates indices. It normalizes (valid) negative indices into the
    interval [0, N) and raises an error if they are outside an allowed range.

    Parameters:
        axes: Axis index inputs, as a Python integer, iterable, or None. None means all allowed
            axes or no axis, depending on `none_means_all`.
        shape: Shape of the array or tensor to index into.
        allowed: Subset of allowed axes in [0, N). Python integer, iterable, or None. None means
            all axes compatible with `shape` are allowed axes.
        none_means_all: Replace an input of `axes=None` with all axes in [0, N). If False,
            `axes=None` will return an empty tuple.

    Returns:
        The set of specified axes normalized into [0, N) as a tuple.

    """
    # Allowed values.
    ndims = len(shape)
    if allowed is None:
        allowed = range(ndims)
    if np.isscalar(allowed):
        allowed = [allowed]
    assert all(ax in range(ndims) for ax in allowed), f'allowed axes {allowed} out of bounds'

    # Axis inputs.
    if axes is None:
        axes = allowed if none_means_all else []
    if np.isscalar(axes):
        axes = [axes]

    # Convert negative indices.
    orig = axes
    axes = [ax + ndims if ax < 0 else ax for ax in axes]

    # Validate.
    for ax, inp in zip(axes, orig):
        if ax not in allowed:
            raise IndexError(f'axis {inp} outside {allowed}')

    # Sort, remove duplicates.
    return tuple(set(axes))
