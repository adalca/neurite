"""
Python utilities for `neurite`.
"""

from datetime import datetime
import os
from pathlib import Path
import tarfile
from typing import Dict, Union
import urllib.request

# Third party imports
import numpy as np
import matplotlib
import torch


def get_cache_dir(*parts: str) -> Path:
    """
    Resolve a path within the user cache directory.

    Parameters
    ----------
    *parts : str
        Optional path components appended to the cache directory.

    Returns
    -------
    pathlib.Path
        Path under `$XDG_CACHE_HOME`, or `~/.cache` when `XDG_CACHE_HOME` is unset.
    """
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_dir.joinpath(*parts)


def load_tutorial_data(
    filename: str = "tutorial_data.npz",
    return_tensors: bool = False,
) -> Union[Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
    """
    Load VoxelMorph tutorial data as NumPy arrays or PyTorch tensors.

    The first uncached call downloads the official archive into the user cache and extracts the
    requested NPZ file.

    Parameters
    ----------
    filename : str, default="tutorial_data.npz"
        Tutorial file to load: `tutorial_data.npz`, `subj1.npz`, or `subj2.npz`.
    return_tensors : bool, default=False
        If True, return PyTorch tensors instead of NumPy arrays.

    Returns
    -------
    dict of str to numpy.ndarray or torch.Tensor
        Values stored in the requested NPZ file.
    """
    allowed_filenames = ("tutorial_data.npz", "subj1.npz", "subj2.npz")
    assert filename in allowed_filenames, f"unknown file: {filename}"

    cache_dir = get_cache_dir("voxelmorph")
    data_path = cache_dir / filename

    if not data_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive_path = cache_dir / "tutorial_data.tar.gz"
        if not archive_path.exists():
            timestamp = datetime.now().isoformat(timespec="seconds")
            print(f"[{timestamp}] Downloading VoxelMorph tutorial data to {archive_path}")
            urllib.request.urlretrieve(
                "https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/tutorial_data.tar.gz",
                archive_path,
            )
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extract(filename, cache_dir)

    with np.load(data_path, allow_pickle=False) as npz_file:
        arrays = {key: npz_file[key] for key in npz_file.files}

    if return_tensors:
        return {key: torch.from_numpy(array) for key, array in arrays.items()}
    return arrays


def softmax(x, axis):
    """
    Compute softmax along one axis.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Input array or tensor.
    axis : int
        Axis over which values are normalized.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Softmax values with the same backend as `x`.
    """

    if isinstance(x, torch.Tensor):
        return torch.softmax(x, dim=axis)
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def rebase_lab(labels):
    """
    Rebase integer labels to contiguous labels starting at 0.

    Parameters
    ----------
    labels : numpy.ndarray or torch.Tensor
        Integer label map. Tensor inputs return tensor lookup tables on the input device.

    Returns
    -------
    tuple
        Pair `(lab_to_ind, ind_to_lab)`. `lab_to_ind[label]` maps original labels to rebased
        labels, and `ind_to_lab[index]` maps rebased labels back to original labels.
    """

    if isinstance(labels, torch.Tensor):
        assert not labels.is_floating_point(), 'non-integer data'
        ind_to_lab = torch.unique(labels, sorted=True)
        max_label = int(ind_to_lab.max().item())
        lab_to_ind = torch.zeros(max_label + 1, dtype=torch.long, device=labels.device)
        for index, label in enumerate(ind_to_lab):
            lab_to_ind[label] = index
        return lab_to_ind, ind_to_lab

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
    Convert a hard segmentation into a FreeSurfer LUT RGB image.

    Parameters
    ----------
    seg : numpy.ndarray or torch.Tensor
        Hard segmentation array. Tensor inputs must be CPU tensors that do not require gradients.
    label_table : dict
        Lookup table keyed by integer label. Each present label must define a `color` key with
        three RGB values in the range 0 to 255.

    Returns
    -------
    numpy.ndarray
        RGB image with shape `seg.shape + (3,)` and dtype `uint8`.
    """
    if isinstance(seg, torch.Tensor):
        assert not seg.requires_grad, 'seg must not have requires_grad=True'
        assert seg.device.type == 'cpu', 'seg tensor must be on CPU'
        seg = seg.numpy()
    else:
        seg = np.asarray(seg)

    unique = np.unique(seg)
    color_seg = np.zeros((*seg.shape, 3), dtype='uint8')
    for sid in unique:
        label = label_table.get(int(sid))
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
