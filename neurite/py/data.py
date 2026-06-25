"""
Data resources for `neurite`.
"""

# Standard library imports
import random
from pathlib import Path

# Third party imports
import numpy as np
import torch


class DataSplit:
    def __init__(self, train=None, val=None, test=None):
        """ initialize DataSplit object, maintains splits of train, val, test

        data can be accessed using member variables, like
        data.train

        or as a victionary, using:
        data['train']

        Args:
            train ([type], optional): [description]. Defaults to None.
            val ([type], optional): [description]. Defaults to None.
            test ([type], optional): [description]. Defaults to None.
        """
        self._splits = []
        self.train = None
        self.val = None
        self.test = None

    def add_split(self, name, data):
        assert name not in self._splits, 'split {} already exists'.format(name)
        self._splits.append(name)
        super().__setattr__(name, data)

    def map_fn(self, lambda_fn, splits=None):
        """ apply function to each of the data splits

        Args:
            lambda_fn (function): function that takes in one input
            splits (list, optional): which splits to do processing on.
                Defaults to ['train', 'val', 'test'].
        """

        if splits is None:
            splits = ['train', 'val', 'test']

        for attr in splits:
            self[attr] = lambda_fn[self[attr]]

    def __getitem__(self, item):
        assert item in self._splits, 'Object only has {}'.format(self._splits)
        return getattr(self, item)

    def __setitem__(self, item, val):
        setattr(self, item, val)

    def __setattr__(self, key, value):
        if key == '_splits':
            assert value == [], 'can only set splits from within class'
            super().__setattr__(key, value)
        elif hasattr(self, key):
            super().__setattr__(key, value)
        else:
            self.add_split(key, value)


def split_dataset(data, ratios, axis=0, randomize=True, rand_seed=None):
    """
    Split a dataset into groups by relative ratios.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor or list or tuple
        Dataset to split. Array and tensor inputs are split along `axis`. List and tuple inputs are
        split along axis 0 and returned as the same container type.
    ratios : sequence of number
        Relative split sizes. Values are normalized by their sum before split indices are computed.
    axis : int, default=0
        Axis to split for NumPy arrays and torch tensors. List and tuple inputs only support axis 0.
    randomize : bool, default=True
        Shuffle item indices before splitting.
    rand_seed : int, default=None
        Seed passed to Python's `random` module when `randomize=True` and the value is truthy.

    Returns
    -------
    list
        Split datasets. NumPy inputs return NumPy arrays, torch inputs return tensors on the input
        device, list inputs return lists, and tuple inputs return tuples.
    """

    nb_groups = len(ratios)
    is_sequence = isinstance(data, (list, tuple))

    if is_sequence:
        nb_items = len(data)
        assert axis == 0, 'if data is a list or tuple, axis needs to be 0. got: %d' % axis
    else:
        assert isinstance(data, (np.ndarray, torch.Tensor)), (
            'data should be list, tuple, numpy array, or torch tensor, got: %s' % type(data)
        )
        nb_items = data.shape[axis]

    cratios = np.cumsum(ratios) / np.sum(ratios)
    sl_idx = [0] + [np.round(c * nb_items).astype(int) for c in cratios]

    rng = list(range(nb_items))
    if randomize:
        if rand_seed:
            random.seed(rand_seed)
        random.shuffle(rng)

    split = []
    for split_idx in range(nb_groups):
        indices = rng[sl_idx[split_idx]:sl_idx[split_idx + 1]]
        split.append(_take_items(data, indices, axis))
    return split


def _take_items(data, indices, axis):
    """Take split indices while preserving the input container type."""
    if isinstance(data, torch.Tensor):
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=data.device)
        return torch.index_select(data, dim=axis, index=index_tensor)
    if isinstance(data, np.ndarray):
        return np.take(data, indices, axis=axis)
    if isinstance(data, tuple):
        return tuple(data[index] for index in indices)
    return [data[index] for index in indices]


def load_dataset(dataset):
    """
    Downloads a dataset and caches it in the user's home directory.
    """
    import urllib.request
    datadir = Path.home().joinpath('.neurite')
    datadir.mkdir(exist_ok=True)

    if dataset == '2D-OASIS-TUTORIAL':
        filename = datadir.joinpath('2D-OASIS-TUTORIAL.npz')
        if not filename.exists():
            url = 'https://surfer.nmr.mgh.harvard.edu/pub/data/voxelmorph/2D-OASIS-TUTORIAL.npz'
            urllib.request.urlretrieve(url, filename)
            print(f'Cached dataset in {datadir}.')
        return np.load(filename)['images']
    else:
        raise ValueError(f'Unknown dataset {dataset}.')
