"""
data resources for neurite.
"""

# internal imports
import random
from pathlib import Path

# third party
import numpy as np
import scipy


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
    split a dataset
    used to split train in train/val, for example

    can input single numpy array or list
    """

    nb_groups = len(ratios)
    is_list = isinstance(data, (list, tuple))

    if is_list:
        nb_items = len(data)
        assert axis == 0, \
            'if data is a list or tuple, axis needs to be 0. got: %d' % axis
    else:
        assert type(data) is np.ndarray, \
            'data should be list, tuple or numpy array, got: %s' % type(data)
        nb_items = data.shape[axis]

    # get slice indices
    cratios = np.cumsum(ratios) / np.sum(ratios)
    sl_idx = [0] + [np.round(c * nb_items).astype(int) for c in cratios]

    # prepare a list of all indices, and shuffle if necessary
    rng = list(range(nb_items))
    if randomize:
        if rand_seed:
            random.seed(rand_seed)
        random.shuffle(rng)

    # extract data
    if is_list:
        split = [data[rng[sl_idx[f]:sl_idx[f + 1]]] for f in range(nb_groups)]
    else:
        split = [np.take(data, rng[sl_idx[f]:sl_idx[f + 1]], axis=axis) for f in range(nb_groups)]

    return split


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
