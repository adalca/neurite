"""
data resources for neurite.
"""

# internal imports
import random

# third party
import numpy as np
import scipy
import tensorflow.keras.datasets
from tqdm import tqdm

# local
import neurite as ne

# TODO:
# MNIST/Fashion-MNIST datasets
# OASIS dataset
# warped-blobs dataset (choose a better name?)
# MNIST overlap dataset
# Kuzushiji-MNIST overall dataset (https://github.com/rois-codh/kmnist)


class KerasDataset:

    def __init__(self, dataset=tensorflow.keras.datasets.mnist, **kwargs):

        if isinstance(dataset, str):
            if dataset == 'mnist':
                dataset = tensorflow.keras.datasets.mnist
            if dataset == 'fashion-mnist':
                dataset = tensorflow.keras.datasets.fashion_mnist

        self.dataset = dataset
        self._load_keras_dataset(**kwargs)

    def _load_keras_dataset(self,
                            tv_ratios=(5 / 6, 1 / 6),
                            sel_class=None,
                            pad_amt=0,
                            zoom_factor=None,
                            randomize=False,):
        """ load and process dataset starting with a tensorflow.keras.datasets dataset

        Args:
            tv_ratios (tuple, optional): [description]. Defaults to (5 / 6, 1 / 6).
            sel_class ([type], optional): [description]. Defaults to None.
            pad_amt (int, optional): [description]. Defaults to 0.
            reshape_fct ([type], optional): [description]. Defaults to None.
            randomize (bool, optional): [description]. Defaults to False.
            dataset ([type], optional): [description]. Defaults to mnist.
        """

        x = ne.py.data.DataSplit()
        y = ne.py.data.DataSplit()

        # load data
        (x.train, y.train), (x.test, y.test) = self.dataset.load_data()

        # split training dataset randomly into true and false
        seed = None
        if randomize:
            seed = np.random.uniform()
        x.train, x.val = ne.py.data.split_dataset(x.train,
                                                  ratios=tv_ratios,
                                                  randomize=randomize,
                                                  rand_seed=seed)
        y.train, y.val = ne.py.data.split_dataset(y.train,
                                                  ratios=tv_ratios,
                                                  randomize=randomize,
                                                  rand_seed=seed)

        # some processing
        data_types = ['train', 'test', 'val']

        # normalize
        for dt in data_types:
            x[dt] = x[dt][..., np.newaxis] / 255

        self.x = x
        self.y = y

        if sel_class is not None:
            self.select_class(sel_class)

        # pad and reshape
        if pad_amt > 0:
            self.pad(pad_amt)

        if zoom_factor is not None:
            self.reshape(zoom_factor)

    def select_class(self, cls, splits=None):

        if splits is None:
            splits = ['train', 'test', 'val']

        # select out a class (digit)
        for dt in splits:
            sel_map = self.y[dt] == cls
            self.x[dt] = self.x[dt][sel_map, :]
            self.y[dt] = self.y[dt][sel_map]

    def pad(self, pad_amt, splits=None):

        if splits is None:
            splits = ['train', 'test', 'val']

        for dt in splits:
            ndims = self.x[dt].ndim - 2
            pad_dim = (pad_amt, ) * ndims
            zero_dim = (0, ) * ndims
            pad_cell = (zero_dim, ) + ((pad_dim, ) * ndims) + (zero_dim, )
            self.x[dt] = np.pad(self.x[dt], pad_cell, mode='constant')

    def reshape(self, zoom_factor, splits=None, order=1, **kwargs):

        if splits is None:
            splits = ['train', 'test', 'val']

        res = [1] + [zoom_factor] * self.x[splits[0]].ndim + [1]
        for dt in splits:
            self.x[dt] = scipy.ndimage.interpolation.zoom(self.x[dt], res, order=order, **kwargs)

    def corrupt_mix(self,
                    corruption_ratio=[0.25, 0.75],
                    nb_corrupt=10,
                    splits=None,
                    output_suffix='olap'):
        """
        corrupt by mixing main image with other images in data:

            im = im * a + mix * b,

        where mix is a mean of several other images.

        Args:
            corruption_ratio (list, optional): ratio of [im, mix] to mix.
                Defaults to [0.25, 0.75].
            nb_corrupt (int, optional): number of mixing images. Defaults to 10.
            splits ([type], optional): splits to include. Defaults to all existing in self.x
            output_suffix (str, optional): [description]. Defaults to 'olap'.
        """
        cr = corruption_ratio
        if splits is None:
            splits = [f for f in self.x._splits]

        for dt in splits:
            out_dt = dt
            if output_suffix is not None:
                out_dt = out_dt + '_' + output_suffix
                self.x[out_dt] = np.zeros(self.x[dt].shape)

            for i in tqdm(range(self.x[dt].shape[0]), desc='overlap corrupting %s' % output_suffix):
                idx = np.random.randint(0, self.x[dt].shape[0], nb_corrupt)

                mean = np.mean(self.x[dt][idx, ...], 0, keepdims=True)
                self.x[out_dt][i, ...] = self.x[dt][i, ...] * cr[0] + mean * cr[1]
            self.y[out_dt] = self.y[dt]

    def corrupt_gaussian(self, mean=0., sigma=0.1, splits=None, output_suffix='noise'):

        if splits is None:
            splits = [f for f in self.x._splits]

        for dt in splits:
            out_dt = dt
            if output_suffix is not None:
                out_dt = out_dt + '_' + output_suffix
                self.x[out_dt] = np.zeros(self.x[dt].shape)

            self.x[out_dt] = self.x[dt] + np.random.normal(mean, sigma, self.x[dt].shape)
            self.x[out_dt] = np.clip(self.x[out_dt], 0, 1)
            self.y[out_dt] = self.y[dt]

    def show_examples(self, nb_examples=10, splits=None):
        """ show several examples.

        WARNING: This is highly experimental right now

        Args:
            nb_examples (int, optional): [description]. Defaults to 10.
            splits ([type], optional): [description]. Defaults to None.
        """

        if splits is None:
            splits = [f for f in self.x._splits]

        for dt in splits:
            print(dt)

            np.random.seed(0)
            idx = np.random.randint(0, self.x[dt].shape[0], nb_examples)
            slices = [self.x[dt][f, ..., 0] for f in idx]
            titles = [self.y[dt][f] for f in idx]
            ne.plot.slices(slices, cmaps=['gray'], titles=titles)
