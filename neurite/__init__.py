"""
---- neurite ----

organization:

we'll have a parent folder for each backend type (e.g. tf or torch). inside each of them is a 
python module for important constructs (e.g. layers), and a utils folder. Inside the folder are
naturally structured modules (e.g. seg.py). 

separately, we'll have a python utilities folder (py), which contains utility modules that are 
in core python/numpy
"""

# set version
__version__ = '0.2'

# check whether pystrum version is valid
from packaging import version
import pystrum
minv = '0.2'
curv = getattr(pystrum, '__version__', None)
if curv is None or version.parse(curv) < version.parse(minv):
    raise ImportError(f'neurite requires pystrum version {minv} or greater, '
                      f'but found version {curv}')

from . import py
from .py import utils
from .py import plot
from .py import dataproc


# import backend-dependent submodules
backend = py.utils.get_backend()
if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the NEURITE_BACKEND
    # environment var to "pytorch"
    try:
        import torch
    except ImportError:
        raise ImportError('Please install pytorch to use this neurite backend')

    from . import torch
else:
    # tensorflow is default backend
    try:
        import tensorflow
    except ImportError:
        raise ImportError('Please install tensorflow to use this neurite backend')

    from . import tf
    from .tf import *
