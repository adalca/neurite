"""
---- neurite ----

organization:

we'll have a parent folder for each backend type (e.g. tf or torch). inside each of them is a 
python module for important constructs (e.g. layers), and a utils folder. Inside the folder are
naturally structured modules (e.g. seg.py). 

separately, we'll have a python utilities folder (py), which contains utility modules that are 
in core python/numpy
"""

from . import py
from .py import utils
from .py import plot
from .py import dataproc


# import backend-dependent submodules
backend = py.utils.get_backend()
if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    from . import torch
else:
    # tensorflow is default backend
    from . import tf
    from .tf import *