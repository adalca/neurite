# ---- neuron ----
from . import py
from .py import utils
from .py import plot


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