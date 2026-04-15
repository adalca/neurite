"""
Neurite: a PyTorch library for medical image processing.
=========================================================

Neurite is organized into two main parts:


- **PyTorch neural network components**: `neurite/nn` contains trainable layers, losses, and
  models.
- **Core Python utilities**: `neurite/py` contains data handling, plotting, and other pure-Python
  helpers.

The package exposes the PyTorch implementation directly on import. The older TensorFlow backend
lives on the `dev-tensorflow`.

Modules
-------
layers
    Operations and augmentations for model training.
losses
    Loss functions for various learning objectives.
random
    Utilities for random sampling, distributions, and random number generation for augmentations and
    model training.
utils
    Helper functions and utilities for common PyTorch operations, including tensor manipulation.
modules
    Foundational, trainable layers for constructing neural networks, including normalization
    layers and convolutional blocks.
models
    Prebuilt neural network architectures
"""

# This module is the top-level initializer for the Neurite package.
# It validates the required pystrum version, imports the PyTorch nn package,
# and re-exports selected helpers from the pure-Python subpackages.


# Define the version of neurite
__version__ = '0.2'

# Third-party imports
from packaging import version

# Custom imports
import pystrum

# Set the minimum allowable pystrum version
_MIN_PYSTRUM_VERSION = "0.2"

current_pystrum_version = getattr(pystrum, '__version__', None)

# Determine if the installed version of pystrum is valid
if (
    current_pystrum_version is None or
    version.parse(current_pystrum_version) <
    version.parse(_MIN_PYSTRUM_VERSION)
):
    raise ImportError(
        f'neurite requires pystrum version {_MIN_PYSTRUM_VERSION} or greater, but found version '
        f'{current_pystrum_version}'
    )

from . import py        # Noqa: F401
from . import nn
from . import utils
from .functional import *

# Importing submodules from `py` to top level. Not `utils` to avoid shadowing
from .py import data, dataproc, plot
