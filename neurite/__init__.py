"""
Neurite: A modular deep learning library for medical image processing.
======================================================================

Neurite is organized into two main parts:

- **Backend-specific modules**: Each backend (e.g., PyTorch, TensorFlow) is contained in their own
  subpackage with functionalities such as layers and models.
- **Core Python utilities**: Common functionality implemented in pure Python and NumPy, organized under
  the 'py' subpackage.

Configuration
-------------
The backend is determined by the environment variable `NEURITE_BACKEND`. The default backend is
TensorFlow unless `NEURITE_BACKEND` is set to 'pytorch'.

`neurite/torch` is the package of neurite that handles its PyTorch implementation.

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
    Prebuilt yet flexible neural network architectures designed for specific tasks, such as image
    segmentation, registration, or classification. models leverage layers and modules from
    other components of the neurite for streamlined object construction.
"""

# Note: This block is intentionally omitted from the module-level docstring to avoid showing it in documentation.
# This module is the top-level initializer for the Neurite package. It performs the following tasks:
# 
#   1. Validates that the required version of pystrum is installed.
#   2. Imports core utilities from the 'py' subpackage.
#   3. Dynamically loads backend-specific modules based on the configured backend.


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
from . import samplers
from . import functional

# Importing submodules from `py` to top level. Not `utils` to avoid shadowing
from .py import data, dataproc, plot

# Public API
__all__ = [
    'py', 'nn', 'utils', 'samplers', 'data', 'dataproc', 'plot',
]
