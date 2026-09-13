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
# It imports the PyTorch nn package and re-exports selected helpers from the
# pure-Python subpackages.


# Define the version of neurite
__version__ = '0.3.3'

from . import nn as nn
from . import py as py
from . import utils as utils
from .functional import *

# Importing submodules from `py` to top level. Not `utils` to avoid shadowing
from .py import data as data
from .py import dataproc as dataproc
from .py import plot as plot
