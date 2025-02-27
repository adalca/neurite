"""
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

from . import samplers          # noqa: F401
from . import utils             # noqa: F401
from . import models            # noqa: F401
from . import modules           # noqa: F401
from . import layers            # noqa: F401
from . import losses            # noqa: F401
