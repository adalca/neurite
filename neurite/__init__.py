"""
Neurite: A modular deep learning library for medical image processing.
======================================================================

Neurite is organized into two main parts:
- Backend-specific modules: Each backend (e.g., PyTorch, TensorFlow) is contained in their own
  subpackages with functionalities such as layers and models.
- Core Python utilities: Common functionality implemented in pure Python and NumPy, organized under
  the 'py' subpackage.

This module is the top-level initializer for the Neurite package. It performs the following tasks:
  1. Validates that the required version of pystrum is installed.
  2. Imports core utilities from the 'py' subpackage.
  3. Dynamically loads backend-specific modules based on the configured backend.

Configuration
-------------
The backend is determined by the environment variable `NEURITE_BACKEND`. The default backend is
TensorFlow unless `NEURITE_BACKEND` is set to 'pytorch'.
"""

# Define the version of neurite
__version__ = '0.2'

# Third-party imports
from packaging import version
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

# Immediate submodules
from . import callbacks
from . import data
from . import generators
from . import layers
from . import losses
from . import metrics
from . import modelio
from . import models
from . import regularizers

# Subpackages
from . import utils
from . import py

# Import `py/` and `utils/` subpackages to namespace to enable non-explicit imports
from .py import plot


__all__ = [
    'callbacks',
    'data',
    'generators',
    'layers',
    'losses',
    'metrics',
    'modelio',
    'models',
    'regularizers',
    'utils',
    'py',
    'plot'
]
