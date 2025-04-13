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

# Standard library imports
import os
import importlib
from importlib import import_module
from typing import TYPE_CHECKING, List, Optional

# Third-party imports
from packaging import version
import pystrum

# Local imports
from . import py
from .py import utils
from .py import plot
from .py import dataproc

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


def _load_backend_module(backend: str) -> None:
    """
    Dynamically load the backend-specifuc module and import its public symbols into the global
    namespace.

    Parameters
    ----------
    backend : str
        The case-insensitive backend identifier stored in the $NEURITE_BACKEND environment variable.
        Default is `pytorch`. Options are:
            - `pytorch`: use the pytorch backend.
            - `tensorflow`: Use the tensorflow backend.

    Returns
    -------
    str
        The module name relative to this __init__.py (e.g. '.pytorch' or '.tf')

    Raises
    ------
    ImportError
        If the required backend library is not installed.
    ValueError
        If an unsupported backend is specified in `NEURITE_BACKEND` environment variable.
    """

    # Convert the string referenced by `NEURITE_BACKEND` to all lowercase (for case insensitivity)
    backend = backend.lower()

    # In the case that $NEURITE_BACKEND == 'pytorch'
    if backend == "pytorch":

        try:
            # Determine if we can import torch, then import the submodule
            import torch                                                            # noqa: C0415
            from . import pytorch                                                   # noqa: C0415

        # Handle the case that pytorch is not importable
        except ImportError as error:
            raise ImportError(
                "PyTorch is required for the neurite PyTorch backend! Please install PyTorch."
            ) from error

        # The neurite module name for the pytorch backend is called `torch`.
        backend_module_name = ".pytorch"

    # In the case that $NEURITE_BACKEND == 'tensorflow'
    elif backend == "tensorflow":

        # Import tensorflow and handle errors
        try:
            import tensorflow                                                       # noqa: C0415

        # Tensorflow is not importable
        except ImportError as error:

            # Throw an error if tensorflow is not importable
            raise ImportError(
                "TensorFlow is required for the neurite TensorFlow backend! "
                "Please install TensorFlow."
            ) from error

        # The neurite module for tensorflow is called `tf`
        backend_module_name = ".tf"

    else:
        raise ValueError(
            f"Unsupported backend `{backend}`! Supported backends are `pytorch` and `tensorflow`. "
            "Please set the environment variable `NEURITE_BACKEND` to one of these"
        )

    # Dynamically import the backend module
    backend_module = importlib.import_module(backend_module_name, __name__)

    # Import public attributes from the backend module into the global namespace.
    for attr in dir(backend_module):
        if not attr.startswith("_"):
            globals()[attr] = getattr(backend_module, attr)

    return backend_module_name


# Determine the backend to use via py.utils.get_backend.
_backend = py.utils.get_backend()
backend_module_name = _load_backend_module(_backend)


# Define __all__ to control the public API.
__all__ = [
    name for name in globals()
    if not name.startswith("_") and
    name not in {
        "pkg_version", "importlib", "logging", "logger", "_MIN_PYSTRUM_VERSION"
        }
    ]

if TYPE_CHECKING:

    _backend = os.getenv("NEURITE_BACKEND", "pytorch").lower()

    if _backend == "tensorflow":
        from .tf import losses, modules
    else:
        from .pytorch import layers, losses, models, modules, samplers
