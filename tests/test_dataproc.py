"""Tests for Neurite data processing utilities."""

import subprocess
import sys

import numpy as np
import pytest
import torch

import neurite as ne


@pytest.mark.parametrize(
    "shape,crop,expected_slices",
    [
        ((10,), (2,), (slice(2, 8),)),
        ((6, 7, 8), (1, 2, 3), (slice(1, 5), slice(2, 5), slice(3, 5))),
        ((6, 7, 8), ((1, 2), (0, 3), (2, 0)), (slice(1, 4), slice(0, 4), slice(2, 8))),
    ],
)
def test_vol_proc_crops_numpy_volumes(shape, crop, expected_slices):
    """Test symmetric, asymmetric, zero-margin, and multidimensional cropping."""
    volume = np.arange(np.prod(shape)).reshape(shape)

    result = ne.dataproc.vol_proc(volume, crop=crop)

    assert np.array_equal(result, volume[expected_slices])


def test_vol_proc_preserves_non_crop_processing():
    """Test NumPy processing that does not use cropping."""
    volume = np.arange(6, dtype=float).reshape(2, 3)
    expected = np.clip((volume + 1) * 2, 0, 8)

    result = ne.dataproc.vol_proc(volume, offset=1, rescale=2, clip=(0, 8))

    assert np.array_equal(result, expected)


def test_vol_proc_still_rejects_tensor_cropping():
    """Test that removing PyStrum does not broaden tensor crop support."""
    with pytest.raises(AssertionError, match="crop is not supported"):
        ne.dataproc.vol_proc(torch.ones(5), crop=(1,))


def test_import_neurite_without_pystrum():
    """Test that importing Neurite does not require PyStrum."""
    code = "import sys; sys.modules['pystrum'] = None; import neurite"
    command = [sys.executable, "-c", code]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
