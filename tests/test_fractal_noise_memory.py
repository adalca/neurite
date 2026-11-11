"""Regression coverage for bounded temporary noise-field lifetimes."""

import weakref

import pytest
import torch

import neurite.nn.functional as nf


@pytest.mark.parametrize("standardize", (False, True))
def test_previous_octave_is_released_before_next_allocation(monkeypatch, standardize):
    """Keep the accumulator alive while releasing each consumed octave buffer."""
    references = []
    original = nf.upsample_noise

    def tracked_sample(*args, **kwargs):
        if len(references) >= 2:
            assert references[-1]() is None
        value = original(*args, **kwargs)
        references.append(weakref.ref(value))
        return value

    monkeypatch.setattr(nf, "upsample_noise", tracked_sample)
    magnitude = torch.tensor(1.3, requires_grad=True)
    options = {"shape": (1, 2, 9, 11, 13), "scales": (2, 4, 8), "method": "upsample"}
    generator = torch.Generator().manual_seed(123)
    options.update(magnitude=magnitude, generator=generator, standardize=standardize)
    noise = nf.fractal_noise(**options)
    noise.square().mean().backward()
    assert torch.isfinite(noise).all()
    assert torch.isfinite(magnitude.grad)
    assert references[-1]() is None
