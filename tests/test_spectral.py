"""Numerical and API-layer contracts for spectral fields."""

import pytest
import torch
import neurite as ne
import neurite.nn.functional as nef


@pytest.mark.parametrize("shape", [(17,), (12, 15), (7, 8, 9)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fft_reference(shape, dtype):
    """Real FFT implementation matches full FFT, including odd sizes and Nyquist planes."""
    generator = torch.Generator().manual_seed(19)
    tensor = torch.randn((2, 3, *shape), dtype=dtype, generator=generator)
    kernel = torch.randn(shape, dtype=dtype, generator=generator)
    power = torch.fft.fftn(kernel, norm="ortho").abs().square()
    axes = tuple(range(2, tensor.ndim))
    transformed = torch.fft.fftn(tensor, dim=axes, norm="ortho")
    reference = torch.fft.ifftn(transformed * power.sqrt(), dim=axes, norm="ortho").real
    result = nef.spectral_filter(tensor, power)
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(nef.spectral_filter(tensor, torch.ones_like(power)), tensor)
    torch.testing.assert_close(nef.spectral_filter(tensor, torch.full_like(power, 9)), 3 * tensor)


@pytest.mark.parametrize("prefix", [(), (3,), (2, 3), (2, 3, 2)])
def test_wrapper_equivalence(prefix):
    """Both top-level wrappers delegate with the same shape and random ordering."""
    shape = (*prefix, 9, 11)
    dims = tuple(range(len(prefix))) if prefix else None
    power = torch.ones(9, 11, dtype=torch.float64)
    tensor = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    direct = nef.spectral_filter(tensor.reshape(-1, 1, 9, 11), power).reshape(shape)
    wrapped = ne.spectral_filter(tensor, power, dims)
    torch.testing.assert_close(wrapped, direct)
    gradient = torch.autograd.grad(wrapped.square().sum(), tensor)[0]
    torch.testing.assert_close(gradient, 2 * tensor)
    first = torch.Generator().manual_seed(42)
    second = torch.Generator().manual_seed(42)
    sampled = ne.spectral_noise(shape, power, dims, dtype=torch.float64, generator=first)
    nn_shape = (tensor.numel() // 99, 1, 9, 11)
    direct = nef.spectral_noise(nn_shape, power, dtype=torch.float64, generator=second)
    torch.testing.assert_close(sampled, direct.reshape(shape), rtol=0, atol=0)


def test_gradients_and_zero_power():
    """Input gradients agree with numerical derivatives; zero spectrum stays finite."""
    tensor = torch.randn(1, 1, 5, 7, dtype=torch.float64, requires_grad=True)
    power = torch.ones(5, 7, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x: nef.spectral_filter(x, power), (tensor,))
    zero = torch.zeros_like(power, requires_grad=True)
    result = nef.spectral_filter(tensor, zero)
    result.square().sum().backward()
    assert torch.isfinite(tensor.grad).all() and torch.isfinite(zero.grad).all()
    assert torch.count_nonzero(result) == 0


def test_empirical_spectrum():
    """An ensemble realizes the specified power without sample normalization."""
    power = torch.ones(15, 17)
    power[0, 0] = 0
    sampled = nef.spectral_noise((256, 1, 15, 17), power)
    empirical = torch.fft.fftn(sampled, dim=(-2, -1), norm="ortho").abs().square().mean((0, 1))
    assert (empirical - power).abs().mean() < 0.09
    assert empirical[0, 0] < 1e-10


@pytest.mark.parametrize("kind", ["negative", "nonfinite", "asymmetric", "shape", "dtype"])
def test_invalid_spectrum(kind):
    """Reject spectra that would silently distort the requested real-field operation."""
    tensor = torch.ones(1, 1, 9, 11)
    power = torch.ones(9, 11)
    if kind == "negative":
        power[0, 0] = -1
    elif kind == "nonfinite":
        power[0, 0] = float("nan")
    elif kind == "asymmetric":
        power[1, 2] = 2
    elif kind == "shape":
        power = power[:8]
    else:
        power = power.double()
    with pytest.raises((ValueError, TypeError)):
        nef.spectral_filter(tensor, power)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_replay_and_gradient():
    """CUDA preserves dtype/device and accepts a device-local generator."""
    power = torch.ones(9, 11, device="cuda", dtype=torch.float64)
    first = torch.Generator(device="cuda").manual_seed(3)
    second = torch.Generator(device="cuda").manual_seed(3)
    a = ne.spectral_noise((2, 3, 9, 11), power, (0, 1), dtype=power.dtype, generator=first)
    b = nef.spectral_noise((6, 1, 9, 11), power, dtype=power.dtype, generator=second)
    torch.testing.assert_close(a, b.reshape(a.shape), rtol=0, atol=0)
    a.requires_grad_()
    nef.spectral_filter(a, power).square().mean().backward()
    assert a.is_cuda and torch.isfinite(a.grad).all()


def test_power_parameter_gradient():
    """A symmetric spectrum parameter remains differentiable at both API layers."""
    tensor = torch.randn(2, 3, 7, 9, dtype=torch.float64)
    gain = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    basis = torch.ones(7, 9, dtype=torch.float64)
    wrapped = ne.spectral_filter(tensor, gain * basis, (0, 1))
    direct = nef.spectral_filter(tensor, gain * basis)
    torch.testing.assert_close(wrapped, direct)
    gradient = torch.autograd.grad(wrapped.square().sum(), gain)[0]
    torch.testing.assert_close(gradient, tensor.square().sum())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_finite_spectrum_preserves_values_and_gradients(dtype):
    """A valid spectrum must not overflow while averaging equal frequency pairs."""
    tensor = torch.ones(1, 1, 5, dtype=dtype, requires_grad=True)
    power_value = 0.75 * torch.finfo(dtype).max
    power = torch.full((5,), power_value, dtype=dtype, requires_grad=True)
    scale = power.detach().sqrt()
    expected = tensor * scale

    result = nef.spectral_filter(tensor, power)
    loss = (result / scale).square().mean()
    loss.backward()

    torch.testing.assert_close(result, expected)
    assert torch.isfinite(tensor.grad).all()
    assert torch.isfinite(power.grad).all()
