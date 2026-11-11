# ::: neurite

## Spectral filtering and sampling

`ne.spectral_filter` and `ne.spectral_noise` support pure spatial tensors or arbitrary leading
non-spatial dimensions. They adapt shapes and delegate to `ne.nn.functional.spectral_filter`
and `ne.nn.functional.spectral_noise`, whose tensors have shape `[B, C, *V]`.
The NN sampler draws white Gaussian noise and calls the NN filter; computation is not duplicated
in the top-level wrappers.

```python
import torch
import neurite as ne

shape = (63, 65)
axes = [torch.fft.fftfreq(size) for size in shape]
grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))
radius = grid.square().sum(0).sqrt()
power = torch.exp(-0.5 * ((radius - 0.1) / 0.02).square())
power[0, 0] = 0
power = power / power.mean()

field = ne.spectral_noise(shape, power, generator=torch.Generator().manual_seed(10))
fields = ne.nn.functional.spectral_noise((2, 3, *shape), power)
filtered = ne.spectral_filter(fields, power, non_spatial_dims=(0, 1))
```

`power` is a shared full spatial grid in unshifted `torch.fft.fftfreq` order. It must match the
input dtype/device and be finite, nonnegative, and symmetric under simultaneous inversion of
all spatial frequencies. Symmetry validation tolerates floating-point roundoff relative to the
maximum spectral power; accepted roundoff is symmetrized before filtering. Float32 and float64,
CPU and CUDA, and one through three spatial axes are supported.

The filter applies `sqrt(power)` to Fourier coefficients using real FFTs. The inverse specifies
the original spatial sizes explicitly, including odd sizes. Unit power is the identity; power
nine multiplies amplitude by three. For unit white Gaussian noise, `power` is the expected
orthonormal-FFT power. For other inputs it is a multiplicative power gain. Mean and variance
are not implicitly normalized. DC power controls the random constant component. Filtering is
periodic; sample a larger grid and explicitly crop if wraparound is undesirable.

Gradients pass through the input and positive power values. Exactly zero power uses a zero
subgradient convention. Per-field spectra are not broadcast by these functions; construct a
shared spectrum or call the function for each distinct spectrum. A generator must use the
sampling device. Reproducibility is within the same device/backend; CPU and CUDA need not
produce identical random streams.
