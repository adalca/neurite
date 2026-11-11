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

## Grids, standardization, and cropped noise

`ne.volshape_to_ndgrid` and `ne.bw_grid` use the current PyTorch default device when
`device=None`. Passing a device explicitly still overrides it. Grid coordinates, axis ordering,
and existing positional arguments are unchanged.

`ne.zscore` keeps its sample-variance and epsilon defaults. The keyword-only `unbiased=False`
selects population variance. Finite singleton groups return zero with zero gradients, including
float16. A singleton is defined by the reduction axes: each row of a `[2, 1]` tensor has one
value, while its column has two.

```python
import torch
import neurite as ne

values = torch.tensor([[4.0], [9.0]], dtype=torch.float16, requires_grad=True)
rows = ne.zscore(values, dim=1)  # [[0.0], [0.0]]: one value per row.
column = ne.zscore(values, dim=0)  # Approximately [[-0.707], [0.707]]: two values.
rows.sum().backward()  # values.grad is zero; each row always centers to zero.
```

Groups with more than one value keep their previous calculation. For constant float16 groups
such as `[5, 5]`, use a representable epsilon such as `eps=1e-4` or standardize in float32.
The default `1e-8` rounds to zero in float16; the singleton fix does not change that default.

Gaussian kernels and smoothing accept zero sigma as an unsmoothed axis; negative or nonfinite
widths raise a clear error. Positive-sigma kernels retain the existing normalization. For
`normalize="gaussian"`, only positive-sigma axes contribute to the continuous normalizer.

`ne.upsample_crop(tensor, size, crop_shape, non_spatial_dims=None)` performs linear interpolation
followed by a central crop without keeping every expanded field in memory at once. Its NN
counterpart accepts `[B, C, *V]`. Oversized 3D grids use direct sampling in depth slabs, which
preserves the interpolation geometry but can introduce small floating-point differences.
Only this path, used above `2**31 - 1` full-grid voxels, promotes float16/bfloat16 sampling to
float32 internally and returns the original dtype. Ordinary interpolation retains its input
precision. This protects voxel-center coordinates from both rounding errors and overflow.

`ne.upsample_noise` and `ne.fractal_noise` accept keyword-only `crop_shape=None`. The default
keeps the previous sampling path. A crop uses the original full-grid scales and random draws;
fractal output standardization happens after cropping. For `method="blur"`, octave smoothing
and octave standardization still use the full grid before cropping.
Both fractal methods validate positive crop sizes before drawing noise; use `None` to disable
cropping.

```python
import neurite as ne

# Sample a padded field while keeping only its central 64-by-64 crop.
field = ne.upsample_noise((96, 96), scale=12, crop_shape=(64, 64))
normalized = ne.zscore(field, unbiased=False)
```
