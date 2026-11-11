# ::: neurite.nn.functional

**NCC numerical warning:** Extreme intensities or large offsets relative to local variation can
cause floating-point errors that `eps` does not prevent. Local squared NCC coefficients are checked
for finite values in [0, 1], with absolute tolerance 1e-5, before averaging or reduction. Incorrect
values within that range are not detected.

The spectral functions `spectral_filter` and `spectral_noise` own the `[B, C, *V]`
implementation. Top-level wrappers delegate here; see [the spectral API guide](../neurite.md#spectral-filtering-and-sampling).
