# ::: neurite.nn.functional

**NCC numerical warning:** Extreme intensities or large offsets relative to local variation can
cause floating-point errors that `eps` does not prevent. Local squared NCC coefficients are checked
for finite values in [0, 1], with absolute tolerance 1e-5, before averaging or reduction. Incorrect
values within that range are not detected.
