
def test_nonexplicit_imports():
    """
    Some imports are not explicit.
    """
    import neurite as ne

    assert hasattr(ne, "plot")
    assert hasattr(ne, "utils")

    # Try initializing a volume3D
    assert hasattr(ne.plot, 'volume3D')
    _ = ne.plot.volume3D

    # Try initializing a `gaussian_kernel`
    _ = ne.utils.gaussian_kernel(42)
