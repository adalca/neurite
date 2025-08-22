
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

    # utils.py
    _ = ne.utils.gaussian_kernel(42)
    _ = ne.utils.barycenter
    _ = ne.utils.interpn
    _ = ne.utils.volshape_to_meshgrid
    _ = ne.utils.resize
    _ = ne.utils.augment.draw_perlin


def test_explicit_submodule_imports():
    import neurite as ne
    _ = ne.py.utils.normalize_axes



def test_primary_modules():

    import neurite as ne

    # metrics.py
    _ = ne.callbacks.ModelCheckpointParallel

    # layers.py
    _ = ne.layers.SampleNormalLogVar
    _ = ne.layers.Negate
    _ = ne.layers.Constant
    _ = ne.layers.LocalParamWithInput
    _ = ne.layers.MeanStream
    _ = ne.layers.HyperConvFromDense

    # modelio.py
    _ = ne.modelio.LoadableModel
    _ = ne.modelio.LoadableModel.ReferenceContainer
    _ = ne.modelio.store_config_args

    # models.py
    _ = ne.models.labels_to_image
    _ = ne.models.labels_to_image_old
    _ = ne.models.conv_dec

    # metrics.py
    _ = ne.metrics.MutualInformation
