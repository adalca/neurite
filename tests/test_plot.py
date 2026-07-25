import numpy as np
import pytest
import torch

from neurite.py.plot import flow, slices, volume3D


@pytest.mark.parametrize(
    'field',
    [
        np.zeros((6, 8, 2)),
        torch.zeros((2, 6, 8)),
    ],
)
def test_flow_accepts_2d_field_representations(field):
    figure, axes = flow(field, indexing='xy', img_indexing=False, show=False)

    assert len(figure.axes) == 1
    assert len(axes) == 1


def test_flow_stride_and_requires_grad_conversion():
    field = torch.ones((2, 7, 9), requires_grad=True)

    figure, _ = flow(field, indexing='xy', img_indexing=False, stride=3, show=False)
    quiver = figure.axes[0].collections[0]

    assert quiver.U.shape == (9,)
    assert quiver.V.shape == (9,)
    expected_x, expected_y = np.meshgrid([0, 3, 6], [0, 3, 6])
    np.testing.assert_array_equal(quiver.X, expected_x.ravel())
    np.testing.assert_array_equal(quiver.Y, expected_y.ravel())


def test_flow_2d_multi_field_preserves_legacy_axes_return():
    fields = [np.zeros((6, 8, 2)), np.zeros((6, 8, 2))]

    _, axes = flow(fields, indexing='xy', img_indexing=False, show=False)

    assert isinstance(axes, np.ndarray)
    assert axes.shape == (2,)


def test_flow_3d_default_planes_layout_and_titles():
    field = np.zeros((3, 4, 5, 6))

    figure, axes = flow(
        field, titles=['Square field'], indexing='xy', img_indexing=False, show=False
    )

    assert len(axes) == 1
    assert len(axes[0]) == 3
    assert [axis.get_title() for axis in figure.axes] == [
        'Square field: axis 0',
        'Square field: axis 1',
        'Square field: axis 2',
    ]


def test_flow_3d_explicit_slice_indices():
    field = np.zeros((3, 4, 5, 6))
    field[0] = 10
    field[1] = 20
    field[2] = 30

    figure, _ = flow(field, slice_nos=[1, 2, 3], indexing='xy', img_indexing=False, show=False)
    quivers = [axis.collections[0] for axis in figure.axes]

    np.testing.assert_array_equal(quivers[0].U, np.full((5, 6), 20).ravel())
    np.testing.assert_array_equal(quivers[0].V, np.full((5, 6), 30).ravel())
    np.testing.assert_array_equal(quivers[1].U, np.full((4, 6), 10).ravel())
    np.testing.assert_array_equal(quivers[1].V, np.full((4, 6), 30).ravel())
    np.testing.assert_array_equal(quivers[2].U, np.full((4, 5), 10).ravel())
    np.testing.assert_array_equal(quivers[2].V, np.full((4, 5), 20).ravel())


def test_flow_does_not_mutate_input_fields():
    field = np.arange(2 * 5 * 7, dtype=float).reshape(2, 5, 7)
    original = field.copy()

    flow(field, show=False)

    np.testing.assert_array_equal(field, original)


def test_volume3d_rotation_and_volume_titles():
    volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    figure, _ = volume3D(
        volume,
        slice_nos=[0, 1, 2],
        rot90=[1, 0, -1],
        volume_titles=['Square'],
        show=False,
    )

    assert [axis.get_title() for axis in figure.axes] == [
        'Square: axis 0',
        'Square: axis 1',
        'Square: axis 2',
    ]
    np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), np.rot90(volume[0], 1))
    np.testing.assert_array_equal(figure.axes[1].images[0].get_array(), volume[:, 1, :])
    np.testing.assert_array_equal(
        figure.axes[2].images[0].get_array(), np.rot90(volume[:, :, 2], -1)
    )


def test_plot_functions_accept_requires_grad_tensors():
    image = torch.randn(1, 1, 8, 9, requires_grad=True)
    volume = torch.randn(1, 1, 4, 5, 6, requires_grad=True)

    slices_figure, _ = slices(image, show=False)
    volume_figure, _ = volume3D(volume, show=False)

    assert len(slices_figure.axes) == 1
    assert len(volume_figure.axes) == 3
