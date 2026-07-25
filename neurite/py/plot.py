"""
Plot utilities for `neurite`.

If you use this code, please cite the first paper this was built for:
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

Contact: adalca [at] csail [dot] mit [dot] edu

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

# Standard library imports
from typing import Union, List, Any, Sequence, Dict, Optional, Tuple

# Third party imports
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable  # plotting


def _as_numpy_plot_array(
    array_like: Union[np.ndarray, torch.Tensor], name: str = 'array'
) -> np.ndarray:
    """
    Return plotting data as a NumPy array without changing the input.

    Parameters
    ----------
    array_like : numpy.ndarray or torch.Tensor
        Plot data. Tensor inputs may require gradients or reside on CUDA; they are detached
        and moved to CPU before conversion.
    name : str, default='array'
        Retained for compatibility with callers that label the plotted input.

    Returns
    -------
    numpy.ndarray
        NumPy array suitable for Matplotlib and NumPy plotting utilities.
    """
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def slices(
    slices_in: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
    titles: Union[str, List[str], None] = None,
    cmaps: Optional[Sequence[Union[str, Colormap]]] = None,
    norms: Any = None,
    do_colorbars: bool = False,
    grid: Union[bool, Sequence[int]] = False,          # grid or single-row layout
    width: int = 15,            # width in in
    show: bool = True,           # option to actually show the plot (plt.show())
    axes_off: bool = True,
    plot_block: bool = True,     # option to plt.show()
    facecolor: Any = None,
    imshow_args: Optional[Dict] = None,
) -> Tuple[Any, Any]:
    '''
    Plot a grid of 2D image slices.

    Parameters
    ----------
    slices_in : array_like or list of array_like
        A 2D image or list of 2D images to plot. Each element must be either a 2D array
        or an RGB image (shape HxWx3). Torch tensors are detached and moved to CPU internally.
    titles : str or list of str, optional
        Title or list of titles for each subplot. A single string is
        applied to all plots.
    cmaps : str or list of str, optional
        Colormap name or list of names for each image. Defaults to 'gray'.
    norms : Normalize instance or list of Normalize, optional
        Normalization(s) for color scaling. If None, default norm used.
    do_colorbars : bool, optional
        If True, add a colorbar to each subplot. Default is False.
    grid : bool or tuple of int, optional
        If True, arrange subplots in a square grid. If tuple
        (rows, cols) is given, use that layout. Default is single row.
    width : float, optional
        Figure width in inches. Default is 15.
    show : bool, optional
        If True, call plt.show() after plotting. Default is True.
    axes_off : bool, optional
        If True, hide axis ticks and labels. Default is True.
    plot_block : bool, optional
        If True, block execution when showing. Passed to
        plt.show(block=plot_block). Default is True.
    facecolor : color spec, optional
        Figure face color. If None, uses default.
    imshow_args : dict or list of dict, optional
        Additional kwargs for ax.imshow. A single dict applies to all.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the subplots.
    axs : array_like of Axes
        Array of Axes objects for each subplot.

    Examples
    --------
    >>> import numpy as np
    >>> slice1 = np.random.rand(64, 64)
    >>> slice2 = np.random.rand(64, 64)
    >>> fig, axs = slices(
    ...     [slice1, slice2],
    ...     titles=['First', 'Second'],
    ...     do_colorbars=True,
    ...     grid=(1, 2),
    ... )
    '''

    if isinstance(slices_in, (np.ndarray, torch.Tensor)):
        slices_in = [slices_in]

    nb_plots = len(slices_in)
    slices_in = [np.squeeze(_as_numpy_plot_array(slice_in, 'slice')) for slice_in in slices_in]

    for _, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, \
                'each slice has to be 2d or RGB (3 channels)'

    def input_check(inputs, nb_plots, name, default=None):
        """change input from None/single-link"""
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [default]
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps', default='gray')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')

    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)

    # Reshape axs to correspond to shape: (rows, cols)
    # For example, with rows = 1 and cols = 8, axs will be of shape: (1, 8) [[Axes(), Axes() ...]]
    # Another example, with rows = 1 and cols = 1, axs will be of shape: (1, 1) [[Axes()]]
    if rows == 1 and cols == 1: 
        axs = [[axs]]
    elif rows == 1: 
        axs = [axs]
    elif cols == 1: 
        axs = [[ax] for ax in axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs[row]
        ax = row_axs[col]

        # turn off axis
        if axes_off:
            ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i],
                          interpolation="nearest", norm=norms[i], **imshow_args[i])

        # colorbars
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        if do_colorbars:  # and cmaps[i] is not None
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_ax, cax=cax)

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)

    if facecolor is not None:
        fig.set_facecolor(facecolor)

    if show:
        plt.tight_layout()
        plt.show(block=plot_block)

    return (fig, axs)


def volume3D(
    vols: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
    slice_nos: Optional[Sequence[Union[int, Sequence[int]]]] = None,
    data_squeeze: bool = True,
    rot90: Optional[Sequence[int]] = None,
    volume_titles: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """
    Plot three orthogonal slices from one or more 3D volumes.

    Parameters
    ----------
    vols : numpy.ndarray, torch.Tensor, or sequence of either
        A 3D volume or a sequence of volumes. Singleton dimensions are removed by default.
        Tensor inputs may require gradients or reside on CUDA.
    slice_nos : sequence of int or sequence of sequence of int, optional
        Three slice indices in axis order, or one three-index sequence per volume. If None,
        the middle slice along each axis is used.
    data_squeeze : bool, default=True
        Remove singleton dimensions before checking the volume shape.
    rot90 : sequence of int, optional
        Three quarter-turn counts applied to the axis-0, axis-1, and axis-2 slices.
    volume_titles : sequence of str, optional
        One title per volume. Generated axis titles are prefixed with the corresponding title.
    **kwargs : Any
        Additional arguments passed to `slices`. Explicit `titles` cannot be combined with
        `volume_titles`.

    Returns
    -------
    Tuple[Any, Any]
        Figure and axes returned by `slices`.
    """
    if not isinstance(vols, (tuple, list)):
        vols = [vols]
    nb_vols = len(vols)
    vols = [_as_numpy_plot_array(vol, 'vol') for vol in vols]
    vols = list(map(np.squeeze if data_squeeze else np.asarray, vols))
    assert all(v.ndim == 3 for v in vols), 'only 3d volumes allowed in volume3D'

    if rot90 is None:
        rot90 = [0, 0, 0]
    assert len(rot90) == 3, 'rot90 must have three values'
    assert all(isinstance(value, (int, np.integer)) for value in rot90), \
        'rot90 values must be integers'
    if volume_titles is not None:
        assert 'titles' not in kwargs, 'titles and volume_titles cannot be combined'
        assert len(volume_titles) == nb_vols, 'volume_titles must match the number of volumes'

    slics = []
    for vi, vol in enumerate(vols):
        these_slice_nos = slice_nos
        if slice_nos is None:
            these_slice_nos = [f // 2 for f in vol.shape]
        elif isinstance(slice_nos[0], (list, tuple)):
            these_slice_nos = slice_nos[vi]
        slics.extend(np.rot90(np.take(vol, these_slice_nos[d], d), rot90[d]) for d in range(3))

    if 'titles' not in kwargs:
        titles = []
        for vi in range(nb_vols):
            prefix = '' if volume_titles is None else '%s: ' % volume_titles[vi]
            titles.extend('%saxis %d' % (prefix, d) for d in range(3))
        kwargs['titles'] = titles

    if 'grid' not in kwargs:
        kwargs['grid'] = [nb_vols, 3]

    return slices(slics, **kwargs)


def flow_legend(plot_block=True):
    """
    show quiver plot to indicate how arrows are colored in the flow() method.
    https://stackoverflow.com/questions/40026718/different-colours-for-arrows-in-quiver-plot
    """
    ph = np.linspace(0, 2 * np.pi, 13)
    x = np.cos(ph)
    y = np.sin(ph)
    u = np.cos(ph)
    v = np.sin(ph)
    colors = np.arctan2(u, v)

    norm = Normalize()
    norm.autoscale(colors)
    # we need to normalize our colors array to match it colormap domain
    # which is [0, 1]

    colormap = cm.winter

    plt.figure(figsize=(6, 6))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.quiver(x, y, u, v, color=colormap(norm(colors)), angles='xy', scale_units='xy', scale=1)
    plt.show(block=plot_block)


def _flow_to_planes(
    field: Union[np.ndarray, torch.Tensor],
    slice_nos: Optional[Sequence[int]] = None,
) -> Tuple[List[np.ndarray], int]:
    """Convert one flow field to channels-last 2D planes for plotting."""
    field = _as_numpy_plot_array(field, 'flow')
    if field.ndim == 3:
        assert slice_nos is None, 'slice_nos is only valid for 3D flow fields'
        if field.shape[-1] == 2:
            return [np.array(field, copy=True)], 2
        assert field.shape[0] == 2, '2D flow fields need two channels'
        return [np.array(np.transpose(field, (1, 2, 0)), copy=True)], 2

    assert field.ndim == 4 and field.shape[0] == 3, '3D flow fields need shape (3, D, H, W)'
    if slice_nos is None:
        slice_nos = [size // 2 for size in field.shape[1:]]
    assert len(slice_nos) == 3, 'slice_nos must have three values'
    # Pair components with the two in-plane axes so each quiver plot uses native vector order.
    planes = [
        np.transpose(field[[1, 2], slice_nos[0], :, :], (1, 2, 0)),
        np.transpose(field[[0, 2], :, slice_nos[1], :], (1, 2, 0)),
        np.transpose(field[[0, 1], :, :, slice_nos[2]], (1, 2, 0)),
    ]
    return [np.array(plane, copy=True) for plane in planes], 3


def _prepare_flow_fields(
    fields: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
    slice_nos: Optional[Sequence[int]] = None,
) -> Tuple[List[np.ndarray], int]:
    """Normalize direct or sequence flow inputs and enforce one spatial dimensionality."""
    if isinstance(fields, (np.ndarray, torch.Tensor)):
        fields = [fields]
    else:
        fields = list(fields)
    assert len(fields) > 0, 'at least one flow field is required'

    planes = []
    spatial_ndim = None
    for field in fields:
        field_planes, field_ndim = _flow_to_planes(field, slice_nos)
        if spatial_ndim is None:
            spatial_ndim = field_ndim
        assert field_ndim == spatial_ndim, \
            'all flow fields must have the same spatial dimensionality'
        planes.extend(field_planes)
    return planes, spatial_ndim


def flow(
    slices_in: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
    titles: Optional[Union[str, Sequence[Optional[str]]]] = None,
    cmaps: Optional[Sequence[Optional[Union[str, Colormap]]]] = None,
    width: Union[int, float] = 15,
    indexing: str = 'ij',
    img_indexing: bool = True,
    grid: Union[bool, Sequence[int]] = False,
    show: bool = True,
    quiver_width: Optional[float] = None,
    plot_block: bool = True,
    scale: Union[float, Sequence[float]] = 1,
    stride: int = 1,
    slice_nos: Optional[Sequence[int]] = None,
) -> Tuple[Any, Any]:
    """
    Plot one or more two-dimensional flow fields as colored quiver plots.

    Parameters
    ----------
    slices_in : numpy.ndarray, torch.Tensor, or sequence of either
        A direct field or a sequence of fields. Legacy 2D fields use `(H, W, 2)`; native
        2D fields use `(2, H, W)`, and native 3D fields use `(3, D, H, W)`.
    titles : str or sequence of str, optional
        Titles for fields. For 3D fields, one title expands to one title per displayed axis.
    cmaps : sequence of str or Colormap, optional
        Colormap settings. Custom colormaps are not implemented, matching legacy behavior.
    width : int or float, default=15
        Figure width in inches.
    indexing : {'ij', 'xy'}, default='ij'
        Vector indexing convention.
    img_indexing : bool, default=True
        Flip displayed rows to match image coordinates.
    grid : bool or sequence of int, default=False
        Subplot layout. Native 3D fields default to one row per field and three columns.
    show : bool, default=True
        If True, display the figure.
    quiver_width : float, optional
        Width passed to `Axes.quiver`.
    plot_block : bool, default=True
        Whether `plt.show` should block.
    scale : float or sequence of float, default=1
        Quiver scale, accepted as one value or one value per displayed field.
    stride : int, default=1
        Positive spacing between displayed vector coordinates.
    slice_nos : sequence of int, optional
        Three plane indices for native 3D fields. The middle plane is used when omitted.

    Returns
    -------
    Tuple[Any, Any]
        Figure and axes containing the quiver plots.

    Notes
    -----
    Fields are copied before indexing or row flips so plotting never mutates caller data. A
    three-axis field with a final axis of length two is interpreted as legacy channels-last.
    """
    assert isinstance(stride, (int, np.integer)) and not isinstance(stride, bool) and stride > 0, \
        'stride must be a positive integer'
    assert indexing in ['ij', 'xy']
    fields, spatial_ndim = _prepare_flow_fields(slices_in, slice_nos)
    nb_fields = len(fields) if spatial_ndim == 2 else len(fields) // 3
    nb_plots = len(fields)

    def input_check(inputs, count, name):
        if inputs is None:
            return [None] * count
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert len(inputs) in (1, count), 'number of %s is incorrect' % name
        if len(inputs) == 1:
            return [inputs[0] for _ in range(count)]
        return list(inputs)

    if spatial_ndim == 3:
        if titles is None:
            titles = ['axis %d' % d for _ in range(nb_fields) for d in range(3)]
        elif not isinstance(titles, (list, tuple)):
            titles = [titles]
        if len(titles) == nb_plots:
            titles = list(titles)
        else:
            titles = input_check(titles, nb_fields, 'titles')
            expanded_titles = []
            for title in titles:
                for axis in range(3):
                    expanded_titles.append(None if title is None else '%s: axis %d' % (title, axis))
            titles = expanded_titles
        cmaps = input_check(cmaps, nb_plots, 'cmaps')
        scale = input_check(scale, nb_fields, 'scale')
        scale = [value for value in scale for _ in range(3)]
    else:
        assert slice_nos is None, 'slice_nos is only valid for 3D flow fields'
        titles = input_check(titles, nb_plots, 'titles')
        cmaps = input_check(cmaps, nb_plots, 'cmaps')
        scale = input_check(scale, nb_plots, 'scale')

    # Copy each field separately because fields may have different spatial shapes.
    fields = [field.copy() for field in fields]
    if indexing == 'ij':
        for field in fields:
            field[..., 1] = -field[..., 1]
    if img_indexing:
        fields = [np.flipud(field) for field in fields]

    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), "grid should either be bool or [rows,cols]"
            rows, cols = grid
    elif spatial_ndim == 3:
        rows, cols = nb_fields, 3
    else:
        rows, cols = 1, nb_plots

    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axes_grid = [[axs]]
        axs = [axs]
    elif rows == 1:
        axes_grid = [axs]
        # Keep the legacy 2D return while retaining a rectangular grid for internal indexing.
        if spatial_ndim == 3:
            axs = [axs]
    elif cols == 1:
        axs = [[ax] for ax in axs]
        axes_grid = axs
    else:
        axes_grid = axs

    for index, field in enumerate(fields):
        row, col = divmod(index, cols)
        ax = axes_grid[row][col]
        ax.axis('off')
        if titles is not None and titles[index] is not None:
            ax.title.set_text(titles[index])

        # Stride reduces rendered arrow density without changing the source field.
        displayed_field = field[::stride, ::stride]
        u, v = displayed_field[..., 0], displayed_field[..., 1]
        x, y = np.meshgrid(np.arange(field.shape[1])[::stride],
                           np.arange(field.shape[0])[::stride])
        colors = np.arctan2(u, v)
        colors[np.isnan(colors)] = 0
        norm = Normalize()
        norm.autoscale(colors)
        if cmaps[index] is None:
            colormap = cm.winter
        else:
            raise Exception("custom cmaps not currently implemented for plt.flow()")

        ax.quiver(x, y, u, v, color=colormap(norm(colors).flatten()), angles='xy', units='xy',
                  width=quiver_width, scale=scale[index])
        ax.axis('equal')

    for index in range(nb_plots, rows * cols):
        row, col = divmod(index, cols)
        axes_grid[row][col].axis('off')

    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()
    if show:
        plt.show(block=plot_block)

    return (fig, axs)


def pca(pca, x, y, plot_block=True):
    x = _as_numpy_plot_array(x, 'x')
    y = _as_numpy_plot_array(y, 'y')
    x_mean = np.mean(x, 0)
    x_std = np.std(x, 0)

    W = pca.components_
    x_mu = W @ pca.mean_  # pca.mean_ is y_mean
    y_hat = x @ W + pca.mean_

    y_err = y_hat - y
    y_rel_err = y_err / np.maximum(0.5 * (np.abs(y) + np.abs(y_hat)), np.finfo('float').eps)

    plt.figure(figsize=(15, 7))
    plt.subplot(2, 3, 1)
    plt.plot(pca.explained_variance_ratio_)
    plt.title('var %% explained')
    plt.subplot(2, 3, 2)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylim([0, 1.01])
    plt.grid()
    plt.title('cumvar explained')
    plt.subplot(2, 3, 3)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylim([0.8, 1.01])
    plt.grid()
    plt.title('cumvar explained')

    plt.subplot(2, 3, 4)
    plt.plot(x_mean)
    plt.plot(x_mean + x_std, 'k')
    plt.plot(x_mean - x_std, 'k')
    plt.title('x mean across dims (sorted)')
    plt.subplot(2, 3, 5)
    plt.hist(y_rel_err.flat, 100)
    plt.title('y rel err histogram')
    plt.subplot(2, 3, 6)
    plt.imshow(W @ np.transpose(W), cmap=plt.get_cmap('gray'))
    plt.colorbar()
    plt.title('W * W\'')
    plt.show(block=plot_block)
