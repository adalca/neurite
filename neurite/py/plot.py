"""
plot utilities for the neuron project

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
import matplotlib.colors
# third party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable  # plotting
from neurite.py.flow_color import flow_uv_to_color
from typing import Any, List, Union


def slices(slices_in,           # the 2D slices
           titles=None,         # list of titles
           cmaps=None,          # list of colormaps
           norms=None,          # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,          # option to plot the images in a grid or a single row
           width=15,            # width in inches
           show=True,           # option to actually show the plot (plt.show())
           axes_off=True,
           yaxis_invert=True,   # flip the y-axis directly
           plot_block=True,     # option to plt.show()
           facecolor=None,
           imshow_args=None):
    """
    plot a grid of slices (2d images)
    """

    # input processing
    if type(slices_in) == np.ndarray:
        slices_in = [slices_in]
    nb_plots = len(slices_in)
    slices_in = list(map(np.squeeze, slices_in))
    for si, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, \
                'each slice has to be 2d or RGB (3 channels)'

    def input_check(inputs, nb_plots, name, default=None):
        """ change input from None/single-link """
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [default]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
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
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
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

        if ax.yaxis_inverted() != yaxis_invert:
            ax.invert_yaxis()

        # colorbars
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        if do_colorbars:  # and cmaps[i] is not None
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_ax, cax=cax)

    # clear axes that are unnecessary
    for i in range(nb_plots, cols * rows):
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


def volume3D(vols, slice_nos=None, data_squeeze=True, **kwargs):
    """
    plot slices of a 3D volume by taking a middle slice of each axis

    Parameters:
        vols: a 3d volume or list of 3d volumes
        slice_nos (optional): a list of 3 elements of the slice numbers for each axis, 
            or list of lists of 3 elements. if None, the middle slices will be used.
        data_squeeze: remove singleton dimensions before plotting
    """
    if not isinstance(vols, (tuple, list)):
        vols = [vols]
    nb_vols = len(vols)
    vols = list(map(np.squeeze if data_squeeze else np.asarray, vols))
    assert all(v.ndim == 3 for v in vols), 'only 3d volumes allowed in volume3D'

    slics = []
    for vi, vol in enumerate(vols):

        these_slice_nos = slice_nos
        if slice_nos is None:
            these_slice_nos = [f // 2 for f in vol.shape]
        elif isinstance(slice_nos[0], (list, tuple)):
            these_slice_nos = slice_nos[vi]
        else:
            these_slice_nos = slice_nos

        slics = slics + [np.take(vol, these_slice_nos[d], d) for d in range(3)]

    if 'titles' not in kwargs.keys():
        kwargs['titles'] = ['axis %d' % d for d in range(3)] * nb_vols

    if 'grid' not in kwargs.keys():
        kwargs['grid'] = [nb_vols, 3]

    slices(slics, **kwargs)


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


def flow(slices_in,            # the 2D slices
         titles=None,          # list of titles
         cmaps=None,           # list of colormaps. set to 'Baker' to use Baker et al. optical flow coloring method
         width=15,             # width in inches
         indexing='ij',        # plot vecs w/ matrix indexing 'ij' or cartesian indexing 'xy'
         img_indexing=True,    # whether to match the image view, i.e. flip y-axis of data.
         yaxis_invert=False,   # instead of affecting the data, flip the y-axis directly
         mode=None,            # set to 'transformer' or 't' to use defaults customized for visualizing transformer flow
         grid=False,           # option to plot the images in a grid or a single row
         show=True,            # option to actually show the plot (plt.show())
         quiver_width=None,
         reduce=1,			   # downsample the flow field by an integer factor.
         plot_block=True,      # option to plt.show()
         axis='off',           # don't display axes
         scale=1,              # note quiver essentially draws quiver length = 1/scale
         clip_flow=None,       # clip flow magnitude at this value
         pivot='tail',         # arrow style parameter passed on to `quiver`
         zero_gray_value=1.0,  # color of zero-flow (e.g. 1.0: totally white. 0.7: slightly gray)
         xlim=None,  # if not None, only plot the flow field that lies in these x-limits
         ylim=None,  # if not None, only plot the flow field that lies in these y-limits
         ):
    """
    plot a grid of flows (2d+2 images)
    """

    if mode == 'transformer' or mode == 't':        
        # Baker et al. coloring assumes xy indexing
        # (first dimension: horizontal flow. Second dim: vertical flow)
        indexing = 'xy' 
        
        # Use yaxis_invert=True to display the flow image with origin (0,0) in the
        # upper left corner, as is typically done with images.
        # Then, flipping the flow field (with img_indexing) is not necessary.
        yaxis_invert = True
        img_indexing = False

        # pivot='tip': better visualize what spatialTransformer does, vs previous
        # default of 'tail'. The flow field value at a particular indicates from which
        # direction the source image is sampled *from*, not where the pixel at a
        # particular location goes *to*. The difference is rather subtle but more
        # noticeable when displacement values are large.
        pivot = 'tip'  

    # input processing
    nb_plots = len(slices_in)
    for slice_in in slices_in:
        assert len(slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
        assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

    def input_check(inputs: Any, nb_plots, name) -> List:
        """ change input from None/single-link """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    assert indexing in ['ij', 'xy']
    slices_in = np.copy(slices_in)  # if img_indexing is True, indexing may modify slices_in in memory
	# indexing and img_indexing are handled in the flow_ax function
    # if indexing == 'ij':
    #    for si, slc in enumerate(slices_in):
    #        # Make y values negative so y-axis will point down in plot
    #        slices_in[si][:, :, 1] = -slices_in[si][:, :, 1]

    # if img_indexing:
    #    for si, slc in enumerate(slices_in):
    #        slices_in[si] = np.flipud(slc)  # Flip vertical order of y values

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    scale = input_check(scale, nb_plots, 'scale')

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
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]
        scale_i = scale[i]

        flow_ax(slices_in[i],  # the 2D slices
                title=titles[i],        # add titles
                indexing=indexing,
                img_indexing=img_indexing,
                yaxis_invert=yaxis_invert,
                cmap=cmaps[i],
                mode=mode,
                quiver_width=quiver_width,
                show=False,  # only show after all plots are drawn
                reduce=reduce,
                ax=ax,
                plot_block=False,  # 
                axis=axis,
                scale=scale_i,
                clip_flow=clip_flow,
                zero_gray_value=zero_gray_value,
                pivot=pivot,
                xlim=xlim,
                ylim=ylim,
                )


    # clear axes that are unnecessary
    for i in range(nb_plots, cols * rows):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    # preserve aspect ratio.
    aspect_ratio = slices_in[0].shape[1] / slices_in[0].shape[0]
    fig.set_size_inches(width * aspect_ratio, rows / cols * width)
    plt.tight_layout()

    if show:
        plt.show(block=plot_block)

    return (fig, axs)


def flow_ax(flow,           # the flow field shape=(H,W,2)
            title=None,     # title for this plot
            cmap=None,    # set to 'Baker' to use Baker et al. optical flow coloring method
            indexing='ij',  # plot vecs w/ matrix indexing 'ij' or cartesian indexing 'xy'
            img_indexing=True,  # whether to match the image view, i.e. flip y-axis of data.
            yaxis_invert=False,  # instead of affecting the data, flip the y-axis directly
            mode=None,      # set to 'transformer' or 't' to use defaults customized for visualizing transformer flow
            quiver_width=None,
            show=True,      # option to actually show the plot (plt.show())
            reduce=1,       # subsample the flow field by this factor.
            ax=None,        # plot the flow field on this axis.
            plot_block=True,  # option to plt.show()
            axis='off',       # don't display axes
            ticks='off',      # don't display x and y ticks
            scale=1,    # note quiver essentially draws quiver length = 1/scale
            clip_flow=None,  # clip flow magnitude at this value
            zero_gray_value=1.0,  # color of zero-flow (e.g. 1.0: totally white. 0.7: slightly gray)
            pivot='tail',         # arrow style parameter passed on to `quiver`
            xlim=None,     # if not None, only plot the flow field that lies in these x-limits
            ylim=None,     # if not None, only plot the flow field that lies in these y-limits
            ):
    """
    # Plots a single flow field on a particular matplotlib axis
    (specified by `ax`. if ax is None, uses the current axis ( plt.gca() )
    """


    if mode == 'transformer' or mode == 't':
        indexing = 'xy'
        img_indexing = False
        yaxis_invert = True
        pivot = 'tip'   # 'tip': better visualize what spatialTransformer does, vs default of 'tail'

    # input processing
    assert len(flow.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
    assert flow.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

    def input_check(inputs, nb_plots, name):
        """ change input from None/single-link """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    assert indexing in ['ij', 'xy']
    flow = flow.copy()  # if img_indexing is True, indexing may modify slices_in in memory

    if indexing == 'ij':
        flow[:, :, 1] = -flow[:, :, 1]

    if img_indexing:
        flow = np.flipud(flow)  # Flip vertical order of y values

    # prepare the subplot
    if ax is None:
        ax = plt.gca()

    # turn off axis
    if axis == 'off':
        ax.axis('off')

    if ticks == 'off':
        ax.set_xticks([])
        ax.set_yticks([])

    if yaxis_invert:
        ax.yaxis.set_inverted(True)

    u, v = flow[..., 0], flow[..., 1]
    if reduce != 1:
        u = u[::reduce, ::reduce]/reduce
        v = v[::reduce, ::reduce]/reduce

    # show figure
    quiver_u, quiver_v = u, v
    if not yaxis_invert:
        quiver_v = -quiver_v

    quiver_args = quiver_u, quiver_v
    if xlim is not None or ylim is not None:
        ny, nx = u.shape
        quiver_x, quiver_y = np.meshgrid(np.arange(nx), np.arange(ny))

        margin = 5
        x_slc, y_slc = slice(None), slice(None)
        if xlim is not None:
            x_slc = slice(max(xlim[0]-margin, 0), xlim[1]+margin)
        if ylim is not None:
            y_slc = slice(max(ylim[0]-margin, 0), ylim[1]+margin)

        quiver_x = quiver_x[y_slc, x_slc]
        quiver_y = quiver_y[y_slc, x_slc]
        quiver_u = quiver_u[y_slc, x_slc]
        quiver_v = quiver_v[y_slc, x_slc]

        quiver_args = quiver_x, quiver_y, quiver_u, quiver_v

    if cmap is None:  # original default colormap
        cmap = 'winter'

    if isinstance(cmap, str) and cmap in matplotlib.colormaps:
        cmap = matplotlib.colormaps[cmap]

    if cmap == 'Baker':
        colors_rgb = flow_uv_to_color(
            quiver_u, quiver_v, alpha=255, clip_flow=clip_flow,
            to_float=True, zero_gray_value=zero_gray_value)
        colors = colors_rgb.reshape([-1, 4])

    elif isinstance(cmap, matplotlib.colors.Colormap):
        colors = np.arctan2(quiver_u, quiver_v)
        colors[np.isnan(colors)] = 0
        norm = Normalize()
        norm.autoscale(colors)
        colors = cmap(norm(colors).flatten())
    else:
        raise ValueError(
            "Invalid argument for `cmap`: it should either be 'Baker', the name of a "
            "matplotlib colormap or a matplotlib colormap instance")

    ax.quiver(*quiver_args,  # quiver_u, quiver_v,
              color=colors,
              angles='xy',
              units='xy',
              width=quiver_width,
              pivot=pivot,
              scale=scale)

    ax.axis('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if yaxis_invert:
        ax.yaxis.set_inverted(True)

    if title is not None:
        ax.title.set_text(title)

    # show the plots
    # aspect_ratio = flow.shape[1] / flow.shape[0]
    # fig.set_size_inches(width * aspect_ratio, rows / cols * width)
    # plt.tight_layout()

    if show:
        plt.show(block=plot_block)
    



def pca(pca, x, y, plot_block=True):
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
