''' plot tools for the fs cnn project '''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # plotting

def slices(slices,              # the 2D slices
           titles=None,         # list of titles
           cmaps=None,          # list of colormaps
           norms=None,          # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           do_grid=False,       # option to plot the images in a grid or a single row
           show=True):          # option to actually show the plot (plt.show())
    ''' plot a grid of slices (2d images) '''

    # input processing
    nb_plots = len(slices)

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        assert (titles is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    norms = input_check(norms, nb_plots, 'norms')

    # figure out the number of rows and columns
    if do_grid:
        rows = np.floor(np.sqrt(nb_plots)).astype(int)
        cols = np.ceil(nb_plots/rows).astype(int)
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i/cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # some cleanup
        if titles is not None: ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices[i], cmap=cmaps[i], interpolation="nearest", norm=norms[i])

        # colorbars
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        if do_colorbars:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_ax, cax=cax)

    # show the plots
    if show:
        fig.set_size_inches(12, rows/cols*12)
        plt.show()

    return (fig, axs)



