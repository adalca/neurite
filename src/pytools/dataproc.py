''' data processing for fs mgh project '''

import sys
import os
import nibabel as nib
import numpy as np
import scipy.ndimage.interpolation
from tqdm import tqdm # for verbosity for forloops

from imp import reload # for re-loading modules, since some of the modules are still in development


# import local ndutils
sys.path.append("C:\\Users\\adalca\\Dropbox (Personal)\\code\\python\\pynd-lib\\src")
import ndutils as nd
reload(nd)


def proc_mgh_vols(inpath, outpath, ext='.mgz',
                  resize_shape=None, interp_order=2, rescale=None, crop=None):
    ''' process mgh data from mgz format and save to numpy format

        1. load file
        2. normalize intensity
        3. resize
        4. save as python block

        TODO: check header info and such.?
        '''

    # get files in input directory
    files = [f for f in os.listdir(inpath) if f.endswith(ext)]

    # go through each file
    for fileidx in tqdm(range(len(files)), ncols=80):

        # load nifti volume
        volnii = nib.load(os.path.join(inpath, files[fileidx]))

        # get the data out
        vol_data = volnii.get_data().astype(float)

        # process volume
        vol_data = vol_proc(vol_data, crop=crop, resize_shape=resize_shape,
                            interp_order=interp_order, rescale=rescale)

        # save numpy file
        outname = os.path.splitext(os.path.join(outpath, files[fileidx]))[0] + '.npz'
        np.savez_compressed(outname, vol_data=vol_data)



def vol_proc(vol_data,
             crop=None,
             resize_shape=None,
             interp_order=None,
             rescale=None):
    ''' process a volume with a series of intensity rescale, resize and crop rescale'''

    # intensity normalize data .* rescale
    if rescale is not None:
        vol_data = np.multiply(vol_data, rescale)

    # resize (downsample) matrices
    if resize_shape is not None and resize_shape != vol_data.shape:
        resize_ratio = np.divide(resize_shape, vol_data.shape)
        vol_data = scipy.ndimage.interpolation.zoom(vol_data, resize_ratio, order=interp_order)

    # crop data if necessary
    if crop is not None:
        vol_data = nd.volcrop(vol_data, crop=crop)

    return vol_data
