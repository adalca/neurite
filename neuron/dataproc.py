''' data processing for neuron project '''

# built-in
import sys
import os
import six

# third party
import nibabel as nib
import numpy as np
import scipy.ndimage.interpolation
from tqdm import tqdm # for verbosity for forloops
from PIL import Image
import matplotlib.pyplot as plt

# import local ndutils
import pynd.ndutils as nd

# from imp import reload # for re-loading modules, since some of the modules are still in development
# reload(nd)


def proc_mgh_vols(inpath, outpath, ext='.mgz', resize_shape=None,
                  interp_order=2, rescale=None, crop=None, offset=None, clip=None):
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
    list_skipped_files = ()
    for fileidx in tqdm(range(len(files)), ncols=80):

        # load nifti volume
        volnii = nib.load(os.path.join(inpath, files[fileidx]))

        # get the data out
        vol_data = volnii.get_data().astype(float)

        if ('dim' in volnii.header) and volnii.header['dim'][4] > 1:
            vol_data = vol_data[:, :, :, -1]

        # process volume
        try:
            vol_data = vol_proc(vol_data, crop=crop, resize_shape=resize_shape,
                                interp_order=interp_order, rescale=rescale,
                                offset=offset, clip=clip)
        except Exception as e:
            list_skipped_files += (files[fileidx], )
            # print("Skipping %s\nError: %s" % (files[fileidx], str(e)), file=sys.stderr)
            continue

        # save numpy file
        outname = os.path.splitext(os.path.join(outpath, files[fileidx]))[0] + '.npz'
        np.savez_compressed(outname, vol_data=vol_data)

    for file in list_skipped_files:
        print("Skipped: %s" % file, file=sys.stderr)


def scans_to_slices(inpath, outpath, slice_nrs, ext='.mgz', 
                    label_idx=None, dim_idx=2, 
                    **kwargs):  # vol_proc args

    # get files in input directory
    files = [f for f in os.listdir(inpath) if f.endswith(ext)]

    # go through each file
    list_skipped_files = ()
    for fileidx in tqdm(range(len(files)), ncols=80):

        # load nifti volume
        volnii = nib.load(os.path.join(inpath, files[fileidx]))

        # get the data out
        vol_data = volnii.get_data().astype(float)

        if ('dim' in volnii.header) and volnii.header['dim'][4] > 1:
            vol_data = vol_data[:, :, :, -1]

        # process volume
        try:
            vol_data = vol_proc(vol_data, **kwargs)
        except Exception as e:
            list_skipped_files += (files[fileidx], )
            print("Skipping %s\nError: %s" % (files[fileidx], str(e)), file=sys.stderr)
            continue
            
        mult_fact = 255
        if label_idx is not None:
            vol_data = (vol_data == label_idx).astype(int)
            mult_fact = 1

        # extract slice
        if slice_nrs is None:
            slice_nrs_sel = range(vol_data.shape[dim_idx])
        else:
            slice_nrs_sel = slice_nrs

        for slice_nr in slice_nrs_sel:
            if dim_idx == 2:  # TODO: fix in one line
                vol_img = np.squeeze(vol_data[:, :, slice_nr])
            elif dim_idx == 1:
                vol_img = np.squeeze(vol_data[:, slice_nr, :])
            else:
                vol_img = np.squeeze(vol_data[slice_nr, :, :])
           
            # save png file
            img = (vol_img*mult_fact).astype('uint8')
            outname = os.path.splitext(os.path.join(outpath, files[fileidx]))[0] + '_slice%d.png' % slice_nr
            Image.fromarray(img).convert('RGB').save(outname)


def vol_proc(vol_data,
             crop=None,
             resize_shape=None, # None (to not resize), or vector. If vector, third entry can be None
             interp_order=None,
             rescale=None,
             rescale_prctle=None,
             resize_slices=None,
             resize_slices_dim=None,
             offset=None,
             clip=None,
             permute=None):
    ''' process a volume with a series of intensity rescale, resize and crop rescale'''

    if offset is not None:
        vol_data = vol_data + offset

    # intensity normalize data .* rescale
    if rescale is not None:
        vol_data = np.multiply(vol_data, rescale)

    if rescale_prctle is not None:
        # print("max:", np.max(vol_data.flat))
        # print("test")
        rescale = np.percentile(vol_data.flat, rescale_prctle)
        # print("rescaling by 1/%f" % (rescale))
        vol_data = np.multiply(vol_data.astype(float), 1/rescale)

    if resize_slices is not None:
        resize_slices = [*resize_slices]
        assert resize_shape is None, "if resize_slices is given, resize_shape has to be None"
        resize_shape = resize_slices
        if resize_slices_dim is None:
            resize_slices_dim = np.where([f is None for f in resize_slices])[0]
            assert len(resize_slices_dim) == 1, "Could not find dimension or slice resize"
            resize_slices_dim = resize_slices_dim[0]
        resize_shape[resize_slices_dim] = vol_data.shape[resize_slices_dim]

    # resize (downsample) matrices
    if resize_shape is not None and resize_shape != vol_data.shape:
        resize_shape = [*resize_shape]
        # allow for the last entry to be None
        if resize_shape[-1] is None:
            resize_ratio = np.divide(resize_shape[0], vol_data.shape[0])
            resize_shape[-1] = np.round(resize_ratio * vol_data.shape[-1]).astype('int')
        resize_ratio = np.divide(resize_shape, vol_data.shape)
        vol_data = scipy.ndimage.interpolation.zoom(vol_data, resize_ratio, order=interp_order)

    # crop data if necessary
    if crop is not None:
        vol_data = nd.volcrop(vol_data, crop=crop)

    # needs to be last to guarantee clip limits. 
    # For e.g., resize might screw this up due to bicubic interpolation if it was done after.
    if clip is not None:
        vol_data = np.clip(vol_data, clip[0], clip[1])
   
    # return with checks
    if clip is not None:
        assert np.max(vol_data) <= clip[1], "clip failed"
        assert np.min(vol_data) >= clip[0], "clip failed"
    return vol_data


def prior_to_weights(prior_filename, nargout=1, min_freq=0, force_binary=False, verbose=False):
    ''' transform a 4D prior (3D + nb_labels) into a class weight vector '''

    # load prior
    if isinstance(prior_filename, six.string_types):
        prior = np.load(prior_filename)['prior']
    else:
        prior = prior_filename

    # assumes prior is 4D.
    assert np.ndim(prior) == 4 or np.ndim(prior) == 3, "prior is the wrong number of dimensions"
    prior_flat = np.reshape(prior, (np.prod(prior.shape[0:(np.ndim(prior)-1)]), prior.shape[-1]))

    if force_binary:
        nb_labels = prior_flat.shape[-1]
        prior_flat[:,1] = np.sum(prior_flat[:,1:nb_labels], 1)
        prior_flat = np.delete(prior_flat, range(2,nb_labels), 1)

    # sum total class votes
    class_count = np.sum(prior_flat, 0)
    class_prior = class_count / np.sum(class_count)
    
    # adding minimum frequency
    class_prior[class_prior < min_freq] = min_freq
    class_prior = class_prior / np.sum(class_prior)

    if np.any(class_prior == 0):
        print("Warning, found a label with 0 support. Setting its weight to 0!", file=sys.stderr)
        class_prior[class_prior == 0] = np.inf

    # compute weights from class frequencies
    weights = 1/class_prior
    weights = weights / np.sum(weights)
    # weights[0] = 0 # explicitly don't care about bg

    # a bit of verbosity
    if verbose:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.bar(range(prior.size), np.log(prior))
        ax1.set_title('log class freq')
        ax2.bar(range(weights.size), weights)
        ax2.set_title('weights')
        ax3.bar(range(weights.size), np.log((weights))-np.min(np.log((weights))))
        ax3.set_title('log(weights)-minlog')
        f.set_size_inches(12, 3)
        plt.show()
        np.set_printoptions(precision=3)

    # return
    if nargout == 1:
        return weights
    else:
        return (weights, prior)
