""" generators for the neuron project """

# general imports
import os
import numpy as np
import nibabel as nib
from keras.utils import np_utils 

# local packages
import pynd.ndutils as nd

# other neuron packages
from . import dataproc as nrn_proc

def get_file_list(volpath, ext):
    """
    get a list of files at the given path with the given extension
    """
    return [f for f in sorted(os.listdir(volpath)) if f.endswith(ext)]



def single_vol(volpath,
               ext='.npz',
               batch_size=1,
               expected_nb_files=-1,
               expected_files=None,
               data_proc_fn=None, # processing function that takes in one arg (the volume)
               relabel=None, # relabeling array
               nb_labels_reshape=0, # reshape to categorial format for keras, need # labels
               name='single_vol', # name, optional
               nb_restart_cycle=None, # number of files to restart after
               verbose_rate=None):
    """
    generator for single volume

    simple volume generator that loads a volume (via npy/mgz/nii/niigz), processes it,
    and prepares it for keras model formats
    """

    # get filenames at given paths
    volfiles = get_file_list(volpath, ext)
    nb_files = len(volfiles)

    if nb_restart_cycle is None:
        nb_restart_cycle = nb_files
    assert nb_restart_cycle <= nb_files, 'restart cycle has to be <= nb_files'

    # check the number of files matches expected (if passed)
    if expected_nb_files >= 0:
        assert nb_files == expected_nb_files, \
            "number of files do not match: %d, %d" % (nb_files, expected_nb_files)
    if expected_files is not None:
        assert volfiles == expected_files, 'file lists did not match'

    # iterate through files
    fileidx = -1
    while 1:
        for batch_idx in range(batch_size):
            fileidx = np.mod(fileidx + 1, nb_restart_cycle)
            if verbose_rate is not None and np.mod(fileidx, verbose_rate) == 0:
                print("%s: %d" %(name, fileidx))

            # read next file (circular)
            if ext == '.npz':
                vol_file = np.load(os.path.join(volpath, volfiles[fileidx]))
                vol_data = vol_file['vol_data']
            elif ext == '.mgz' or ext == '.nii' or ext == '.nii.gz':
                vol_med = nib.load(os.path.join(volpath, volfiles[fileidx]))
                vol_data = vol_med.get_data()
            else:
                raise ValueError("Unexpected extension %s" % ext)

            # process volume
            if data_proc_fn is not None:
                vol_data = data_proc_fn(vol_data)

            # need to reshape for Keras model.
            vol_data = np.expand_dims(vol_data, axis=0)

            # the original segmentation files have non-sequential relabel (i.e. some relabel are
            # missing to avoid exploding our model, we only care about the relabel that exist.
            if relabel is not None:
                resized_seg_data_fix = np.copy(vol_data)
                for idx, val in np.ndenumerate(relabel): vol_data[resized_seg_data_fix == val] = idx

            # reshape output layer as categorical
            if nb_labels_reshape > 0:
                vol_data = np_utils.to_categorical(vol_data, nb_labels_reshape)
                vol_data = np.expand_dims(vol_data, axis=0)
            else:
                vol_data = np.expand_dims(vol_data, axis=4)

            # add to batch of volume data, unless the batch is currently empty
            if batch_idx == 0:
                vol_data_batch = vol_data
            else:
                vol_data_batch = np.vstack([vol_data_batch, vol_data])

        # output matrix volume, position (pre-computed but crop)
        yield vol_data_batch




def vol_loc_seg(volpath,
                segpath,
                ext='.npz',
                batch_size=1,
                crop=None, resize_shape=None, rescale=None, # processing parameters
                relabel=None, # relabeling array
                nb_labels_reshape=0, # reshape to categorial format for keras, need # labels
                name='vol_loc_seg', # name, optional
                prior='location', # prior type: None, 'location', npz filename
                nb_restart_cycle=None, # number of files to restart after
                verbose_rate=None):
    """
    generator with ((volume, location), segmentation)
    """

    # compute processing function
    proc_vol_fn = lambda x: nrn_proc.vol_proc(x, crop=crop, resize_shape=resize_shape,
                                              interp_order=2, rescale=rescale)
    proc_seg_fn = lambda x: nrn_proc.vol_proc(x, crop=crop, resize_shape=resize_shape,
                                              interp_order=0, rescale=rescale)

    # get vol generator
    vol_gen = single_vol(volpath, ext=ext, data_proc_fn=proc_vol_fn, relabel=relabel,
                         batch_size=batch_size, nb_restart_cycle=nb_restart_cycle,
                         nb_labels_reshape=-1, name='vol', verbose_rate=None)

    # get seg generator, matching nb_files
    vol_files = [f.replace('norm', 'aseg') for f in get_file_list(volpath, ext)]
    nb_files = len(vol_files)
    seg_gen = single_vol(segpath, ext=ext, data_proc_fn=proc_seg_fn, relabel=relabel,
                         expected_files=vol_files, batch_size=batch_size,
                         nb_restart_cycle=nb_restart_cycle,
                         nb_labels_reshape=nb_labels_reshape, name='seg', verbose_rate=None)

    # get prior
    if prior == 'location':
        loc_vol = nd.volsize2ndgrid(vol_size)
        loc_vol = np.transpose(loc_vol, [1, 2, 3, 0])
        loc_vol = np.expand_dims(loc_vol, axis=0) # reshape for model

    elif prior is not None: # assumes a npz filename passed in
        data = np.load(prior)
        loc_vol = data['prior'].astype('float16')
        loc_vol = np.expand_dims(loc_vol, axis=0) # reshape for model

    # on next (while):
    idx = -1
    while 1:
        idx = np.mod(idx + 1, nb_files)
        if verbose_rate is not None and np.mod(idx, verbose_rate) == 0:
            print("%s: %d" %(name, idx))

        # get input and output (seg) vols
        input_vol = next(vol_gen).astype('float16')
        output_vol = next(seg_gen).astype('int8')

        #    make sure the samples match size
        # assert input_vol.shape == output_vol.shape, "data sizes do not match"

        # output input and output
        if prior is None:
            yield (input_vol, output_vol)
        else:
            yield ([input_vol, loc_vol], output_vol)
