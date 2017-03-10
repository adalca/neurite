''' generators for the fs_cnn project '''
import os
import numpy as np
import nibabel as nib
from keras.utils import np_utils 
from . import dataproc


# generator for single volume
def single_vol(volpath,
               ext='.npz',
               expected_nb_files=-1,
               data_proc_fn=None, # processing function that takes in one arg (the volume)         
               relabel=None, # relabeling array
               nb_labels_reshape=0, # reshape to categorial format for keras, need # labels
               name='single_vol', # name, optional
               verbose_rate=None):
    ''' simple volume generator that loads a volume (via npy/mgz/nii/niigz), processes it,
        and prepares it for keras model formats'''

    # get filenames at given paths
    volfiles = [f for f in os.listdir(volpath) if f.endswith(ext)]
    nb_files = len(volfiles)

    # check the number of files matches expected (if passed)
    if expected_nb_files >= 0:
        assert nb_files == expected_nb_files, \
            "number of files do not match: %d, %d" % (nb_files, expected_nb_files)

    # iterate through files
    fileidx = -1
    while 1:
        fileidx = np.mod(fileidx + 1, nb_files)
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
        # TODO: not sure why this needs these specific dimensions
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

        # output matrix volume, position (pre-computed but crop)
        yield vol_data




def vol_loc_seg(volpath,
                segpath,
                ext='.npz',
                crop=None, resize_shape=None, rescale=None, # processing parameters
                relabel=None, # relabeling array
                nb_labels_reshape=0, # reshape to categorial format for keras, need # labels
                name='vol_loc_seg', # name, optional
                prior='location', # prior type: None, 'location', npz filename
                verbose_rate=None):
    '''prepare a generator with ((volume, location), segmentation)'''

    # compute processing function
    proc_vol_fn = lambda x: dataproc.vol_proc(x, crop=crop, resize_shape=resize_shape,
                                              interp_order=2, rescale=rescale)
    proc_seg_fn = lambda x: dataproc.vol_proc(x, crop=crop, resize_shape=resize_shape,
                                              interp_order=0, rescale=rescale)

    # get vol generator
    vol_gen = single_vol(volpath, ext=ext, data_proc_fn=proc_vol_fn, relabel=relabel,
                         nb_labels_reshape=-1, name='vol', verbose_rate=None)

    # get seg generator, matching nb_files
    nb_files = len([f for f in os.listdir(volpath) if f.endswith(ext)])
    seg_gen = single_vol(segpath, ext=ext, data_proc_fn=proc_seg_fn, relabel=relabel,
                         expected_nb_files=nb_files,
                         nb_labels_reshape=nb_labels_reshape, name='seg', verbose_rate=None)

    # get prior
    if prior == 'location':
        loc_vol = np.mgrid[0:vol_size[0], 0:vol_size[1], 0:vol_size[2]]
        loc_vol = np.transpose(loc_vol, [1, 2, 3, 0])
        loc_vol = np.expand_dims(loc_vol, axis=0) # reshape for model

    elif prior is not None: # assumes a npz filename passed in
        data = np.load(prior)
        loc_vol = data['prior']
        loc_vol = np.expand_dims(loc_vol, axis=0) # reshape for model

    # on next (while):
    fileidx = -1
    while 1:
        fileidx = np.mod(fileidx + 1, nb_files)
        if verbose_rate is not None and np.mod(fileidx, verbose_rate) == 0:
            print("%s: %d" %(name, fileidx))

        # get input and output (seg) vols
        input_vol = next(vol_gen)
        output_vol = next(seg_gen)

        #    make sure the samples match size
        # assert input_vol.shape == output_vol.shape, "data sizes do not match"

        # output input and output
        if prior is None:
            yield (input_vol, output_vol)
        else:
            yield ([input_vol, loc_vol], output_vol)
