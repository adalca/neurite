""" generators for the neuron project """

# general imports
import sys
import os

# third party imports
import numpy as np
import nibabel as nib
import scipy
from keras.utils import np_utils 
import keras
import keras.preprocessing
import keras.preprocessing.image
from keras.models import Model

# local packages
import pynd.ndutils as nd
import pytools.patchlib as pl

# reload patchlib (it's often updated right now...)
from imp import reload
reload(pl)

# other neuron (this project) packages
from . import dataproc as nrn_proc
from . import models as nrn_models




def vol(volpath,
        ext='.npz',
        batch_size=1,
        expected_nb_files=-1,
        expected_files=None,
        data_proc_fn=None,  # processing function that takes in one arg (the volume)
        relabel=None,       # relabeling array
        nb_labels_reshape=0,  # reshape to categorial format for keras, need # labels
        name='single_vol',  # name, optional
        nb_restart_cycle=None,  # number of files to restart after
        patch_size=None,     # split the volume in patches? if so, get patch_size
        patch_stride=1,  # split the volume in patches? if so, get patch_stride
        verbose_rate=None):
    """
    generator for single volume (or volume patches) from a list of files

    simple volume generator that loads a volume (via npy/mgz/nii/niigz), processes it,
    and prepares it for keras model formats
    """

    # get filenames at given paths
    volfiles = _get_file_list(volpath, ext)
    nb_files = len(volfiles)
    assert nb_files > 0, "Could not find any files"

    # compute subvolume split
    vol_data = _load_medical_volume(os.path.join(volpath, volfiles[0]), ext)
    nb_patches_per_vol = 1
    if patch_size is not None and all(f is not None for f in patch_size):
        nb_patches_per_vol = np.prod(pl.gridsize(vol_data.shape, patch_size, patch_stride))
    if nb_restart_cycle is None:
        nb_restart_cycle = nb_files * nb_patches_per_vol

    assert nb_restart_cycle <= (nb_files * nb_patches_per_vol), \
        '%s restart cycle (%s) too big (%s) in %s' % (name, nb_restart_cycle, nb_files * nb_patches_per_vol, volpath)

    # check the number of files matches expected (if passed)
    if expected_nb_files >= 0:
        assert nb_files == expected_nb_files, \
            "number of files do not match: %d, %d" % (nb_files, expected_nb_files)
    if expected_files is not None:
        assert volfiles == expected_files, 'file lists did not match'

    # iterate through files
    fileidx = -1
    batch_idx = -1
    while 1:
        fileidx = np.mod(fileidx + 1, nb_restart_cycle)
        if verbose_rate is not None and np.mod(fileidx, verbose_rate) == 0:
            print("%s fileidx: %d [/%d]" %(name, fileidx, nb_restart_cycle))

        # read next file (circular)
        vol_data = _load_medical_volume(os.path.join(volpath, volfiles[fileidx]), ext)

        # process volume
        if data_proc_fn is not None:
            vol_data = data_proc_fn(vol_data)

        # the original segmentation files have non-sequential relabel (i.e. some relabel are
        # missing to avoid exploding our model, we only care about the relabel that exist.
        if relabel is not None:
            resized_seg_data_fix = np.copy(vol_data)
            for idx, val in np.ndenumerate(relabel):
                vol_data[resized_seg_data_fix == val] = idx

        # split volume into patches if necessary and yield
        if patch_size is not None and all(f is not None for f in patch_size):
            patch_gen = patch(vol_data, patch_size, patch_stride=patch_stride,
                              nb_labels_reshape=nb_labels_reshape, batch_size=1,
                              infinite=False)
        else:
            # reshape output layer as categorical
            if nb_labels_reshape > 1:
                lpatch = np_utils.to_categorical(vol_data, nb_labels_reshape)
            elif nb_labels_reshape == 1:
                lpatch = np.expand_dims(vol_data, axis=-1)
            lpatch = np.expand_dims(lpatch, axis=0)
            patch_gen = (lpatch, )

        empty_gen = True
        for lpatch in patch_gen:
            empty_gen = False
            # add to batch of volume data, unless the batch is currently empty
            if batch_idx == -1:
                vol_data_batch = lpatch
            else:
                vol_data_batch = np.vstack([vol_data_batch, lpatch])

            # yield patch
            batch_idx += 1
            if batch_idx == batch_size - 1:
                batch_idx = -1
                yield vol_data_batch

        if empty_gen:
            raise ValueError('Patch generator was empty for file %s', volfiles[fileidx])


def patch(vol_data,             # the volume
          patch_size,           # patch size
          patch_stride=1,       # patch stride (spacing)
          nb_labels_reshape=1,  # number of labels for categorical resizing. 0 if no resizing
          batch_size=1,         # batch size
          infinite=False):      # whether the generator should continue (re)-generating patches
    """
    generate patches from volume for keras package

    Yields:
        patch: nd array of shape [batch_size, *patch_size], unless resized via nb_labels_reshape
    """

    # some parameter setup
    assert batch_size >= 1, "batch_size should be at least 1"
    patch_size = vol_data.shape if patch_size is None else patch_size
    batch_idx = -1

    # do while. if not infinite, will break at the end
    while True:
        # create patch generator
        gen = pl.patch_gen(vol_data, patch_size, stride=patch_stride)

        # go through the patch generator
        empty_gen = True
        for lpatch in gen:
            empty_gen = False

            # reshape output layer as categorical
            if nb_labels_reshape > 1:
                lpatch = np_utils.to_categorical(lpatch, nb_labels_reshape)
            elif nb_labels_reshape == 1:
                lpatch = np.expand_dims(lpatch, axis=-1)

            # reshape for Keras model.
            lpatch = np.expand_dims(lpatch, axis=0)

            # add this patch to the stack
            if batch_idx == -1:
                patch_data_batch = lpatch
            else:
                patch_data_batch = np.vstack([patch_data_batch, lpatch])

            # yield patch
            batch_idx += 1
            if batch_idx == batch_size - 1:
                batch_idx = -1
                yield patch_data_batch

        if empty_gen:
            raise Exception('generator was empty')

        # if not infinite generation, yield the last batch and break the while
        if not infinite:
            if batch_idx >= 0:
                yield patch_data_batch
            break


def vol_seg(volpath,
            segpath,
            crop=None, resize_shape=None, rescale=None, # processing parameters
            verbose_rate=None,
            name='vol_seg', # name, optional
            ext='.npz',
            nb_restart_cycle=None,  # number of files to restart after
            nb_labels_reshape=-1,
            **kwargs): # named arguments for vol(...), except verbose_rate, data_proc_fn, ext, nb_labels_reshape and name (which this function will control when calling vol()) 
    """
    generator with (volume, segmentation)
    """

    # compute processing function
    proc_vol_fn = lambda x: nrn_proc.vol_proc(x, crop=crop, resize_shape=resize_shape,
                                              interp_order=2, rescale=rescale)
    proc_seg_fn = lambda x: nrn_proc.vol_proc(x, crop=crop, resize_shape=resize_shape,
                                              interp_order=0, rescale=rescale)

    # get vol generator
    vol_gen = vol(volpath, **kwargs, ext=ext, nb_restart_cycle=nb_restart_cycle,
                  data_proc_fn=proc_vol_fn, nb_labels_reshape=1, name='vol', verbose_rate=None)

    # get seg generator, matching nb_files
    vol_files = [f.replace('norm', 'aseg') for f in _get_file_list(volpath, ext)]
    nb_files = len(vol_files)
    seg_gen = vol(segpath, **kwargs, ext=ext, nb_restart_cycle=nb_restart_cycle,
                  data_proc_fn=proc_seg_fn, nb_labels_reshape=nb_labels_reshape,
                  expected_files=vol_files, name='seg', verbose_rate=None)

    # on next (while):
    idx = -kwargs['batch_size']
    while 1:
        idx = np.mod(idx + kwargs['batch_size'], nb_restart_cycle)
        if verbose_rate is not None and np.mod(idx, verbose_rate) == 0:
            print("%s [vol_seg]: %d [/ %d]" %(name, idx, nb_restart_cycle))

        # get input and output (seg) vols
        input_vol = next(vol_gen).astype('float16')
        output_vol = next(seg_gen).astype('int8')

        # output input and output
        yield (input_vol, output_vol)


def vol_cat(volpaths, # expect two folders in here
            crop=None, resize_shape=None, rescale=None, # processing parameters
            verbose_rate=None,
            name='vol_cat', # name, optional
            ext='.npz',
            nb_labels_reshape=-1,
            **kwargs): # named arguments for vol(...), except verbose_rate, data_proc_fn, ext, nb_labels_reshape and name (which this function will control when calling vol()) 
    """
    generator with (volume, binary_bit) (random order)
    ONLY works with abtch size of 1 for now
    """

    folders = [f for f in sorted(os.listdir(volpaths))]

    # compute processing function
    proc_vol_fn = lambda x: nrn_proc.vol_proc(x, crop=crop, resize_shape=resize_shape,
                                              interp_order=2, rescale=rescale)

    # get vol generators
    generators = ()
    generators_len = ()
    for folder in folders:
        vol_gen = vol(os.path.join(volpaths, folder), **kwargs, ext=ext,
                      data_proc_fn=proc_vol_fn, nb_labels_reshape=1, name=folder, verbose_rate=None)
        generators_len += (len(_get_file_list(os.path.join(volpaths, folder), '.npz')), )
        generators += (vol_gen, )

    bake_data_test = False
    if bake_data_test:
        print('fake_data_test', file=sys.stderr)

    # on next (while):
    while 1:
        # build the random order stack
        order = np.hstack((np.zeros(generators_len[0]), np.ones(generators_len[1]))).astype('int')
        np.random.shuffle(order) # shuffle
        for idx in order:
            gen = generators[idx]

        # for idx, gen in enumerate(generators):
            z = np.zeros([1, 2]) #1,1,2 for categorical binary style
            z[0,idx] = 1 #
            # z[0,0,0] = idx

            data = next(gen).astype('float32')
            if bake_data_test and idx == 0:
                # data = data*idx
                data = -data

            yield (data, z)


def vol_seg_prior(*args,
                  prior_type='location',  # file-static, file-gen, location
                  prior_file=None,  # prior filename
                  prior_feed='input',  # input or output
                  patch_stride=1,
                  patch_size=None,
                  batch_size=1,
                  **kwargs):
    """
    generator that appends prior to (volume, segmentation) depending on input
    e.g. could be ((volume, prior), segmentation)
    """

    nb_patch_elems = np.prod(batch_size)

    # prepare the vol_seg
    gen = vol_seg(*args, **kwargs,
                  patch_size=patch_size, patch_stride=patch_stride, batch_size=batch_size)

    # get prior
    if prior_type == 'location':
        prior_vol = nd.volsize2ndgrid(vol_size)
        prior_vol = np.transpose(prior_vol, [1, 2, 3, 0])
        prior_vol = np.expand_dims(prior_vol, axis=0) # reshape for model

    else: # assumes a npz filename passed in prior_file
        data = np.load(prior_file)
        prior_vol = data['prior'].astype('float16')
    nb_channels = prior_vol.shape[-1]

    # get the prior to have the right volume [x, y, z, nb_channels]
    assert np.ndim(prior_vol) == 4, "prior is the wrong size"

    # prior generator
    if patch_size is None:
        patch_size = prior_vol.shape[0:3]
    prior_gen = patch(prior_vol, patch_size + (nb_channels,),
                      patch_stride=patch_stride, batch_size=batch_size, infinite=True, nb_labels_reshape=0)

    # generator loop
    while 1:

        # generate input and output volumes
        input_vol, output_vol = next(gen)

        # generate patch batch
        prior_batch = next(prior_gen)

        # reshape for model
        # prior_batch = np.reshape(prior_batch, (batch_size, np.prod(patch_size), nb_channels))

        if prior_feed == 'input':
            yield ([input_vol, prior_batch], output_vol)
        else:
            assert prior_feed == 'output'
            yield (input_vol, [output_vol, prior_batch])


def max_patch_in_vol_cat(volpaths,
                         patch_size,
                         patch_stride,
                         model,
                         tst_model,
                         out_layer,
                         name='max_patch_in_vol_cat', # name, optional
                         **kwargs): # named arguments for vol_cat(...)
    """
    given a model by reference
    goes to the next volume and, given a patch_size, finds
    the highest prediction of out_layer_name, and yields that patch index to the model

    TODO: this doesn't work if called while training, something about running through the graph
    perhaps need to do a different tf session?
    """

    # strategy: call vol_cat on full volume
    vol_cat_gen = vol_cat(volpaths, **kwargs)

    # need to make a deep copy:
    #model_tmp = Model(inputs=model.inputs, outputs=model.get_layer(out_layer).output)
    # nrn_models.copy_weights(model_tmp, tst_model)
    #nrn_models.copy_weights(tst_model, tst_model)
    #asdasd
    tst_model = model

    while 1:
        # get next volume
        try:
            sample = next(vol_cat_gen)
        except:
            print("Failed loading file. Skipping", file=sys.stderr)
            continue
        sample_vol = np.squeeze(sample[0])
        patch_gen = patch(sample_vol, patch_size, patch_stride=patch_stride)

        # go through its patches
        max_resp = -np.infty
        max_idx = np.nan
        max_out = None
        for idx, ptc in enumerate(patch_gen):
            # predict
            res = np.squeeze(tst_model.predict(ptc))[1]
            if res > max_resp:
                max_ptc = ptc
                max_out = sample[1]
                max_idx = idx

        # yield the right patch
        yield (max_ptc, max_out)


def img_seg(volpath,
            segpath,
            batch_size=1,
            verbose_rate=None,
            name='img_seg', # name, optional
            ext='.png'):
    """
    generator with (image, segmentation)
    """

    def imggen(path, ext):
        files = _get_file_list(path, ext)
        idx = -1
        while 1:
            idx = np.mod(idx+1, len(files))
            im = scipy.misc.imread(os.path.join(path, files[idx]))[:, :, 0]
            yield im.reshape((1,) + im.shape)

    img_gen = imggen(volpath, ext)
    seg_gen = imggen(segpath, ext)

    # on next (while):
    while 1:
        input_vol = np.vstack([next(img_gen).astype('float16')/255 for i in range(batch_size)])
        input_vol = np.expand_dims(input_vol, axis=-1)

        output_vols = [np_utils.to_categorical(next(seg_gen).astype('int8'), num_classes=2) for i in range(batch_size)]
        output_vol = np.vstack([np.expand_dims(f, axis=0) for f in output_vols])

        # output input and output
        yield (input_vol, output_vol)


# Some internal use functions

def _get_file_list(volpath, ext=None):
    """
    get a list of files at the given path with the given extension
    """
    return [f for f in sorted(os.listdir(volpath)) if ext is None or f.endswith(ext)]


def _load_medical_volume(filename, ext):
    """
    load a medical volume from one of a number of file types
    """
    if ext == '.npz':
        vol_file = np.load(filename)
        vol_data = vol_file['vol_data']
    elif ext == 'npy':
        vol_data = np.load(filename)
    elif ext == '.mgz' or ext == '.nii' or ext == '.nii.gz':
        vol_med = nib.load(filename)
        vol_data = vol_med.get_data()
    else:
        raise ValueError("Unexpected extension %s" % ext)

    return vol_data


