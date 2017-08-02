"""
sandbox (often changing and very specific) functions for neuron project 
"""

import os
from imp import reload
import six

# third party imports
import numpy as np
from tqdm import tqdm
import keras
from keras_tqdm import TQDMNotebookCallback
import keras.callbacks as keras_callbacks
import matplotlib.pylab as plt
import matplotlib

# personal import
import medipy
import neuron.callbacks as nrn_callbacks; reload(nrn_callbacks)
import neuron.generators as nrn_gen; reload(nrn_gen)
import neuron.metrics as nrn_metrics; reload(nrn_metrics)
import neuron.models as nrn_models; reload(nrn_models)
import neuron.utils as nrn_utils; reload(nrn_utils)
import neuron.dataproc as nrn_dataproc; reload(nrn_dataproc)
import neuron.plot as nrn_plt; reload(nrn_plt)
import pytools.iniparse as ini
import pytools.patchlib as pl
from pytools import plotting

import pynd.ndutils as nd; reload(nd)
import pynd.segutils as su; reload(su)
import pytools.timer as timer



def nb_steps_and_files(nb_files, nb_patches_per_volume, batch_size):
    """
    computes the number of steps per epoch and the number of files going through,
    with a bunch of checks
    """

    # get the number of files in the path
    if isinstance(nb_files, six.string_types):
        nb_files = len(os.listdir(nb_files))
    nb_steps = ((nb_files * nb_patches_per_volume - 1) // batch_size) + 1

    """
    Keeping some old sets around in case they are useful, but in the end we decided
    against these constraints, as long as the generators output a smaller back size for
    the last batch in a folder

    # if the batch size is smaller than the number of patches per volume, then
    # we require the ppv to be a multiple of the batch size. Otherwise, we get
    # into some ugly math and are not able to complete nice consistent cycles
    if batch_size < nb_patches_per_volume:
        assert np.mod(nb_patches_per_volume, batch_size) == 0, \
        "%d batches does not fit into  %d patches_per_volume" % (batch_size, nb_patches_per_volume)
        nb_steps = nb_files * nb_patches_per_volume // batch_size

        # # paranoia assert. This should be true given the previous assert. But I'm tired...
        assert np.mod(nb_files * nb_patches_per_volume, batch_size) == 0, \
        "%d batches does not fit into  %d patches_per_volume" % (batch_size, nb_patches_per_volume)

    # if the batch size is larger than the number of patches per volume, then
    # as long as we have enough batches in the dataset, then find the number of voxels
    else:
        assert nb_files * nb_patches_per_volume >= batch_size, "not enough batches in the data"
        nb_steps = nb_files * nb_patches_per_volume // batch_size
        nb_files = batch_size * nb_steps // nb_patches_per_volume

        # old, inappropriate asserts
        # assert np.mod(batch_size, nb_patches_per_volume) == 0, \
        # "%d patches_per_volume does not fit into  %d batches" % (nb_patches_per_volume, batch_size)
        # assert np.mod(nb_files * nb_patches_per_volume, batch_size) == 0, \
        # 'batch_size %d does not fit in prod %d' % (batch_size, nb_files * nb_patches_per_volume)
        # nb_steps = nb_files * nb_patches_per_volume // batch_size

    assert batch_size * nb_steps == nb_files * nb_patches_per_volume, "something went wrong"
    """

    # add to the list of tuples
    return (nb_steps, nb_files)


def prepare_run_params(setup_file, model_file, data_file, run_file, 
                       procname='t1_proc',
                       outname='t1_output',
                       prior_filename="seg_prior.npz",
                       verbose=True):
    """
    prepare the setup and parameter structures for neuron experimentation
    see examples in config folder for each of these files.

    TODO_maybe: procname and outname should really be replaced with just a single proc and output name
    """

    # parse files
    paths = ini.ini_to_struct(setup_file).paths
    model = ini.ini_to_struct(model_file).model
    data = ini.ini_to_struct(data_file).data
    run = ini.ini_to_struct(run_file).run

    # cleanup parameters
    data.labels = np.array(data.labels)
    data.nb_labels = data.labels.size
    data.vol_size = list(data.vol_size)

    # patch size and stride checks
    if run.patch_size is not None and run.patch_size[0] is not None:
        run.patch_size = list(run.patch_size)
    if not isinstance(run.patch_stride, list):
        run.patch_stride = (run.patch_stride,)
    if run.patch_size is not None and len(run.patch_stride) == 1 and len(run.patch_size) > 1:
        run.patch_stride = [run.patch_stride[0] for f in run.patch_size]

    # compute the grid size
    if run.patch_size is None or run.patch_size[0] is None:
        run.grid_size = [1] * len(data.vol_size)
    elif len(run.patch_size) == 2 and len(data.vol_size) == 3:
        run.grid_size = pl.gridsize(data.vol_size,
                                    [*run.patch_size, 1],
                                    patch_stride=[*run.patch_stride, 1])
    else:
        run.grid_size = pl.gridsize(data.vol_size,
                                    run.patch_size,
                                    patch_stride=run.patch_stride)
    run.nb_patches_per_volume = np.prod(run.grid_size)

    # prepare a datalink to the folder structure
    # a lambda function that takes in split_type (train/test/validate) and io_type (vols/asegs)
    # paths.datalink('train', 'vols') should take you to the training volumes, etc
    assert not hasattr(paths, 'datalink'), 'paths.datalink is reserved for neuron operation'
    paths.datalink = lambda x, y: os.path.join(getattr(paths, procname), data.folder, x, y)

    # prepare prior
    paths.prior = os.path.join(getattr(paths, procname), data.folder, 'train', prior_filename)

    # model output folder
    name = model.name + "___" + data.folder + "___" + run.name
    paths.output = os.path.join(getattr(paths, outname), name)
    if verbose:
        print("Model folder name: \n%s" % paths.output)
    if not os.path.isdir(paths.output): os.mkdir(paths.output)

    return (paths, model, data, run)


def prep_run_output_dir(model_folder, increment_run, existing_run_id=None):
    """
    prepare the output dirs for this run
    """

    # dump or add to a file with the run_id in the main folder
    runs_dir = os.path.join(model_folder, 'runs')
    if not os.path.isdir(runs_dir):
        os.mkdir(runs_dir)

    # get current run id
    if existing_run_id is None:
        existing_run_id = len(os.listdir(runs_dir))

    # increment the run or not
    run_id = existing_run_id + 1 if increment_run else existing_run_id
    this_run_dir = os.path.join(runs_dir, 'run_%d' % run_id)
    if not os.path.isdir(this_run_dir):
        os.mkdir(this_run_dir)

    # file_mod = 'w' if increment_run else 'a'
    # with(os.path.join(notes_dir, "run_%d" % run_id), file_mod):
        # print(params_file, file)

    return this_run_dir


def seg_callbacks(run_dir,
                  run,
                  data,
                  batch_size,
                  generators,
                  train_nb_dice=None,
                  validation_nb_dice=None,
                  metric_callbacks_batch_size=1,
                  print_period=10,
                  metrics_period=10,
                  at_batch_end=None,
                  seg_verbose=False):
    """
    usual callbacks for segmentation
    """
    nb_labels = data.nb_labels

    

    callbacks = {}

    # model saving
    hdf_dir = os.path.join(run_dir, 'hdf5')
    if not os.path.isdir(hdf_dir):
        os.mkdir(hdf_dir)
    filename = os.path.join(hdf_dir, 'model.{epoch:02d}-{iter:02d}.hdf5')
    callbacks['save'] = nrn_callbacks.ModelCheckpoint(filename,
                                                      monitor='val_loss',  # is this needed?
                                                      verbose=seg_verbose,
                                                      save_best_only=False,
                                                      save_weights_only=False,
                                                      at_batch_end=at_batch_end,
                                                      mode='auto',
                                                      period=1)

    # png on test data
    png_dir = os.path.join(run_dir, 'png-test')
    if not os.path.isdir(png_dir):
        os.mkdir(png_dir)
    filename = os.path.join(png_dir, 'test.{epoch:02d}-{iter:02d}-{axis:s}-{slice_nr:d}.png')
    callbacks['print'] = nrn_callbacks.PlotTestSlices(filename,
                                                      generators['print'],
                                                      run.patch_size,
                                                      run,
                                                      data,
                                                      at_batch_end=at_batch_end,
                                                      period=print_period,
                                                      verbose=seg_verbose)

    # metrics callbacks
    # do volume if patch is 2d and volume is 3d
    if len(run.patch_size) == 2 and len(data.vol_size) == 3:
        vol_params = {'patch_size': [*run.patch_size, 1],
                      'patch_stride': [*run.patch_stride, 1],
                      'grid_size': run.grid_size}
    else:
        vol_params = {'patch_size': run.patch_size,
                      'patch_stride': run.patch_stride,
                      'grid_size': run.grid_size}
    dice_metric = lambda x, y: medipy.metrics.dice(x, y, labels=range(nb_labels))
    predict_metrics = nrn_callbacks.PredictMetrics
    dice_metric.__name__ = 'hardDiceTraining'
    callbacks['metrics_train'] = predict_metrics(None,
                                                 [dice_metric],
                                                 data_generator=generators['metrics_train'],
                                                 nb_samples=train_nb_dice,
                                                 nb_labels=nb_labels,
                                                 batch_size=metric_callbacks_batch_size,
                                                 vol_params=vol_params,
                                                 period=metrics_period,
                                                 at_batch_end=at_batch_end,
                                                 verbose=seg_verbose)
    dice_metric.__name__ = 'hardDiceValidation'
    callbacks['metrics_validate'] = predict_metrics(None,
                                                    [dice_metric],
                                                    data_generator=generators['metrics_validate'],
                                                    nb_samples=validation_nb_dice,
                                                    nb_labels=nb_labels,
                                                    batch_size=metric_callbacks_batch_size,
                                                    vol_params=vol_params,
                                                    period=metrics_period,
                                                    at_batch_end=at_batch_end,
                                                    verbose=seg_verbose)

    # weight checking.
    callbacks['weight_checking'] = nrn_callbacks.ModelWeightCheck(at_batch_end=True)

    # tensorboard
    log_dir = os.path.join(run_dir, 'log')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    callbacks['tensorboard'] = keras_callbacks.TensorBoard(log_dir=log_dir,
                                                           histogram_freq=0,
                                                           write_graph=True,
                                                           write_images=True)

    # tqdm
    callbacks['tqdm'] = TQDMNotebookCallback()

    # return the dictionary of callbacks
    return callbacks


def seg_generators(paths, model, data, run, batch_size,
                   nb_train_metric_files,
                   nb_validate_metric_files,
                   metric_callbacks_batch_size=1,
                   verbose=False,
                   nb_train_files=None,  # if None, will be estimated below
                   nb_validate_files=None,  # if None, will be estimated below
                   nb_test_files=None,  # if None, will be estimated below
                   rand_seed_vol=None,
                   label_blur_sigma=None,
                   nb_input_feats=1,
                   gen_verbose=False,
                   extra_gens={}):
    """
    usual generators for segmentation
    """

    # compute patch size, and dimension-collapse if necessary
    patch_size = run.patch_size
    patch_stride = run.patch_stride
    collapse_2d = None
    vol_proc = lambda x: x
    if len(patch_size) == 2 and len(data.vol_size) == 3:
        collapse_2d = run.collapse_2d
        assert run.collapse_2d == 2, "Some of this code assumes collapse_2d is 2"
        patch_size = [*patch_size, 1]
        patch_stride = [*patch_stride, 1]

        # process volumes with a crop
        if hasattr(run, 'extract_slices') and run.extract_slices is not None:
            assert isinstance(run.extract_slices, (list, tuple))
            extract_ix = nd.arange([0 for f in data.vol_size], data.vol_size)
            extract_ix[run.collapse_2d] = run.extract_slices
            vol_proc = lambda x: nrn_dataproc.vol_proc(x, extract_nd=extract_ix)

    if rand_seed_vol is None:
        rand_seed_vol = np.random.randint(0, 1000)

    # prepare arguments for generators
    gen_args = {'ext' : data.ext,
                'relabel': data.labels,
                'nb_labels_reshape': data.nb_labels,
                'batch_size': batch_size,
                'patch_size': patch_size,
                'patch_stride': patch_stride,
                'proc_vol_fn':vol_proc,
                'proc_seg_fn':vol_proc,
                'collapse_2d':collapse_2d,
                'rand_seed_vol':rand_seed_vol,
                'nb_input_feats':nb_input_feats,
                'verbose':gen_verbose}

    # prepare the generator function depending on whether a prior is used
    genfcn = nrn_gen.vol_seg
    genfcn_vol = nrn_gen.vol_count
    genfcn_ext = nrn_gen.vol_ext_data
    if model.include_prior:
        if verbose:
            print('Using prior')
        gen_args['prior_type'] = 'file'
        gen_args['prior_file'] = paths.prior
        gen_args['prior_feed'] = 'input'
        genfcn = nrn_gen.vol_seg_prior
        genfcn_vol = nrn_gen.vol_count_prior
        genfcn_ext = nrn_gen.vol_ext_data_prior
    if data.ext == '.png':
        genfcn = nrn_gen.img_seg

    # get generators
    generators = {}

    # main training generator
    if nb_train_files is None:
        _, nb_train_files = nb_steps_and_files(paths.datalink('train', 'vols'),
                                               run.nb_patches_per_volume,
                                               batch_size)
    gen_args['nb_restart_cycle'] = nb_train_files  # sample the same training files
    generators['train'] = genfcn(paths.datalink('train', 'vols'),
                                 paths.datalink('train', 'asegs'),
                                 name='training_gen',
                                 **gen_args)
    generators['train-vol'] = genfcn_vol(paths.datalink('train', 'vols'),
                                 paths.datalink('train', 'asegs'),
                                 name='training_gen',
                                 label_blur_sigma=label_blur_sigma,
                                 **gen_args)
    generators['train-ext'] = genfcn_ext(paths.datalink('train', 'vols'),
                                 paths.datalink('train', 'external'),
                                 name='training_gen',
                                 **gen_args)

    # main validation generator
    if nb_validate_files is None:
        _, nb_validate_files = nb_steps_and_files(paths.datalink('validate', 'vols'),
                                                  run.nb_patches_per_volume,
                                                  batch_size)
    gen_args['nb_restart_cycle'] = nb_validate_files  # sample the same validation files
    generators['validate'] = genfcn(paths.datalink('validate', 'vols'),
                                    paths.datalink('validate', 'asegs'),
                                    name='validation_gen',
                                    **gen_args)
    generators['validate-vol'] = genfcn_vol(paths.datalink('validate', 'vols'),
                                    paths.datalink('validate', 'asegs'),
                                    name='validation_gen',
                                    label_blur_sigma=label_blur_sigma,
                                    **gen_args)
    generators['validate-ext'] = genfcn_ext(paths.datalink('validate', 'vols'),
                                    paths.datalink('validate', 'external'),
                                    name='validation_gen',
                                    **gen_args)

    # test
    if nb_test_files is None:
        _, nb_test_files = nb_steps_and_files(paths.datalink('test', 'vols'),
                                              run.nb_patches_per_volume,
                                              batch_size)
    gen_args['nb_restart_cycle'] = nb_test_files  # sample the same validation files
    gen_args['batch_size'] = 1
    generators['test'] = genfcn(paths.datalink('test', 'vols'),
                                paths.datalink('test', 'asegs'),
                                name='test_gen',
                                **gen_args)
    generators['test-2'] = genfcn(paths.datalink('test', 'vols'),
                                  paths.datalink('test', 'asegs'),
                                  name='test_gen',
                                  
                                  **gen_args)
    for k,v in extra_gens.items():
        gen_args['rand_seed_vol'] = None
        _, nb_test_files_extra = nb_steps_and_files(paths.datalink(v, 'vols'),
                                              run.nb_patches_per_volume,
                                              batch_size)
        gen_args['nb_restart_cycle'] = nb_test_files_extra
        generators[k] = genfcn(paths.datalink(v, 'vols'),
                                  paths.datalink(v, 'asegs'),
                                  name='%s_gen'%v,
                                  **gen_args)

    # generators for metrics
    gen_args['batch_size'] = metric_callbacks_batch_size
    gen_args['nb_restart_cycle'] = nb_train_metric_files  # sample the same training files
    generators['metrics_train'] = genfcn(paths.datalink('train', 'vols'),
                                         paths.datalink('train', 'asegs'),
                                         name='metric_train_gen',
                                         **gen_args)

    gen_args['batch_size'] = metric_callbacks_batch_size
    gen_args['nb_restart_cycle'] = nb_validate_metric_files  # sample the same training files
    generators['metrics_validate'] = genfcn(paths.datalink('validate', 'vols'),
                                            paths.datalink('validate', 'asegs'),
                                            name='metric_validate_gen',
                                            **gen_args)

    # test separately used for callbacks
    gen_args['batch_size'] = 1
    gen_args['nb_restart_cycle'] = 1
    generators['print'] = genfcn(paths.datalink('test', 'vols'),
                                 paths.datalink('test', 'asegs'),
                                 name='test_gen',
                                 **gen_args)

    return generators


def seg_losses(nb_labels,
               prior_filename=None,
               weights=None,
               patch_size=None,
               disc=None,
               dice_mix_weights=[0.01,1]):
    """
    usual losses for segmentation models
    """

    # compute weights
    if prior_filename is not None:
        assert weights is None, "cannot provide both weights and prior"
        weights, prior = nrn_dataproc.prior_to_weights(prior_filename,
                                                       nargout=2,
                                                       min_freq=0.001)
    assert weights is not None, "weights cannot be None"

    # prepare weights with 0-weighted bg (first label)
    weights0bg = list(weights)
    weights0bg[0] = 0

    # losses
    losses = {}

    # CC losses
    loss = nrn_metrics.CategoricalCrossentropy().loss
    losses['cc'] = _loss_with_name(loss, 'cc_loss')
    loss = nrn_metrics.CategoricalCrossentropy(weights=weights).loss
    losses['cc_wt'] = _loss_with_name(loss, 'cc_wt_loss')
    loss = nrn_metrics.CategoricalCrossentropy(weights=weights0bg).loss
    losses['cc_wt0bg'] = _loss_with_name(loss, 'cc_wt0bg_loss')
    # pwcce = nrn_metrics.CategoricalCrossentropy(weights=weights, prior=prior).loss
    # ppwcce = nrn_metrics.CategoricalCrossentropy(weights=weights, prior=prior_filename,
    #   patch_size=patch_size, patch_stride=patch_stride,batch_size=batch_size).loss

    dice = nrn_metrics.Dice
    loss = dice(nb_labels, dice_type='hard').loss
    losses['dice_hard'] = _loss_with_name(loss, 'dice_hard_loss')
    loss = dice(nb_labels, dice_type='soft').loss
    losses['dice_soft'] = _loss_with_name(loss, 'dice_soft_loss')
    loss = dice(nb_labels, weights=weights, dice_type='hard').loss
    losses['dice_wt_hard'] = _loss_with_name(loss, 'dice_wt_hard_loss')
    loss = dice(nb_labels, weights=weights0bg, dice_type='hard').loss
    losses['dice_wt0bg_hard'] = _loss_with_name(loss, 'dice_wt0bg_hard_loss')
    loss = dice(nb_labels, weights=weights, dice_type='soft').loss
    losses['dice_wt_soft'] = _loss_with_name(loss, 'dice_wt_soft_loss')
    loss = dice(nb_labels, weights=weights0bg, dice_type='soft').loss
    losses['dice_wt0bg_soft'] = _loss_with_name(loss, 'dice_wt0bg_soft_loss')

    # TODO: we can do this automatically when setting up the models by having 
    # two losses and specifying the weight!
    mix_losses = (losses['dice_wt_soft'], losses['cc_wt'])
    mix_weights = dice_mix_weights
    loss = nrn_metrics.Mix(mix_losses, mix_weights).loss
    losses['mix_dice_wt_soft_cc_wt'] = _loss_with_name(loss, 'mix_dice_wt_soft_cc_wt_loss')

    if patch_size is not None:
        # dist from center
        patch_center = (np.array(patch_size) - 1) / 2
        ndgrid = nd.volsize2ndgrid(patch_size)
        nddst = np.zeros(patch_size)
        for f, _ in enumerate(ndgrid):
            nddst += np.square(ndgrid[f]-patch_center[f])
        # how low should half way between the center and the edge be in the vote?
        quarter_vote = 0.5
        # figure out the lambda in exp(-lambda * r) = quater_vote, where r = patch-size/2
        lbd = np.mean(- 2 / np.array(patch_size) * np.log(quarter_vote))
        # compute patch_size weight
        ex = np.exp(-lbd * nddst)
        ex = np.tile(np.expand_dims(ex.flat, 1), [1, nb_labels])

        # compute losses with vox_weights
        # cw = center_weighed
        loss = nrn_metrics.CategoricalCrossentropy(weights=weights, vox_weights=ex).loss
        losses['cc_wt_cw'] = _loss_with_name(loss, 'cc_wt_cw_loss')
        loss = dice(nb_labels, weights=weights, dice_type='soft').loss
        losses['dice_wt_soft_cw'] = _loss_with_name(loss, 'dice_wt_soft_loss_loss')
        mix_losses = (losses['dice_wt_soft_cw'], losses['cc_wt_cw'])
        loss = nrn_metrics.Mix(mix_losses, mix_weights).loss
        losses['mix_dice_wt_soft_cw_cc_wt_cw'] = _loss_with_name(loss, 'mix_dice_wt_soft_cw_cc_wt_cw_loss')


    loss = nrn_metrics.WGAN_GP(disc).loss
    losses['wgan-gp'] = _loss_with_name(loss, 'wgan-gp_loss')

    return losses




def seg_models(model, run, data, load_loss, seed=0, nb_input_features=1):
    """
    prepare models for segmentation tasks

    related: how to load a model
    load_file = '/path/to/model.99-0.00.hdf5'
    loss = losses['dice']
    models['seg'] = keras.models.load_model(load_file,  custom_objects={'loss': loss})
    """

    # 
    if seed is not None:
        np.random.seed(seed)

    # a template for create a u-net model (since we create several unet models here)
    unet_template = lambda nb_labels, dict: nrn_models.design_unet(model.nb_features,
                                                                   run.patch_size,
                                                                   model.nb_levels,
                                                                   model.conv_size,
                                                                   nb_labels,
                                                                   feat_mult=model.feat_mult,
                                                                   pool_size=model.pool_size,
                                                                   use_residuals=model.use_residuals,
                                                                   use_logp=model.use_logp,
                                                                   **dict)

    models = {}
    if hasattr(run, 'load_path') and run.load_path is not None:
        print('loading model %s' % run.load_path)
        models['seg'] = keras.models.load_model(run.load_path,
                                                custom_objects={'loss': load_loss})

    else:
        dct = {'name':'seg', 'add_prior_layer':model.include_prior, 'nb_input_features':nb_input_features}
        models['seg'] = unet_template(data.nb_labels, dct)


        # cycleGAN:
        if run.patch_size is not None and run.patch_size[0] is not None:
            # get S = segmentor Unet
            models['cg-seg'] = unet_template(data.nb_labels, {'name':'cg-seg', 'add_prior_layer':model.include_prior})
            # get G = generator (S->I)
            models['cg-gen'] = unet_template(1, {'name':'cg-gen', 'nb_input_features':data.nb_labels, 'add_prior_layer':False, 'final_pred_activation':None})
            # get D = discriminator 
            models['cg-disc'] = nrn_models.design_dnn(model.nb_features,
                                                    run.patch_size,
                                                    model.nb_levels,
                                                    model.conv_size,
                                                    data.nb_labels,
                                                    final_layer='dense-tanh',
                                                    feat_mult=model.feat_mult,
                                                    pool_size=model.pool_size,
                                                    nb_input_features=data.nb_labels,
                                                    name='cg-disc')
            models['cg-cdisc'] = nrn_models.design_dnn(model.nb_features,
                                                    run.patch_size,
                                                    model.nb_levels,
                                                    model.conv_size,
                                                    data.nb_labels,
                                                    final_layer='dense-tanh',
                                                    feat_mult=model.feat_mult,
                                                    pool_size=model.pool_size,
                                                    nb_input_features=data.nb_labels+1,
                                                    name='cg-disc')

    if hasattr(run, 'load_weights') and run.load_weights is not None:
        print('loading weights %s' % run.load_weights)
        models['seg'].load_weights(run.load_weights, by_name=True)

    return models


def show_example_prediction_result(test_models,
                                   test_generator,
                                   run,
                                   data,
                                   test_batch_size=1,
                                   test_model_names=None,
                                   test_grid_size=None,
                                   ccmap=None,
                                   collapse_2d=2,
                                   slice_nr=None,
                                   plt_width=17,
                                   verbose=False):

    # some more parameters
    if not isinstance(test_models, (list, tuple)):
        test_models = [test_models]

    if ccmap is None:
        [ccmap, scrambled_cmap] = plotting.jitter(data.nb_labels, nargout=2)
        scrambled_cmap[0,:] = np.array([0,0,0,1])
        ccmap = matplotlib.colors.ListedColormap(scrambled_cmap)
    if test_model_names is None:
        test_model_names = [f.name for f in test_models]

    # parameters - being careful with 2d-ness
    do_3d = len(data.vol_size) == 3
    test_patch_size = run.patch_size
    test_patch_stride = run.patch_stride
    if do_3d and len(test_patch_size) == 2:
        test_patch_size = [*test_patch_size, 1]
        test_patch_stride = [*test_patch_stride, 1]
    if test_grid_size is None:  # default: just one patch !
        assert test_batch_size == 1
        test_grid_size = [1 for _ in test_patch_size]

    # visualize this just like a volume prediction with batch_size of 1 and grid_size of 1 ?
    with timer.Timer('Vol Prediction', verbose):
        vols = nrn_utils.predict_volumes(test_models,
                                         test_generator,
                                         test_batch_size,
                                         test_patch_size,
                                         test_patch_stride,
                                         test_grid_size,
                                         do_extra_vol=True,
                                         do_prob_of_true=True,
                                         verbose=verbose)
        if len(test_models) == 1:
            vols = [vols]
        do_prior = len(vols[0]) == 6
        if do_3d:
            assert len(vols[0][0].shape) == 3, 'volume is not 3D, something went wrong'

    rcmap = np.linspace(0, 1, data.nb_labels)
    outline_cmap = plt.get_cmap(ccmap)(rcmap)[:, 0:3]

    # Warning: this is slow in 3D!. Should really only compute the overlap for the slices...
    outline_fn = lambda x: su.seg_overlap(vols[0][2], x, cmap=scrambled_cmap)

    # extract specific volumes
    # empty_vol = np.zeros((vols[0][2].shape))  # empty canvas
    plt_titles = ["vol", "true_seg", "true_seg_outlines"]
    plt_vols = [vols[0][2], vols[0][0], outline_fn(vols[0][0])]
    cmaps = ["gray", ccmap, None]
    ia = lambda x: {'vmin':0, 'vmax':x}
    imshow_args = [ia(1), ia(data.nb_labels), {}]
    if do_prior:  # not doing prior since it seems it wasn't in there.
        plt_titles = [*plt_titles, "prior_prob_of_true", "prior_seg", "prior_seg_outlines"]
        plt_vols = [*plt_vols, vols[0][5], vols[0][3], outline_fn(vols[0][3])]
        cmaps = [*cmaps, "gray", ccmap, None]
        imshow_args += [ia(1), ia(data.nb_labels), {}]

    for midx, _ in enumerate(test_models):
        vm = vols[midx]
        mname = test_model_names[midx]
        pred_prov_idx = 3 + do_prior
        plt_vols = [*plt_vols, vm[pred_prov_idx], vm[1], outline_fn(vm[1])]
        plt_titles = [*plt_titles, "%s_pred_prob_of_true" % mname, "%s_pred_seg" % mname, "%s_pred_seg_outlines" % mname]
        cmaps = [*cmaps, "gray", ccmap, None]
        imshow_args += [ia(1), ia(data.nb_labels), {}]

    # go through dimensions to plot
    if not isinstance(collapse_2d, (list, tuple)):
        collapse_2d = [collapse_2d]

    grid = [len(test_models) + 1 + do_prior, 3]
    ret = (plt_vols, )
    for c_2d in collapse_2d:

        # extract 2d slices for visualization
        plt_vols_s = plt_vols
        if do_3d:
            assert len(plt_vols[0].shape) == 3, 'volume is not 3D'
            if slice_nr is None:
                print("printing. vol shape:", plt_vols[0].shape)
                slice_nr_ = (plt_vols[0].shape[c_2d] // 2)
            else:
                slice_nr_ = slice_nr
            print(slice_nr_)
            extract_ix = nd.arange([0 for f in plt_vols[0].shape], plt_vols[0].shape)
            extract_ix[c_2d] = [slice_nr_]
            vol_proc = lambda x: np.squeeze(nrn_dataproc.vol_proc(x, extract_nd=extract_ix))
            plt_vols_s = [vol_proc(f) for f in plt_vols]

        # plot
        # imshow_args={'vmin':0, 'vmax':1} #data.nb_labels}
        f = nrn_plt.slices(plt_vols_s,
                           titles=plt_titles,
                           width=plt_width,
                           cmaps=cmaps,
                           do_colorbars=True,
                           grid=grid,
                           imshow_args=imshow_args)
        ret += (f[0], )

    return ret




def _sample_to_disc_data(sample, seg_model, patch_size):
    
    true = sample[1]
    pred = seg_model.predict(sample[0])
    
    # batch size
    bs = pred.shape[0]
    
    # prepare 0s and 1s and data stack
    z = np.vstack((-np.ones((bs,1)), np.ones((bs,1))))
    data = np.vstack((pred, true))
    data = np.reshape(data, (bs*2, *patch_size, -1))
    
    # randomize stack
    idx = np.arange(0, bs*2)
    np.random.shuffle(idx)  # in-place

    # return
    return (data[idx,:], z[idx], pred, true)


def _loss_with_name(loss, name):
    x = lambda x, y: loss(x, y)
    x.__name__ = name
    return loss

# some code to compute dice on volumes...
# nb_volumes = 3
# import medipy

# test_patch_size = [*run.patch_size, 1]
# test_patch_stride = [*run.patch_stride, 1]
# dice_scores = np.empty((0, data.nb_labels))
# for _ in range(nb_volumes):
#     vols = nrn_utils.predict_volumes([models['seg']],
#                                     generators['test'],
#                                     batch_size=8,
#                                     patch_size=test_patch_size,
#                                     patch_stride=test_patch_stride,
#                                     grid_size=run.grid_size,
#                                     do_extra_vol=False,
#                                     do_prob_of_true=False,
#                                     verbose=False)
#     dsc = medipy.metrics.dice(vols[0], vols[1], labels=np.arange(data.nb_labels))
#     dice_scores = np.vstack((dice_scores, dsc))



