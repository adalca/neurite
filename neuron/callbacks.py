''' callbacks for the neuron project '''

'''
We'd like the following callback actions for neuron:

- print metrics on the test and validation, especially surface-specific dice
--- Perhaps doable with CSVLogger?
- output graph up to current iteration for each metric
--- Perhaps call CSVLogger or some metric computing callback?
- save dice plots on validation
--- again, expand CSVLogger or similar
- save screenshots of a single test subject [Perhaps just do this as a separate callback?]
--- new callback, PlotSlices

'''

import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings

import pynd.ndutils as nd
import pynd.segutils as su
import pytools.patchlib as pl

# the neuron folder should be on the path
import neuron.plot as nrn_plt

class ModelCheckpoint(keras.callbacks.Callback):
    """
    A modification of kera's ModelCheckpoint that allow for saving on_batch_end
    changes include:
    - optional at_batch_end, at_epoch_end arguments,
    - filename now must includes 'iter'

    Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 at_batch_end=None,     # None or number indicate when to execute (i.e. at_batch_end = 10 means execute every 10 batches)
                 at_epoch_end=True,     # logical, whether to execute at epoch end
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.steps_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

        self.at_batch_end = at_batch_end
        self.at_epoch_end = at_epoch_end
        self.current_epoch = 0

    def on_batch_end(self, batch, logs=None):
        if self.at_batch_end is not None and np.mod(batch + 1, self.at_batch_end) == 0:
            self.on_model_save(self.current_epoch, batch + 1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.at_epoch_end:
            self.on_model_save(epoch, 0, logs=logs)
        self.current_epoch = epoch

    def on_model_save(self, epoch, iter, logs=None):
        logs = logs or {}
        self.steps_since_last_save += 1
        if self.steps_since_last_save >= self.period:
            self.steps_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, iter=iter, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d Iter%05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, iter, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d Iter%05d: %s did not improve' %
                                  (epoch, iter, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class PlotTestSlices(keras.callbacks.Callback):
    '''
    plot slices of a test subject from several directions
    '''

    def __init__(self, savefilepath, generator, vol_size,
                 at_batch_end=None,     # None or number indicate when to execute (i.e. at_batch_end = 10 means execute every 10 batches)
                 at_epoch_end=True,     # logical, whether to execute at epoch end
                 prior=None):
        super().__init__()

        # save some parameters
        self.savefilepath = savefilepath
        self.generator = generator
        self.vol_size = vol_size

        self.at_batch_end = at_batch_end
        self.at_epoch_end = at_epoch_end
        self.current_epoch = 0

        # prepare prior
        self.prior = None
        if prior is not None:
            data = np.load(prior)
            loc_vol = data['prior']
            self.prior = np.expand_dims(loc_vol, axis=0) # reshape for model

    def on_batch_end(self, batch, logs=None):
        if self.at_batch_end is not None and np.mod(batch + 1, self.at_batch_end) == 0:
            self.on_plot_save(self.current_epoch, batch + 1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.at_epoch_end:
            self.on_plot_save(epoch, 0, logs=logs)
        self.current_epoch = epoch

    def on_plot_save(self, epoch, iter, logs=None):
        vol_size = self.vol_size

        # get the next sample
        sample = next(self.generator)
        sample_inputs = sample[0]
        sample_output = sample[1]
        nb_classes = sample_output.shape[-1]

        # if sample_input is a tuple, we assume it's (vol, prior)
        do_explicit_prior = isinstance(sample_inputs, list)
        if do_explicit_prior:
            sample_vol = np.squeeze(sample_inputs[0])
            sample_prior = np.squeeze(sample_inputs[1])
        else:
            sample_vol = np.squeeze(sample_inputs)
            sample_prior = self.prior

        # extract true segmentation
        seg_true = np.argmax(sample_output, -1).reshape(vol_size)

        # predict
        seg_pred_prob_all = self.model.predict(sample_inputs)
        seg_pred_prob_NDp1 = seg_pred_prob_all.reshape(vol_size + (nb_classes, ))
        seg_pred_prob = np.max(seg_pred_prob_NDp1, -1)
        seg_pred_label = np.argmax(seg_pred_prob_NDp1, -1) #.reshape(q[0][0].shape)
        seg_pred_label = seg_pred_label.reshape(vol_size)

        # compute the prior map for the predicted labels
        if sample_prior is not None:
            idxes = np.arange(np.prod(vol_size))
            arr = np.array([idxes, seg_pred_label.flatten()])
            locs = np.ravel_multi_index(arr, seg_pred_prob_all.shape[1:-1], order='C')
            seg_pred_label_prior = (sample_prior.flat[locs]).reshape(vol_size)
        else:
            seg_pred_label_prior = np.zeros(vol_size)

        # display
        vols = [sample_vol, seg_true, seg_pred_label,
                seg_true == seg_pred_label, seg_pred_prob, seg_pred_label_prior]
        titles = ['sample slice', 'true seg', 'pred seg',
                  'true vs pred', 'pred seg prob', 'prior @ pred seg']
        cmaps = ['gray', 'Set3', 'Set3', 'gray', 'gray', 'gray']
        label_norm = matplotlib.colors.Normalize(vmin=0, vmax=nb_classes-1)
        norms = [None, label_norm, label_norm, None, None, None]

        # prepare some overlap params
        kwargs = {'cmaps':cmaps, 'do_colorbars':True, 'titles':titles, 'norms':norms, 'show':False}
        filename = self.savefilepath
        if len(vol_size) == 3:  # 3D

            slice_nrs = np.floor(np.array(vol_size) / 2).astype('int')

            slices = list(map(lambda x: np.transpose(x[:, :, slice_nrs[2]]), vols))
            _print_and_save_slice(slices, epoch, 'coronal', slice_nrs[2], filename, nb_classes, **kwargs)

            slices = list(map(lambda x: np.rot90(x[:, slice_nrs[1], :]), vols))
            _print_and_save_slice(slices, epoch, 'axial', slice_nrs[1], filename, nb_classes, **kwargs)

            slices = list(map(lambda x: x[slice_nrs[0], :, :], vols))
            _print_and_save_slice(slices, epoch, 'saggital', slice_nrs[0], filename, nb_classes, **kwargs)
        else:
            slices = vols
            _print_and_save_slice(slices, epoch, 'slice', 0, filename, nb_classes, **kwargs)




class PredictMetrics(keras.callbacks.Callback):
    '''
    Compute metrics, like Dice, and save to CSV/log

    '''

    def __init__(self,
                 filepath,              # filepath with epoch and metric
                 metrics,               # list of metrics (functions)
                 validation_generator,  # validation generator
                 nb_validation,         # number of validation samples to get (# of times to call next())
                 nb_labels,             # number of labels
                 label_ids=None,
                 vol_params=None,
                 crop=None,             # allow for cropping of volume (volume edges are troublesome)
                 at_batch_end=None,     # None or number indicate when to execute (i.e. at_batch_end = 10 means execute every 10 batches)
                 at_epoch_end=True,     # logical, whether to execute at epoch end
                 vol_size=None):        # if cropping, need volume size

        # pass in the parameters to object variables
        self.metrics = metrics
        self.validation_generator = validation_generator
        self.nb_validation = nb_validation
        self.filepath = filepath
        self.nb_labels = nb_labels
        if label_ids is None:
            self.label_ids = list(range(nb_labels))
        else:
            self.label_ids = label_ids
        self.vol_size = vol_size
        self.vol_params = vol_params

        self.current_epoch = 1
        self.at_batch_end = at_batch_end
        self.at_epoch_end = at_epoch_end

    def on_batch_end(self, batch, logs={}):
        if self.at_batch_end is not None and np.mod(batch + 1, self.at_batch_end) == 0:
            self.on_metric_call(self.current_epoch, batch + 1, logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        if self.at_epoch_end:
            self.on_metric_call(epoch, 0, logs=logs)
        self.current_epoch = epoch

    def on_metric_call(self, epoch, iter, logs={}):
        # prepare metric
        met = np.zeros((self.nb_validation, self.nb_labels, len(self.metrics)))

        # prepare generator
        gen = _generate_predictions(self.model,
                                    self.validation_generator,
                                    self.nb_validation,
                                    self.vol_params)
        batch_idx = 0
        for (vol_pred, vol_true) in gen:
            for idx, metric in enumerate(self.metrics):
                met[batch_idx, :, idx] = metric(vol_pred, vol_true)
            batch_idx += 1

        # write metric to csv file
        if self.filepath is not None:
            for idx, metric in enumerate(self.metrics):
                filen = self.filepath.format(epoch=epoch, iter=iter, metric=metric.__name__)
                np.savetxt(filen, met[:, :, idx], fmt='%f', delimiter=',')
        else:
            meanmet = np.nanmean(met, axis=0)
            for midx, metric in enumerate(self.metrics):
                for idx in range(self.nb_labels):
                    varname = '%s_label_%d' % (metric.__name__, self.label_ids[idx])
                    logs[varname] = meanmet[idx, midx]


##################################################################################################
# helper functions
##################################################################################################

def _generate_predictions(model, validation_generator, nb_validation, vol_params):
    if vol_params is not None:
        for batch_idx in range(nb_validation):  # assumes nr volume
            vol_pred, vol_true = _predict_volume(model, validation_generator,
                                                 vol_params["patch_size"],
                                                 vol_params["patch_stride"],
                                                 vol_params["grid_size"])
            yield (vol_pred, vol_true)

    else:
        for batch_idx in range(nb_validation):  # assumes nr batches
            vol_pred, vol_true = _batch2lab(model, validation_generator)
            yield (vol_pred, vol_true)


def _quilt(patches, patch_size, grid_size, patch_stride, **kwargs):
    patches = np.vstack([f.flatten() for f in patches])
    return pl.quilt(patches, patch_size, grid_size, patch_stride=patch_stride, **kwargs)


def _predict_volume(model, validation_generator, patch_size, patch_stride, gridsize, nargout=2):

    # compute the number of patches
    nb_batches = np.prod(gridsize)
    med1 = lambda x: np.nanmedian(x, 0)
    nb_elems = np.prod(patch_size)

    # prepare all the data
    all_pred = np.empty((0, nb_elems), int)
    all_true = np.empty((0, nb_elems), int)

    # go through the batches
    for _ in range(nb_batches):
        lab = _batch2lab(model, validation_generator, nargout)
        pred_maxlabel, true_maxlabel = lab[0], lab[1]  # there might be a third argument.

        # append predictions
        all_pred = np.vstack((all_pred, np.reshape(pred_maxlabel, (1, nb_elems))))
        all_true = np.vstack((all_true, np.reshape(true_maxlabel, (1, nb_elems))))

    # combine to volumes
    vol_pred = _quilt(all_pred, patch_size, gridsize, patch_stride, nan_func=med1)
    vol_true = _quilt(all_true, patch_size, gridsize, patch_stride, nan_func=med1)

    # return data
    return (vol_pred, vol_true)


def _batch2lab(model, validation_generator, nargout=2):

    # predict output for a new sample
    sample = next(validation_generator)
    res = model.predict(sample[0])

    # compute resulting volume(s)
    pred_maxlabel = np.squeeze(np.argmax(res, -1))
    true_maxlabel = np.squeeze(np.argmax(sample[1], -1))

    # return
    if nargout == 2:
        return (pred_maxlabel, true_maxlabel)
    else:  # also return the sample input
        return (pred_maxlabel, true_maxlabel, sample[0])


def _print_and_save_slice(slices, epoch, axis, slice_nr, savefilepath, nb_labels, **kwargs):

    # slices
    f, _ = nrn_plt.slices(slices, **kwargs)
    f.savefig(savefilepath.format(epoch=epoch, axis='axial', slice_nr=slice_nr))
    plt.close()

    # with overlap
    olap_cmap = plt.get_cmap('Set3')(np.linspace(0, 1, nb_labels))[:,0:3]
    qslices = [su.seg_overlap(slices[0], f.astype('int'), cmap=olap_cmap) for f in slices[1:-1]]
    f, _ = nrn_plt.slices([slices[0], *qslices], **kwargs)
    f.savefig(savefilepath.format(epoch=epoch, axis='axial', slice_nr=slice_nr))
    plt.close()
