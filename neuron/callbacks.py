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
import pynd.ndutils as nd

# the neuron folder should be on the path
import neuron.plot as n_plt

class PlotTestSlices(keras.callbacks.Callback):
    '''
    plot slices of a test subject from several directions
    '''

    def __init__(self, savefilepath, generator, vol_size, prior=None):
        super().__init__()

        # save some parameters
        self.savefilepath = savefilepath
        self.generator = generator
        self.vol_size = vol_size
        self.prior = None
        if prior is not None:
            data = np.load(prior)
            loc_vol = data['prior']
            self.prior = np.expand_dims(loc_vol, axis=0) # reshape for model


    def on_epoch_end(self, epoch, logs=None):
        log = logs or {}
        vol_size = self.vol_size
        nb_classes = self.sample_output.shape[-1]


        sample = next(self.generator)
        sample_inputs = sample[0]
        sample_output = sample[1]

        # TODO: Separate the part that outputs slices.

        # if sample_input is a tuple, we assume it's (vol, prior)
        do_explicit_prior = isinstance(self.sample_inputs, list)

        # extract volume and prior
        if do_explicit_prior:
            sample_vol = np.squeeze(self.sample_inputs[0])
            sample_prior = np.squeeze(self.sample_inputs[1])
        else:
            sample_vol = np.squeeze(self.sample_inputs)
            sample_prior = None

            if self.prior is not None:
                sample_prior = self.prior

        # extract true segmentation
        seg_true = np.argmax(self.sample_output, 2).reshape(vol_size)

        # predict
        seg_pred_prob_all = self.model.predict(self.sample_inputs)
        seg_pred_prob_4D = seg_pred_prob_all.reshape(vol_size + (nb_classes, ))
        seg_pred_prob = np.max(seg_pred_prob_4D, 3)
        seg_pred_label = np.argmax(seg_pred_prob_4D, 3) #.reshape(q[0][0].shape)
        seg_pred_label = seg_pred_label.reshape(vol_size)

        # compute the prior map for the predicted labels
        if sample_prior is not None:
            idxes = np.arange(np.prod(vol_size))
            arr = np.array([idxes, seg_pred_label.flatten()])
            locs = np.ravel_multi_index(arr, seg_pred_prob_all.shape[1:3], order='C')
            seg_pred_label_prior = (sample_prior.flat[locs]).reshape(vol_size)
        else:
            seg_pred_label_prior = np.zeros(vol_size)

        # show some accuracy
        # TODO: MOVE THESE TO OTHER CALLBACKS.
        # print('seg accuracy %3.2f:' % \
            # (np.mean(seg_true == seg_pred_label)))
        #  perhaps achieve by calling Nonbg and then normal accuracy.
        # print('seg accuracy within brain %3.2f:' % \
            # (np.sum(~np.equal(seg_true, 0) & (seg_true == seg_pred_label)) / np.sum(~np.equal(seg_true, 0))))

        # display
        vols = [sample_vol, seg_true, seg_pred_label,
                seg_true == seg_pred_label, seg_pred_prob, seg_pred_label_prior]
        titles = ['sample slice', 'true seg', 'pred seg',
                  'true vs pred', 'pred seg prob', 'prior @ pred seg']
        cmaps = ['gray', 'Set3', 'Set3', 'gray', 'gray', 'gray']
        label_norm = matplotlib.colors.Normalize(vmin=0, vmax=nb_classes-1)
        norms = [None, label_norm, label_norm, None, None, None]

        kwargs = {'cmaps':cmaps, 'do_colorbars':True, 'titles':titles, 'norms':norms, 'show':False}

        slice_nrs = np.floor(np.array(vol_size) / 2).astype('int')
        slices = list(map(lambda x: np.transpose(x[:, :, slice_nrs[2]]), vols))
        f, _ = n_plt.slices(slices, **kwargs)
        f.savefig(self.savefilepath.format(epoch=epoch, axis='coronal', slice_nr=slice_nrs[2]))
        plt.close()

        slices = list(map(lambda x: np.rot90(x[:, slice_nrs[1], :]), vols))
        f, _ = n_plt.slices(slices, **kwargs)
        f.savefig(self.savefilepath.format(epoch=epoch, axis='axial', slice_nr=slice_nrs[1]))
        plt.close()

        slices = list(map(lambda x: x[slice_nrs[0], :, :], vols))
        f, _ = n_plt.slices(slices, **kwargs)
        f.savefig(self.savefilepath.format(epoch=epoch, axis='saggital', slice_nr=slice_nrs[0]))
        plt.close()



class PredictMetrics(keras.callbacks.Callback):
    '''
    Compute metrics and save to CSV/log
    '''

    def __init__(self, 
                 filepath,  # filepath with epoch and metric
                 metrics,  # list of metrics (functions)
                 validation_generator,  # validation generator
                 nb_validation,  # number of validation samples to get (# of times to call next())
                 nb_labels,  # number of labels
                 label_ids=None,
                 crop=None,  # allow for cropping of volume (volume edges are troublesome)
                 vol_size=None):  # if cropping, need volume size
        self.metrics = metrics
        self.validation_generator = validation_generator
        self.nb_validation = nb_validation
        self.filepath = filepath
        self.nb_labels = nb_labels
        if label_ids is None:
            self.label_ids = list(range(nb_labels))
        else:
            self.label_ids = label_ids
        self.crop = crop
        self.vol_size = vol_size
        if crop is not None:
            assert vol_size is not None, "if cropping, need volume size"

    def on_epoch_end(self, epoch, logs={}):
        # prepare files
        for metric in self.metrics:

            # prepare metric
            met = np.zeros((self.nb_validation, self.nb_labels))
            for batch_idx in range(self.nb_validation):
                # predict output for a new sample
                sample = next(self.validation_generator)
                res = self.model.predict(sample[0])

                # compute resulting volume(s)
                pred_maxlabel = np.argmax(res, -1)
                true_maxlabel = np.argmax(sample[1], -1)

                # crop volumes if required
                if self.crop is not None:
                    pred_maxlabel = np.reshape(np.squeeze(pred_maxlabel), self.vol_size)
                    pred_maxlabel = nd.volcrop(pred_maxlabel, crop=self.crop)
                    true_maxlabel = np.reshape(np.squeeze(true_maxlabel), self.vol_size)
                    true_maxlabel = nd.volcrop(true_maxlabel, crop=self.crop)

                # compute metric
                met[batch_idx, :] = metric(pred_maxlabel, true_maxlabel)

            # write metric to csv file
            if self.filepath is not None:
                filen = self.filepath.format(epoch=epoch, metric=metric.__name__)
                np.savetxt(filen, met, fmt='%f', delimiter=',')
            else:
                meanmet = np.mean(met, axis=0)
                for idx in range(self.nb_labels):
                    varname = 'dice_label_%d' % self.label_ids[idx]
                    logs[varname] = meanmet[idx]
