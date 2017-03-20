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
import pytools.plot as n_plt

class PlotTestSlices(keras.callbacks.Callback):
    '''
    plot slices of a test subject from several directions
    '''

    def __init__(self, filepath, sample_inputs, sample_output, vol_size, prior=None):
        self.filepath = filepath
        self.sample_inputs = sample_inputs
        self.sample_output = sample_output
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

        # TODO: Separate the part that outputs slices.

        # if sample_input is a tuple, we assume it's (vol, prior)
        do_explicit_prior = isinstance(self.sample_inputs, tuple)

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
        f.savefig(self.filepath.format(epoch=epoch, axis='coronal', slice_nr=slice_nrs[2]))
        # plt.close()

        slices = list(map(lambda x: np.rot90(x[:, slice_nrs[1], :]), vols))
        f, _ = n_plt.slices(slices, **kwargs)
        f.savefig(self.filepath.format(epoch=epoch, axis='axial', slice_nr=slice_nrs[1]))
        # plt.close()

        slices = list(map(lambda x: x[slice_nrs[0], :, :], vols))
        f, _ = n_plt.slices(slices, **kwargs)
        f.savefig(self.filepath.format(epoch=epoch, axis='saggital', slice_nr=slice_nrs[0]))
        # plt.close()