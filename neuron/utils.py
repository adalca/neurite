""" various utilities for the neuron project """


# third party imports
import numpy as np
from tqdm import tqdm

# local imports
import pynd.ndutils as nd





def gather_model_patch_preds(test_generator, nb_patches, patch_size, nb_labels, models):
    """
    for a single volume, compute the patches for:
    - (output[0]):  image patches
    - (output[1]): (#models+1 list): each list entry is
        patches of most likely label
    - (output[2]): (#models+1 list): each list entry is
        patches of probability of "true" label
    """
    shape = patch_size + (nb_labels, )

    # gather patches for the original image
    imgs = []
    maxlabels = []
    prob_of_fs_labels = []

    # go through the patches
    for _ in tqdm(range(nb_patches)):
        # get a sample
        sample = next(test_generator)

        # predict with various models
        preds = []
        for model in models:
            preds.append(np.reshape(model.predict(sample[0]), shape))

        # "true" result
        true = np.reshape(sample[1], shape)

        # compute max label and prob of fs label
        argmaxvols = [np.argmax(v, axis=-1) for v in [*models, true]]

        # compute the probability of the "true" label
        label_prob_vols = [prob_of_label(f, argmaxvols[-1]) for f in [*models, true]]

        # append patch
        imgs.append(sample[0][0])
        maxlabels.append(argmaxvols)
        prob_of_fs_labels.append(label_prob_vols)

    # go from lists of len nb_patches to len nb_models
    maxlabels = list(map(list, zip(*maxlabels)))
    prob_of_fs_labels = list(map(list, zip(*prob_of_fs_labels)))

    # return
    return (imgs, maxlabels, prob_of_fs_labels)



def prob_of_label(vol, labelvol):
    """
    compute the probability of the labels in labelvol in each of the volumes in vols

    labelvol is a nd volume. vols is a list of nd+1 volume with a prob dist
    for each of the voxels in the nd vols
    
    returns a list of 3d volumes of probabilities.
    """

    # allow for vol to be a list of volumes
    if isinstance(vol, (list, tuple)):
        return [prob_of_label(f, labelvol) for f in vol]

    # check dimensions
    ndims = np.ndim(labelvol)
    assert np.ndim(vol) == ndims + 1, "dimensions do not match"
    shp = vol.shape
    nb_voxels = np.prod(shp[0:ndims])
    nb_labels = shp[-1]

    # reshape volume to be [nb_voxels, nb_labels]
    flat_vol = np.reshape(vol, (nb_voxels, nb_labels))

    # normalize accross second dimension
    rows_sums = flat_vol.sum(axis=1)
    flat_vol_norm = flat_vol / rows_sums[:, np.newaxis]

    # index into the flattened volume
    idx = list(range(nb_voxels))
    return flat_vol_norm[idx, labelvol.flat]
