"""
Data processing for `neurite`.
"""

# Standard library imports
import os
import re
import shutil
import sys

# Third-party imports
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage.interpolation
import six
import torch
from tqdm import tqdm_notebook as tqdm  # for verbosity for forloops


def proc_mgh_vols(inpath,
                  outpath,
                  ext='.mgz',
                  label_idx=None,
                  **kwargs):
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
            vol_data = vol_proc(vol_data, **kwargs)
        except Exception as e:
            list_skipped_files += (files[fileidx], )
            print("Skipping %s\nError: %s" % (files[fileidx], str(e)), file=sys.stderr)
            continue

        if label_idx is not None:
            vol_data = (vol_data == label_idx).astype(int)

        # save numpy file
        outname = os.path.splitext(os.path.join(outpath, files[fileidx]))[0] + '.npz'
        np.savez_compressed(outname, vol_data=vol_data)

    for file in list_skipped_files:
        print("Skipped: %s" % file, file=sys.stderr)


def scans_to_slices(inpath, outpath, slice_nrs,
                    ext='.mgz',
                    label_idx=None,
                    dim_idx=2,
                    out_ext='.png',
                    slice_pad=0,
                    vol_inner_pad_for_slice_nrs=0,
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

        if slice_pad > 0:
            assert (out_ext != '.png'), "slice pad can only be used with volumes"

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
            slice_nrs_sel = range(vol_inner_pad_for_slice_nrs + slice_pad,
                                  vol_data.shape[dim_idx] - slice_pad - vol_inner_pad_for_slice_nrs)
        else:
            slice_nrs_sel = slice_nrs

        for slice_nr in slice_nrs_sel:
            slice_nr_out = range(slice_nr - slice_pad, slice_nr + slice_pad + 1)
            if dim_idx == 2:  # TODO: fix in one line
                vol_img = np.squeeze(vol_data[:, :, slice_nr_out])
            elif dim_idx == 1:
                vol_img = np.squeeze(vol_data[:, slice_nr_out, :])
            else:
                vol_img = np.squeeze(vol_data[slice_nr_out, :, :])

            # save file
            if out_ext == '.png':
                # save png file
                img = (vol_img * mult_fact).astype('uint8')
                outname = os.path.splitext(os.path.join(outpath, files[fileidx]))[
                    0] + '_slice%d.png' % slice_nr
                try:
                    from PIL import Image
                    Image.fromarray(img).convert('RGB').save(outname)
                except ImportError:
                    raise ImportError(
                        'Could not save "%s" since PIL has not been installed' % outname)
            else:
                if slice_pad == 0:  # dimenion has collapsed
                    assert vol_img.ndim == 2
                    vol_img = np.expand_dims(vol_img, dim_idx)
                # assuming nibabel saving image
                nii = nib.Nifti1Image(vol_img, np.diag([1, 1, 1, 1]))
                outname = os.path.splitext(os.path.join(outpath, files[fileidx]))[
                    0] + '_slice%d.nii.gz' % slice_nr
                nib.save(nii, outname)


def vol_proc(vol_data,
             crop=None,
             # None (to not resize), or vector. If vector, third entry can be None
             resize_shape=None,
             interp_order=None,
             rescale=None,
             rescale_prctle=None,
             resize_slices=None,
             resize_slices_dim=None,
             offset=None,
             clip=None,
             extract_nd=None,  # extracts a particular section
             force_binary=None,  # forces anything > 0 to be 1
             permute=None):
    """
    Process a volume with intensity scaling, resizing, cropping, clipping, and extraction.

    Parameters
    ----------
    vol_data : numpy.ndarray or torch.Tensor
        Volume data. Tensor inputs support options with direct torch equivalents: `offset`,
        `rescale`, `rescale_prctle`, `clip`, `extract_nd`, and `force_binary`.
    crop : sequence of int or sequence of tuple[int, int], default=None
        Per-axis crop margins. Integers crop both ends equally, while pairs specify separate start
        and end margins. Tensor inputs do not support crop.
    resize_shape : sequence of int, default=None
        Target shape for scipy-based resizing. Tensor inputs only allow `None` or the current shape.
    interp_order : int, default=None
        Interpolation order passed to scipy resizing for NumPy inputs.
    rescale : number, default=None
        Multiplicative intensity scale.
    rescale_prctle : number, default=None
        Percentile used as an inverse multiplicative intensity scale.
    resize_slices : sequence, default=None
        Slice-resize shorthand for NumPy inputs. Tensor inputs do not support this option.
    resize_slices_dim : int, default=None
        Dimension inferred or used with `resize_slices`.
    offset : number, default=None
        Additive intensity offset.
    clip : tuple of number, default=None
        Inclusive `(min, max)` clipping range.
    extract_nd : sequence of index sequences, default=None
        Per-axis indices to extract with NumPy `ix_` semantics.
    force_binary : bool, default=None
        Convert positive values to 1 and non-positive values to 0.
    permute : optional
        Existing unused argument preserved for API compatibility.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Processed volume with the same backend as `vol_data`.
    """
    if isinstance(vol_data, torch.Tensor):
        return _vol_proc_torch(
            vol_data,
            crop=crop,
            resize_shape=resize_shape,
            rescale=rescale,
            rescale_prctle=rescale_prctle,
            resize_slices=resize_slices,
            offset=offset,
            clip=clip,
            extract_nd=extract_nd,
            force_binary=force_binary,
        )

    if offset is not None:
        vol_data = vol_data + offset

    # intensity normalize data .* rescale
    if rescale is not None:
        vol_data = np.multiply(vol_data, rescale)

    if rescale_prctle is not None:
        rescale = np.percentile(vol_data.flat, rescale_prctle)
        vol_data = np.multiply(vol_data.astype(float), 1 / rescale)

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
        assert len(crop) == vol_data.ndim, "crop must contain one entry per volume dimension"
        if isinstance(crop[0], (list, tuple)):
            crop_margins = crop
        else:
            crop_margins = [(margin, margin) for margin in crop]

        crop_slices = []
        for size, (start_margin, end_margin) in zip(vol_data.shape, crop_margins):
            crop_slices.append(slice(start_margin, size - end_margin))
        vol_data = vol_data[tuple(crop_slices)]

    # needs to be last to guarantee clip limits.
    # For e.g., resize might screw this up due to bicubic interpolation if it was done after.
    if clip is not None:
        vol_data = np.clip(vol_data, clip[0], clip[1])

    if extract_nd is not None:
        vol_data = vol_data[np.ix_(*extract_nd)]

    if force_binary:
        vol_data = (vol_data > 0).astype(float)

    # return with checks. this check should be right at the end before rturn
    if clip is not None:
        assert np.max(vol_data) <= clip[1], "clip failed"
        assert np.min(vol_data) >= clip[0], "clip failed"
    return vol_data


def _vol_proc_torch(vol_data,
                    crop=None,
                    resize_shape=None,
                    rescale=None,
                    rescale_prctle=None,
                    resize_slices=None,
                    offset=None,
                    clip=None,
                    extract_nd=None,
                    force_binary=None):
    """Process tensor volumes with operations that have direct torch equivalents."""
    assert crop is None, 'crop is not supported for torch tensor vol_data'
    assert resize_slices is None, 'resize_slices is not supported for torch tensor vol_data'
    if resize_shape is not None:
        assert tuple(resize_shape) == tuple(vol_data.shape), (
            'resize_shape is not supported for torch tensor vol_data unless it equals '
            'vol_data.shape'
        )

    if offset is not None:
        vol_data = vol_data + offset

    if rescale is not None:
        vol_data = vol_data * rescale

    if rescale_prctle is not None:
        quantile = torch.as_tensor(rescale_prctle / 100, device=vol_data.device)
        rescale = torch.quantile(vol_data.flatten().to(torch.float64), quantile)
        vol_data = vol_data.to(torch.float64) * (1 / rescale)

    if clip is not None:
        vol_data = torch.clamp(vol_data, min=clip[0], max=clip[1])

    if extract_nd is not None:
        vol_data = vol_data[_torch_ix(extract_nd, vol_data.device)]

    if force_binary:
        vol_data = (vol_data > 0).to(dtype=vol_data.dtype)

    if clip is not None:
        assert torch.max(vol_data) <= clip[1], 'clip failed'
        assert torch.min(vol_data) >= clip[0], 'clip failed'
    return vol_data


def _torch_ix(indices, device):
    """Create broadcastable tensor indices with NumPy `ix_` semantics."""
    grids = []
    ndim = len(indices)
    for axis, values in enumerate(indices):
        index = torch.as_tensor(values, dtype=torch.long, device=device)
        shape = [1] * ndim
        shape[axis] = -1
        grids.append(index.reshape(shape))
    return tuple(grids)


def prior_to_weights(prior_filename, nargout=1, min_freq=0, force_binary=False, verbose=False):
    """
    Transform a spatial class prior into class weights.

    Parameters
    ----------
    prior_filename : str or numpy.ndarray or torch.Tensor
        Path to an `.npz` file containing `prior`, or an in-memory prior. Tensor priors return
        tensor weights on the same device.
    nargout : int, default=1
        If 1, return only weights. Any other value returns `(weights, prior)`.
    min_freq : number, default=0
        Minimum class frequency before weights are computed.
    force_binary : bool, default=False
        Merge labels 1..N into one foreground class before computing weights.
    verbose : bool, default=False
        Plot NumPy diagnostic histograms. Tensor priors do not support verbose plotting.

    Returns
    -------
    numpy.ndarray or torch.Tensor or tuple
        Class weights, or `(weights, prior)` when `nargout != 1`.
    """

    # load prior
    if isinstance(prior_filename, six.string_types):
        prior = np.load(prior_filename)['prior']
    else:
        prior = prior_filename

    if isinstance(prior, torch.Tensor):
        return _prior_to_weights_torch(prior, nargout, min_freq, force_binary, verbose)

    # assumes prior is 4D.
    assert np.ndim(prior) == 4 or np.ndim(prior) == 3, "prior is the wrong number of dimensions"
    prior_flat = np.reshape(prior, (np.prod(prior.shape[0:(np.ndim(prior) - 1)]), prior.shape[-1]))

    if force_binary:
        nb_labels = prior_flat.shape[-1]
        prior_flat[:, 1] = np.sum(prior_flat[:, 1:nb_labels], 1)
        prior_flat = np.delete(prior_flat, range(2, nb_labels), 1)

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
    weights = 1 / class_prior
    weights = weights / np.sum(weights)
    # weights[0] = 0 # explicitly don't care about bg

    # a bit of verbosity
    if verbose:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.bar(range(prior.size), np.log(prior))
        ax1.set_title('log class freq')
        ax2.bar(range(weights.size), weights)
        ax2.set_title('weights')
        ax3.bar(range(weights.size), np.log((weights)) - np.min(np.log((weights))))
        ax3.set_title('log(weights)-minlog')
        f.set_size_inches(12, 3)
        plt.show()
        np.set_printoptions(precision=3)

    # return
    if nargout == 1:
        return weights
    else:
        return (weights, prior)


def _prior_to_weights_torch(prior, nargout, min_freq, force_binary, verbose):
    """Compute class weights for tensor priors without crossing into NumPy reductions."""
    assert not verbose, 'verbose is not supported for torch tensor priors'
    assert prior.ndim == 4 or prior.ndim == 3, 'prior is the wrong number of dimensions'

    prior_flat = prior.reshape((-1, prior.shape[-1]))
    if force_binary:
        nb_labels = prior_flat.shape[-1]
        foreground = torch.sum(prior_flat[:, 1:nb_labels], dim=1, keepdim=True)
        prior_flat = torch.cat((prior_flat[:, 0:1], foreground), dim=1)

    class_count = torch.sum(prior_flat, dim=0)
    class_prior = class_count / torch.sum(class_count)

    min_freq_tensor = torch.as_tensor(min_freq, dtype=class_prior.dtype, device=class_prior.device)
    class_prior = torch.where(class_prior < min_freq_tensor, min_freq_tensor, class_prior)
    class_prior = class_prior / torch.sum(class_prior)

    zero_mask = class_prior == 0
    if torch.any(zero_mask):
        print("Warning, found a label with 0 support. Setting its weight to 0!", file=sys.stderr)
        class_prior = class_prior.clone()
        class_prior[zero_mask] = torch.inf

    weights = 1 / class_prior
    weights = weights / torch.sum(weights)
    if nargout == 1:
        return weights
    return (weights, prior)


def filestruct_change(in_path, out_path, re_map,
                      mode='subj_to_type',
                      use_symlinks=False, name=""):
    """
    change from independent subjects in a folder to breakdown structure 

    example: filestruct_change('/../in_path', '/../out_path',
        {'asegs.nii.gz':'asegs', 'norm.nii.gz':'vols'})


    input structure: 
        /.../in_path/subj_1 --> with files that match regular repressions defined in re_map.keys()
        /.../in_path/subj_2 --> with files that match regular repressions defined in re_map.keys()
        ...
    output structure:
        /.../out_path/asegs/subj_1.nii.gz, subj_2.nii.gz
        /.../out_path/vols/subj_1.nii.gz, subj_2.nii.gz

    Parameters:
        in_path (string): input path
        out_path (string): output path
        re_map (dictionary): keys are reg-exs that match files in the input folders. 
            values are the folders to put those files in the new structure. 
            values can also be tuples, in which case values[0] is the dst folder, 
            and values[1] is the extension of the output file
        mode (optional)
        use_symlinks (bool): whether to just use symlinks rather than copy files
            default:True
    """

    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    # go through folders
    for subj in tqdm(os.listdir(in_path), desc=name):

        # go through files in a folder
        files = os.listdir(os.path.join(in_path, subj))
        for file in files:

            # see which key matches. Make sure only one does.
            matches = [re.match(k, file) for k in re_map.keys()]
            nb_matches = sum([f is not None for f in matches])
            assert nb_matches == 1, "Found %d matches for file %s/%s" % (nb_matches, file, subj)

            # get the matches key
            match_idx = [i for i, f in enumerate(matches) if f is not None][0]
            matched_dst = re_map[list(re_map.keys())[match_idx]]
            _, ext = os.path.splitext(file)
            if isinstance(matched_dst, tuple):
                ext = matched_dst[1]
                matched_dst = matched_dst[0]

            # prepare source and destination file
            src_file = os.path.join(in_path, subj, file)
            dst_path = os.path.join(out_path, matched_dst)
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            dst_file = os.path.join(dst_path, subj + ext)

            if use_symlinks:
                # on windows there are permission problems.
                # Can try : call(['mklink', 'LINK', 'TARGET'], shell=True)
                # or note https://stackoverflow.com/questions/6260149/os-symlink-support-in-windows
                os.symlink(src_file, dst_file)

            else:
                shutil.copyfile(src_file, dst_file)


def ml_split(in_path, out_path,
             cat_titles=['train', 'validate', 'test'],
             cat_prop=[0.5, 0.3, 0.2],
             use_symlinks=False,
             seed=None,
             tqdm=tqdm):
    """
    split dataset 
    """

    if seed is not None:
        np.random.seed(seed)

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # get subjects and randomize their order
    subjs = sorted(os.listdir(in_path))
    nb_subj = len(subjs)
    subj_order = np.random.permutation(nb_subj)

    # prepare split
    cat_tot = np.cumsum(cat_prop)
    if not cat_tot[-1] == 1:
        print("split_prop sums to %f, re-normalizing" % cat_tot)
        cat_tot = np.array(cat_tot) / cat_tot[-1]
    nb_cat_subj = np.round(cat_tot * nb_subj).astype(int)
    cat_subj_start = [0, *nb_cat_subj[:-1]]

    # go through each category
    for cat_idx, cat in enumerate(cat_titles):
        if not os.path.isdir(os.path.join(out_path, cat)):
            os.mkdir(os.path.join(out_path, cat))

        cat_subj_idx = subj_order[cat_subj_start[cat_idx]:nb_cat_subj[cat_idx]]
        for subj_idx in tqdm(cat_subj_idx, desc=cat):
            src_folder = os.path.join(in_path, subjs[subj_idx])
            dst_folder = os.path.join(out_path, cat, subjs[subj_idx])

            if use_symlinks:
                # on windows there are permission problems.
                # Can try : call(['mklink', 'LINK', 'TARGET'], shell=True)
                # or note https://stackoverflow.com/questions/6260149/os-symlink-support-in-windows
                os.symlink(src_folder, dst_folder)

            else:
                if os.path.isdir(src_folder):
                    shutil.copytree(src_folder, dst_folder)
                else:
                    shutil.copyfile(src_folder, dst_folder)
