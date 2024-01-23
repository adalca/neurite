import torch
import numpy as np

class MutualInfomation(torch.nn.Module):
    """
    Soft Mutual Information approximation for intensity volumes and probabilistic volumes
    (e.g. probabilistic segmentaitons)
    More information/citation:
    - Courtney K Guo.
      Multi-modal image registration with unsupervised deep learning.
      PhD thesis, Massachusetts Institute of Technology, 2019.
    - M Hoffmann, B Billot, JE Iglesias, B Fischl, AV Dalca.
      Learning image registration without images.
      arXiv preprint arXiv:2004.10282, 2020. https://arxiv.org/abs/2004.10282
    - https://github.com/adalca/neurite/blob/dev/neurite/tf/metrics.py

    - Pytorch version was created by Junyu Chen (Email: jchen245@jhmi.edu)
    """
    def __init__(self, type='volumes', bin_centers=None, nb_bins=None, min_clip=None, max_clip=None, soft_bin_alpha=1):
        super(MutualInfomation, self).__init__()
        """
        Initialize the mutual information class
        Arguments below are related to soft quantizing of volumes, which is done automatically 
        in functions that comptue MI over volumes (e.g. volumes(), volume_seg(), channelwise()) 
        using these parameters
        Args:
           bin_centers (np.float32, optional): array or list of bin centers. 
               Defaults to None.
           nb_bins (int, optional):  number of bins, if bin_centers is not specified. 
               Defaults to 16.
           min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
           max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
           soft_bin_alpha (int, optional): alpha in RBF of soft quantization. Defaults to 1.
        """
        self.type = type
        self.bin_centers = None
        if bin_centers is not None:
            self.bin_centers = torch.from_numpy(bin_centers).cuda().float()
            assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]

        self.nb_bins = nb_bins
        if bin_centers is None and nb_bins is None:
            self.nb_bins = 16

        self.min_clip = min_clip
        if self.min_clip is None:
            self.min_clip = -np.inf

        self.max_clip = max_clip
        if self.max_clip is None:
            self.max_clip = np.inf

        self.soft_bin_alpha = soft_bin_alpha

    def volumes(self, x, y):
        """
        Mutual information for each item in a batch of volumes.
        Algorithm:
        - use neurite.utils.soft_quantize() to create a soft quantization (binning) of
          intensities in each channel
        - channelwise()
        Parameters:
            x and y:  [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = x.shape[1]
        tensor_channels_y = y.shape[1]
        msg = 'volume_mi requires two single-channel volumes. See channelwise().'
        assert tensor_channels_x == 1, msg
        assert tensor_channels_y == 1, msg

        # volume mi
        return torch.flatten(self.channelwise(x, y))

    def segs(self, x, y):
        """
        Mutual information between two probabilistic segmentation maps.
        Wraps maps()
        Parameters:
            x and y:  [bs, nb_labels, ...]
        Returns:
            Tensor of size [bs]
        """
        # volume mi
        return self.maps(x, y)

    def volume_seg(self, x, y):
        """
        Mutual information between a volume and a probabilistic segmentation maps.
        Wraps maps()
        Parameters:
            x and y: a volume and a probabilistic (soft) segmentation. Either:
              - x: [bs, ..., 1] and y: [bs, ..., nb_labels], Or:
              - x: [bs, ..., nb_labels] and y: [bs, ..., 1]
        Returns:
            Tensor of size [bs]
        """
        # check shapes
        tensor_channels_x = x.shape[1]
        tensor_channels_y = y.shape[1]
        msg = 'volume_seg_mi requires one single-channel volume.'
        assert min(tensor_channels_x, tensor_channels_y) == 1, msg
        msg = 'volume_seg_mi requires one multi-channel segmentation.'
        assert max(tensor_channels_x, tensor_channels_y) > 1, msg

        # transform volume to soft-quantized volume
        if tensor_channels_x == 1:
            x = self._soft_sim_map(x[:, 0, ...])  # [bs, B, ...]
        else:
            y = self._soft_sim_map(y[:, 0, ...])  # [bs, B, ...]

        return self.maps(x, y)  # [bs]

    def channelwise(self, x, y):
        """
        Mutual information for each channel in x and y. Thus for each item and channel this
        returns retuns MI(x[...,i], x[...,i]). To do this, we use neurite.utils.soft_quantize() to
        create a soft quantization (binning) of the intensities in each channel
        Parameters:
            x and y:  [bs, ..., C]
        Returns:
            Tensor of size [bs, C]
        """
        # check shapes
        tensor_shape_x = x.shape
        tensor_shape_y = y.shape
        assert tensor_shape_x == tensor_shape_y, 'volume shapes do not match'
        # reshape to [bs, V, C]
        if len(tensor_shape_x) != 3:
            x = torch.reshape(x, (tensor_shape_x[0], tensor_shape_x[1], -1))  # [bs, C, V]
            x = x.permute(0, 2, 1)# [bs, V, C]
            y = torch.reshape(y, (tensor_shape_x[0], tensor_shape_x[1], -1))  # [bs, C, V]
            y = y.permute(0, 2, 1)  # [bs, V, C]

        # move channels to first dimension
        cx = x.permute(2, 0, 1) # [C, bs, V]
        cy = y.permute(2, 0, 1) # [C, bs, V]

        # soft quantize
        cxq = self._soft_sim_map(cx)  # [C, bs, V, B]
        cyq = self._soft_sim_map(cy)  # [C, bs, V, B]
        # get mi
        cout = []
        for i in range(cxq.shape[0]):
            cout.append(self.maps(cxq[i:i+1, ...], cyq[i:i+1, ...]))
        cout = torch.stack(cout, dim=0) # [C, bs]

        # permute back
        return cout.permute(1, 0) # [bs, C]

    def maps(self, x, y):
        """
        Computes mutual information for each entry in batch, assuming each item contains
        probability or similarity maps *at each voxel*. These could be e.g. from a softmax output
        (e.g. when performing segmentaiton) or from soft_quantization of intensity image.
        Note: the MI is computed separate for each itemin the batch, so the joint probabilities
        might be  different across inputs. In some cases, computing MI actoss the whole batch
        might be desireable (TODO).
        Parameters:
            x and y are probability maps of size [bs, ..., B], where B is the size of the
              discrete probability domain grid (e.g. bins/labels). B can be different for x and y.
        Returns:
            Tensor of size [bs]
        """

        # check shapes
        tensor_shape_x = x.shape
        tensor_shape_y = y.shape
        assert tensor_shape_x == tensor_shape_y, 'volume shapes do not match'
        assert torch.min(x) >= 0, 'voxel values must be non-negative'
        assert torch.min(y) >= 0, 'voxel values must be non-negative'

        eps = 1e-6

        # reshape to [bs, V, B]
        if len(tensor_shape_x) != 3:
            x = torch.reshape(x, (tensor_shape_x[1], tensor_shape_x[2], tensor_shape_x[3])) # [bs, V, B1]
            y = torch.reshape(y, (tensor_shape_x[1], tensor_shape_x[2], tensor_shape_x[3])) # [bs, V, B2]

        # x probability for each batch entry
        px = torch.sum(x, 1, keepdim=True)  # [bs, 1, B1]
        px = px / (torch.sum(px, dim=2, keepdim=True)+eps)
        # y probability for each batch entry
        py = torch.sum(y, 1, keepdim=True)  # [bs, 1, B2]
        py = py / (torch.mean(py, dim=2, keepdim=True)+eps)

        # joint probability for each batch entry
        x_trans = x.permute(0, 2, 1)  # [bs, B1, V]
        pxy = torch.bmm(x_trans, y)  # [bs, B1, B2]
        pxy = pxy / (torch.sum(pxy, dim=[1, 2], keepdim=True) + eps)  # [bs, B1, B2]

        # independent xy probability
        px_trans = px.permute(0, 2, 1)  # [bs, B1, 1]
        pxpy = torch.bmm(px_trans, py)  # [bs, B1, B2]
        pxpy_eps = pxpy + eps

        # mutual information
        log_term = torch.log(pxy / pxpy_eps + eps)  # [bs, B1, B2]
        mi = torch.sum(pxy * log_term, dim=[1, 2])  # [bs]
        return mi

    def _soft_log_sim_map(self, x):
        """
        soft quantization of intensities (values) in a given volume
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image.
        Returns:
            volume with one more dimension [bs, ..., B]
        """

        return self.soft_quantize(x,
                                  alpha=self.soft_bin_alpha,
                                  bin_centers=self.bin_centers,
                                  nb_bins=self.nb_bins,
                                  min_clip=self.min_clip,
                                  max_clip=self.max_clip,
                                  return_log=True)  # [bs, ..., B]

    def _soft_sim_map(self, x):
        """
        See neurite.utils.soft_quantize
        Parameters:
            x [bs, ...]: intensity image.
        Returns:
            volume with one more dimension [bs, ..., B]
        """
        return self.soft_quantize(x,
                                  alpha=self.soft_bin_alpha,
                                  bin_centers=self.bin_centers,
                                  nb_bins=self.nb_bins,
                                  min_clip=self.min_clip,
                                  max_clip=self.max_clip,
                                  return_log=False)  # [bs, ..., B]

    def _soft_prob_map(self, x, **kwargs):
        """
        normalize a soft_quantized volume at each voxel, so that each voxel now holds a prob. map
        Parameters:
            x [bs, ..., B]: soft quantized volume
        Returns:
            x [bs, ..., B]: renormalized so that each voxel adds to 1 across last dimension
        """
        eps = 1e-6
        x_hist = self._soft_sim_map(x, **kwargs)  # [bs, ..., B]
        x_hist_sum = torch.sum(x_hist, -1, keepdim=True) + eps  # [bs, ..., B]
        x_prob = x_hist / x_hist_sum  # [bs, ..., B]
        return x_prob

    def soft_quantize(self, x,
                      bin_centers=None,
                      nb_bins=16,
                      alpha=1,
                      min_clip=-np.inf,
                      max_clip=np.inf,
                      return_log=False):
        """
        (Softly) quantize intensities (values) in a given volume, based on RBFs.
        In numpy this (hard quantization) is called "digitize".

        Code modified based on:
        https://github.com/adalca/neurite/blob/3858b473fcdc89354fe645a453d75ad01c794c8a/neurite/tf/utils/utils.py#L860
        """
        if bin_centers is not None:
            if not torch.is_tensor(bin_centers):
                bin_centers = torch.from_numpy(bin_centers).cuda().float()
            else:
                bin_centers = bin_centers.cuda().float()
            #assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]
        else:
            if nb_bins is None:
                nb_bins = 16
            # get bin centers dynamically
            minval = torch.min(x)
            maxval = torch.max(x)
            bin_centers = torch.linspace(minval.item(), maxval.item(), nb_bins)
        #print(bin_centers)

        # clipping at bin values
        x = x[..., None]  # [..., 1]
        x = torch.clamp(x, min_clip, max_clip)

        # reshape bin centers to be (1, 1, .., B)
        new_shape = [1] * (len(x.shape) - 1) + [nb_bins]
        bin_centers = torch.reshape(bin_centers, new_shape)  # [1, 1, ..., B]

        # compute image terms
        bin_diff = torch.square(x - bin_centers.cuda())  # [..., B]
        log = -alpha * bin_diff  # [..., B]

        if return_log:
            return log  # [..., B]
        else:
            return torch.exp(log)  # [..., B]

    def forward(self, y_pred, y_true):
        if self.type.lower() == 'volumes':
            mi = self.volumes(y_pred, y_true)
        elif self.type.lower() == 'segmentation':
            mi = self.segs(y_pred, y_true)
        elif self.type.lower() == 'volume segmentation':
            mi = self.volume_seg(y_pred, y_true)
        elif self.type.lower() == 'channelwise':
            mi = self.channelwise(y_pred, y_true)
        else:
            raise Exception("Type not implemented!")
        return mi.mean()