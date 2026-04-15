# Neurite

A neural networks toolbox with a focus on medical image analysis in pytorch.

> ⚠️ **Warning**: neurite is under active development. We are in the process of finalizing the structure for PyTorch -- interfaces may change. 

## Install

To use the Neurite library, either clone this repository and install the requirements listed in `setup.py` or install directly with pip.

```
pip install neurite
```

**For users who want to use the stable TensorFlow version**, use either `pip install neurite`, or pull/clone from the `dev-tensorflow` branch.

## Main tools
- [nn.functional](neurite/nn/functional.py): tensor ops for smoothing, resampling, interpolation, masking, and spatial math.
- [nn.modules](neurite/nn/modules.py): reusable stateful layers, losses, and preprocessing blocks.
- [nn.models](neurite/nn/models.py): prebuilt architectures for arbitrary spatial dimensions (1d, 2d, 3d).
- [py.plot](neurite/py/plot.py): plotting tools for tensor slices, volumes, and flow fields.
- [utils.utils](neurite/utils/utils.py): lightweight factory and helper functions and tools.


## Papers:
If you use this code, please cite:

**Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation**  
[Adrian V. Dalca](http://adalca.mit.edu), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
CVPR 2018.  
[ [PDF](http://www.mit.edu/~adalca/files/papers/cvpr2018_priors.pdf) | [arxiv](http://arxiv.org/abs/1903.03148) | [bibtex](citations.bib) ]

If you are using any of the sparse/imputation functions, please cite:  

**Unsupervised Data Imputation via Variational Inference of Deep Subspaces**  
[Adrian V. Dalca](http://adalca.mit.edu), [John Guttag](https://people.csail.mit.edu/guttag/), [Mert R. Sabuncu](http://sabuncu.engineering.cornell.edu/)  
Arxiv preprint 2019  
[ [arxiv](https://arxiv.org/abs/1903.03503) | [bibtex](citations.bib) ]


## Development:
We welcome contributions; please make sure your code respects `pep8`, except for `E731,W291,W503,W504`, by running:  
```pycodestyle --ignore E731,W291,W503,W504 --max-line-length 100 /path/to/neurite```  
Please open an [issue](https://github.com/adalca/neurite/issues) [preferred] or contact Adrian Dalca at adalca@csail.mit.edu for question related to `neurite`.


## Use/demos:
Parts of `neurite` were used in [VoxelMorph](http://voxelmorph.mit.edu) and [brainstorm](https://github.com/xamyzhao/brainstorm/), which we encourage you to check out!
