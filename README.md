# Neurite

A neural networks toolbox with a focus on medical image analysis in tensorflow/keras for now.


## Install

To use the Neurite library, either clone this repository and install the requirements listed in `setup.py` or install directly with pip.

```
pip install neurite
```

## Main tools
- [layers](neurite/tf/layers.py): various network layers, sparse operations (e.g. `SpatiallySparse_Dense`), and `LocallyConnected3D` currently not included in `keras`  
- [utils](neurite/tf/utils/utils.py): various utilities, including `interpn`: N-D gridded interpolation, and several nonlinearities  
  - [model](neurite/tf/utils/model.py): `stack_models`: keras model stacking  
  - [vae](neurite/tf/utils/vae.py): tools for analyzing (V)AE style models  
  - [seg](neurite/tf/utils/seg.py): segmentation tools  
- [models](neurite/tf/models.py): flexible models (many parameters to play with) particularly useful in medical image analysis, such as UNet/hourglass model, convolutional encoders and decoders   
- [generators](neurite/tf/generators.py): generators for medical image volumes and various combinations of volumes, segmentation, categorical and other output  
- [callbacks](neurite/tf/callbacks.py): a set of callbacks for `keras` training to help with understanding your fit, such as Dice measurements and volume-segmentation overlaps  
- [dataproc](neurite/py/dataproc.py): a set of tools for processing medical imaging data for preparation for training/testing  
- [metrics](neurite/tf/metrics.py): metrics (most of which can be used as loss functions), such as Dice or weighted categorical crossentropy  
- [plot](neurite/py/plot.py): plotting tools, mostly for debugging models  


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
