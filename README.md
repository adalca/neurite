# neuron
A Neural networks toolbox for anatomical image analysis (**currently in development**)  

A set of tools and infrastructure for medical image analysis with neural networks. While the tools are somewhat general, `neuron` will generally run with `keras` on top of `tensorflow`.

### Main tools
`layers`: various network layers, including a rich `SpatialTransformer` layer for N-D (dense and affine) spatial transforms, a vector integration layer `VecInt`, and `LocallyConnected3D` currently not included in `keras`  
`models`: flexible models (many parameters to play with) particularly useful in medical image analysis, such as UNet/hourglass model, convolutional encoders and decoders   
`generators`: generators for medical image volumes and various combinations of volumes, segmentation, categorical and other output  
`callbacks`: a set of callbacks for `keras` training to help with understanding your fit, such as Dice measurements and volume-segmentation overlaps  
`dataproc`: a set of tools for processing medical imaging data for preparation for training/testing  
`metrics`: metrics (most of which can be used as loss functions), such as Dice or weighted categorical crossentropy  
`plot`: plotting tools, mostly for debugging models  
`utils`: various utilities, including `interpn`: N-D gridded interpolation, `transform`: warp images, `integrate_vec`: vector field integration, `stack_models`: keras model stacking  

### Requirements:
- tensorflow, keras and all of their requirements (e.g. hyp5) 
- numpy, scipy, tqdm  
- [python libraries](https://github.com/search?q=user%3Aadalca+topic%3Apython) from @adalca github account  
 
### Development:
Please contact Adrian Dalca, adalca@csail.mit.edu for question related to `neuron`

### Papers:
**Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation**  
AV Dalca, J Guttag, MR Sabuncu  
*Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.*

**Spatial Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation**  
A.V. Dalca, J. Guttag, and M. R. Sabuncu  
*NIPS ML4H: Machine Learning for Health. 2017.* 
