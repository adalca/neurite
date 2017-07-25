# neuron
neural networks for brain image analysis

This toolbox is **currently in development**, with the goal providing a set of tools with infrastructure for medical image analysis with neural network. While the tools are somewhat general, `neuron` will generally run with `keras` on top of `tensorflow`.

### Main tools
`callbacks`: a set of callbacks during keras training to help with understanding your fit, such as Dice measurements and volume-segmentation overlaps  
`generators`: generators for medical image volumes and various combinations of volumes, segmentation, categorical and other output  
`dataproc`: a set of tools for processing medical imaging data for preparation for training/testing  
`metrics`: metrics (most of which can be used as loss functions), such as dice or weighted categorical crossentropy.  
`models`: a set of flexible models (many parameters to play with...) particularly useful in medical image analysis, such as a U-net/hourglass model and a standard classifier. 

Other utilities and a few `jupyter` notebooks are also provided.

### Requirements:
numpy, python libraries from @adalca github account
 
### Development:
Please contact Adrian Dalca, adalca@csail.mit.edu for question related to `neuron`
