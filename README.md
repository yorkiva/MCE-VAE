## Introducing MCEVAE

Multi-Clustering Equivariant Variational AutoEncoder (MCEVAE) is agenerative model that can perform transformation-invariant clustering of images while learning spatial transformations on a Lie manifold. Constrained learning of Lie transformations allows this model to create a caonical, invariant reconstruction of images from each cluster.

This codebase provides the basic implementation of MCEVAE using PyTorch and training the model on MNIST handwritten digits dataset.


## Instructions for getting the data

The data is obatined and formatted by the script `data/gen_mnist.py`. Running this scipt will create necessary `.npy` arrays for training the model with SO(2) and SE(2) transformed MNIST images. 

## Training the model

Before training the model, create empty directories called `models` and `losses`. To train the model on MNIST images, run the script `train_mnist.py`. To see the arguments allowed by this script, run `python train_mnist.py --help`

## Visualization of the results

The attached notebook `Tester.ipynb` has the necessary codeblocks to produce visualization of the performance of the model. Adjust the parameters so that it loads the correct dataset and loads the correct model.
