# Conditional Mutual Information Neural Estimator

## Introduction
In this repository you may find the method explained in [here](https://arxiv.org/abs/2006.07225) to estimate conditional mutual information.
This technique is based on variational lower bounds for relative entropy known as Donsker-Varadhan bound **(DV bound)** and **NWJ bound**. 
We use the k- nearest neighbor technique to help us design a neural classifier that is the basis of our estimation.

The model that we used in our simulations is a Gaussian model:

![The model](model.png?raw=true "Title")

## Implementation
The neural network is implemented with **PyTorch**.

## How to run
To run the code and reproduce the results in the paper use the help below:

### estimate I(X;Y|Z)
python main.py --d 5 --k=20 n=80000 scenario=0, seed=123

### estimate I(X;Z|Y)
python main.py --d 5 --k=20 n=80000 scenario=1, seed=123
