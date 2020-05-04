# Conditional Mutual Information Neural Estimator

## Introduction
In this repository I am explaining the method explained in [here](https://arxiv.org/abs/1911.02277) and another paper which is under submission process.
This technique is based on variational lower bounds for relative entropy known as Donsker-Varadhan bound **(DV bound)** and **NWJ bound**. 

## What is included?
In this repository, I am including my implementations for estimating mutual information (MI) using neural networks. Then I extend the method to estimate the conditional mutual information. 
The main contribution of this work is using k nearest neighborhood technique to construct sample batches for training the neural estimator.

## Implementation
The neural network is implemented with **PyTorch**.

