# CMI_Neural_Estimator
Conditional Mutual Informaation Neural Estimator

## Introduction
In this repository I am explaining the method explained in [here](https://arxiv.org/abs/1911.02277) and another paper which is under submission process.
This technique is based on variational lower bounds for relative entropy known as Donsker-Varadhan (DV):
<img src="https://latex.codecogs.com/svg.latex?\Large&space;I(X;Y|Z)\geq E_{\djoint}\left[f(x,y,z)\right]-\log E_{\dprod}\big[ \exp f(x,y,z) \big]" />

$I(X;Y|Z)\geq E_{\djoint}\left[f(x,y,z)\right]-\log E_{\dprod}\big[ \exp f(x,y,z) \big]$,

\begin{equation}
I(X;Y|Z)\geq E_{\djoint}\left[f(x,y,z)\right]-\log E_{\dprod}\big[ \exp f(x,y,z) \big],
\end{equation}
and NWJ bound characterized as following for the conditional mutual information:
\begin{equation}
I(X;Y|Z)\geq E_{\djoint}\left[f(x,y,z)\right]- e^{-1}E_{\dprod}\big[ \exp f(x,y,z) \big].
\end{equation}


The neural network is implemented with **PyTorch**.

