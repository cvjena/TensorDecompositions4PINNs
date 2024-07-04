# TensorDecompositions4PINNs

Using Tensor decompositions for variable separation in PINNs. 


## Abstract:

Physics-Informed Neural Networks (PINNs) have shown great
promise in approximating partial differential equations (PDEs), although
they remain constrained by the curse of dimensionality. In this paper,
we propose a generalized PINN version of the classical variable separable
method. To do this, we first show that, using the universal approxima-
tion theorem, a multivariate function can be approximated by the outer
product of neural networks, whose inputs are separated variables. We
leverage tensor decomposition forms to separate the variables in a PINN
setting. By employing Canonic Polyadic (CP), Tensor-Train (TT), and
Tucker decomposition forms within the PINN framework, we create ro-
bust architectures for learning multivariate functions from separate neu-
ral networks connected by outer products. Our methodology significantly
enhances the performance of PINNs, as evidenced by improved results
on complex high-dimensional PDEs, including the 3D Helmholtz and 5D
Poisson equations, among others. This research underscores the poten-
tial of tensor decomposition-based variably separated PINNs to surpass
the state-of-the-art, offering a compelling solution to the dimensionality
challenge in PDE approximation.

## Required packages:

-tqdm
-jax
-pina 
-matplotlib











Forward gradients and structure taken from: https://proceedings.neurips.cc/paper_files/paper/2023/file/4af827e7d0b7bdae6097d44977e87534-Paper-Conference.pdf
