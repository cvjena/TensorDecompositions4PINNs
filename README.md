# TensorDecompositions4PINNs

## TL;DR

Functional Tensor Decompositions explores how tensor decomposition techniques can unlock new possibilities for variable separation in Physics-Informed Neural Networks (PINNs), overcoming the curse of dimensionality in solving high-dimensional PDEs

## Abstract

Physics-Informed Neural Networks (PINNs) have shown great promise in approximating partial differential equations (PDEs), although they remain constrained by the curse of dimensionality.
In this paper, we propose a generalized PINN version of the classical variable separable method.
To do this, we first show that, using the universal approximation theorem, a multivariate function can be approximated by the outer product of neural networks, whose inputs are separated variables.
We leverage tensor decomposition forms to separate the variables in a PINN setting.
By employing Canonic Polyadic (CP), Tensor-Train (TT), and Tucker decomposition forms within the PINN framework, we create robust architectures for learning multivariate functions from separate neural networks connected by outer products.
Our methodology significantly enhances the performance of PINNs, as evidenced by improved results on complex high-dimensional PDEs, including the 3D Helmholtz and 5D Poisson equations, among others.
This research underscores the potential of tensor decomposition-based variably separated PINNs to surpass the state-of-the-art, offering a compelling solution to the dimensionality challenge in PDE approximation.

## Links

For more details, refer to our paper:

- **Published version:** [Springer Link](https://dl.acm.org/doi/10.1007/978-3-031-78389-0_3)
- **Preprint version:** [arXiv](https://arxiv.org/abs/2408.13101)

## Installation

We use Anaconda to utilize virtual environments. Any other `venv` framework will also likely be suitable.
We recommend `Python 3.11` as the desired version.
Our framework has minimal requirements you can install after cloning this repository with:

```bash
# Optional
conda create -n TD4PINN python=3.11 pip -y
conda activate TD4PINN
# Required
git clone git@github.com:cvjena/TensorDecompositions4PINNs.git
pip install -r requirements.txt
```

All models and experiments are implemented in `jax` and during the installation the needed Nvidia packages should be installed.
See the according Jupyter notebooks to reproduce the experiments from the paper (per dataset).

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{vemuri2024,
  author = {Vemuri, Sai Karthikeya and B\"{u}chner, Tim and Niebling, Julia and Denzler, Joachim},
  title = {Functional Tensor Decompositions for Physics-Informed Neural Networks},
  year = {2024},
  isbn = {978-3-031-78388-3},
  publisher = {Springer-Verlag},
  address = {Berlin, Heidelberg},
  url = {https://doi.org/10.1007/978-3-031-78389-0_3},
  doi = {10.1007/978-3-031-78389-0_3},
  abstract = {Physics-Informed Neural Networks (PINNs) have shown continuous and increasing promise in approximating partial differential equations (PDEs), although they remain constrained by the curse of dimensionality. In this paper, we propose a generalized PINN version of the classical variable separable method. To do this, we first show that, using the universal approximation theorem, a multivariate function can be approximated by the outer product of neural networks, whose inputs are separated variables. We leverage tensor decomposition forms to separate the variables in a PINN setting. By employing Canonic Polyadic (CP), Tensor-Train (TT), and Tucker decomposition forms within the PINN framework, we create robust architectures for learning multivariate functions from separate neural networks connected by outer products. Our methodology significantly enhances the performance of PINNs, as evidenced by improved results on complex high-dimensional PDEs, including the 3d Helmholtz and 5d Poisson equations, among others. This research underscores the potential of tensor decomposition-based variably separated PINNs to surpass the state-of-the-art, offering a compelling solution to the dimensionality challenge in PDE approximation.},
  booktitle = {Pattern Recognition: 27th International Conference, ICPR 2024, Kolkata, India, December 1–5, 2024, Proceedings, Part XXV},
  pages = {32–46},
  numpages = {15},
  keywords = {Tensor Decomposition, Physics-Informed Neural Networks},
  location = {Kolkata, India}
}
