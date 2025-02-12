__all__ = ["CP_PINN", "TT_PINN", "Tucker_PINN"]

import string
from abc import ABC

import jax.numpy as jnp
from flax import linen as nn


class FTD_PINN(ABC, nn.Module):
    features: list[int]
    input_dim: int


class CP_PINN(FTD_PINN):
    def setup(self):
        # create einsum string for the CP decomposition
        # depending on the number of inputs
        # r denotes the kernel dimension
        # eg,
        #   - for 3 inputs it would be: "ra,rb,rc->abc"
        #   - for 5 inputs it would be: "ra,rb,rc,rd,re->abcde"

        einsum_start = ",".join([f"r{string.ascii_lowercase[i]}" for i in range(self.input_dim)])
        einsum_end = "".join([f"{string.ascii_lowercase[i]}" for i in range(self.input_dim)])

        self.decomp = f"{einsum_start}->{einsum_end}"

    @nn.compact
    def __call__(self, *inputs: jnp.ndarray):
        init = nn.initializers.xavier_uniform()

        outputs = []
        for x in inputs:
            for fs in self.features[:-1]:
                x = nn.Dense(fs, kernel_init=init)(x)
                x = nn.activation.tanh(x)

            x = nn.Dense(self.features[-1], kernel_init=init)(x)

            outputs += [jnp.transpose(x, (1, 0))]

        return jnp.einsum(self.decomp, *outputs)


class TT_PINN(FTD_PINN):
    def setup(self):
        # create the einsum string for the TensorTrain decomposition
        # depending on the number of inputs
        # eg,
        #   - for 3 inputs it would be: "a1,b12,c2->abc"
        #   - for 5 inputs it would be: "a1,b12,c23,d34,e4->abcde"
        #   - for 7 inputs it would be: "a1,b12,c23,d34,e45,f56,g6->abcdefg"

        einsum_start = ""
        for i in range(self.input_dim):
            letter = string.ascii_lowercase[i]

            if i == 0:
                einsum_start += f"{letter}{str(i + 1)}"
            elif i == self.input_dim - 1:
                einsum_start += f",{letter}{str(i)}"
            else:
                einsum_start += f",{letter}{str(i)}{str(i + 1)}"

        einsum_end = "".join([f"{string.ascii_lowercase[i]}" for i in range(self.input_dim)])
        self.decomp = f"{einsum_start}->{einsum_end}"

    @nn.compact
    def __call__(self, *inputs: jnp.ndarray):
        init = nn.initializers.xavier_uniform()

        outputs = []
        for i, X in enumerate(inputs):
            for fs in self.features[:-1]:
                X = nn.Dense(fs, kernel_init=init)(X)
                X = nn.activation.tanh(X)
            if i != 0 and i != 4:
                X = nn.DenseGeneral((self.features[-1], self.features[-1]), kernel_init=init)(X)
            else:
                X = nn.Dense(self.features[-1], kernel_init=init)(X)
            outputs += [X]

        return jnp.einsum(self.decomp, *outputs)


class Tucker_PINN(FTD_PINN):
    def setup(self):
        self.core = self.param(
            "core",
            nn.initializers.orthogonal(),
            tuple(self.features[-1] for _ in range(self.input_dim)),
        )

        # create the einsum string for the Tucker decomposition
        # depending on the number of inputs
        # eg,
        #   - for 3 inputs it would be: "klm,ka,lb,mc->abc"
        #   - for 5 inputs it would be: "klmno,ka,lb,mc,nd,oe->abcde"
        #   - for 7 inputs it would be: "klmnopq,ka,lb,mc,nd,oe,pf,qg->abcdefg"

        # we use UPPERCASE letters to denote the core tensor
        # and lowercase letters to denote the outputs
        str_rev = string.ascii_lowercase[::-1]
        einsum_start = "".join([f"{str_rev[i]}" for i in range(self.input_dim)])

        for i in range(self.input_dim):
            einsum_start += f",{str_rev[i]}{string.ascii_lowercase[i]}"

        einsum_end = "".join([f"{string.ascii_lowercase[i]}" for i in range(self.input_dim)])
        self.decomp = f"{einsum_start}->{einsum_end}"

    @nn.compact
    def __call__(self, *inputs: jnp.ndarray):
        init = nn.initializers.xavier_normal()

        outputs = []
        for X in inputs:
            for fs in self.features[:-1]:
                X = nn.Dense(fs, kernel_init=init)(X)
                X = nn.activation.tanh(X)
            X = nn.Dense(self.features[-1], kernel_init=init)(X)

            outputs += [jnp.transpose(X, (1, 0))]

        return jnp.einsum(self.decomp, self.core, *outputs)
