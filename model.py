__all__ = ["CP_PINN", "TTPINN", "TuckerPINN"]

import string
from abc import ABC

import jax.numpy as jnp
from flax import linen as nn


class FTD_PINN(ABC, nn.Module):
    features: list[int]


class CP_PINN(FTD_PINN):
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

        # create einsum string based on the amount of inputs
        # r denotes the kernel dimension
        # eg,
        #   - for 3 inputs it would be: "ra,rb,rc->abc"
        #   - for 5 inputs it would be: "ra,rb,rc,rd,re->abcde"

        einsum_start = ",".join([f"r{string.ascii_lowercase[i]}" for i in range(len(inputs))])
        einsum_end = "".join([f"{string.ascii_lowercase[i]}" for i in range(len(inputs))])
        return jnp.einsum(f"{einsum_start}->{einsum_end}", *outputs)


class TTPINN(FTD_PINN):
    @nn.compact
    def __call__(self, a, b, c, d, e):
        inputs, outputs = [a, b, c, d, e], []
        init = nn.initializers.xavier_uniform()
        for i, X in enumerate(inputs):
            for fs in self.features[:-1]:
                X = nn.Dense(fs, kernel_init=init)(X)
                X = nn.activation.tanh(X)
            if i != 0 and i != 4:
                X = nn.DenseGeneral((self.features[-1], self.features[-1]), kernel_init=init)(X)
            else:
                X = nn.Dense(self.features[-1], kernel_init=init)(X)
            outputs += [X]
        return jnp.einsum(
            "a1,b12,c23,d34,e4->abcde",
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
            outputs[4],
        )


class TuckerPINN(FTD_PINN):
    def setup(self):
        self.core = self.param(
            "core",
            nn.initializers.orthogonal(),
            (
                self.features[-1],
                self.features[-1],
                self.features[-1],
                self.features[-1],
                self.features[-1],
            ),
        )

    @nn.compact
    def __call__(self, a, b, c, d, e):
        inputs, outputs = [a, b, c, d, e], []
        init = nn.initializers.xavier_normal()
        for X in inputs:
            for fs in self.features[:-1]:
                X = nn.Dense(fs, kernel_init=init)(X)
                X = nn.activation.tanh(X)
            X = nn.Dense(self.features[-1], kernel_init=init)(X)

            outputs += [jnp.transpose(X, (1, 0))]
        return jnp.einsum(
            "klmno,ka,lb,mc,nd,oe->abcde",
            self.core,
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
            outputs[4],
        )
