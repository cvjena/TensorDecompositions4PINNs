import os
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
import tqdm
from flax import linen as nn
from jax import value_and_grad

from model import CP_PINN, TT_PINN, Tucker_PINN
from pde import Poisson5D, PDE, FlowMixing3D, Helmholtz3D, KleinGordon3D


def relative_l2(u, u_gt):
    return jnp.linalg.norm(u - u_gt) / jnp.linalg.norm(u_gt)


# optimizer step function
@partial(jax.jit, static_argnums=(0,))
def update_model(optim, gradient, params, state):
    updates, state = optim.update(gradient, state)
    params = optax.apply_updates(params, updates)
    return params, state


def main(model: nn.Module, pde: PDE, NC: int, NC_TEST: int, SEED: int, LR: float, EPOCHS: int, LOG_ITER: int):
    # force jax to use one device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # random key
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)

    params = model.init(subkey, *[jax.random.uniform(key, (NC, 1)) for _ in range(pde.input_dim)])

    # optimizer
    optim = optax.adam(LR)
    state = optim.init(params)

    key, subkey = jax.random.split(key, 2)
    train_data = pde.train_generator(NC, subkey)

    test_input, _, u_gt = pde.test_generator(NC_TEST)
    logger = []

    apply_fn = jax.jit(model.apply)
    loss_fn = pde.loss(apply_fn, *train_data)

    @jax.jit
    def train_one_step(params, state):
        # compute loss and gradient
        loss, gradient = value_and_grad(loss_fn)(params)
        # update state
        params, state = update_model(optim, gradient, params, state)
        return loss, params, state

    start = time.time()

    pbar = tqdm.tqdm(total=EPOCHS)
    error = np.nan

    for iters in range(1, EPOCHS + 1):
        # single run
        loss, params, state = train_one_step(params, state)

        if iters % LOG_ITER == 0 or iters == 1:
            u = apply_fn(params, *test_input)
            error = relative_l2(u, u_gt)
            logger.append([iters, loss, error])

        pbar.set_postfix({"loss": f"{loss:0.8f}", "error": f"{error:0.8f}"}, refresh=False)
        pbar.update(1)
    pbar.close()

    end = time.time()
    print(f"Runtime: {((end-start)/EPOCHS*1000):.2f} ms/iter.")
    return logger


if __name__ == "__main__":
    N_LAYERS = 4

    points = 24
    for pde_cls in [Poisson5D, FlowMixing3D, Helmholtz3D, KleinGordon3D]:
        pde = pde_cls()
        out_folder = Path(pde.name) / "results"
        out_folder.mkdir(exist_ok=True, parents=True)

        for model_cls in [CP_PINN]:
            for rank in [6]:
                for run in range(1):
                    # feature sizes
                    feat_sizes = [rank for _ in range(N_LAYERS)]

                    model = model_cls(feat_sizes, pde.input_dim)
                    model_name = model.__class__.__name__

                    print(f"Running {model_name} with rank {rank} and run {run}")

                    save_folder = out_folder / model_name / f"Rank_{rank:03d}"
                    save_folder.mkdir(exist_ok=True, parents=True)

                    logs = main(model=model, pde=pde, NC=points, NC_TEST=32, SEED=444444 + run, LR=1e-3, EPOCHS=10_000, LOG_ITER=5000)

                    out_file = save_folder / f"{model_name}-Rank_{rank:02d}-Points_{points:02d}-run_{run:02d}.csv"
                    pd.DataFrame(logs, columns=["Iter", "Loss", "Error"]).to_csv(out_file, index=False, float_format="%.16f")
