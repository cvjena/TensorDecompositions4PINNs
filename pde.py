__all__ = ["PDE", "Poisson5D", "FlowMixing3D"]

from abc import ABC, abstractmethod
import os
import time
from functools import partial
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
import tqdm
from flax import linen as nn
from jax import jvp, value_and_grad


def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    # g = lambda primals: jvp(f, (primals,), tangents)[1]
    def g(primals):
        return jvp(f, (primals,), tangents)[1]

    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out

    return tangents_out


class PDE(ABC):
    name: str

    @abstractmethod
    def exact(self):
        pass

    @abstractmethod
    def train_generator(self, nc, key):
        pass

    @abstractmethod
    def test_generator(self) -> tuple[tuple, tuple, jnp.ndarray]:
        pass

    @abstractmethod
    def loss(self, apply_fn, *train_data):
        pass


class Poisson5D(PDE):
    name = "Poisson5D"
    input_dim = 5

    def exact(self, a, b, c, d, e):
        sol = 0
        for i in [a, b, c, d, e]:
            sol += jnp.sin((jnp.pi / 2) * i)
        return sol

    def train_generator(self, nc, key):
        keys = jax.random.split(key, 6)
        ac = jax.random.uniform(keys[0], (nc,), minval=0.0, maxval=1.0)
        bc = jax.random.uniform(keys[1], (nc,), minval=0.0, maxval=1.0)
        cc = jax.random.uniform(keys[2], (nc,), minval=0.0, maxval=1.0)
        dc = jax.random.uniform(keys[3], (nc,), minval=0.0, maxval=1.0)
        ec = jax.random.uniform(keys[4], (nc,), minval=0.0, maxval=1.0)

        acm, bcm, ccm, dcm, ecm = jnp.meshgrid(ac, bc, cc, dc, ec, indexing="ij")
        source_term = 0
        for i in [acm, bcm, ccm, dcm, ecm]:
            source_term = source_term + ((jnp.pi * jnp.pi / 4) * jnp.sin((jnp.pi / 2) * i))

        ac = ac.reshape(-1, 1)
        bc = bc.reshape(-1, 1)
        cc = cc.reshape(-1, 1)
        dc = dc.reshape(-1, 1)
        ec = ec.reshape(-1, 1)

        ab = [jnp.array([[0.0]]), jnp.array([[1.0]]), ac, ac, ac, ac, ac, ac, ac, ac]
        bb = [bc, bc, jnp.array([[0.0]]), jnp.array([[1.0]]), bc, bc, bc, bc, bc, bc]
        cb = [cc, cc, cc, cc, jnp.array([[0.0]]), jnp.array([[1.0]]), cc, cc, cc, cc]
        db = [dc, dc, dc, dc, dc, dc, jnp.array([[0.0]]), jnp.array([[1.0]]), dc, dc]
        eb = [ec, ec, ec, ec, ec, ec, ec, ec, jnp.array([[0.0]]), jnp.array([[1.0]])]

        ub = []
        for i in range(10):
            abm, bbm, cbm, dbm, ebm = jnp.meshgrid(ab[i].ravel(), bb[i].ravel(), cb[i].ravel(), db[i].ravel(), eb[i].ravel(), indexing="ij")
            ub += [self.exact(abm, bbm, cbm, dbm, ebm)]

        return ac, bc, cc, dc, ec, source_term, ab, bb, cb, db, eb, ub

    def test_generator(self, nc_test):
        a = jnp.linspace(0, 1, nc_test)
        b = jnp.linspace(0, 1, nc_test)
        c = jnp.linspace(0, 1, nc_test)
        d = jnp.linspace(0, 1, nc_test)
        e = jnp.linspace(0, 1, nc_test)
        am, bm, cm, dm, em = jnp.meshgrid(a, b, c, d, e, indexing="ij")

        u_gt = self.exact(am, bm, cm, dm, em)

        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        c = c.reshape(-1, 1)
        d = d.reshape(-1, 1)
        e = e.reshape(-1, 1)
        return (a, b, c, d, e), (am, bm, cm, dm, em), u_gt

    def loss(self, apply_fn, *train_data) -> callable:
        def residual_loss(params, a, b, c, d, e, source):
            # tangent vector dx/dx
            # v_f = jnp.ones(f.shape)
            v_a = jnp.ones(a.shape)
            v_b = jnp.ones(b.shape)
            v_c = jnp.ones(c.shape)
            v_d = jnp.ones(d.shape)
            v_e = jnp.ones(e.shape)

            uaa = hvp_fwdfwd(lambda a: apply_fn(params, a, b, c, d, e), (a,), (v_a,))
            ubb = hvp_fwdfwd(lambda b: apply_fn(params, a, b, c, d, e), (b,), (v_b,))
            ucc = hvp_fwdfwd(lambda c: apply_fn(params, a, b, c, d, e), (c,), (v_c,))
            udd = hvp_fwdfwd(lambda d: apply_fn(params, a, b, c, d, e), (d,), (v_d,))
            uee = hvp_fwdfwd(lambda e: apply_fn(params, a, b, c, d, e), (e,), (v_e,))
            # uff = hvp_fwdfwd(lambda t: apply_fn(params,a,b,c,d,e,f), (f,), (v_f,))
            nabla_u = uaa + ubb + ucc + udd + uee
            return jnp.mean((nabla_u + source) ** 2)

        def boundary_loss(params, a, b, c, d, e, u):
            loss = 0
            for i in range(10):
                loss += jnp.mean((apply_fn(params, a[i], b[i], c[i], d[i], e[i]) - u[i]) ** 2)
                return loss

        ac, bc, cc, dc, ec, source_term, ab, bb, cb, db, eb, ub = train_data
        loss_fn = lambda params: residual_loss(params, ac, bc, cc, dc, ec, source_term) + boundary_loss(params, ab, bb, cb, db, eb, ub)
        return loss_fn


class FlowMixing3D(PDE):
    name = "FlowMixing3D"
    input_dim = 3
    v_max = 0.385

    def exact(self, x, y, t, omega):
        return -jnp.tanh((y / 2) * jnp.cos(omega * t) - (x / 2) * jnp.sin(omega * t))

    def params(self, t, x, y, v_max=0.385, require_ab=False):
        # t, x, y must be meshgrid
        r = jnp.sqrt(x**2 + y**2)
        v_t = ((1 / jnp.cosh(r)) ** 2) * jnp.tanh(r)
        omega = (1 / r) * (v_t / v_max)
        a, b = None, None
        if require_ab:
            a = -(v_t / v_max) * (y / r)
            b = (v_t / v_max) * (x / r)
        return omega, a, b

    def train_generator(self, nc, key):
        keys = jax.random.split(key, 3)
        # collocation points
        tc = jax.random.uniform(keys[0], (nc, 1), minval=0.0, maxval=4.0)
        xc = jax.random.uniform(keys[1], (nc, 1), minval=-4.0, maxval=4.0)
        yc = jax.random.uniform(keys[2], (nc, 1), minval=-4.0, maxval=4.0)
        tc_mesh, xc_mesh, yc_mesh = jnp.meshgrid(tc.ravel(), xc.ravel(), yc.ravel(), indexing="ij")

        _, a, b = self.params(tc_mesh, xc_mesh, yc_mesh, v_max=self.v_max, require_ab=True)

        # initial points
        ti = jnp.zeros((1, 1))
        xi = xc
        yi = yc
        ti_mesh, xi_mesh, yi_mesh = jnp.meshgrid(ti.ravel(), xi.ravel(), yi.ravel(), indexing="ij")
        omega_i, _, _ = self.params(ti_mesh, xi_mesh, yi_mesh, v_max=self.v_max)
        ui = self.exact(ti_mesh, xi_mesh, yi_mesh, omega_i)
        # boundary points (hard-coded)
        tb = [tc, tc, tc, tc]
        xb = [jnp.array([[-4.0]]), jnp.array([[4.0]]), xc, xc]
        yb = [yc, yc, jnp.array([[-4.0]]), jnp.array([[4.0]])]
        ub = []
        for i in range(4):
            tb_mesh, xb_mesh, yb_mesh = jnp.meshgrid(tb[i].ravel(), xb[i].ravel(), yb[i].ravel(), indexing="ij")
            omega_b, _, _ = self.params(tb_mesh, xb_mesh, yb_mesh, v_max=self.v_max)
            ub += [self.exact(tb_mesh, xb_mesh, yb_mesh, omega_b)]

        return tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b

    def test_generator(self, nc_test) -> tuple[tuple, tuple, jnp.ndarray]:
        v_max = 0.385
        t = jnp.linspace(0, 4, nc_test)
        x = jnp.linspace(-4, 4, nc_test)
        y = jnp.linspace(-4, 4, nc_test)
        t = jax.lax.stop_gradient(t)
        x = jax.lax.stop_gradient(x)
        y = jax.lax.stop_gradient(y)
        tm, xm, ym = jnp.meshgrid(t, x, y, indexing="ij")

        omega, _, _ = self.params(tm, xm, ym, v_max)
        u_gt = self.exact(tm, xm, ym, omega)
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        return (t, x, y), (tm, xm, ym), u_gt

    def loss(self, apply_fn, *train_data) -> callable:
        def residual_loss(params, t, x, y, a, b):
            # tangent vector dx/dx
            v_t = jnp.ones(t.shape)
            v_x = jnp.ones(x.shape)
            v_y = jnp.ones(y.shape)
            # 1st derivatives of u
            ut = jvp(lambda t: apply_fn(params, t, x, y), (t,), (v_t,))[1]
            ux = jvp(lambda x: apply_fn(params, t, x, y), (x,), (v_x,))[1]
            uy = jvp(lambda y: apply_fn(params, t, x, y), (y,), (v_y,))[1]
            return jnp.mean((ut + a * ux + b * uy) ** 2)

        def initial_loss(params, t, x, y, u):
            return jnp.mean((apply_fn(params, t, x, y) - u) ** 2)

        def boundary_loss(params, t, x, y, u):
            loss = 0.0
            for i in range(4):
                loss += jnp.mean((apply_fn(params, t[i], x[i], y[i]) - u[i]) ** 2)
            return loss

        # unpack data
        print("Received", len(train_data))

        tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b = train_data

        # isolate loss func from redundant arguments
        loss_fn = lambda params: 10 * residual_loss(params, tc, xc, yc, a, b) + initial_loss(params, ti, xi, yi, ui) + boundary_loss(params, tb, xb, yb, ub)

        return loss_fn
