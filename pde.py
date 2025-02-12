__all__ = ["PDE", "Poisson5D", "FlowMixing3D", "KleinGordon", "Helmholtz3D"]

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax import jvp


def hvp_fwdfwd(f, primals, tangents, return_primals=False):
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
        # print("Received", len(train_data))

        tc, xc, yc, ti, xi, yi, ui, tb, xb, yb, ub, a, b = train_data

        # isolate loss func from redundant arguments
        loss_fn = lambda params: 10 * residual_loss(params, tc, xc, yc, a, b) + initial_loss(params, ti, xi, yi, ui) + boundary_loss(params, tb, xb, yb, ub)

        return loss_fn


class KleinGordon3D(PDE):
    name = "KleinGordon3D"
    input_dim = 3

    def exact(self, t, x, y):
        return (x + y) * jnp.cos(2 * t) + (x * y) * jnp.sin(2 * t)

    def source(self, t, x, y):
        u = self.exact(t, x, y)
        return u**2 - 4 * u

    def train_generator(self, nc, key):
        keys = jax.random.split(key, 3)
        # collocation points
        tc = jax.random.uniform(keys[0], (nc, 1), minval=0.0, maxval=10.0)
        xc = jax.random.uniform(keys[1], (nc, 1), minval=-1.0, maxval=1.0)
        yc = jax.random.uniform(keys[2], (nc, 1), minval=-1.0, maxval=1.0)
        tc_mesh, xc_mesh, yc_mesh = jnp.meshgrid(tc.ravel(), xc.ravel(), yc.ravel(), indexing="ij")
        uc = self.source(tc_mesh, xc_mesh, yc_mesh)
        # initial points
        ti = jnp.zeros((1, 1))
        xi = xc
        yi = yc
        ti_mesh, xi_mesh, yi_mesh = jnp.meshgrid(ti.ravel(), xi.ravel(), yi.ravel(), indexing="ij")
        ui = self.exact(ti_mesh, xi_mesh, yi_mesh)
        # boundary points (hard-coded)
        tb = [tc, tc, tc, tc]
        xb = [jnp.array([[-1.0]]), jnp.array([[1.0]]), xc, xc]
        yb = [yc, yc, jnp.array([[-1.0]]), jnp.array([[1.0]])]
        ub = []
        for i in range(4):
            tb_mesh, xb_mesh, yb_mesh = jnp.meshgrid(tb[i].ravel(), xb[i].ravel(), yb[i].ravel(), indexing="ij")
            ub += [self.exact(tb_mesh, xb_mesh, yb_mesh)]
        return tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub

    def test_generator(self, nc_test) -> tuple[tuple, tuple, jnp.ndarray]:
        t = jnp.linspace(0, 10, nc_test)
        x = jnp.linspace(-1, 1, nc_test)
        y = jnp.linspace(-1, 1, nc_test)
        t = jax.lax.stop_gradient(t)
        x = jax.lax.stop_gradient(x)
        y = jax.lax.stop_gradient(y)
        tm, xm, ym = jnp.meshgrid(t, x, y, indexing="ij")
        u_gt = self.exact(tm, xm, ym)
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        return (t, x, y), (tm, xm, ym), u_gt

    def loss(self, apply_fn, *train_data):
        def residual_loss(params, t, x, y, source_term):
            # calculate u
            u = apply_fn(params, t, x, y)
            # tangent vector dx/dx
            # assumes t, x, y have same shape (very important)
            v = jnp.ones(t.shape)
            # 2nd derivatives of u
            utt = hvp_fwdfwd(lambda t: apply_fn(params, t, x, y), (t,), (v,))
            uxx = hvp_fwdfwd(lambda x: apply_fn(params, t, x, y), (x,), (v,))
            uyy = hvp_fwdfwd(lambda y: apply_fn(params, t, x, y), (y,), (v,))
            return jnp.mean((utt - uxx - uyy + u**2 - source_term) ** 2)

        def initial_loss(params, t, x, y, u):
            return jnp.mean((apply_fn(params, t, x, y) - u) ** 2)

        def boundary_loss(params, t, x, y, u):
            loss = 0.0
            for i in range(4):
                loss += (1 / 4.0) * jnp.mean((apply_fn(params, t[i], x[i], y[i]) - u[i]) ** 2)
            return loss

        # unpack data
        tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data

        # isolate loss function from redundant arguments
        loss_fn = lambda params: residual_loss(params, tc, xc, yc, uc) + initial_loss(params, ti, xi, yi, ui) + boundary_loss(params, tb, xb, yb, ub)

        return loss_fn


class Helmholtz3D(PDE):
    name = "Helmholtz3D"
    input_dim = 3

    def __init__(self, a1: int = 4, a2: int = 4, a3: int = 3, lda: float = 1.0):
        super().__init__()
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.lda = 1.0

    def exact(self, x, y, z) -> jnp.ndarray:
        return jnp.sin(self.a1 * jnp.pi * x) * jnp.sin(self.a2 * jnp.pi * y) * jnp.sin(self.a3 * jnp.pi * z)

    def source(self, x, y, z) -> jnp.ndarray:
        u_gt = self.exact(x, y, z)
        uxx = -((self.a1 * jnp.pi) ** 2) * u_gt
        uyy = -((self.a2 * jnp.pi) ** 2) * u_gt
        uzz = -((self.a3 * jnp.pi) ** 2) * u_gt
        return uxx + uyy + uzz + self.lda * u_gt

    def train_generator(self, nc, key):
        keys = jax.random.split(key, 3)
        # collocation points
        xc = jax.random.uniform(keys[0], (nc,), minval=-1.0, maxval=1.0)
        yc = jax.random.uniform(keys[1], (nc,), minval=-1.0, maxval=1.0)
        zc = jax.random.uniform(keys[2], (nc,), minval=-1.0, maxval=1.0)
        # source term
        xcm, ycm, zcm = jnp.meshgrid(xc, yc, zc, indexing="ij")
        uc = self.source(xcm, ycm, zcm)
        xc, yc, zc = xc.reshape(-1, 1), yc.reshape(-1, 1), zc.reshape(-1, 1)
        # boundary (hard-coded)
        xb = [jnp.array([[1.0]]), jnp.array([[-1.0]]), xc, xc, xc, xc]
        yb = [yc, yc, jnp.array([[1.0]]), jnp.array([[-1.0]]), yc, yc]
        zb = [zc, zc, zc, zc, jnp.array([[1.0]]), jnp.array([[-1.0]])]
        return xc, yc, zc, uc, xb, yb, zb

    def test_generator(self, nc_test) -> tuple[tuple, tuple, jnp.ndarray]:
        x = jnp.linspace(-1.0, 1.0, nc_test)
        y = jnp.linspace(-1.0, 1.0, nc_test)
        z = jnp.linspace(-1.0, 1.0, nc_test)
        x = jax.lax.stop_gradient(x)
        y = jax.lax.stop_gradient(y)
        z = jax.lax.stop_gradient(z)
        xm, ym, zm = jnp.meshgrid(x, y, z, indexing="ij")
        u_gt = self.exact(xm, ym, zm)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
        return (x, y, z), (xm, ym, zm), u_gt

    def loss(self, apply_fn, *train_data):
        def residual_loss(params, x, y, z, source_term, lda=1.0):
            # compute u
            u = apply_fn(params, x, y, z)
            # tangent vector dx/dx
            v_x = jnp.ones(x.shape)
            v_y = jnp.ones(y.shape)
            v_z = jnp.ones(z.shape)
            # 2nd derivatives of u
            uxx = hvp_fwdfwd(lambda x: apply_fn(params, x, y, z), (x,), (v_x,))
            uyy = hvp_fwdfwd(lambda y: apply_fn(params, x, y, z), (y,), (v_y,))
            uzz = hvp_fwdfwd(lambda z: apply_fn(params, x, y, z), (z,), (v_z,))
            return jnp.mean(((uzz + uyy + uxx + lda * u) - source_term) ** 2)

        def boundary_loss(params, x, y, z):
            loss = 0.0
            for i in range(6):
                loss += jnp.mean(apply_fn(params, x[i], y[i], z[i]) ** 2)
            return loss

        # print("Received", len(train_data))

        # unpack data
        xc, yc, zc, uc, xb, yb, zb = train_data

        # isolate loss func from redundant arguments
        loss_fn = lambda params: residual_loss(params, xc, yc, zc, uc) + boundary_loss(params, xb, yb, zb)

        return loss_fn
