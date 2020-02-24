import tensorflow as tf
import gpflow
import t3f

import math
import time
import os

from gtilde import gtilde, euler_mascheroni

pi = tf.constant(math.pi, dtype=tf.float64)


def kernel(x, l, y=None):
    if y is None:
        y = x
    xs = x / l
    ys = y / l
    dist = -2 * tf.matmul(xs, ys, transpose_b=True)
    dist = dist + xs ** 2 + tf.transpose(ys) ** 2
    return tf.exp(-0.5 * dist)


def lower_tril(mat):
    cores = []
    batch = isinstance(mat, t3f.TensorTrainBatch)
    for c in mat.tt_cores:
        if batch:
            cores.append(
                tf.linalg.band_part(c[:, 0, :, :, 0], -1, 0)[
                    :, tf.newaxis, :, :, tf.newaxis
                ]
            )
        else:
            cores.append(
                tf.linalg.band_part(c[0, :, :, 0], -1, 0)[tf.newaxis, :, :, tf.newaxis]
            )
    if batch:
        return t3f.TensorTrainBatch(cores, tt_ranks=[1] * (len(cores) + 1))
    else:
        return t3f.TensorTrain(cores, tt_ranks=[1] * (len(cores) + 1))


def softplus_inverse(x):
    return tf.math.log(tf.math.exp(x) - 1.0)


class PoissonTTSVGPR:
    def __init__(self, X, Y, n_inducers=10):  # TODO: at least 5 inducers
        self.X = X
        self.Y = tf.cast(Y, X.dtype)

        self._len = tf.Variable(1.0, dtype=X.dtype)
        self._sigma = tf.Variable(1.0, dtype=X.dtype)
        self.u = tf.Variable(1.0, dtype=X.dtype)
        self.r = 4

        self.n_inducers_per_dim = n_inducers
        self.n_inducers = n_inducers ** X.shape[1]
        self.data_minrange, self.data_maxrange = (
            tf.squeeze(tf.math.reduce_min(X, axis=0, keepdims=True)),
            tf.squeeze(tf.math.reduce_max(X, axis=0, keepdims=True)),
        )
        self.inducers_grid = [
            tf.linspace(
                self.data_minrange[i], self.data_maxrange[i], self.n_inducers_per_dim,
            )
            for i in range(X.shape[1])
        ]
        self.spacing = (self.data_maxrange - self.data_minrange) / (
            self.n_inducers_per_dim - 1
        )
        self.abs_Tau = tf.reduce_prod(tf.abs(self.data_maxrange - self.data_minrange))

        K_mm = self.K_mm()
        self.Sigma = t3f.get_variable("Sigma", initializer=K_mm)
        self.ones = self.ones_tt()

        mus = [
            0.1 * tf.ones(shape=(1, self.n_inducers_per_dim, 1, self.r), dtype=X.dtype)
        ]
        for i in range(X.shape[1] - 2):
            mus.append(
                0.1
                * tf.ones(
                    shape=(self.r, self.n_inducers_per_dim, 1, self.r), dtype=X.dtype
                )
            )
        mus.append(
            0.1 * tf.ones(shape=(self.r, self.n_inducers_per_dim, 1, 1), dtype=X.dtype)
        )
        self.m = t3f.get_variable(
            "m",
            initializer=t3f.TensorTrain(
                mus, tt_ranks=[1] + [self.r] * (X.shape[1] - 1) + [1]
            ),
        )

        Ws = []
        gridpoints = tf.cast((X - self.data_minrange) // self.spacing, tf.int32)
        n_points = X.shape[0]
        for i in range(len(self.inducers_grid)):
            s = (
                X[:, i] - tf.gather(self.inducers_grid[i], gridpoints[:, i])
            ) / self.spacing[i]
            idx = gridpoints[tf.newaxis, :, i] + tf.constant(
                [-1, 0, 1, 2], dtype=tf.int32, shape=(4, 1)
            )

            equal_idx = tf.cast(tf.where(s == 0), tf.int32)
            left_idx = tf.cast(tf.where((idx[0, :] < 0) & (s != 0)), tf.int32)
            right_idx = tf.cast(
                tf.where((idx[3, :] >= self.n_inducers_per_dim) & (s != 0)), tf.int32
            )

            include_idx = tf.where(
                (idx[0, :] >= 0) & (idx[3, :] < self.n_inducers_per_dim) & (s != 0)
            )

            s3 = s ** 3
            s2 = s ** 2
            vals = tf.stack(
                [
                    -0.5 * s3 + s2 - 0.5 * s,
                    1.5 * s3 - 2.5 * s2 + 1,
                    -1.5 * s3 + 2 * s2 + 0.5 * s,
                    0.5 * (s3 - s2),
                ],
                axis=0,
            )

            vals = tf.gather(vals, include_idx, axis=1)
            idx = tf.gather(idx, include_idx, axis=1)
            idx = tf.concat(
                [
                    tf.tile(tf.gather(tf.range(n_points), include_idx), [4, 1]),
                    tf.reshape(idx, [-1, 1]),
                ],
                axis=1,
            )
            vals = tf.reshape(vals, [-1])

            idx = tf.concat(
                [
                    idx,
                    tf.concat(
                        [
                            tf.tile(left_idx, (3, 1)),
                            tf.repeat([0, 1, 2], left_idx.shape[0])[:, tf.newaxis],
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
            ts3 = tf.gather(s3, left_idx)
            ts2 = tf.gather(s2, left_idx)
            ts = tf.gather(s, left_idx)
            vals = tf.concat(
                [
                    vals,
                    tf.reshape(
                        tf.stack(
                            [0.5 * ts2 - 1.5 * ts + 1, -ts2 + 2 * ts, 0.5 * (ts2 - ts)],
                            axis=0,
                        ),
                        [-1],
                    ),
                ],
                axis=0,
            )

            idx = tf.concat(
                [
                    idx,
                    tf.concat(
                        [
                            tf.tile(right_idx, (3, 1)),
                            tf.repeat(
                                [
                                    self.n_inducers_per_dim - 3,
                                    self.n_inducers_per_dim - 2,
                                    self.n_inducers_per_dim - 1,
                                ],
                                right_idx.shape[0],
                            )[:, tf.newaxis],
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
            ts3 = tf.gather(s3, right_idx)
            ts2 = tf.gather(s2, right_idx)
            ts = tf.gather(s, right_idx)
            vals = tf.concat(
                [
                    vals,
                    tf.reshape(
                        tf.stack(
                            [0.5 * (ts2 - ts), -ts2 + 1, 0.5 * (ts2 + ts)], axis=0
                        ),
                        [-1],
                    ),
                ],
                axis=0,
            )

            idx = tf.concat(
                [
                    idx,
                    tf.concat(
                        [equal_idx, tf.gather(gridpoints[:, i], equal_idx)], axis=1
                    ),
                ],
                axis=0,
            )
            vals = tf.concat(
                [vals, tf.cast(tf.repeat([1.0], equal_idx.shape[0]), vals.dtype)],
                axis=0,
            )

            Ws.append(
                tf.scatter_nd(idx, vals, (n_points, self.n_inducers_per_dim))[
                    :, tf.newaxis, :, tf.newaxis, tf.newaxis
                ]
            )

        self.W = t3f.TensorTrainBatch(Ws, tt_ranks=[1] * (len(Ws) + 1))

    @property
    def l(self):
        return tf.math.softplus(self._len)

    @property
    def sigma(self):
        return tf.math.softplus(self._sigma)

    def _Sigma(self):
        S_L = lower_tril(self.Sigma)
        return t3f.ops.matmul(S_L, t3f.ops.transpose(S_L))

    def K_mm(self, clip=1e-3):
        kerns = []
        for i in range(len(self.inducers_grid)):
            k = kernel(self.inducers_grid[i][:, tf.newaxis], self.l)[
                tf.newaxis, :, :, tf.newaxis
            ]
            kerns.append(tf.where(k < clip, tf.cast(0.0, k.dtype), k))
        kerns[0] *= self.sigma
        return t3f.TensorTrain(kerns, tt_ranks=[1] * (len(self.inducers_grid) + 1))

    def Psi(self):
        Psis = []
        for i in range(len(self.inducers_grid)):
            xs = self.inducers_grid[i] / self.l
            z = xs[:, tf.newaxis] - xs[tf.newaxis, :]
            zbar = 0.5 * (xs[:, tf.newaxis] + xs[tf.newaxis, :])
            Psi_i = (
                -0.5
                * tf.sqrt(tf.cast(pi, self.l.dtype))
                * self.l
                * tf.exp(-0.25 * z ** 2)
                * (
                    tf.math.erf(zbar - self.data_maxrange[i] / self.l)
                    - tf.math.erf(zbar - self.data_minrange[i] / self.l)
                )
            )
            Psis.append(Psi_i[tf.newaxis, :, :, tf.newaxis])
        Psis[0] *= self.sigma ** 2
        return t3f.TensorTrain(Psis, tt_ranks=[1] * (len(Psis) + 1))

    def phi(self):
        phis = []
        s2 = tf.sqrt(tf.constant(2, dtype=self.l.dtype))
        for i in range(len(self.inducers_grid)):
            phi_i = (
                tf.sqrt(tf.cast(pi, self.l.dtype))
                * self.l
                / s2
                * (
                    tf.math.erf(
                        (self.data_maxrange[i] - self.inducers_grid[i]) / (s2 * self.l)
                    )
                    - tf.math.erf(
                        (self.data_minrange[i] - self.inducers_grid[i]) / (s2 * self.l)
                    )
                )
            )
            phis.append(phi_i[tf.newaxis, :, tf.newaxis, tf.newaxis])
        phis[0] *= self.sigma
        return t3f.TensorTrain(phis, tt_ranks=[1] * (len(phis) + 1))

    def ones_tt(self):
        components = []
        for i in range(len(self.inducers_grid)):
            components.append(
                tf.ones((1, self.n_inducers_per_dim, 1, 1), dtype=self.Sigma.dtype)
            )
        return t3f.TensorTrain(components, tt_ranks=[1] * (len(components) + 1))

    def _elbo(self, trace=False):
        S = self._Sigma()

        K_mm = self.K_mm()
        K_mm_inv = t3f.kronecker.inv(K_mm)
        Psi = (
            self.Psi()
        )  # + t3f.multiply(t3f.ones_like(K_mm), 2 * self.sigma * 1e-7) + t3f.multiply(t3f.eye([self.n_inducers_per_dim] * len(self.inducers_grid), dtype=K_mm.dtype), 1e-10)
        phi = self.phi()

        mu2 = (
            t3f.ops.flat_inner(self.m, self.W)
            + self.u * (1.0 - t3f.ops.flat_inner(self.W, self.ones))
        ) ** 2
        sigma2 = self.sigma - t3f.ops.flat_inner(
            self.W, t3f.ops.matmul(K_mm - S, self.W)
        )

        K_inv_m = t3f.ops.matmul(K_mm_inv, self.m)
        K_inv_ones = t3f.ops.matmul(K_mm_inv, self.ones)

        E_f2 = (
            t3f.ops.bilinear_form(Psi, K_inv_m, K_inv_m)
            + 2 * self.u * t3f.ops.bilinear_form(K_mm_inv, phi, self.m)
            - 2 * self.u * t3f.ops.bilinear_form(Psi, K_inv_m, K_inv_ones)
            + self.u ** 2 * self.abs_Tau
            - 2 * self.u ** 2 * t3f.ops.bilinear_form(K_mm_inv, phi, self.ones)
            + self.u ** 2 * t3f.ops.bilinear_form(Psi, K_inv_ones, K_inv_ones)
        )

        Var_f = self.sigma * self.abs_Tau - t3f.ops.flat_inner(
            t3f.eye([self.n_inducers_per_dim] * len(self.inducers_grid), K_mm.dtype)
            - t3f.ops.transpose(t3f.ops.matmul(K_mm_inv, S)),
            t3f.ops.matmul(K_mm_inv, Psi),
        )

        u_minus_m = t3f.ops.multiply(self.ones, self.u) - self.m
        KLdiv = 0.5 * (
            t3f.ops.flat_inner(S, K_mm_inv)
            + t3f.kronecker.slog_determinant(K_mm)[1]
            - t3f.kronecker.slog_determinant(S)[1]
            - self.n_inducers
            + t3f.ops.bilinear_form(K_mm_inv, u_minus_m, u_minus_m)
        )

        E_logf2 = tf.math.reduce_sum(
            tf.squeeze(self.Y)
            * (
                -tf.cast(gtilde(-0.5 * mu2 / sigma2), sigma2.dtype)
                + tf.math.log(0.5 * sigma2)
                - tf.cast(euler_mascheroni, sigma2.dtype)
            )
            - tf.math.lgamma(tf.squeeze(self.Y) + 1)
        )

        elbo = (-(E_f2 + Var_f) + E_logf2 - KLdiv) / self.X.shape[0]

        if trace:
            tf.summary.scalar("elbo", elbo)
            tf.summary.scalar("lengthscale", self.l)
            tf.summary.scalar("sigma", self.sigma)
            tf.summary.scalar("u", self.u)
            tf.summary.scalar("E_f2", E_f2)
            tf.summary.scalar("Var_f", Var_f)
            tf.summary.scalar("KLdiv", KLdiv)
            tf.summary.scalar("E_logf2", E_logf2)

            tf.summary.image("Psi", t3f.full(Psi)[tf.newaxis, :, :, tf.newaxis])
            tf.summary.image("K_mm", t3f.full(K_mm)[tf.newaxis, :, :, tf.newaxis])
            tf.summary.image("S", t3f.full(S)[tf.newaxis, :, :, tf.newaxis])

            tf.summary.histogram("mu2", mu2)
            tf.summary.histogram("sigma2", sigma2)
            tf.summary.histogram("m", t3f.full(self.m))
            tf.summary.histogram("Phi", t3f.full(phi))

        return elbo

    def elbo(self):
        return self._elbo()

    def neg_elbo(self):
        return -self.elbo()

    def _neg_elbo(self, trace=False):
        negelbo = -self._elbo(trace)
        if trace:
            tf.summary.scalar("neg_elbo", negelbo)
        return negelbo

    def optimize(self, logdir=None):
        bfgs = gpflow.optimizers.Scipy()
        vars = [self._len, self._sigma, self.u]
        vars.extend(self.Sigma.tt_cores)
        vars.extend(self.m.tt_cores)

        trace = logdir is not None
        if trace:
            writer = tf.summary.create_file_writer(
                os.path.join(logdir, time.strftime("%Y-%m-%d_%H%M%S"))
            )
        else:
            writer = tf.summary.create_noop_writer()

        step = tf.Variable(0, dtype=tf.int64)

        def objective():
            tf.summary.experimental.set_step(step)
            negelbo = self._neg_elbo(trace)
            step.assign_add(1)
            return negelbo

        with writer.as_default():
            bfgs.minimize(
                objective, variables=vars, method="bfgs", options={"disp": True},
            )

    def predict_mean(self, x):
        S = self._Sigma()

        K_mm = self.K_mm()
        K_mm_inv = t3f.kronecker.inv(K_mm)

        k_zx = [
            kernel(x[:, i, tf.newaxis], self.l, self.inducers_grid[i][:, tf.newaxis])[
                :, tf.newaxis, :, tf.newaxis, tf.newaxis
            ]
            for i in range(len(self.inducers_grid))
        ]
        k_zx[0] *= self.sigma
        k_zx = t3f.TensorTrainBatch(k_zx, tt_ranks=[1] * (len(self.inducers_grid) + 1))

        K_mm_inv_K_zx = t3f.ops.matmul(K_mm_inv, k_zx)
        m_K_mm_inv_K_zx = t3f.ops.flat_inner(self.m, K_mm_inv_K_zx)
        mu2 = (
            m_K_mm_inv_K_zx ** 2
            + 2
            * m_K_mm_inv_K_zx
            * self.u
            * (1 - t3f.ops.flat_inner(self.ones, K_mm_inv_K_zx))
            + self.u ** 2 * (1 - t3f.ops.flat_inner(self.ones, K_mm_inv_K_zx)) ** 2
        )
        sigma2 = (
            self.sigma
            - t3f.ops.flat_inner(k_zx, K_mm_inv_K_zx)
            + t3f.ops.flat_inner(K_mm_inv_K_zx, t3f.ops.matmul(S, K_mm_inv_K_zx))
        )

        return mu2 + sigma2


class PoissonSVGPR:
    def __init__(self, X, Y, n_inducers=10):  # TODO: at least 5 inducers
        self.X = tf.constant(X)
        self.Y = tf.cast(tf.constant(Y), self.X.dtype)

        self.n_inducers_per_dim = n_inducers
        self.n_inducers = n_inducers ** X.shape[1]
        self.data_minrange, self.data_maxrange = (
            tf.math.reduce_min(X, axis=0),
            tf.math.reduce_max(X, axis=0),
        )
        self.inducers_grid = [
            tf.linspace(
                self.data_minrange[i], self.data_maxrange[i], self.n_inducers_per_dim,
            )
            for i in range(X.shape[1])
        ]
        grids = tf.meshgrid(*self.inducers_grid)
        self.inducers = tf.stack([tf.reshape(x, [-1]) for x in grids], axis=1)
        self.spacing = (self.data_maxrange - self.data_minrange) / (
            self.n_inducers_per_dim - 1
        )
        self.abs_Tau = tf.reduce_prod(tf.abs(self.data_maxrange - self.data_minrange))

        self._len = tf.Variable(1.0, dtype=X.dtype)
        self._beta = tf.Variable(
            softplus_inverse(0.66 * tf.math.sqrt(tf.reduce_sum(self.Y) / self.abs_Tau)),
            dtype=X.dtype,
        )
        self._sigma = tf.Variable(
            0.5 * tf.reduce_sum(self.Y) / self.abs_Tau, dtype=X.dtype
        )
        K_mm = self.K_mm()
        self.Sigma = tf.Variable(K_mm)
        self.m = tf.Variable(tf.cast([0] * self.n_inducers, X.dtype))

    @property
    def l(self):
        return tf.math.softplus(self._len)

    @l.setter
    def l(self, l):
        self._len.assign(softplus_inverse(tf.cast(l, self._len.dtype)))

    @property
    def beta(self):
        return tf.math.softplus(self._beta)

    @beta.setter
    def beta(self, beta):
        self._beta.assign(softplus_inverse(tf.cast(beta, self._beta.dtype)))

    @property
    def sigma(self):
        return tf.math.softplus(self._sigma)

    @sigma.setter
    def sigma(self, sigma):
        self._sigma.assign(softplus_inverse(tf.cast(sigma, self._sigma.dtype)))

    def kernel(self, x, y=None):
        if y is None:
            y = x
        xs = x / self.l
        ys = y / self.l
        dist = -2 * tf.matmul(xs, ys, transpose_b=True)
        dist = (
            dist
            + tf.reduce_sum(xs ** 2, axis=1)[:, tf.newaxis]
            + tf.reduce_sum(ys ** 2, axis=1)[tf.newaxis, :]
        )
        return self.sigma * tf.exp(-0.5 * dist)

    def _Sigma(self):
        S_L = tf.linalg.band_part(self.Sigma, -1, 0)
        return (
            tf.matmul(S_L, S_L, transpose_b=True)
            + tf.eye(S_L.shape[0], dtype=S_L.dtype) * 1e-6
        )

    def K_mm(self):
        K_mm = self.kernel(self.inducers)
        return K_mm + tf.eye(K_mm.shape[0], dtype=K_mm.dtype) * 1e-6

    def Psi(self):
        Psis = []
        for i in range(len(self.inducers_grid)):
            xs = self.inducers[:, i] / self.l
            z = xs[:, tf.newaxis] - xs[tf.newaxis, :]
            zbar = 0.5 * (xs[:, tf.newaxis] + xs[tf.newaxis, :])
            Psi_i = (
                -0.5
                * tf.sqrt(tf.cast(pi, self.l.dtype))
                * self.l
                * tf.exp(-0.25 * z ** 2)
                * (
                    tf.math.erf(zbar - self.data_maxrange[i] / self.l)
                    - tf.math.erf(zbar - self.data_minrange[i] / self.l)
                )
            )
            Psis.append(Psi_i)
        Psi = Psis[0] * self.sigma ** 2
        for i in range(1, len(Psis)):
            Psi = Psi * Psis[i]
        return Psi

    def phi(self):
        phis = []
        s2 = tf.sqrt(tf.constant(0.5, dtype=self.l.dtype))
        for i in range(len(self.inducers_grid)):
            inducers = self.inducers[:, i]
            phi_i = (
                tf.sqrt(tf.cast(pi, self.l.dtype))
                * self.l
                * s2
                * (
                    tf.math.erf((self.data_maxrange[i] - inducers) / self.l * s2)
                    - tf.math.erf((self.data_minrange[i] - inducers) / self.l * s2)
                )
            )
            phis.append(phi_i)
        phi = phis[0] * self.sigma
        for i in range(1, len(phis)):
            phi = phi * phis[i]
        return phi

    def ones_tt(self):
        return tf.ones(shape=(self.Sigma.shape[1],), dtype=self.Sigma.dtype)

    def _elbo(self, trace=False):
        S = self._Sigma()

        K_mm = self.K_mm()
        K_mm_inv = tf.linalg.inv(K_mm)
        Psi = self.Psi()
        phi = self.phi()

        K_xz = self.kernel(self.X, self.inducers)

        mu2 = (tf.tensordot(K_xz @ K_mm_inv, self.m, axes=[1, 0])) ** 2
        sigma2 = (
            self.sigma
            - tf.einsum("ij,ij->i", K_xz @ (K_mm_inv - K_mm_inv @ S @ K_mm_inv), K_xz)
        )

        K_inv_m = tf.tensordot(K_mm_inv, self.m, axes=[1, 0])
        Psi_K_inv_m = tf.tensordot(Psi, K_inv_m, axes=[1, 0])

        E_f2 = tf.tensordot(K_inv_m, Psi_K_inv_m, axes=1)

        Var_f = (
            self.sigma * self.abs_Tau
            - tf.linalg.trace(K_mm_inv @ (Psi - S @ K_mm_inv @ Psi))
        )

        KLdiv = 0.5 * (
            tf.linalg.trace(K_mm_inv @ S)
            + tf.linalg.logdet(K_mm)
            - tf.linalg.logdet(S)
            - self.n_inducers
            + tf.tensordot(self.m, tf.tensordot(K_mm_inv, self.m, axes=[1, 0]), axes=1)
        )

        E_logf2 = tf.math.reduce_sum(
            tf.squeeze(self.Y)
            * (
                -tf.cast(gtilde(-0.5 * mu2 / sigma2), sigma2.dtype)
                + tf.math.log(0.5 * sigma2)
                - tf.cast(euler_mascheroni, sigma2.dtype)
            )
            - tf.math.lgamma(tf.squeeze(self.Y) + 1)
        )

        elbo = (
            -(
                E_f2
                + Var_f
                + 2 * self.beta * tf.tensordot(phi, K_inv_m, axes=1)
                + self.beta ** 2 * self.abs_Tau
            )
            + E_logf2
            - KLdiv
        ) / self.X.shape[0]

        if trace:
            tf.summary.scalar("elbo", elbo)
            tf.summary.scalar("lengthscale", self.l)
            tf.summary.scalar("sigma", self.sigma)
            tf.summary.scalar("beta", self.beta)
            tf.summary.scalar("E_f2", E_f2)
            tf.summary.scalar("Var_f", Var_f)
            tf.summary.scalar("KLdiv", KLdiv)
            tf.summary.scalar("E_logf2", E_logf2)

            tf.summary.image("Psi", Psi[tf.newaxis, :, :, tf.newaxis])
            tf.summary.image("K_mm", K_mm[tf.newaxis, :, :, tf.newaxis])
            tf.summary.image("S", S[tf.newaxis, :, :, tf.newaxis])

            tf.summary.histogram("mu2", mu2)
            tf.summary.histogram("sigma2", sigma2)
            tf.summary.histogram("m", self.m)
            tf.summary.histogram("Phi", phi)
            tf.summary.histogram("Psi_h", Psi)

        return elbo

    def _neg_elbo(self, trace=False):
        negelbo = -self._elbo(trace)
        if trace:
            tf.summary.scalar("neg_elbo", negelbo)
        return negelbo

    def elbo(self):
        return self._elbo()

    def neg_elbo(self):
        return -self.elbo()

    def optimize(self, logdir=None):
        bfgs = gpflow.optimizers.Scipy()
        vars = [self._len, self._sigma, self._beta, self.Sigma, self.m]

        trace = logdir is not None
        if trace:
            writer = tf.summary.create_file_writer(
                os.path.join(logdir, time.strftime("%Y-%m-%d_%H%M%S"))
            )
        else:
            writer = tf.summary.create_noop_writer()

        step = tf.Variable(0, dtype=tf.int64)

        def objective():
            tf.summary.experimental.set_step(step)
            negelbo = self._neg_elbo(trace)
            step.assign_add(1)
            return negelbo

        with writer.as_default():
            bfgs.minimize(
                objective, variables=vars, method="l-bfgs-b", options={"disp": True},
            )

    def predict_mean(self, x):
        S = self._Sigma()

        K_mm = self.K_mm()
        K_mm_inv = tf.linalg.inv(K_mm)

        K_xz = self.kernel(x, self.inducers)

        mu = tf.tensordot(K_xz @ K_mm_inv, self.m, axes=[1, 0])
        sigma2 = (
            self.sigma
            - tf.einsum("ij,ij->i", K_xz @ (K_mm_inv - K_mm_inv @ S @ K_mm_inv), K_xz)
        )

        return mu ** 2 + sigma2 + 2 * self.beta * mu + self.beta ** 2
