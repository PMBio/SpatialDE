from importlib import resources
import pickle

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

euler_mascheroni = tf.constant(0.57721566490153286060651209008240243104215933593992, dtype=tf.float64)

__gtilde_pickle_fn = 'gtilde.pkl'
__gtilde_csv_fn = 'gtilde.csv'

class lininterp1d:
    def __init__(self, x, y, fill_value=np.nan):
        tf.debugging.Assert(tf.debugging.is_numeric_tensor(x), x)
        tf.debugging.Assert(tf.debugging.is_numeric_tensor(y), y)
        tf.debugging.assert_rank_in(x, [1,2])
        tf.debugging.assert_rank_in(y, [1,2])
        tf.debugging.assert_equal(tf.size(x), tf.size(y))

        if tf.rank(x).numpy() > 1:
            tf.debugging.assert_equal(tf.reduce_min(tf.shape(x)), 1)
        if tf.rank(y).numpy() > 1:
            tf.debugging.assert_equal(tf.reduce_min(tf.shape(y)), 1)

        x = tf.reshape(x, [-1])
        idx = tf.argsort(x)
        self.x = tf.gather(x, idx)
        self.y = tf.gather(tf.reshape(y, [-1]), idx)
        if not self.x.dtype.is_floating:
            self.x = tf.cast(self.x, tf.float32)
        if not self.y.dtype.is_floating:
            self.y = tf.cast(self.y, tf.float32)

        if isinstance(fill_value, tuple):
            if len(fill_value) != 2:
                self.fill_value = (fill_value[0], fill_value[0])
            else:
                self.fill_value = fill_value
        else:
            self.fill_value = (fill_value, fill_value)

    def __call__(self, xnew):
        tf.debugging.Assert(tf.debugging.is_numeric_tensor(xnew), [xnew])
        tf.debugging.assert_rank_in(xnew, [1,2])
        tf.debugging.Assert((tf.rank(xnew) <= 1) | (tf.reduce_min(tf.shape(xnew)) == 1), [xnew])
        xnew = tf.cast(tf.reshape(xnew ,[-1]), self.x.dtype)
        idx = tf.searchsorted(self.x, xnew)
        lidx = tf.where((idx > 0) & (idx < tf.size(self.x)), idx - 1, 0)
        ridx = tf.where((idx > 0) & (idx < tf.size(self.x) - 1), idx, tf.size(self.x) - 1)
        alpha = (xnew - tf.gather(self.x, lidx)) / (tf.gather(self.x, ridx) - tf.gather(self.x, lidx))

        return tf.where((idx == 0) & (xnew < self.x[0]),
                        self.fill_value[0],
                        tf.where(idx == 0,
                                 self.x[0],
                                 tf.where((idx == tf.size(self.x)) & (xnew > self.x[-1]),
                                          self.fill_value[1],
                                          tf.where(idx == tf.size(self.x),
                                                   self.y[-1],
                                                   (1 - alpha) * tf.gather(self.y, lidx) + alpha * tf.gather(self.y, ridx)))))


with resources.open_binary(__package__, __gtilde_pickle_fn) as pkl:
    _gtilde_table = tf.constant(pickle.load(pkl))

_gtilde_neglogz, _gtilde_value, _grad_gtilde_value = _gtilde_table
assert not tf.math.is_inf(tf.math.reduce_min(_gtilde_neglogz))
_gtilde_neglogz_0, _gtilde_value_0, _grad_gtilde_value_0 = -np.inf, tf.constant(0.0), tf.constant(2.)
_gtilde_neglogz_range = (tf.math.reduce_min(_gtilde_neglogz),tf.math.reduce_max(_gtilde_neglogz))
imin = tf.math.argmin(_gtilde_neglogz)
tf.debugging.assert_equal(imin, tf.cast(0, imin.dtype))
tf.debugging.assert_near(_gtilde_value_0, _gtilde_value[imin])
tf.debugging.assert_near(_grad_gtilde_value_0, _grad_gtilde_value[imin])
_gtilde_interp = lininterp1d(_gtilde_neglogz, _gtilde_value, fill_value=(_gtilde_value_0, np.nan))
_grad_gtilde_interp = lininterp1d(_gtilde_neglogz, _grad_gtilde_value, fill_value=(_grad_gtilde_value_0, np.nan))


def grad_gtilde(z):
    """get the value of grad of gtilde at -z by intersection"""
    tf.debugging.assert_less_equal(z, tf.cast(0., z.dtype))
    lognegz = tf.cast(tf.math.log(-z), _gtilde_neglogz.dtype)
    tf.debugging.assert_less_equal(lognegz, _gtilde_neglogz_range[1], message=str((tf.math.reduce_min(lognegz), tf.math.reduce_max(lognegz), _gtilde_neglogz_range)))
    rval = _grad_gtilde_interp(lognegz)
    tf.debugging.assert_all_finite(rval, message=str((tf.math.reduce_min(z), tf.math.reduce_max(z), tf.math.reduce_min(lognegz), tf.math.reduce_max(lognegz))))
    return rval

@tf.custom_gradient
def gtilde(z):
    """get the value of gtilde at -z by intersection"""
    tf.debugging.Assert(tf.debugging.is_numeric_tensor(z), [z])
    tf.debugging.assert_less_equal(z, tf.cast(0., z.dtype))
    lognegz = tf.cast(tf.math.log(-z), _gtilde_neglogz.dtype)
    tf.debugging.assert_less_equal(lognegz, _gtilde_neglogz_range[1], message=str((tf.math.reduce_min(lognegz), tf.math.reduce_max(lognegz), _gtilde_neglogz_range)))
    rval = _gtilde_interp(lognegz)
    tf.debugging.assert_all_finite(rval, message=str(rval))

    def grad(dy):
        return dy * tf.reshape(tf.cast(grad_gtilde(z), dy.dtype), z.shape)
    return tf.reshape(tf.cast(rval, z.dtype), z.shape), grad
