""" Main underlying functions for SpatialDE functionality.
"""
import sys
import logging
from time import time
import warnings
from typing import Optional, Dict, Tuple

import numpy as np
from scipy import stats
from scipy.special import logsumexp
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

from tqdm.auto import tqdm

from .kernels import SquaredExponential, Cosine, Linear
from .models import Model, Constant, Null, model_factory
from .util import bh_adjust, Kernel, GP, SGPIPM, GPControl
from .score_test import (
    ScoreTest,
    GaussianConstantScoreTest,
    GaussianNullScoreTest,
    NegativeBinomialScoreTest,
)
from .gpflow_helpers import *

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def get_l_limits(X):
    Xsq = np.sum(np.square(X), 1)
    R2 = -2.0 * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 0, np.inf)
    R_vals = np.unique(R2.flatten())
    R_vals = R_vals[R_vals > 1e-8]

    l_min = np.sqrt(R_vals.min()) * 2.0
    l_max = np.sqrt(R_vals.max()) * 2.0

    return l_min, l_max


def inducers_grid(X, ninducers):
    rngmin = X.min(0)
    rngmax = X.max(0)
    xvals = np.linspace(rngmin[0], rngmax[0], int(np.ceil(np.sqrt(ninducers))))
    yvals = np.linspace(rngmin[1], rngmax[1], int(np.ceil(np.sqrt(ninducers))))
    xx, xy = np.meshgrid(xvals, yvals)
    return np.hstack((xx.reshape((xx.size, 1)), xy.reshape((xy.size, 1))))


def fit_model(model: Model, exp_tab: pd.DataFrame, raw_counts: pd.DataFrame):
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with model:
            for i, gene in enumerate(tqdm(exp_tab.columns)):
                y = exp_tab.iloc[:, i].to_numpy()
                rawy = raw_counts.iloc[:, i].to_numpy()
                model.sety(y, rawy)
                t0 = time()

                res = model.optimize()
                t = time() - t0
                res = {
                    "gene": gene,
                    "max_ll": model.log_marginal_likelihood,
                    "max_delta": model.delta,
                    "max_mu_hat": model.mu,
                    "max_s2_t_hat": model.sigma_s2,
                    "max_s2_e_hat": model.sigma_n2,
                    "time": t,
                    "n": model.n,
                    "FSV": model.FSV,
                    "s2_FSV": model.s2_FSV,
                    "s2_logdelta": model.s2_logdelta,
                    "converged": res.success,
                    "M": model.n_parameters,
                }
                for (k, v) in vars(model.kernel).items():
                    if k not in res:
                        res[k] = v

                results.append(res)
    return pd.DataFrame(results)


def factory(kern: str, X: np.ndarray, lengthscale: Optional[float] = None):
    Z = None
    if X.shape[0] > 1000:
        Z = inducers_grid(X, np.maximum(100, np.sqrt(X.shape[0])))

    if kern == "linear":
        return model_factory(X, Z, Linear())
    elif kern == "SE":
        return model_factory(X, Z, SquaredExponential(lengthscale))
    elif kern == "PER":
        return model_factory(X, Z, Cosine(lengthscale))
    elif kern == "const":
        return Constant(X)
    elif kern == "null":
        return Null(X)
    else:
        raise ValueError("unknown kernel")


def kspace_walk(kernel_space: dict, X: np.ndarray):
    for kern, lengthscales in kernel_space.items():
        try:
            for l in lengthscales:
                yield factory(kern, X, l), kern
        except TypeError:
            yield factory(kern, X, lengthscales), kern


def score_test_fast_models(
    results: pd.DataFrame,
    exp_tab: pd.DataFrame,
    raw_counts: pd.DataFrame,
    tests: Dict[Tuple[str, float], ScoreTest],
    testskey: str,
):
    with tqdm(total=results.shape[0]) as pbar:

        def test(df):
            results = []
            with tests[df.name] as test:
                for gene in df.gene:
                    t0 = time()
                    test.model.sety(exp_tab[gene].to_numpy(), raw_counts[gene].to_numpy())
                    stest = test()
                    t = time() - t0
                    res = {
                        "gene": gene,
                        "test_time": t,
                    }
                    res.update(stest.to_dict())
                    results.append(res)
                    pbar.update()
            return pd.DataFrame(results)

        testresults = results.groupby(testskey, sort=False).apply(test)
        results = pd.concat((results.set_index("gene"), testresults.set_index("gene")), axis=1)
        results.time += results.test_time
        results.index.name = "gene"  # FIXME: https://github.com/pandas-dev/pandas/issues/21629
        return results.drop(columns="test_time").reset_index()


def score_test_detailed_models(results: pd.DataFrame, test: ScoreTest, modelkey):
    testresults = []

    with tqdm(total=results.shape[0]) as pbar:
        for row in results.itertuples():
            test.model = getattr(row, modelkey)
            with test:
                t0 = time()
                stest = test()
                t = time() - t0
                res = {
                    "gene": row.gene,
                    "test_time": t,
                }
                res.update(stest.to_dict())
                testresults.append(res)
                pbar.update()
        testresults = pd.DataFrame(testresults)
        results = pd.concat((results.set_index("gene"), testresults.set_index("gene")), axis=1)
        results.time += results.test_time
        results.index.name = "gene"  # FIXME: https://github.com/pandas-dev/pandas/issues/21629
        return results.drop(columns="test_time").reset_index()


def run_gpflow(
    X: pd.DataFrame,
    exp_tab: pd.DataFrame,
    raw_counts: Optional[pd.DataFrame] = None,
    control: Optional[GPControl] = GPControl(),
    rng: np.random.Generator = np.random.default_rng(),
):
    if control.gp is None:
        if X.shape[0] < 750:
            control.gp = GP.GPR
        else:
            control.gp = GP.SGPR

    results = DataSetResults()
    X = tf.constant(X.to_numpy(), dtype=gpflow.config.default_float())
    colnames = exp_tab.columns.to_numpy()
    t = tqdm(colnames)
    opt = gpflow.optimizers.Scipy()

    logging.info("Fitting gene models")
    if control.gp == GP.GPR:
        for g, gene in enumerate(t):
            t.set_description(gene, refresh=False)
            model = GPR(
                X,
                Y=tf.constant(
                    exp_tab.iloc[:, g].to_numpy()[:, np.newaxis],
                    dtype=gpflow.config.default_float(),
                ),
                rawY=tf.constant(raw_counts.iloc[:, g].to_numpy()[:, np.newaxis]) if raw_counts is not None else None,
                n_kernel_components=control.ncomponents,
                ard=control.ard,
            )
            results[gene] = GeneGP(model, opt.minimize, method="bfgs")
    elif control.gp == GP.SGPR:
        ninducers = (
            np.ceil(np.sqrt(X.shape[0])).astype(np.int32)
            if control.ninducers is None
            else control.ninducers
        )
        if control.ipm == SGPIPM.free or control.ipm == SGPIPM.random:
            inducers = X.iloc[rng.integers(0, X.shape[0], ninducers), :].to_numpy()
        elif control.ipm == SGPIPM.grid:
            rngmin = tf.reduce_min(X, axis=0)
            rngmax = tf.reduce_max(X, axis=0)
            xvals = tf.linspace(rngmin[0], rngmax[0], int(np.ceil(np.sqrt(ninducers))))
            yvals = tf.linspace(rngmin[1], rngmax[1], int(np.ceil(np.sqrt(ninducers))))
            xx, xy = tf.meshgrid(xvals, yvals)
            inducers = tf.stack((tf.reshape(xx, (-1,)), tf.reshape(xy, (-1,))), axis=1)
            inducers = gpflow.inducing_variables.InducingPoints(inducers)
        if control.ipm != SGPIPM.free:
            gpflow.utilities.set_trainable(inducers, False)

        method = "BFGS"
        if control.ipm == SGPIPM.free and ninducers > 1e3:
            method = "L-BFGS-B"

        for g, gene in enumerate(t):
            t.set_description(gene, refresh=False)
            model = SGPR(
                X,
                Y=tf.constant(
                    exp_tab.iloc[:, g].to_numpy()[:, np.newaxis],
                    dtype=gpflow.config.default_float(),
                ),
                rawY=tf.constant(raw_counts.iloc[:, g].to_numpy()[:, np.newaxis]) if raw_counts is not None else None,
                inducing_variable=inducers,
                n_kernel_components=control.ncomponents,
                ard=control.ard,
            )
            results[gene] = GeneGP(model, opt.minimize, method=method)

    logging.info("Finished fitting models to %i genes" % len(colnames))
    return results


def run(
    X: pd.DataFrame,
    exp_tab: pd.DataFrame,
    raw_counts: pd.DataFrame,
    score_test: str = "nb",
    kernel_space: Optional[dict] = None,
) -> pd.DataFrame:
    """ Perform SpatialDE test

    X : matrix of spatial coordinates times observations
    exp_tab : Expression table, assumed appropriatealy normalised.
    raw_counts : Unnormalized expression table

    The grid of covariance matrices to search over for the alternative
    model can be specifiec using the kernel_space parameter.
    """
    if kernel_space == None:
        l_min, l_max = get_l_limits(X)
        kernel_space = {
            "SE": np.logspace(np.log10(l_min), np.log10(l_max), 10),
            #'PER': np.logspace(np.log10(l_min), np.log10(l_max), 10),
            #'linear': None
        }

    logging.info("Performing DE test")
    results = []

    if score_test == "nb":
        stest_class = NegativeBinomialScoreTest
    else:
        stest_class = GaussianConstantScoreTest

    logging.info("Fitting gene models")
    n_models = 0
    stests = {}
    for model, mname in kspace_walk(kernel_space, X.to_numpy()):
        res = fit_model(model, exp_tab, raw_counts)
        stests[model] = stest_class(X.to_numpy(), exp_tab.to_numpy(), raw_counts.to_numpy(), model)
        res["model"] = mname
        res["_model"] = model
        results.append(res)
        n_models += 1

    n_genes = exp_tab.shape[1]
    logging.info("Finished fitting {} models to {} genes".format(n_models, n_genes))

    results = pd.concat(results, sort=True).reset_index(drop=True)
    sizes = (
        results.groupby(["model", "gene"], sort=False).size().groupby("model", sort=False).unique()
    )
    results = results.set_index("model")
    results.loc[sizes > 1, "M"] += 1
    results = results.reset_index()
    results["BIC"] = -2 * results["max_ll"] + results["M"] * np.log(results["n"])

    results = results.loc[results.groupby(["model", "gene"], sort=False)["max_ll"].idxmax()]
    results = results.loc[results.groupby("gene", sort=False)["BIC"].idxmin()]

    logging.info("Performing score test")
    results = score_test_fast_models(results, exp_tab, raw_counts, stests, "_model")
    results["p.adj"] = bh_adjust(results["pval"].to_numpy())

    return results.drop(columns="_model").reset_index(drop=True)


def run_detailed(
    X: pd.DataFrame,
    exp_tab: pd.DataFrame,
    raw_counts: pd.DataFrame,
    score_test: str = "nb",
    control: GPControl = GPControl(),
    rng: np.random.Generator = np.random.default_rng(),
):
    logging.info("Fitting gene models")
    res = run_gpflow(X, exp_tab, raw_counts, control, rng)
    logging.info("Finished fitting models to {} genes".format(X.shape[0]))
    results = res.to_df(modelcol="model")

    if score_test == "nb":
        stest_class = NegativeBinomialScoreTest
    else:
        stest_class = GaussianConstantScoreTest

    stest = stest_class(X.to_numpy(), exp_tab.to_numpy(), raw_counts.to_numpy())
    results = score_test_detailed_models(results, stest, "model")
    results["p.adj"] = bh_adjust(results["pval"].to_numpy())

    return results.reset_index(drop=True)


def fit_mixture_kernel(
    X: pd.DataFrame,
    exp_tab: pd.DataFrame,
    control: GPControl = GPControl(),
    rng: np.random.Generator = np.random.default_rng(),
):
    return run_gpflow(X, exp_tab, control=control, rng=rng)
