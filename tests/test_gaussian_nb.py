"""The GPU Gaussian naive Bayes of the benchmark computes what scikit-learn's does."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.datasets import make_classification  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402

PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "gaussian_nb.py"
spec = importlib.util.spec_from_file_location("gaussian_nb", PATH)
gaussian_nb = importlib.util.module_from_spec(spec)
sys.modules["gaussian_nb"] = gaussian_nb  # dataclasses look the module up
spec.loader.exec_module(gaussian_nb)


def implementations():
    yield "lumapy", lambda: gaussian_nb.LumaGaussianNB()
    yield "numpy-float64", lambda: gaussian_nb.NumpyGaussianNB(np.float64)
    yield "numpy-float32", lambda: gaussian_nb.NumpyGaussianNB(np.float32)
    if gaussian_nb.numba is not None:
        yield "numba", lambda: gaussian_nb.NumbaGaussianNB()


def host(a):
    return a.to_numpy() if hasattr(a, "to_numpy") else np.asarray(a)


@pytest.mark.parametrize("name, make", list(implementations()), ids=[n for n, _ in implementations()])
@pytest.mark.parametrize("n, d, k", [(500, 4, 2), (5_000, 20, 7)])
def test_matches_sklearn(name, make, n, d, k):
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=d // 2, n_redundant=0, n_classes=k, random_state=1
    )
    X = X.astype(np.float32)
    labels = np.array([3, 10, 42, 7, 1, 99, 5])[:k][y]  # non-contiguous labels map back
    sk = GaussianNB().fit(X, labels)
    model = make().fit(X, labels)

    np.testing.assert_array_equal(host(model.classes_), sk.classes_)
    np.testing.assert_allclose(host(model.class_prior_), sk.class_prior_, rtol=1e-6)
    np.testing.assert_allclose(host(model.theta_), sk.theta_, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(host(model.var_), sk.var_, rtol=1e-4, atol=1e-6)
    if hasattr(model, "joint_log_likelihood"):
        np.testing.assert_allclose(
            host(model.joint_log_likelihood(X)), sk._joint_log_likelihood(X), rtol=1e-4, atol=1e-3
        )
    assert np.mean(model.predict(X) == sk.predict(X)) > 0.999
