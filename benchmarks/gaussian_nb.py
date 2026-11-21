"""Gaussian naive Bayes on the GPU with lumapy, against CPU implementations of the same model.

    python benchmarks/gaussian_nb.py                     # the default sizes
    python benchmarks/gaussian_nb.py --quick             # small sizes, a smoke run
    python benchmarks/gaussian_nb.py --sizes 1000000x64 --classes 20 --repeat 9
    python benchmarks/gaussian_nb.py --threads 1         # the CPU baselines on one core
    python benchmarks/gaussian_nb.py --json results.json

Every implementation fits the same model as scikit-learn's ``GaussianNB``: class priors from the class
frequencies, per-class feature means and two-pass variances, ``var_smoothing`` times the largest feature
variance added to every variance, and the joint log-likelihood

    log P(c) - 1/2 sum_j log(2 pi var_cj) - 1/2 sum_j (x_j - mean_cj)^2 / var_cj

whose argmax over the classes is the prediction.

To separate what the GPU contributes from what a better formulation contributes, the benchmark runs:

* **scikit-learn ``GaussianNB``** as shipped: float64 arithmetic, a Python loop over the classes (one
  pass over the data per class), mostly single-threaded NumPy.
* **NumPy, matmul form**, float64 and float32: the formulation the GPU version uses, on the CPU. Fit
  and predict are matrix products against a one-hot class matrix, which NumPy runs on a multithreaded
  BLAS (all cores unless ``--threads`` says otherwise); the rest is NumPy's (single-threaded)
  elementwise work.

      fit:     counts = 1ᵀY, means = YᵀX / counts, var = Yᵀ(X - means[y])² / counts
      predict: argmax((X²) A + X B + c), A = -1/(2 var)ᵀ, B = (mean / var)ᵀ, c per-class constants

* **Numba, fused loops** (float32 data, float64 sums; needs ``numba``): the same model as compiled,
  parallel loops over all cores, one fused pass per phase. The strongest CPU baseline here.

* **LumaPy** (float32), the same formulation on the GPU, once from NumPy arrays (uploads included,
  the comparable case when the data starts on the host) and once with the data already on the GPU.

Speedups are quoted against the **fastest CPU implementation** for that phase, at the same float32
precision where possible. Each measurement reports the median and interquartile range of ``--repeat``
runs after ``--warmup`` seconds of warm-up runs, and the first call separately (for LumaPy it includes
compiling the kernels for these shapes, unless Luma's on-disk pipeline cache already has them). Every
implementation's test predictions are compared with scikit-learn's.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field

import numpy as np

import lumapy as lp

try:
    from sklearn.datasets import make_classification
    from sklearn.naive_bayes import GaussianNB
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:  # pragma: no cover - the benchmark needs scikit-learn
    sys.exit("this benchmark needs scikit-learn: pip install scikit-learn (or pip install --group bench)")
try:
    import numba
except ImportError:  # the Numba baseline is optional
    numba = None


class LumaGaussianNB:
    """Gaussian naive Bayes with the arithmetic of ``sklearn.naive_bayes.GaussianNB``, on the GPU.

    ``fit`` and ``predict`` take NumPy arrays or ``lumapy.Array``s; ``predict`` returns a NumPy array
    of the class labels. Computation is in float32.
    """

    def __init__(self, var_smoothing: float = 1e-9, ctx: lp.Context | None = None) -> None:
        self.var_smoothing = var_smoothing
        self.ctx = ctx or lp.default_context()

    def fit(self, X: np.ndarray | lp.Array, y: np.ndarray | lp.Array) -> LumaGaussianNB:
        X = lp.asarray(X, dtype="float32", ctx=self.ctx)
        y = lp.asarray(y, ctx=self.ctx)
        n, d = X.shape
        # The distinct labels (sorted) and every sample's index into them; reads the class count back.
        self.classes_, codes = y.factorize()
        k = self.classes_.size

        # One-hot (n, k) class matrix, computed inside the kernels that read it.
        onehot = (codes.reshape(n, 1) == lp.range((1, k), axis=1, dtype="uint32")).astype("float32")
        counts = onehot.sum(axis=0).reshape(k, 1)  # (k, 1)
        theta = (onehot.T @ X) / counts  # (k, d) means
        dev = X - onehot @ theta  # (n, d) deviations from the own class mean
        var = (onehot.T @ (dev * dev)) / counts  # (k, d) population variances
        # The variance of every feature over all samples from the class statistics (law of total variance),
        # without another pass over X; scikit-learn smooths with var_smoothing times the largest of them.
        mean = (counts * theta).sum(axis=0) / n
        total_var = (counts * (var + (theta - mean) ** 2)).sum(axis=0) / n
        self.epsilon_ = total_var.max() * self.var_smoothing
        var = var + self.epsilon_

        self.theta_, self.var_ = theta, var
        self.class_prior_ = counts.reshape(k) / n
        inv = 1.0 / var
        self._a = (inv * -0.5).T  # (d, k), multiplies x²
        self._b = (theta * inv).T  # (d, k), multiplies x
        self._c = lp.log(self.class_prior_) - 0.5 * (
            lp.log(var * (2 * math.pi)).sum(axis=1) + (theta * theta * inv).sum(axis=1)
        )  # (k,)
        lp.eval_all([self._a, self._b, self._c, self.theta_, self.var_])
        return self

    def joint_log_likelihood(self, X: np.ndarray | lp.Array) -> lp.Array:
        """``log P(c) + log P(x | c)`` for every sample and class, shape (n, k); lazy."""
        X = lp.asarray(X, dtype="float32", ctx=self.ctx)
        return (X * X) @ self._a + X @ self._b + self._c

    def predict(self, X: np.ndarray | lp.Array) -> np.ndarray:
        best = self.joint_log_likelihood(X).argmax(axis=1)
        return self.classes_.take(best).to_numpy()


def _total_variance(counts: np.ndarray, theta: np.ndarray, var: np.ndarray, n: int) -> np.ndarray:
    """Every feature's variance over all samples, from the per-class counts (k, 1), means and variances."""
    mean = (counts * theta).sum(axis=0) / n
    return (counts * (var + (theta - mean) ** 2)).sum(axis=0) / n


class NumpyGaussianNB:
    """The formulation of ``LumaGaussianNB`` in NumPy, on the CPU, in ``dtype`` (the CPU baseline).

    The matrix products run on NumPy's BLAS, multithreaded; the class means are gathered by index
    (``theta[codes]``) instead of by a product, as is cheaper on a CPU.
    """

    def __init__(self, dtype: type[np.floating] = np.float32, var_smoothing: float = 1e-9) -> None:
        self.dtype = dtype
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> NumpyGaussianNB:
        X = np.asarray(X, dtype=self.dtype)
        n, _ = X.shape
        self.classes_, codes = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        onehot = np.zeros((n, k), dtype=self.dtype)
        onehot[np.arange(n), codes] = 1
        counts = onehot.sum(axis=0)[:, None]
        theta = (onehot.T @ X) / counts
        dev = X - theta[codes]
        var = (onehot.T @ (dev * dev)) / counts
        var += self.var_smoothing * _total_variance(counts, theta, var, n).max()

        self.theta_, self.var_ = theta, var
        self.class_prior_ = counts[:, 0] / n
        inv = 1.0 / var
        self._a = (inv * -0.5).T
        self._b = (theta * inv).T
        self._c = np.log(self.class_prior_) - 0.5 * (
            np.log(var * (2 * math.pi)).sum(axis=1) + (theta * theta * inv).sum(axis=1)
        )
        return self

    def joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.dtype)
        return (X * X) @ self._a + X @ self._b + self._c

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]


if numba is not None:

    @numba.njit(parallel=True, cache=True)
    def _numba_class_stats(X, codes, k, chunks):  # type: ignore[no-untyped-def]
        """Per-class counts, means and two-pass variances; each thread accumulates a chunk of rows."""
        n, d = X.shape
        step = (n + chunks - 1) // chunks
        sums = np.zeros((chunks, k, d))
        cnts = np.zeros((chunks, k))
        for t in numba.prange(chunks):
            for i in range(t * step, min(n, (t + 1) * step)):
                c = codes[i]
                cnts[t, c] += 1
                for j in range(d):
                    sums[t, c, j] += X[i, j]
        counts = np.zeros(k)
        theta = np.zeros((k, d))
        for t in range(chunks):
            for c in range(k):
                counts[c] += cnts[t, c]
                for j in range(d):
                    theta[c, j] += sums[t, c, j]
        for c in range(k):
            for j in range(d):
                theta[c, j] /= counts[c]
        sq = np.zeros((chunks, k, d))
        for t in numba.prange(chunks):
            for i in range(t * step, min(n, (t + 1) * step)):
                c = codes[i]
                for j in range(d):
                    delta = X[i, j] - theta[c, j]
                    sq[t, c, j] += delta * delta
        var = np.zeros((k, d))
        for t in range(chunks):
            for c in range(k):
                for j in range(d):
                    var[c, j] += sq[t, c, j]
        for c in range(k):
            for j in range(d):
                var[c, j] /= counts[c]
        return counts, theta, var

    @numba.njit(parallel=True, cache=True)
    def _numba_predict(X, theta, inv, const):  # type: ignore[no-untyped-def]
        """The class of the largest joint log-likelihood, one row per iteration, rows in parallel."""
        n, d = X.shape
        k = theta.shape[0]
        out = np.empty(n, np.int64)
        for i in numba.prange(n):
            best = -np.inf
            arg = 0
            for c in range(k):
                s = const[c]
                for j in range(d):
                    delta = X[i, j] - theta[c, j]
                    s -= 0.5 * delta * delta * inv[c, j]
                if s > best:
                    best = s
                    arg = c
            out[i] = arg
        return out


class NumbaGaussianNB:
    """The same model as fused, multithreaded loops compiled by Numba: the strongest CPU baseline here.

    The data stays float32; sums are accumulated in float64 (free on a CPU). Fit is two parallel passes
    over the rows; predict computes ``log P(c) - 1/2 sum (x - mean)^2 / var`` per row and class without
    materializing the (n, k) log-likelihoods.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> NumbaGaussianNB:
        X = np.ascontiguousarray(X, dtype=np.float32)
        n = X.shape[0]
        self.classes_, codes = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        counts, theta, var = _numba_class_stats(X, codes, k, 4 * numba.get_num_threads())
        var += self.var_smoothing * _total_variance(counts[:, None], theta, var, n).max()
        self.theta_, self.var_ = theta, var
        self.class_prior_ = counts / n
        self._theta32 = theta.astype(np.float32)
        self._inv32 = (1.0 / var).astype(np.float32)
        self._const = np.log(self.class_prior_) - 0.5 * np.log(var * (2 * math.pi)).sum(axis=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self.classes_[_numba_predict(X, self._theta32, self._inv32, self._const)]


# -- measuring ---------------------------------------------------------------------------------------

#: Seconds each measurement is warmed up for (``--warmup``).
WARMUP = 1.0


@dataclass
class Timing:
    """Seconds: the first call, and the median and quartiles of the timed runs after the warm-up."""

    first: float
    median: float
    q1: float
    q3: float
    runs: list[float] = field(repr=False)


def measure(f: Callable[[], object], repeat: int) -> Timing:
    # The first call is reported on its own (it compiles kernels, fills caches and buffer pools). Then
    # warm up for a while, not for one call: after the CPU-only implementations ran, the GPU needs time
    # at work to clock back up (a single warm-up call left the following runs 10x slow).
    t = time.perf_counter()
    f()
    first = time.perf_counter() - t
    start = time.perf_counter()
    while time.perf_counter() - start < WARMUP:
        f()
    runs = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        runs.append(time.perf_counter() - t)
    quartiles = statistics.quantiles(runs, n=4) if len(runs) > 1 else [runs[0]] * 3
    return Timing(first, statistics.median(runs), quartiles[0], quartiles[2], runs)


@dataclass
class Row:
    name: str
    device: str
    dtype: str
    fit: Timing
    predict: Timing
    agreement: float  # fraction of test labels equal to scikit-learn's
    accuracy: float


@dataclass
class Result:
    n_train: int
    n_test: int
    features: int
    classes: int
    rows: list[Row]


def run(n: int, d: int, k: int, repeat: int, seed: int) -> Result:
    n_test = max(n // 4, 1)
    X_all, y_all = make_classification(
        n_samples=n + n_test,
        n_features=d,
        n_informative=max(2, d // 2),
        n_redundant=0,
        n_classes=k,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=seed,
    )
    # One float32 dataset for everyone: the float64 implementations convert it themselves, inside the
    # timed fit, as they would any float32 input.
    X_all = X_all.astype(np.float32)
    y_all = y_all.astype(np.int32)
    X, X_test, y, y_test = X_all[:n], X_all[n:], y_all[:n], y_all[n:]
    ctx = lp.default_context()
    rows: list[Row] = []

    def add(name: str, device: str, dtype: str, fit: Callable[[], object], predict: Callable[[], np.ndarray]) -> None:
        fit_t = measure(fit, repeat)
        predict_t = measure(predict, repeat)
        labels = predict()
        rows.append(Row(name, device, dtype, fit_t, predict_t, 0.0, float(np.mean(labels == y_test))))
        rows[-1].agreement = float(np.mean(labels == reference))

    sk = GaussianNB().fit(X, y)
    reference = sk.predict(X_test)
    add("scikit-learn GaussianNB", "cpu", "float64", lambda: sk.fit(X, y), lambda: sk.predict(X_test))

    for dtype in (np.float64, np.float32):
        model = NumpyGaussianNB(dtype).fit(X, y)
        add(
            "NumPy, matmul form",
            "cpu",
            np.dtype(dtype).name,
            lambda m=model: m.fit(X, y),
            lambda m=model: m.predict(X_test),
        )

    if numba is not None:
        nb_model = NumbaGaussianNB().fit(X, y)
        add("Numba, fused loops", "cpu", "float32", lambda: nb_model.fit(X, y), lambda: nb_model.predict(X_test))

    host = LumaGaussianNB()

    def host_fit() -> None:
        host.fit(X, y)
        ctx.sync()

    add("LumaPy, from NumPy arrays", "gpu", "float32", host_fit, lambda: host.predict(X_test))

    gX, gy, gX_test = lp.asarray(X), lp.asarray(y), lp.asarray(X_test)
    lp.eval_all([gX, gX_test])
    gy.eval()
    ctx.sync()
    resident = LumaGaussianNB()

    def resident_fit() -> None:
        resident.fit(gX, gy)
        ctx.sync()

    add("LumaPy, data on the GPU", "gpu", "float32", resident_fit, lambda: resident.predict(gX_test))
    return Result(n, n_test, d, k, rows)


# -- reporting ---------------------------------------------------------------------------------------


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def fmt_timing(t: Timing) -> str:
    spread = (t.q3 - t.q1) / t.median * 50 if t.median else 0.0  # half the IQR, in % of the median
    return f"{fmt_time(t.median):>9} +/-{spread:2.0f}%"


def print_result(r: Result) -> None:
    print(f"\n{r.n_train:,} train x {r.features} features, {r.classes} classes, {r.n_test:,} test samples")
    head = (
        f"  {'implementation':<27} {'dtype':<8} {'fit':>16} {'predict':>16} "
        f"{'1st fit':>10} {'1st pred':>10} {'same labels':>11} {'accuracy':>9}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for row in r.rows:
        print(
            f"  {row.name:<27} {row.dtype:<8} {fmt_timing(row.fit):>16} {fmt_timing(row.predict):>16} "
            f"{fmt_time(row.fit.first):>10} {fmt_time(row.predict.first):>10} "
            f"{row.agreement:>11.2%} {row.accuracy:>9.1%}"
        )
    cpu = [row for row in r.rows if row.device == "cpu"]
    best_fit = min(cpu, key=lambda row: row.fit.median)
    best_pred = min(cpu, key=lambda row: row.predict.median)
    print(f"  fastest CPU: fit {best_fit.name} ({best_fit.dtype}), predict {best_pred.name} ({best_pred.dtype})")
    for row in r.rows:
        if row.device == "gpu":
            print(
                f"  speedup of {row.name} over the fastest CPU: "
                f"fit {best_fit.fit.median / row.fit.median:.1f}x, "
                f"predict {best_pred.predict.median / row.predict.median:.1f}x"
            )


def parse_size(text: str) -> tuple[int, int]:
    n, d = text.lower().split("x")
    return int(n), int(d)


@contextlib.contextmanager
def cpu_threads(threads: int | None) -> Iterator[None]:
    if threads is None:
        yield
    else:
        if numba is not None:
            numba.set_num_threads(threads)
        with threadpool_limits(limits=threads):
            yield


def blas_description() -> str:
    np.ones((64, 64)) @ np.ones((64, 64))  # loads the BLAS so that it is listed
    libs = [f"{i['internal_api']} {i.get('version', '')} ({i['num_threads']} threads)" for i in threadpool_info()]
    return ", ".join(libs) or "unknown"


def main() -> None:
    global WARMUP
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=parse_size,
        default=[(10_000, 16), (100_000, 32), (1_000_000, 32), (200_000, 256)],
        help="training sizes as SAMPLESxFEATURES (the test set is a quarter of it)",
    )
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=9, help="timed runs per measurement (median reported)")
    parser.add_argument("--warmup", type=float, default=WARMUP, help="seconds of warm-up runs per measurement")
    parser.add_argument("--threads", type=int, help="limit the BLAS threads of the CPU implementations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="small sizes and few repeats, for a smoke run")
    parser.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    args = parser.parse_args()
    if args.quick:
        args.sizes, args.repeat, args.warmup = [(2_000, 8), (20_000, 16)], 3, 0.2
    WARMUP = args.warmup

    import sklearn

    print("Gaussian naive Bayes: LumaPy (GPU) vs CPU implementations of the same model")
    lp.show_versions()
    with cpu_threads(args.threads):
        blas = blas_description()
        numba_info = f", numba {numba.__version__} ({numba.get_num_threads()} threads)" if numba else ", no numba"
        print(f"sklearn {sklearn.__version__}, BLAS {blas}{numba_info}")
        print(f"median +/- half the interquartile range of {args.repeat} runs after {args.warmup:g} s of warm-up")
        results = [run(n, d, args.classes, args.repeat, args.seed) for n, d in args.sizes]
    for r in results:
        print_result(r)
    print(
        "\n'same labels' compares each implementation's test predictions with scikit-learn's; '1st' is the "
        "first call\n(for LumaPy: compiling the kernels of these shapes, unless Luma's pipeline cache has them)."
    )
    if args.json:
        meta = {
            "python": platform.python_version(),
            "lumapy": lp.__version__,
            "luma": lp.__luma_version__,
            "backend": lp.default_context().backend,
            "device": lp.default_context().device_name,
            "sklearn": sklearn.__version__,
            "blas": blas,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": [asdict(r) for r in results]}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
