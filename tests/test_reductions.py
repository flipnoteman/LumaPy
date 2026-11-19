"""Reductions, scans, sorting, searching, group-by, scatters, checked against NumPy."""

import numpy as np
import pytest

import lumapy as lp

RNG = np.random.default_rng(0)
H = RNG.standard_normal((4, 5, 6)).astype(np.float32)


def close(a, b, rtol=1e-4, atol=1e-5):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


@pytest.mark.parametrize("name", ["sum", "min", "max", "mean"])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, -1, (0, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_reductions(name, axis, keepdims):
    got = getattr(lp.asarray(H), name)(axis=axis, keepdims=keepdims)
    ref = getattr(np, name)(H, axis=axis, keepdims=keepdims)
    assert got.shape == np.shape(ref)
    close(got, ref)


@pytest.mark.parametrize("name", ["var", "std", "median"])
@pytest.mark.parametrize("axis", [None, 0, 2])
def test_single_axis_reductions(name, axis):
    got = getattr(lp, name)(lp.asarray(H), axis=axis)
    close(got, getattr(np, name)(H, axis=axis))


def test_quantile():
    close(lp.asarray(H).quantile(0.3), np.quantile(H, 0.3))
    close(lp.asarray(H).quantile(0.9, axis=1), np.quantile(H, 0.9, axis=1))


def test_argmax_argmin():
    a = lp.asarray(H)
    assert a.argmax().item() == np.argmax(H)
    assert a.argmin().item() == np.argmin(H)
    np.testing.assert_array_equal(a.argmax(axis=1).to_numpy(), np.argmax(H, axis=1))
    np.testing.assert_array_equal(a.argmin(axis=-1).to_numpy(), np.argmin(H, axis=-1))


def test_bool_reductions():
    m = lp.asarray(H) > 0
    assert m.sum().item() == (H > 0).sum()
    assert m.sum().dtype == "uint32"
    np.testing.assert_array_equal(m.any(axis=0).to_numpy(), (H > 0).any(axis=0))
    np.testing.assert_array_equal(m.all(axis=2).to_numpy(), (H > 0).all(axis=2))
    assert lp.count_nonzero(lp.asarray([0, 1, 2, 0])).item() == 2


def test_integer_sums():
    h = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    np.testing.assert_array_equal(lp.asarray(h).sum(axis=1).to_numpy(), h.sum(axis=1))
    close(lp.asarray(h).mean(), h.mean())


def test_multi_reduce():
    a = lp.asarray(H)
    lo, hi, s = a.reduce(["min", "max", "sum"]).to_numpy()
    close([lo, hi, s], [H.min(), H.max(), H.sum()])
    mn, sq = a.reduce_axis(1, ["min", "sumsq"])
    close(mn, H.min(axis=1))
    close(sq, (H * H).sum(axis=1))


def test_norm_and_compensated():
    a = lp.asarray(H)
    close(a.norm(), np.linalg.norm(H.ravel()))
    close(a.sum_compensated(), H.astype(np.float64).sum())


def test_nan_reductions():
    h = H.copy()
    h[0, 0, :3] = np.nan
    a = lp.asarray(h)
    close(lp.nansum(a), np.nansum(h))
    close(lp.nanmin(a), np.nanmin(h))
    close(lp.nanmax(a), np.nanmax(h))
    close(lp.nanmean(a), np.nanmean(h))
    close(a.nansum(axis=2), np.nansum(h, axis=2))


def test_sum_where():
    a = lp.asarray(H)
    close(a.sum_where(a > 0), H[H > 0].sum())


def test_cumsum_and_scan():
    h = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    a = lp.asarray(h)
    close(a.cumsum(), np.cumsum(h))
    close(a.cumsum(axis=0), np.cumsum(h, axis=0))
    close(a.cumsum(axis=1), np.cumsum(h, axis=1))
    close(a.scan(lambda x, y: lp.maximum(x, y), axis=1), np.maximum.accumulate(h, axis=1))


def test_sort():
    h = RNG.standard_normal((3, 50)).astype(np.float32)
    a = lp.asarray(h)
    close(a.sort(), np.sort(h, axis=-1))
    close(a.sort(axis=0), np.sort(h, axis=0))
    close(a.sort(axis=None), np.sort(h, axis=None))
    v = lp.asarray(h[0])
    np.testing.assert_array_equal(v.argsort().to_numpy(), np.argsort(h[0], kind="stable"))
    close(v.sort(descending=True), np.sort(h[0])[::-1])
    vals, idx = v.topk(5)
    close(vals, np.sort(h[0])[::-1][:5])
    close(v.take(idx), vals)


def test_unique_factorize():
    h = np.array([3, 1, 3, 2, 1, 3], np.int32)
    a = lp.asarray(h)
    np.testing.assert_array_equal(a.unique().to_numpy(), np.unique(h))
    uniq, codes = a.factorize()
    np.testing.assert_array_equal(uniq.take(codes).to_numpy(), h)


def test_search():
    edges = np.array([0.0, 1.0, 2.0, 3.0], np.float32)
    x = np.array([-0.5, 0.0, 0.5, 2.0, 3.5], np.float32)
    e = lp.asarray(edges)
    np.testing.assert_array_equal(e.searchsorted(x).to_numpy(), np.searchsorted(edges, x))
    np.testing.assert_array_equal(e.searchsorted(x, side="right").to_numpy(), np.searchsorted(edges, x, side="right"))
    np.testing.assert_array_equal(lp.digitize(x, edges).to_numpy(), np.digitize(x, edges))
    np.testing.assert_array_equal(lp.asarray(x).isin([0.0, 2.0]).to_numpy(), np.isin(x, [0.0, 2.0]))


def test_group_by_readme():
    codes = lp.asarray(np.array([0, 1, 0, 1], np.uint32))
    price = lp.asarray([10.0, 20.0, 30.0, 50.0])
    close(price.group_mean(codes, 2), [20.0, 35.0])
    close(price.group_sum(codes, 2), [40.0, 70.0])
    np.testing.assert_array_equal(price.group_count(codes, 2).to_numpy(), [2, 2])
    close(price.group_max(codes, 2), [30.0, 50.0])
    assert price.median().item() == 25.0


def test_histogram_bincount():
    h = RNG.uniform(0, 1, 10_000).astype(np.float32)
    counts = lp.histogram(h, 10, 0.0, 1.0).to_numpy()
    np.testing.assert_array_equal(counts, np.histogram(h, 10, (0.0, 1.0))[0])
    idx = np.array([0, 2, 2, 5], np.uint32)
    np.testing.assert_array_equal(lp.bincount(idx, 6).to_numpy(), np.bincount(idx, minlength=6))


def test_scatters():
    a = lp.zeros(5)
    a.put_add([0, 2, 2, 4], [1.0, 2.0, 3.0, 4.0])
    close(a, [1, 0, 5, 0, 4])
    a.put_max([0, 1], 10.0)
    close(a, [10, 10, 5, 0, 4])
    a.put([3], [7.0])
    close(a, [10, 10, 5, 7, 4])


def test_filter():
    h = np.array([3.0, -1.0, 4.0, -1.5], np.float32)
    a = lp.asarray(h)
    close(a.filter(a > 0), h[h > 0])
    assert a.filter(a > 100) is None
    vals, count = a.filter_padded(a > 0, -9.0)
    close(vals, [3.0, 4.0, -9.0, -9.0])
    assert count.item() == 2
