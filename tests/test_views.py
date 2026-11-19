"""Views, indexing, assignment, joins, matmul and linear solves, checked against NumPy."""

import numpy as np
import pytest

import lumapy as lp

H = np.arange(24, dtype=np.float32).reshape(2, 3, 4)


def close(a, b, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "key",
    [
        0,
        -1,
        (1, 2),
        (0, slice(1, 3)),
        (slice(None), 1),
        (Ellipsis, 2),
        (Ellipsis, slice(1, None)),
        (slice(None, None, -1),),
        (0, slice(None, None, -1), slice(1, 3)),
        (None, 0),
        (0, None, 1),
        (slice(2, 0, -1),),
    ],
)
def test_getitem_matches_numpy(key):
    got = lp.asarray(H)[key]
    ref = H[key]
    assert got.shape == ref.shape
    close(got, ref)


def test_mask_and_take_indexing():
    a = lp.asarray(H)
    close(a[a > 10], H[H > 10])
    v = lp.asarray(np.arange(10, dtype=np.float32))
    close(v[[1, 3, 5]], [1, 3, 5])


def test_setitem():
    h = H.copy()
    a = lp.asarray(h)
    a[0, 1:3] = 100.0
    h[0, 1:3] = 100.0
    a[1, :, 0] = lp.asarray([7.0, 8.0, 9.0])
    h[1, :, 0] = [7.0, 8.0, 9.0]
    a[a > 20] = -1.0
    h[h > 20] = -1.0
    close(a, h)
    a[...] = 0.0
    close(a, np.zeros_like(h))


def test_setitem_1d_index_array():
    a = lp.zeros(5)
    a[[1, 3]] = 2.0
    close(a, [0, 2, 0, 2, 0])


def test_reshape_transpose():
    a = lp.asarray(H)
    close(a.reshape(6, 4), H.reshape(6, 4))
    close(a.reshape((4, -1)), H.reshape(4, -1))
    close(a.T, H.T)
    close(a.transpose(1, 0, 2), H.transpose(1, 0, 2))
    close(a.transpose((2, 0, 1)), H.transpose(2, 0, 1))
    close(a.swapaxes(0, 2), H.swapaxes(0, 2))
    close(a.ravel(), H.ravel())
    close(a.expand_dims(1), np.expand_dims(H, 1))
    close(lp.asarray(np.ones((1, 3, 1))).squeeze(), np.ones(3))


def test_flip_pad_roll_broadcast():
    a = lp.asarray(H)
    close(a.flip(), np.flip(H))
    close(a.flip(1), np.flip(H, 1))
    m = H[0]
    b = lp.asarray(m)
    for mode, np_mode in [("constant", "constant"), ("edge", "edge"), ("wrap", "wrap"), ("symmetric", "symmetric")]:
        close(b.pad(((1, 2), (2, 1)), mode), np.pad(m, ((1, 2), (2, 1)), np_mode))
    close(b.pad(1), np.pad(m, 1))
    close(b.roll(1, axis=1), np.roll(m, 1, axis=1))
    close(b.roll((1, -2)), np.roll(m, (1, -2), axis=(0, 1)))
    close(lp.asarray([1.0, 2.0, 3.0, 4.0]).shift_fill(1, 9.0), [9, 1, 2, 3])
    close(lp.asarray([1.0, 2.0, 3.0, 4.0]).shift_clamp(-1), [2, 3, 4, 4])
    close(lp.asarray([1.0, 2.0, 3.0]).broadcast_to((2, 3)), np.broadcast_to([1, 2, 3], (2, 3)))


def test_index_axis():
    a = lp.zeros((2, 3))
    np.testing.assert_array_equal(a.index(0).to_numpy(), np.indices((2, 3))[0])
    np.testing.assert_array_equal(a.index(1).to_numpy(), np.indices((2, 3))[1])


def test_concatenate_stack():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    y = x + 10
    close(lp.concatenate([x, y], axis=0), np.concatenate([x, y], axis=0))
    close(lp.concatenate([x, y], axis=1), np.concatenate([x, y], axis=1))
    close(lp.stack([x, y]), np.stack([x, y]))
    close(lp.stack([x, y], axis=1), np.stack([x, y], axis=1))
    close(lp.stack([x, y], axis=-1), np.stack([x, y], axis=-1))


def test_where():
    h = np.array([-1.0, 2.0, -3.0, 4.0], np.float32)
    a = lp.asarray(h)
    close(lp.where(a > 0, a, 0.0), np.where(h > 0, h, 0.0))
    close(lp.where(a > 0, 1.0, -1.0), np.where(h > 0, 1.0, -1.0))
    close(lp.where(a > 0, a, a * 10), np.where(h > 0, h, h * 10))


def test_matmul_numpy_semantics():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 3)).astype(np.float32)
    B = rng.standard_normal((3, 4)).astype(np.float32)
    v = rng.standard_normal(3).astype(np.float32)
    w = rng.standard_normal(5).astype(np.float32)
    a, b = lp.asarray(A), lp.asarray(B)
    close(a @ b, A @ B, rtol=1e-4)
    close(a @ v, A @ v, rtol=1e-4)
    close(w @ a, w @ A, rtol=1e-4)
    close(lp.dot(v, v), v @ v, rtol=1e-4)
    close(a.T @ a, A.T @ A, rtol=1e-4)  # README: the Gram matrix, transpose read inside the kernel
    assert (a @ b).shape == (5, 4)
    assert (a @ v).shape == (5,)


def test_solve_tridiagonal():
    n = 16
    rng = np.random.default_rng(2)
    lo = rng.uniform(-1, 0, n).astype(np.float32)
    up = rng.uniform(-1, 0, n).astype(np.float32)
    diag = (4 + rng.uniform(0, 1, n)).astype(np.float32)
    rhs = rng.standard_normal(n).astype(np.float32)
    M = np.diag(diag) + np.diag(lo[1:], -1) + np.diag(up[:-1], 1)
    x = lp.solve_tridiagonal(lo, diag, up, rhs)
    close(x, np.linalg.solve(M.astype(np.float64), rhs), rtol=1e-4, atol=1e-5)


def test_views_are_lazy_reads_of_the_source():
    a = lp.asarray(np.arange(6, dtype=np.float32))
    s = a[1:4]
    a[...] = 0.0
    # Recorded before the write: keeps the old values (Luma's semantics).
    close(s, [1, 2, 3])
