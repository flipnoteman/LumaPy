"""Construction, host transfer, dtypes, operators and elementwise math, checked against NumPy."""

import numpy as np
import pytest

import lumapy as lp


def close(a, b, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


def test_readme_example():
    x = lp.asarray([1.0, 2.0, 3.0, 4.0])
    y = lp.asarray([10.0, 20.0, 30.0, 40.0])
    close(lp.sqrt(x * 2 + y), np.sqrt(np.array([1, 2, 3, 4.0]) * 2 + [10, 20, 30, 40]))
    assert (x * y).sum().item() == 300.0


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 3, 2, 2)])
def test_roundtrip_shapes(shape):
    h = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    a = lp.asarray(h)
    assert a.shape == shape
    assert a.ndim == len(shape)
    assert a.size == h.size
    np.testing.assert_array_equal(a.to_numpy(), h)
    np.testing.assert_array_equal(np.asarray(a), h)


def test_luma_shape_is_reversed():
    a = lp.zeros((4, 3))
    assert a.luma_shape == [3, 4, 1, 1]


@pytest.mark.parametrize(
    "host, dtype",
    [
        (np.array([1.5, 2.5], np.float64), "float32"),
        (np.array([1, 2], np.int64), "int32"),
        (np.array([1, 2], np.uint8), "uint32"),
        (np.array([True, False]), "bool"),
        ([1, 2, 3], "int32"),
        ([1.0, 2.0], "float32"),
    ],
)
def test_dtype_inference(host, dtype):
    assert lp.asarray(host).dtype == dtype


def test_explicit_dtypes():
    h = np.array([1.0, 2.0, 3.0])
    assert lp.asarray(h, dtype="float64").to_numpy().dtype == np.float64
    d = lp.asarray(np.array([1.0 + 1e-10, 2.0]), dtype=lp.df32)
    assert d.dtype == "df32"
    # df32 keeps about 48 bits: 1 + 1e-10 survives, float32 would round it to 1.
    assert abs(d.to_numpy()[0] - (1.0 + 1e-10)) < 1e-13
    assert lp.asarray([1, 2], dtype=np.uint32).dtype == "uint32"
    assert lp.asarray([1, 2], dtype=float).dtype == "float32"


def test_int64_out_of_range_raises():
    with pytest.raises(ValueError):
        lp.asarray(np.array([2**40]))


def test_empty_raises():
    with pytest.raises(ValueError):
        lp.asarray(np.zeros((0,), np.float32))


def test_zero_d():
    s = lp.asarray(np.float32(3.5))
    assert s.shape == ()
    assert s.item() == 3.5
    assert float(s) == 3.5


def test_constructors():
    np.testing.assert_array_equal(lp.zeros((2, 3), dtype="int32").to_numpy(), np.zeros((2, 3), np.int32))
    np.testing.assert_array_equal(lp.ones(4).to_numpy(), np.ones(4, np.float32))
    np.testing.assert_array_equal(lp.full((2, 2), 7).to_numpy(), np.full((2, 2), 7, np.int32))
    np.testing.assert_array_equal(lp.arange(5).to_numpy(), np.arange(5))
    close(lp.arange(1.0, 2.0, 0.25), np.arange(1.0, 2.0, 0.25))
    close(lp.linspace(0, 1, 5), np.linspace(0, 1, 5))
    np.testing.assert_array_equal(
        lp.range((2, 3), axis=1, dtype="uint32").to_numpy(), np.broadcast_to(np.arange(3), (2, 3))
    )
    np.testing.assert_array_equal(
        lp.range((2, 3), axis=0, dtype="uint32").to_numpy(), np.broadcast_to(np.arange(2)[:, None], (2, 3))
    )


def test_random_is_reproducible():
    a = lp.uniform((1000,), seed=7).to_numpy()
    b = lp.uniform((1000,), seed=7).to_numpy()
    np.testing.assert_array_equal(a, b)
    assert 0.0 <= a.min() and a.max() < 1.0
    n = lp.normal(100_000, seed=1).to_numpy()
    assert abs(n.mean()) < 0.02 and abs(n.std() - 1) < 0.02
    assert lp.random_bits(8, seed=3).dtype == "uint32"


H = np.array([[1.0, -2.0, 3.5], [4.0, 5.0, -6.25]], np.float32)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: x + 1,
        lambda x: 1 + x,
        lambda x: x - 2.5,
        lambda x: 2.5 - x,
        lambda x: x * x,
        lambda x: 3 * x,
        lambda x: x / 2,
        lambda x: 2 / x,
        lambda x: -x,
        lambda x: abs(x),
        lambda x: x**2,
        lambda x: x // 2,
        lambda x: x % 1.5,
    ],
)
def test_float_operators(f):
    close(f(lp.asarray(H)), f(H))


def test_broadcasting_numpy_rules():
    row = np.array([10.0, 20.0, 30.0], np.float32)
    col = np.array([[1.0], [2.0]], np.float32)
    close(lp.asarray(H) + lp.asarray(row), H + row)
    close(lp.asarray(H) * lp.asarray(col), H * col)
    close(lp.asarray(col) + lp.asarray(row), col + row)


def test_numpy_on_the_left_defers():
    r = np.ones(3, np.float32) + lp.asarray(np.arange(3, dtype=np.float32))
    assert isinstance(r, lp.Array)
    close(r, np.arange(3) + 1.0)


def test_int_operators_match_python_semantics():
    h = np.array([-7, -3, 0, 3, 7], np.int32)
    a = lp.asarray(h)
    np.testing.assert_array_equal((a // 2).to_numpy(), h // 2)
    np.testing.assert_array_equal((a % 3).to_numpy(), h % 3)
    np.testing.assert_array_equal((a // -2).to_numpy(), h // -2)
    np.testing.assert_array_equal((a % -3).to_numpy(), h % -3)
    np.testing.assert_array_equal(a.rem(3).to_numpy(), np.fmod(h, 3))
    close(a / 2, h / 2)  # true division gives float
    assert (a / 2).dtype == "float32"
    np.testing.assert_array_equal((a**3).to_numpy(), h**3)


def test_bitwise():
    h = np.array([0b1100, 0b1010, 0xFFFF0000], np.uint32)
    a = lp.asarray(h)
    np.testing.assert_array_equal((a & 0b1000).to_numpy(), h & 0b1000)
    np.testing.assert_array_equal((a | a).to_numpy(), h | h)
    np.testing.assert_array_equal((a ^ 0xFF).to_numpy(), h ^ 0xFF)
    np.testing.assert_array_equal((~a).to_numpy(), ~h)
    np.testing.assert_array_equal((a << 2).to_numpy(), h << 2)
    np.testing.assert_array_equal((a >> 1).to_numpy(), h >> 1)
    rot = a.rotate_left(4).to_numpy()
    np.testing.assert_array_equal(rot, ((h << 4) | (h >> 28)).astype(np.uint32))


def test_comparisons_and_bool():
    a = lp.asarray(H)
    m = a > 0
    assert m.dtype == "bool"
    np.testing.assert_array_equal(m.to_numpy(), H > 0)
    np.testing.assert_array_equal((a == a).to_numpy(), np.ones_like(H, bool))
    np.testing.assert_array_equal(((a > 0) & (a < 4)).to_numpy(), (H > 0) & (H < 4))
    np.testing.assert_array_equal((~m).to_numpy(), ~(H > 0))
    # bool joins arithmetic as 0/1
    close(a * m, H * (H > 0))


def test_promotion():
    i = lp.asarray(np.array([1, 2, 3], np.int32))
    f = lp.asarray(np.array([0.5, 0.5, 0.5], np.float32))
    assert (i + f).dtype == "float32"
    assert (i + 0.5).dtype == "float32"
    assert (i + 1).dtype == "int32"
    with pytest.raises(TypeError):
        i + lp.asarray(np.array([1, 2, 3], np.uint32))


def test_in_place_updates_shared_data():
    a = lp.asarray(np.arange(4, dtype=np.float32))
    alias = a
    a += 1
    a *= a
    np.testing.assert_array_equal(alias.to_numpy(), (np.arange(4) + 1.0) ** 2)
    with pytest.raises(TypeError):
        i = lp.asarray([1, 2])
        i /= 2


def test_unary_math():
    h = np.array([0.25, 1.0, 2.0, 9.0], np.float32)
    a = lp.asarray(h)
    for name, ref in [
        ("sqrt", np.sqrt),
        ("exp", np.exp),
        ("log", np.log),
        ("log1p", np.log1p),
        ("expm1", np.expm1),
        ("sin", np.sin),
        ("cos", np.cos),
        ("tan", np.tan),
        ("tanh", np.tanh),
        ("floor", np.floor),
        ("ceil", np.ceil),
        ("trunc", np.trunc),
        ("round", np.round),
        ("sign", np.sign),
    ]:
        close(getattr(lp, name)(a), ref(h), rtol=1e-4)
    close(lp.asarray([1, 4, 9]).sqrt(), [1, 2, 3])  # integers become float32
    close(a.atan2(1.0), np.arctan2(h, 1.0), rtol=1e-4)  # software drivers are less exact
    close(a.hypot(a), np.hypot(h, h))
    close(a.clip(0.5, 2.0), np.clip(h, 0.5, 2.0))
    close(lp.minimum(a, 1.5), np.minimum(h, 1.5))
    close(lp.maximum(a, a * 0.5), h)


def test_nan_classes():
    h = np.array([1.0, np.nan, np.inf, -np.inf], np.float32)
    a = lp.asarray(h)
    np.testing.assert_array_equal(lp.isnan(a).to_numpy(), np.isnan(h))
    np.testing.assert_array_equal(lp.isinf(a).to_numpy(), np.isinf(h))
    np.testing.assert_array_equal(lp.isfinite(a).to_numpy(), np.isfinite(h))
    assert not lp.isnan(lp.asarray([1, 2])).any().item()


def test_astype():
    a = lp.asarray(np.array([-1.7, 0.0, 2.9], np.float32))
    np.testing.assert_array_equal(a.astype("int32").to_numpy(), [-1, 0, 2])
    np.testing.assert_array_equal(a.astype("bool").to_numpy(), [True, False, True])
    np.testing.assert_array_equal(a.astype("float64").to_numpy(), a.to_numpy().astype(np.float64))
    assert a.astype("df32").astype("int32").dtype == "int32"


def test_scalar_uniform():
    x = lp.asarray([1.0, 2.0])
    s = lp.Scalar(2.0)
    a = x * s
    s.set(3.0)
    b = x * s
    close(a, [2, 4])
    close(b, [3, 6])
    assert s.get() == 3.0


def test_df32_and_f64_arithmetic():
    h = np.array([1.0, 2.0, 3.0])
    d = lp.asarray(h, dtype="df32")
    close((d * d + 1).to_numpy(), h * h + 1, rtol=1e-12)
    close(d.sqrt(), np.sqrt(h), rtol=1e-12)
    if "SHADER_F64" in lp.default_context().features:
        f = lp.asarray(h, dtype="float64")
        close((f * 2 - 1).to_numpy(), h * 2 - 1, rtol=1e-15)
        np.testing.assert_array_equal((f < 2).to_numpy(), h < 2)


def test_repr_and_tolist():
    a = lp.asarray([[1, 2], [3, 4]])
    assert a.tolist() == [[1, 2], [3, 4]]
    assert "int32" in repr(a)
    with pytest.raises(ValueError):
        bool(a)
    assert len(a) == 2


def test_reflected_power():
    h = np.array([0.0, 1.0, 2.5], np.float32)
    close(2.0 ** lp.asarray(h), 2.0**h, rtol=1e-5)
