"""Errors, evaluation, contexts, profiling, loops and custom kernels."""

import numpy as np
import pytest

import lumapy as lp


def close(a, b, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)


def test_poisoned_arrays_raise_at_readback():
    a = lp.zeros((2, 3))
    b = lp.zeros((4, 5))
    c = a + b  # shape mismatch: recorded as a poisoned array, no exception yet
    assert c.error is not None
    with pytest.raises(lp.LumaError):
        c.to_numpy()
    with pytest.raises(lp.LumaError):
        c.check()
    with pytest.raises(lp.LumaError):
        (c * 2).eval()
    assert issubclass(lp.LumaError, RuntimeError)


def test_eval_and_eval_all():
    a = lp.asarray([1.0, 2.0]) * 3
    assert a.eval() is a
    b = lp.asarray([1, 2]) + 1
    lp.eval_all([a, b])
    lp.flush()
    lp.default_context().sync()
    close(a, [3, 6])


def test_copy_is_independent():
    a = lp.asarray([1.0, 2.0])
    b = a.copy()
    a += 1
    close(b, [1, 2])


def test_write_and_read_into():
    a = lp.zeros(4)
    a.write(np.array([1, 2, 3, 4], np.float32))
    out = np.empty(4, np.float32)
    a.read_into(out)
    np.testing.assert_array_equal(out, [1, 2, 3, 4])


def test_context_options():
    ctx = lp.Context(calibrate=False)
    a = lp.asarray([1.0, 2.0], ctx=ctx)
    assert (a * 2).sum().item() == 6.0
    assert ctx.tuning_profile in {"apple", "nvidia", "generic", "cpu", "unknown"}
    assert isinstance(ctx.features, list)


def test_profile():
    ctx = lp.default_context()
    if not ctx.profiling_supported:
        pytest.skip("no timestamp queries")
    x = lp.uniform(1 << 16)
    result, prof = ctx.profile(lambda: (x * 2 + 1).sqrt().sum().item())
    assert result > 0
    assert prof.launches >= 1
    assert prof.total > 0
    assert prof.kernels[0].label
    assert str(prof)


def test_fusion_limit():
    x = lp.ones(8)
    y = lp.fusion_limit(1, lambda: ((x + 1.0) * 2.0).sqrt())
    close(y, np.full(8, 2.0))


def test_iterate():
    x = lp.asarray([1.0, 2.0, 3.0])
    close(x.iterate(5, lambda v: v * 2), [32, 64, 96])


def test_iterate_until():
    x = lp.asarray([16.0, 64.0])
    r = x.iterate_until(200, 1e-6, lambda v: v * 0.5)
    assert r.to_numpy().max() < 1e-5


def test_iterate_all_mandelbrot():
    c = lp.asarray([0.1, -0.5, 0.3, 0.25])
    zero = lp.zeros(4)

    def step(x, count):
        inside = x <= 2.0
        return [lp.where(inside, x * x + c, x), count + lp.where(inside, 1.0, 0.0)]

    x, count = lp.iterate_all([zero, zero], 64, step)
    # the reference on the host
    xr, cr = np.zeros(4, np.float32), np.zeros(4, np.float32)
    ch = np.array([0.1, -0.5, 0.3, 0.25], np.float32)
    for _ in range(64):
        inside = xr <= 2
        xr = np.where(inside, xr * xr + ch, xr)
        cr = cr + inside
    close(count, cr)


def test_callback_errors_propagate():
    x = lp.asarray([1.0])
    with pytest.raises(ZeroDivisionError):
        x.iterate(3, lambda v: 1 / 0)


KERNEL = """
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@group(1) @binding(2) var<uniform> scale: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= luma.size.x) { return; }
    y[id.x] = x[id.x] * scale + y[id.x];
}
"""


def test_custom_kernel():
    k = lp.Kernel(KERNEL, label="axpy")
    assert [b["kind"] for b in k.bindings] == ["read", "read_write", "uniform"]
    assert k.workgroup == [64, 1, 1]
    x = lp.asarray(np.arange(100, dtype=np.float32))
    y = lp.ones(100)
    alias = y
    k.launch(100, x, y, lp.Uniform(2.0))
    close(alias, np.arange(100) * 2.0 + 1)
    k.launch(100, x, y, 1.0)  # a bare float is an f32 uniform
    close(y, np.arange(100) * 3.0 + 1)


def test_luma_version_is_recorded():
    import re

    assert re.fullmatch(r"\d+\.\d+\.\d+.*", lp.__luma_version__)
    assert lp.__luma_source__ in {"git", "crates.io", "local"}
    if lp.__luma_source__ == "git":
        assert re.fullmatch(r"[0-9a-f]{40}", lp.__luma_commit__)
    lp.show_versions()
