"""Running your own WGSL kernels on lumapy arrays (a port of Luma's `examples/custom_kernel.rs`).

Luma binds a `luma` uniform in `@group(0)` (`luma.size`, the problem size passed to `launch`)
and provides `luma_idx2(i, j)` / `luma_idx3(i, j, k)` for indexing (x fastest). Your bindings
go in `@group(1)`, numbered from 0 in argument order:

- an `Array` bound `var<storage, read>` is read;
- an `Array` bound `var<storage, read_write>` is written in place (every handle on it sees it);
- `lumapy.Uniform(value, dtype)` (or a bare float / int) is a `var<uniform>` scalar.

Arguments are checked against the shader at launch. Launches are recorded in the same batch
as built-in operations, so they can be freely mixed.
"""

import lumapy as lp

# 1. A single-entry kernel with uniforms: y = alpha * x + y. (Luma's Rust example passes one
#    uniform struct; lumapy passes scalar uniforms, one binding each.)
SAXPY = """
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@group(1) @binding(2) var<uniform> alpha: f32;
@group(1) @binding(3) var<uniform> n: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= n) { return; }
    y[gid.x] = alpha * x[gid.x] + y[gid.x];
}
"""

# 2. One module with several entry points. Each entry point's arguments are only the globals it
#    uses, and those must be numbered contiguously from 0: `fill_index` takes (dst) and
#    `transpose` takes (dst, src).
GRID = """
@group(1) @binding(0) var<storage, read_write> dst: array<f32>;
@group(1) @binding(1) var<storage, read> src: array<f32>;

// Writes each element's flat index, on a 2D grid.
@compute @workgroup_size(8, 8)
fn fill_index(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x || gid.y >= luma.size.y) { return; }
    let i = luma_idx2(gid.x, gid.y);
    dst[i] = f32(i);
}

// dst (h x w) = transpose of src (w x h); luma.size is the shape of src.
@compute @workgroup_size(8, 8)
fn transpose(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x || gid.y >= luma.size.y) { return; }
    dst[gid.y + gid.x * luma.size.y] = src[luma_idx2(gid.x, gid.y)];
}
"""

# --- saxpy ---------------------------------------------------------------------------------------
x = lp.asarray([1.0, 2.0, 3.0, 4.0])
y = lp.full(4, 10.0)

saxpy = lp.Kernel(SAXPY)
saxpy.launch(4, x, y, lp.Uniform(2.0), lp.Uniform(4, "uint32"))

# Operators and launches share one ordered batch; nothing has run on the GPU yet.
y = y * 0.5
print(f"saxpy:     {y.to_numpy()}")  # [6. 7. 8. 9.]

# --- several entry points, 2D launches -----------------------------------------------------------
fill = lp.Kernel(GRID, entry="fill_index")
transpose = lp.Kernel(GRID, entry="transpose", label="transpose")

w, h = 4, 3
# Kernels index the data in Luma's order (x fastest), so a w x h grid is a NumPy (h, w) array.
a = lp.zeros((h, w))
t = lp.zeros((w, h))

fill.launch((w, h), a)
transpose.launch((w, h), t, a)

print(f"a ({w}x{h}):\n{a.to_numpy()}")
print(f"t ({h}x{w}):\n{t.to_numpy()}")  # the NumPy transpose of a

# Built-in reductions work on the output like on any other array.
print(f"sum of t:  {t.sum().item()}")  # 0 + 1 + ... + 11 = 66

# Mistakes in the arguments are reported as errors.
try:
    transpose.launch((w, h), a, a, a)
except lp.LumaError as err:
    print(f"bad args:  {err}")
