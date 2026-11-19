"""Expressions, select and reductions (a port of Luma's `examples/basics.rs`)."""

import time

import lumapy as lp

t = time.perf_counter()

a = lp.asarray([1, 2, 3], dtype="uint32")
b = lp.asarray([10, 20, 30], dtype="uint32")

# One expression, all GPU-resident, no CPU copies in between:
#
#   a = [1, 2, 3]
#   b = [10, 20, 30]
#
#   (a + b)          -> [11, 22, 33]
#   * a              -> [11, 44, 99]
#   + 1              -> [12, 45, 100]
#   * 2              -> [24, 90, 200]
#   - b              -> [14, 70, 170]
#   // 2             -> [7, 35, 85]
#
# Every step is recorded into the context's batch; to_numpy() submits it and waits.
result = ((((a + b) * a + 1) * 2 - b) // 2).to_numpy()
print(f"Chain result: {result}")

# Float math, comparisons, selection and reductions work the same way.
x = lp.asarray([-2.0, -0.5, 0.0, 1.5, 4.0])
relu = lp.where(x > 0, x, lp.zeros(5))
print(f"relu(x)   = {relu.to_numpy()}")
print(f"sqrt(|x|) = {abs(x).sqrt().to_numpy()}")
print(
    "sum, min, max, mean, norm = "
    f"{x.sum().item()}, {x.min().item()}, {x.max().item()}, {x.mean().item()}, {x.norm().item()}"
)

print(f"Program Time: {time.perf_counter() - t:.3f} s")
