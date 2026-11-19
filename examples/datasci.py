"""A table of data through the array API (the second example of Luma's README)."""

import numpy as np

import lumapy as lp

# Four rows of three features (NumPy order: rows first).
x = lp.asarray(np.array([[1, 2, 0], [2, 1, 1], [0, 0, 3], [4, 2, 1]], np.float32))
gram = x.T @ x  # XᵀX, (3, 3); the transpose is read inside the kernel
variance = x.var(axis=0)  # per feature, in two passes

codes = lp.asarray([0, 1, 0, 1], dtype="uint32")
price = lp.asarray([10.0, 20.0, 30.0, 50.0])
print(price.group_mean(codes, 2).to_numpy())  # [20. 35.]
print(price.median().item())  # 25.0
print(price.sort(descending=True).to_numpy())  # [50. 30. 20. 10.]
print(gram.shape, variance.size)  # (3, 3) 3
