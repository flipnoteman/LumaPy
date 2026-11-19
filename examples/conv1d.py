"""1D convolution (a port of Luma's `examples/conv1d.rs`)."""

import lumapy as lp

# A signal with a spike pattern [1, 3, 1] hidden in flat regions. The kernel matches that exact
# pattern: the convolution peaks wherever the pattern appears.
input_data = [0.0, 0.0, 1.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0]
kernel_data = [1.0, 3.0, 1.0]  # the pattern we're looking for

print(f"Signal: {input_data}")
print(f"Kernel: {kernel_data}")
print()

signal = lp.asarray(input_data)
kernel = lp.asarray(kernel_data)

result = signal.conv1d(kernel, stride=1, padding=0).to_numpy()

print(f"Conv1D output ({len(result)} elements):")
print(result)
print()

# Find the peak: that's where the pattern matched.
peak = result.max()
print(f"Peak value: {peak}")
print("Match locations (output index -> input index):")
for i, v in enumerate(result):
    if v == peak:
        print(f"  output[{i}] = {v} <- pattern found at input[{i}..{i + 3}]")
