from pathlib import Path
import numpy as np
from gguf import GGUFReader

path = Path("models/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf")
reader = GGUFReader(str(path))

print(f"Model file: {path}")
print(f"Tensor count: {len(reader.tensors)}")
print("\nTensor checks:")

checked = 0
for tensor in reader.tensors:
    name = tensor.name
    if any(key in name for key in ("ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v")):
        data = tensor.data
        if data is None:
            print(f"{name}: data unavailable")
            continue

        values = np.asarray(data)
        total = values.size
        nonzero = int(np.count_nonzero(values))
        print(f"{name}: elements={total}, nonzero={nonzero}, dtype={values.dtype}")
        checked += 1

print(f"\nSelected tensors checked: {checked}")
