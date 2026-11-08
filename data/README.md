# Captured arrays

One compressed file per prompt: `cos.npz`, `dict.npz`, `control.npz`, written by
`notebooks/02_generate_and_capture.ipynb`.

| key | shape | dtype | meaning |
|---|---|---|---|
| `attn_window` | (100, 28, 100, 100) | float16 | head 17's attention, 3×3 average-pooled, bottom-right 100×100 block, per generation step and layer |
| `last_hidden` | (100, 28, 3072) | float16 | last token's residual stream after the final RMSNorm, per generation step and layer |
| `text` | () | str | prompt plus the generated continuation |
| `new_token_ids` | (100,) | int | the token chosen at each generation step |

Generated with `mlx-community/Llama-3.2-3B-Instruct`, greedy decoding, 100 new tokens. Generation is
deterministic, so re-running notebook 02 reproduces these files bit for bit.

Arrays are computed in float32 and stored as float16 to keep each file under GitHub's 100 MB limit.
`attnscope.plotting.load_slices` converts them back to float32 on load:

```python
from attnscope.plotting import load_slices
cos = load_slices("data", "cos")   # {"attn_window": ..., "last_hidden": ...} as float32
```
