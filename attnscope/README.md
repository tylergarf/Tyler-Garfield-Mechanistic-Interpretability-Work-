# `attnscope`

The probe behind this experiment: a hand-written Llama forward pass that exposes every attention head,
a generation loop that records attention and the residual stream at every step and layer, and the
code that turns those recordings into the comparison figures.

The `# Original:` comments in this folder quote the code I wrote when I first built this experiment in
summer 2026. I then worked with Claude to refactor it into the cleaner version here.
[`scripts/check_against_original.py`](../scripts/check_against_original.py) runs the original and
refactored code side by side and confirms they produce identical outputs. The full original code is
preserved in [`original/tylerscope_summer2026.py`](../original/tylerscope_summer2026.py).

## Files

| File | Needs | What it does | Original it replaces |
|---|---|---|---|
| [`forward.py`](forward.py) | MLX | `layer_forward` runs one decoder block by hand and returns its attention; `embed` / `unembed` go from text to the residual stream and back to vocabulary logits | `TylerScope.manually_run_foward`, `tokenize_input`, the next-token line of `gen_from_prompt` |
| [`scope.py`](scope.py) | MLX | `generate_and_capture` greedily generates while recording one head's attention window and the last token's hidden state; `end_window`, `focus_window` and `most_active_head` are the attention "cameras" | `TylerScope.gen_from_prompt`, `static_cam_at_end_conv`, `dynamic_camera_atten_slice` |
| [`plotting.py`](plotting.py) | numpy | loads `data/`, tiles the 100 attention windows into a grid, draws the per-layer comparison figure and GIF, plus the single-pass head grids | cells 1–5 of `final_lamma_vis.ipynb` (Part 2 of the original file) |
| [`analysis.py`](analysis.py) | numpy | `dominant_dims`: which hidden dimension lights up in each layer | new |

`forward.py` and `scope.py` need MLX (Apple Silicon). `plotting.py` and `analysis.py` need only numpy and
matplotlib, so the figures can be rebuilt on any machine. `__init__.py` deliberately imports nothing, so
`import attnscope.plotting` works without MLX installed.

## Reading the comments

Each file opens with a table mapping its functions to the original ones and listing the variable renames
(for example `casual_att` → `attn`, `G_K` → `group`). Inside each function, comments explain what every
line or small block does, give array shapes for Llama-3.2-3B where they help, and quote the original
line(s) it replaces:

```python
# Grouped-query attention: each KV head is shared by ``group`` query heads.
# ...
# Original (4 lines, same result):
#           K = K.reshape(batch_len,seq_len,K_num_heads,head_dim)          (B, S, 8, 128)
#           K = mx.expand_dims(K,axis=-2)                                  (B, S, 8, 1, 128)
#           ...
k = mx.repeat(k.reshape(B, S, n_kv, head_dim), group, axis=2).transpose(0, 2, 1, 3)
```

## Verifying the refactor

From the repo root on Apple Silicon:

```bash
python scripts/check_against_original.py           # pooling, attention windows, head choice (seconds)
python scripts/check_against_original.py --model   # + all 28 layers and 3 generation steps (loads the model)
```

The script holds verbatim copies of the original functions (with the original comments) and requires
exact equality, not "close enough":

* `end_window`, `focus_window` and `most_active_head` against the original cameras, on random causal
  attention maps of several sizes (including ones small enough to hit every edge case) and on real
  windows from `data/cos.npz`;
* `layer_forward` against `manually_run_foward` on all 28 layers, comparing attention and residual
  stream;
* `generate_and_capture` against the original generation loop for 3 tokens, comparing the generated
  text, attention windows and hidden states.

## What wasn't carried over

The live Qt viewer (`TensorVisualizer` and the `time.sleep(5)` calls in the generation loop),
`construct_atten_plots`, the `track_option == 1` branch of the dynamic camera, and the trackers that were
never saved (second-to-last token's hidden state, the full attention tensors). All of it is still in
[`original/tylerscope_summer2026.py`](../original/tylerscope_summer2026.py).
