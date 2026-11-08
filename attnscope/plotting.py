"""Figures for the COS / DICT / CONTROL comparison (numpy + matplotlib only).

Nothing in this module needs MLX, so the final figures can be rebuilt on any
machine from the arrays in ``data/``.
"""

# =============================================================================
# HOW THIS FILE MAPS TO THE ORIGINAL SUMMER CODE
#
# These functions are NOT in TylerScope. They come from the plotting notebooks:
#
#   this file                   original
#   --------------------------  -------------------------------------------------
#   load_slices(...)            final_lamma_vis.ipynb cell 1: mx.load(".../cos_atten_slice.npy") etc.
#   softmax(...)                mx.softmax(...) used in final_lamma_vis.ipynb cells 2 and 4
#   global_ranges(...)          final_lamma_vis.ipynb cell 2: atten_glob_vmin/vmax, mlp_glob_vmin/vmax
#   tile_steps(...)             final_lamma_vis.ipynb cell 4: the canvas_cos / canvas_dict /
#                               canvas_control loop that tiles the 100 steps into a 10 x 10 grid
#   attention_cmap()            final_lamma_vis.ipynb cell 4: sns.cubehelix_palette(...) + set_bad
#   layer_comparison_figure()   final_lamma_vis.ipynb cell 4: the body of the per-layer loop
#                               (subplot_mosaic layout, the six imshow panels, colorbars)
#   save_all_layers(...)        final_lamma_vis.ipynb cell 4: the `for layer_num in range(num_layers)`
#                               loop + plt.savefig(file_path)
#   make_gif(...)               final_lamma_vis.ipynb cell 5: PIL padding + imageio.mimsave
#   head_grid(...)              llama_vis.ipynb: the "Max Attention Heads COS/DICT" grids
#   specialization_map(...)     llama_vis.ipynb / llama_temp_vis.ipynb: the "Average
#                               Specialization" map, mean |max_heads - median(max_heads)|
#
# Names, original -> here:
#   cos_atten / dict_atten / control_atten   -> slices["cos"]["attn_window"] etc.
#   cos_last / dict_last / control_last      -> slices["cos"]["last_hidden"] etc.
#   atten_glob_vmin / atten_glob_vmax        -> a_min / a_max
#   mlp_glob_vmin / mlp_glob_vmax            -> h_min / h_max
#   "COS MLP SOFTMAX" panel title            -> "COS: last-token hidden state, softmax over d_model"
#     (renamed because the stored vector is the final-normed residual stream,
#      not the MLP output, and the softmax is over hidden dims, not vocabulary)
# =============================================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import seaborn as sns

# The three prompts, in the order they appear in the figures, and their panel labels.
PROMPTS = ("cos", "dict", "control")
LABELS = {"cos": "COS", "dict": "DICT", "control": "CONTROL"}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_slices(data_dir: str | Path, name: str) -> dict[str, np.ndarray]:
    """Load ``attn_window`` and ``last_hidden`` for one prompt as float32.

    Accepts either the compressed ``<name>.npz`` shipped in the repo or the
    original ``<name>_atten_slice.npy`` / ``<name>_lastmlp_slice.npy`` pair.
    """
    # Original (final_lamma_vis.ipynb cell 1):
    #   cos_atten = mx.load(f"{atten_mlxs_pth}/cos_atten_slice.npy")
    #   cos_last  = mx.load(f"{atten_mlxs_pth}/cos_lastmlp_slice.npy")    (and the same for dict, control)
    # Returns {"attn_window": (100, 28, 100, 100), "last_hidden": (100, 28, 3072)}.
    data_dir = Path(data_dir)
    npz = data_dir / f"{name}.npz"
    if npz.exists():
        # Repo format: one compressed file per prompt, stored as float16.
        # Cast back to float32 so all maths downstream is in float32 like the originals.
        with np.load(npz) as z:
            return {k: z[k].astype(np.float32) for k in ("attn_window", "last_hidden")}
    # Fallback: the original two-file .npy format from the summer run.
    return {
        "attn_window": np.load(data_dir / f"{name}_atten_slice.npy").astype(np.float32),
        "last_hidden": np.load(data_dir / f"{name}_lastmlp_slice.npy").astype(np.float32),
    }


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # numpy version of mx.softmax, which the original used on cos_last etc.
    # Subtract the max first so exp() can't overflow; it doesn't change the result
    # because softmax(x) == softmax(x - c) for any constant c.
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    # Normalise so the values along `axis` sum to 1.
    return e / e.sum(axis=axis, keepdims=True)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def tile_steps(windows: np.ndarray, border: int = 1) -> ma.MaskedArray:
    """Tile ``(steps, h, w)`` maps row-major into a square grid.

    Each tile gets a ``border``-pixel frame; the frame is masked so it can be
    drawn in a neutral colour (``cmap.set_bad``).
    """
    # Original (final_lamma_vis.ipynb cell 4), the canvas loop:
    #   rows = cols = int(np.ceil(np.sqrt(seq_len)))
    #   canvas_cos = mx.full((cam_height_with_pad*rows, cam_width_with_pad*cols), vals=border_val, ...)
    #   for idx in range(seq_len):
    #       row_offset = (idx // rows); col_offset = (idx % cols)
    #       row_ind = (cam_height_with_pad * row_offset)+1; col_ind = (cam_width_with_pad * col_offset)+1
    #       canvas_cos[row_ind:(row_ind+cam_height), col_ind:(col_ind+cam_width)] = cos_atten[idx,layer_num,:,:]
    #   canvas_masked_cos = ma.masked_where(canvas_cos == -1.0, canvas_cos)
    # Here: windows is one layer's (100 steps, 100, 100); the caller slices the layer.
    # One difference in method, not result: the original filled the border with
    # -1.0 and masked == -1; this fills with NaN and masks NaN (can't collide with
    # a real attention value).

    # steps = 100, h = w = 100
    steps, h, w = windows.shape
    # Grid side: ceil(sqrt(100)) = 10, so a 10 x 10 grid of tiles.
    # Original: rows = cols = int(np.ceil(np.sqrt(seq_len)))
    n = int(np.ceil(np.sqrt(steps)))
    # Tile size including the frame: 100 + 2*1 = 102.
    # Original: cam_height_with_pad = cam_height + 2
    th, tw = h + 2 * border, w + 2 * border
    # Blank canvas, 1020 x 1020, all NaN (everything not overwritten is border).
    canvas = np.full((th * n, tw * n), np.nan, dtype=np.float32)
    for idx in range(steps):
        # Step index -> grid position, row-major: step 0 top-left, 9 top-right,
        # 10 starts the second row, ..., 99 bottom-right.
        # Original: row_offset = (idx // rows); col_offset = (idx % cols)
        r, c = divmod(idx, n)
        # Paste this step's 100 x 100 window inside its tile, 1 pixel in from the
        # tile's top-left corner (the +border), leaving the frame as NaN.
        # Original: canvas_cos[row_ind:(row_ind+cam_height), col_ind:(col_ind+cam_width)] = cos_atten[idx,layer_num,:,:]
        canvas[r * th + border : r * th + border + h, c * tw + border : c * tw + border + w] = windows[idx]
    # Mark the NaN frame as "bad" so imshow draws it in cmap's bad colour (light gray).
    # Original: ma.masked_where(canvas_cos == -1.0, canvas_cos)
    return ma.masked_invalid(canvas)


def attention_cmap():
    # Colour map for the attention panels, same settings as the original.
    # Original: cmap = sns.cubehelix_palette(start=0.3, hue=0.9, rot=0.8, as_cmap=True)
    cmap = sns.cubehelix_palette(start=0.3, hue=0.9, rot=0.8, as_cmap=True)
    # Masked (border) pixels drawn light gray.
    # Original: cmap.set_bad(color='lightgray')
    cmap.set_bad(color="lightgray")
    return cmap


# --------------------------------------------------------------------------
# main comparison figure
# --------------------------------------------------------------------------
def global_ranges(slices: dict[str, dict[str, np.ndarray]], step: int = -1):
    """Shared colour ranges so the three prompts are directly comparable.

    Attention: min/max over every step, layer and pixel of all three prompts.
    Hidden-state panel: min/max of the softmax at generation ``step`` over all
    layers of all three prompts.
    """
    # Original (final_lamma_vis.ipynb cell 2):
    #   atten_glob_vmin = min([mx.min(cos_atten,axis=(-4,-3,-2,-1)), mx.min(dict_atten,...), mx.min(control_atten,...)])
    #   atten_glob_vmax = max([... mx.max ...])
    # Smallest / largest attention value anywhere in the three prompts' arrays.
    attn_min = min(float(s["attn_window"].min()) for s in slices.values())
    attn_max = max(float(s["attn_window"].max()) for s in slices.values())
    # Original: mlp_glob_vmin = min([mx.min(mx.softmax(cos_last[-1,:,:],axis=-1),axis=(-2,-1)), ...])
    #           mlp_glob_vmax = max([... mx.max ...])
    # last_hidden[step] is (28 layers, 3072) at the final generation step (step=-1);
    # softmax over the 3072 dims for every layer, then the global min/max.
    probs = [softmax(s["last_hidden"][step]) for s in slices.values()]
    hid_min = min(float(p.min()) for p in probs)
    hid_max = max(float(p.max()) for p in probs)
    return (attn_min, attn_max), (hid_min, hid_max)


def layer_comparison_figure(
    slices: dict[str, dict[str, np.ndarray]],
    layer: int,
    head: int = 17,
    step: int = -1,
    ranges=None,
    hidden_grid: tuple[int, int] = (24, 128),
):
    """One layer: tiled attention windows for every generation step of the
    three prompts (top), and the last token's hidden state at generation
    ``step`` for each prompt (bottom).

    Bottom panels: the final-RMSNormed residual stream of the last token
    (d_model = 3072, reshaped to 24 x 128), passed through a softmax across
    the hidden dimensions. This is *not* a vocabulary distribution: it
    highlights which residual dimensions dominate in magnitude.
    """
    # Original: the body of `for layer_num in range(num_layers):` in final_lamma_vis.ipynb cell 4.
    # `head` only goes into the titles; the data was already captured for head 17.

    # Use the shared colour ranges if given (save_all_layers computes them once),
    # otherwise compute them here.
    (a_min, a_max), (h_min, h_max) = ranges or global_ranges(slices, step)
    # Number of generation steps (100); only used in titles.
    # Original: seq_len,num_layers,cam_height,cam_width = cos_atten.shape
    steps = next(iter(slices.values()))["attn_window"].shape[0]

    # Panel layout, same as the original. Each letter is one panel; '.' is empty.
    #   A, B = COS and DICT attention grids (top row)
    #   C    = CONTROL attention grid (centred below)
    #   D, E, F = COS, DICT, CONTROL hidden-state heatmaps (full width)
    #   c, m = colour bars for attention and hidden-state panels
    layout = """
    .AAAABBBB..c
    ...CCCC....c
    DDDDDDDDDD.m
    EEEEEEEEEE.m
    FFFFFFFFFF.m
    """
    # Original: fig, ax_dict = plt.subplot_mosaic(layout, figsize=(12, 18), layout="constrained",
    #               gridspec_kw={"width_ratios": [1,1,1,1,1,1,1,1,1,1,0.2,0.3], "height_ratios": [4,4,1.5,1.5,1.5]})
    # ([1] * 10 + [0.2, 0.3] is that same width list: 10 plot columns, a spacer, a colour-bar column.)
    fig, ax = plt.subplot_mosaic(
        layout,
        figsize=(12, 18),
        layout="constrained",
        gridspec_kw={
            "width_ratios": [1] * 10 + [0.2, 0.3],
            "height_ratios": [4, 4, 1.5, 1.5, 1.5],
        },
    )

    # Colour maps. Original: cmap = sns.cubehelix_palette(...); cmap.set_bad('lightgray')
    #                        sns.color_palette("mako", as_cmap=True) for the MLP panels
    cmap = attention_cmap()
    hid_cmap = sns.color_palette("mako", as_cmap=True)
    attn_im = hid_im = None
    # One pass per prompt: ("cos", "A", "D"), ("dict", "B", "E"), ("control", "C", "F").
    # The original wrote the three blocks out separately (COS, then DICT, then CONTROL).
    for name, a_key, h_key in zip(PROMPTS, "ABC", "DEF"):
        s = slices[name]
        # Top panel: this layer's 100 attention windows tiled 10 x 10, with the
        # GLOBAL vmin/vmax so colours mean the same thing in all three prompts.
        # s["attn_window"][:, layer] is (100 steps, 100, 100).
        # Original: atten_vis_cos = ax_dict["A"].imshow(canvas_masked_cos, cmap=cmap, interpolation='nearest',
        #                                              vmin=atten_glob_vmin, vmax=atten_glob_vmax)
        attn_im = ax[a_key].imshow(
            tile_steps(s["attn_window"][:, layer]), cmap=cmap, interpolation="nearest", vmin=a_min, vmax=a_max
        )
        # Original: ax_dict["A"].set_xticks([]); ax_dict["A"].set_yticks([])
        ax[a_key].set_xticks([])
        ax[a_key].set_yticks([])
        # Original title: "COS ATTEN CONV 1:1"
        ax[a_key].set_title(f"{LABELS[name]}: head {head} attention, steps 0-{steps - 1}")

        # Bottom panel: last token's normed hidden state at generation `step`
        # (default -1 = the final, 100th step) for this layer: (3072,)
        # -> softmax over the 3072 dims -> reshape to 24 x 128 so it fits a heatmap.
        # Original: mlp_cos_curr = mx.softmax(cos_last[-1,layer_num,:])
        #           mlp_cos_curr.reshape(24,128)
        probs = softmax(s["last_hidden"][step, layer]).reshape(hidden_grid)
        # Original: mlp_vs_cos = ax_dict["D"].imshow(mlp_cos_curr.reshape(24,128), cmap=sns.color_palette("mako",as_cmap=True),
        #                                            interpolation='nearest', vmin=mlp_glob_vmin, vmax=mlp_glob_vmax)
        hid_im = ax[h_key].imshow(probs, cmap=hid_cmap, interpolation="nearest", vmin=h_min, vmax=h_max)
        # Original title: "COS MLP SOFTMAX"
        ax[h_key].set_title(f"{LABELS[name]}: last-token hidden state, softmax over d_model")

    # Original: fig.suptitle(f"Gen tokens 0-{seq_len} head 17 & final token mlp; model layer {layer_num}",
    #                        fontsize=18, fontweight='bold')
    fig.suptitle(
        f"Llama-3.2-3B layer {layer}: head {head} (0-indexed) over {steps} generated tokens",
        fontsize=18,
        fontweight="bold",
    )
    # One colour bar per panel type. Any of the three images works as the source
    # because they share vmin/vmax; the last one drawn is used.
    # Original: fig.colorbar(atten_vis_cos, cax=ax_dict['c'])
    #           fig.colorbar(mlp_vs_cos, cax=ax_dict['m'])
    fig.colorbar(attn_im, cax=ax["c"], label="attention (3x3 avg-pooled)")
    fig.colorbar(hid_im, cax=ax["m"], label="softmax prob")
    return fig


def save_all_layers(slices, out_dir: str | Path, head: int = 17, step: int = -1, dpi: int = 100) -> list[Path]:
    """Write one comparison figure per layer; returns the file paths."""
    # Original: the outer loop of final_lamma_vis.ipynb cell 4:
    #   for layer_num in range(num_layers): ... plt.savefig(file_path, bbox_inches="tight"); fig_paths.append(file_path)
    out_dir = Path(out_dir)
    # Create the output folder if needed.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Compute the shared colour ranges ONCE, so all 28 frames use the same scale.
    ranges = global_ranges(slices, step)
    # 28 layers: axis 1 of attn_window (steps, layers, h, w).
    n_layers = next(iter(slices.values()))["attn_window"].shape[1]
    paths = []
    for layer in range(n_layers):
        fig = layer_comparison_figure(slices, layer, head=head, step=step, ranges=ranges)
        # Original file name: final_vis/conv_scope_gen_len_h17_mlplast_lay{layer_num}.png
        # :02d zero-pads (layer_00 ... layer_27) so files sort in layer order.
        p = out_dir / f"layer_{layer:02d}.png"
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
        # Close each figure so 28 big figures don't pile up in memory.
        # (The original called plt.show() instead, for the notebook display.)
        plt.close(fig)
        paths.append(p)
    return paths


def make_gif(paths, out_path: str | Path, duration_ms: int = 1000, loop: int = 0) -> Path:
    """Stitch frames into a GIF, padding each to the largest frame on white."""
    # Original: final_lamma_vis.ipynb cell 5 (same steps, written as a function).
    import imageio.v2 as imageio
    from PIL import Image

    # Original: pil_images = [Image.open(img) for img in fig_paths]
    # (.convert("RGB") drops any alpha channel so every frame has the same format.)
    frames = [Image.open(p).convert("RGB") for p in paths]
    # bbox_inches="tight" makes frame sizes differ slightly; find the largest.
    # Original: max_width = max(img.width for img in pil_images)
    #           max_height = max(img.height for img in pil_images)
    W = max(f.width for f in frames)
    H = max(f.height for f in frames)
    padded = []
    for f in frames:
        # Paste each frame onto a white canvas of the largest size, top-left
        # aligned, so every GIF frame is identical in size.
        # Original: background = Image.new('RGB', (max_width, max_height), (255, 255, 255))
        #           background.paste(img, (0, 0))
        bg = Image.new("RGB", (W, H), (255, 255, 255))
        bg.paste(f, (0, 0))
        padded.append(np.asarray(bg))
    # Write the GIF. duration = ms per frame; loop=0 means loop forever
    # (the original used duration=1000, loop=5).
    # Original: imageio.mimsave('final_vis/conv_scope_atten_gen_len_h17_mlplast.gif', gif_images, duration=1000, loop=5)
    imageio.mimsave(out_path, padded, duration=duration_ms, loop=loop)
    return Path(out_path)


# --------------------------------------------------------------------------
# single-pass (stage 1) figures
# --------------------------------------------------------------------------
def head_grid(maps: np.ndarray, title: str, ncols: int = 4):
    """Grid of per-head maps, ``maps`` shape ``(heads, seq, seq)``.

    Panels are labelled with 0-indexed head numbers so they match the head
    index used in the generation experiment.
    """
    # Original: the "Max Attention Heads COS" / "Max Attention Heads DICT" grids
    # in llama_vis.ipynb (6 rows x 4 columns for 24 heads). The original titled
    # panels "Head {j+1}" (1-indexed); this uses "Head {j}" (0-indexed).

    # n = 24 heads; nrows = ceil(24 / 4) = 6.
    n = maps.shape[0]
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), constrained_layout=True)
    cmap = sns.color_palette("cubehelix", as_cmap=True)
    # axs.flat walks the grid row by row; j is the head index.
    for j, a in enumerate(axs.flat):
        a.set_xticks([])
        a.set_yticks([])
        if j < n:
            # Draw head j's (seq, seq) map.
            a.imshow(maps[j], cmap=cmap)
            a.set_title(f"Head {j}")
        else:
            # Hide any spare panels if heads don't fill the grid exactly.
            a.axis("off")
    fig.suptitle(title)
    return fig


def specialization_map(max_heads: np.ndarray) -> np.ndarray:
    """Mean over heads of |head - median head| for ``(heads, seq, seq)`` maps."""
    # Original: the "Average Specialization" figure, from
    #   mx.abs(max_cos_heads - mx.median(max_cos_heads, axis=0)).mean(axis=0)
    # Read inside-out:
    #   np.median(max_heads, axis=0)   (seq, seq)         the "typical head" at every (query, key) cell
    #   max_heads - median             (24, seq, seq)     how far each head deviates from typical
    #   np.abs(...).mean(axis=0)       (seq, seq)         average deviation across heads
    # Bright cells = places where heads disagree, i.e. where some heads specialise.
    return np.abs(max_heads - np.median(max_heads, axis=0)).mean(axis=0)
