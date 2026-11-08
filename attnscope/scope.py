"""Capture attention windows and per-layer hidden states during generation.

This is a cleaned-up version of the original ``TylerScope.gen_from_prompt``.
Behaviour is kept identical so the saved arrays in ``data/`` can be
regenerated:

* greedy decoding, up to ``max_new_tokens`` tokens, stopping early on
  ``<|eot_id|>`` / EOS;
* no KV cache: the full prompt + generated text is re-tokenized and re-run
  through every layer at each step;
* at every step and every layer, the attention matrix of one fixed head is
  smoothed with a 3x3 average pool (stride 1, zero padding 1) and its
  bottom-right ``window x window`` block is kept (the most recent query
  positions attending to the most recent key positions);
* at every step and every layer, the last token's residual stream is passed
  through the model's final RMSNorm and stored.

Requires Apple Silicon + ``mlx`` / ``mlx-lm``.
"""

# =============================================================================
# HOW THIS FILE MAPS TO THE ORIGINAL SUMMER CODE
#
#   this file                      original
#   -----------------------------  ---------------------------------------------
#   load_prompt(path)              the `with open(promt_path,"r")` block in gen_from_prompt
#   pool_attention(head)           the AvgPool2d lines shared by static_cam_at_end_conv
#                                  and dynamic_camera_atten_slice
#   end_window(head)               TylerScope.static_cam_at_end_conv(layer_att, radius, track_head)
#   most_active_head(layer_att)    the `if tracking_head is None:` block of
#                                  dynamic_camera_atten_slice
#   focus_window(head)             TylerScope.dynamic_camera_atten_slice with
#                                  track_option=0 and the head already chosen
#   ScopeResult                    the trackers self.atten_out / self.mlp_out_mone /
#                                  self.text_gen, plus mx.save(...) in llama_temp_vis
#   forward_all_layers(...)        the single-pass loop in llama_vis.ipynb (one forward
#                                  pass, keep every layer's attention)
#   generate_and_capture(...)      TylerScope.gen_from_prompt(...)
#
# Not carried over (only used for the live Qt viewer or never saved):
#   self.mlp_out_mtwo (second-to-last token), self.tracked_heads (always 17),
#   self.orig_atten_out (full attention, too large to save), the TensorVisualizer
#   update calls + time.sleep(5), and construct_atten_plots.
#
# scripts/check_against_original.py checks end_window, focus_window,
# most_active_head and generate_and_capture against copies of the originals.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .forward import embed, layer_forward, unembed


def load_prompt(path: str | Path) -> str:
    """Read a prompt file exactly the way the original experiment did.

    The original code used ``" ".join(f.readlines())``, which inserts a space
    after every newline. The model therefore saw each line after the first
    prefixed by one space. Keep this so results are reproducible.
    """
    # Original (gen_from_prompt): with open(promt_path,"r") as f:
    #                                 prompt = " ".join(f.readlines())
    # readlines() keeps each line's "\n"; joining with " " puts a space after
    # every newline, so "a\nb\n" becomes "a\n b\n". Kept on purpose.
    with open(path, "r") as f:
        return " ".join(f.readlines())


def pool_attention(attn_head: mx.array, kernel_size: int = 3) -> mx.array:
    """3x3 average pool of one ``(seq, seq)`` attention map, same size out.

    This is the original pooling code. ``nn.AvgPool2d`` expects images shaped
    ``(batch, height, width, channels)``, so the map is given a batch axis in
    front and a channel axis at the back, pooled, then squeezed back to 2-D.
    Padding is ``(kernel_size - 1) // 2`` zeros on each side, so the output
    keeps the input's shape.
    """
    # Zeros to add on each side: (3 - 1) // 2 = 1. With stride 1 this keeps the
    # output the same size as the input.
    # Original: padding_sz = int(((kernel_size-1)//2))
    padding_sz = int((kernel_size - 1) // 2)
    # The 3x3 mean filter: each output pixel = average of its 3x3 neighbourhood
    # (padded zeros count toward the average, so edges come out a bit dimmer).
    # Original: conv_lay = mlx.nn.AvgPool2d(kernel_size=(kernel_size,kernel_size),padding=(padding_sz),stride=1)
    conv_lay = nn.AvgPool2d(kernel_size=(kernel_size, kernel_size), padding=padding_sz, stride=1)
    # AvgPool2d wants images shaped (batch, height, width, channels), so add a
    # batch axis at position 0 and a channel axis at position 3.
    # Original: conv_head = mlx.core.expand_dims(layer_att[:,:],axis=(0,3))    (static_cam_at_end_conv)
    #           good_head = mlx.core.expand_dims(layer_att[tracking_head,:,:],axis=(0,3))  (dynamic_camera)
    conv_head = mx.expand_dims(attn_head, axis=(0, 3))  # (1, seq, seq, 1)
    # Pool, then drop the two size-1 axes to get back a plain (seq, seq) map.
    # Original: mx.squeeze(conv_lay(conv_head))
    return mx.squeeze(conv_lay(conv_head))  # (seq, seq)


def end_window(attn_head: mx.array, window: int = 100, kernel_size: int = 3) -> mx.array:
    """Pool a ``(seq, seq)`` attention map and crop its bottom-right block.

    Same as the original ``static_cam_at_end_conv``. If ``seq < window`` the
    whole (smaller) map is returned.
    """
    # Equivalent to: TylerScope.static_cam_at_end_conv(layer_att, radius, track_head)
    # Differences in the call, not the result:
    #   - the original took the full (1, heads, seq, seq) tensor and picked the
    #     head inside (layer_att = layer_att[0,track_head,:,:]); this takes the
    #     one head's (seq, seq) map, already picked by the caller.
    #   - original `radius` is called `window` here (100 in the experiment).
    #   - the original returned (track_head, conv_head); this returns just the map.
    # [-window:, -window:] keeps the last 100 rows (most recent queries) and last
    # 100 columns (most recent keys): the bottom-right corner of the map.
    # Original: conv_head = mx.squeeze(conv_lay(conv_head))[-radius:,-radius:]
    return pool_attention(attn_head, kernel_size)[-window:, -window:]


def most_active_head(layer_att: mx.array) -> int:
    """Head whose attention is most concentrated (original selection rule).

    For each head: max over keys for every query, then median over queries;
    pick the head with the largest value. ``layer_att``: ``(heads, seq, seq)``.
    """
    # Equivalent to the `if tracking_head is None:` block of
    # TylerScope.dynamic_camera_atten_slice.
    #   mx.max(layer_att, axis=-1)     (heads, seq)  strongest attention of each query (each row)
    #   mx.median(..., axis=-1)        (heads,)      the typical row's peak, per head
    #   mx.squeeze(...)                drop any size-1 axes
    # A high value means most queries in that head focus sharply on one key.
    # Original: median_max_att = mlx.core.squeeze(mlx.core.median(mlx.core.max(layer_att,axis=-1),axis=-1))
    median_max_att = mx.squeeze(mx.median(mx.max(layer_att, axis=-1), axis=-1))
    # Head index with the largest median-of-max. .item() turns the 0-d MLX array
    # into a plain Python int (the original kept it as an MLX array).
    # Original: tracking_head = median_max_att.argmax(axis=-1)
    return median_max_att.argmax(axis=-1).item()


def focus_window(attn_head: mx.array, cam_radius: int = 30, kernel_size: int = 3) -> mx.array:
    """Crop a square window centred on the hottest spot of an attention map.

    Same as the original ``dynamic_camera_atten_slice`` with the head already
    chosen: pool the map, centre the "camera" on the pooled argmax, cut out a
    ``2 * ((cam_radius - 1) // 2) + 1`` square, and zero-pad wherever it runs
    off the edge so every crop has the same size.
    """
    # Equivalent to: TylerScope.dynamic_camera_atten_slice(layer_att, orig_seq_len,
    #     tracking_head=<given>, kernel_size, cam_radius, track_option=0)
    # Only the track_option == 0 branch is kept (that is the one the saved
    # sliding-window figures used). Head selection lives in most_active_head().
    # Here attn_head is the (seq, seq) map; the original's good_head before and
    # after its squeeze.

    # Where should the camera point? The argmax of the pooled map, as a flat index.
    #   pool_attention(...)            (seq, seq)   smoothed map, so one noisy pixel can't win
    #   mx.flatten(..., start_axis=0)  (seq*seq,)   one long row-by-row vector
    #   mx.argmax(...).item()          int          flat position of the hottest pooled pixel
    # Original: # Now we are going to calculate where the camera should currently center on for each head independently.
    #           center_flat_ind =  mlx.core.argmax(mlx.core.flatten(mlx.core.squeeze(conv_lay(good_head)),start_axis=0)).item()
    center_flat_ind = mx.argmax(mx.flatten(pool_attention(attn_head, kernel_size), start_axis=0)).item()

    # Map size. Original: _,good_head_height, good_head_width,_ = good_head.shape
    #   (read from the 4-D version there; same numbers)
    head_height, head_width = attn_head.shape
    # Flat index -> (row, col): row = index // width, col = index % width.
    # Original: # Now we figure out where the camera is going to be centered.
    #           camera_row_cent = int(center_flat_ind // good_head_width)
    camera_row_cent = int(center_flat_ind // head_width)
    # Original: camera_col_cent = int(center_flat_ind % good_head_width)
    camera_col_cent = int(center_flat_ind % head_width)
    # How far the window reaches from its centre: (30 - 1) // 2 = 14, so the
    # window is 14 + 1 + 14 = 29 pixels wide.
    # Original: # Caclulate the actual amount we'll need to subtract for the padding.
    #           sub_amt = (cam_radius-1)//2
    sub_amt = (cam_radius - 1) // 2

    # Clip the window to the map's edges...
    # The ideal window spans rows centre-14 .. centre+14; np.clip pulls any
    # bound that falls off the map back to 0 or the last index.
    # Original: # Next calc the clipped row and col slices
    #           (the same four np.clip lines, with good_head_height / good_head_width)
    row_min = np.clip((camera_row_cent - sub_amt), 0, (head_height - 1)).astype(int)
    row_max = np.clip((camera_row_cent + sub_amt), 0, (head_height - 1)).astype(int)
    col_min = np.clip((camera_col_cent - sub_amt), 0, (head_width - 1)).astype(int)
    col_max = np.clip((camera_col_cent + sub_amt), 0, (head_width - 1)).astype(int)
    # Cut out the part of the window that is actually on the map. Slices from
    # the UNpooled map: pooling was only used to decide where to look.
    # +1 because Python slice ends are exclusive.
    # Original: good_sliced_mat = good_head[row_min:(row_max+1),col_min:(col_max+1)]
    sliced = attn_head[row_min : (row_max + 1), col_min : (col_max + 1)]

    # ...then pad back with zeros by however much was clipped on each side.
    # Each pad = how far the ideal edge fell past the map edge (0 if it didn't):
    #   pad_top:    if centre-14 < 0, row_min was clipped to 0 and abs(centre-14) - 0 rows are missing;
    #               otherwise row_min == centre-14 and this is 0.
    #   pad_bottom: (centre+14) - row_max, the rows clipped off the bottom.
    #   left/right: same idea for columns.
    # Original: # Need to calc top bottom left right padding here
    #           (the same four lines)
    pad_top = np.abs(camera_row_cent - sub_amt) - row_min
    pad_bottom = (camera_row_cent + sub_amt) - row_max
    pad_left = np.abs(camera_col_cent - sub_amt) - col_min
    pad_right = (camera_col_cent + sub_amt) - col_max
    # Add the zeros so every window is exactly 29 x 29, wherever the camera points.
    # The int() casts only turn numpy integers into Python ints; values unchanged.
    # Original: # Now actually make what we'll see
    #           good_lens_padded = mlx.core.pad(good_sliced_mat,pad_width=((pad_top,pad_bottom),(pad_left,pad_right)),constant_values=0)
    #           (the original returned (tracking_head, good_lens_padded); this returns only the window)
    return mx.pad(
        sliced,
        pad_width=((int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right))),
        constant_values=0,
    )


@dataclass
class ScopeResult:
    """Arrays captured by :func:`generate_and_capture`."""

    # Replaces the original trackers on the TylerScope object:
    #   attn_window   <- self.atten_out      (stacked to (steps, layers, 100, 100))
    #   last_hidden   <- self.mlp_out_mone   (stacked to (steps, layers, 3072))
    #   text          <- self.text_gen[-1] / curr_gen  (prompt + everything generated)
    #   new_token_ids    new: the greedy token id chosen at each step
    attn_window: np.ndarray  # (steps, layers, window, window)
    last_hidden: np.ndarray  # (steps, layers, d_model), final-RMSNorm applied
    text: str  # prompt + generated continuation
    new_token_ids: list[int] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Save as compressed ``.npz`` (float16 attention, float16 hidden)."""
        # Replaces the original mx.save(f'{atten_mlxs_pth}/cos_atten_slice.npy', my_scope.atten_out)
        # and mx.save(... cos_lastmlp_slice.npy, my_scope.mlp_out_mone) calls.
        # One file per prompt instead of two, compressed, float16 so it fits under
        # GitHub's 100 MB file limit (see data/README.md for the precision check).
        np.savez_compressed(
            path,
            attn_window=self.attn_window.astype(np.float16),
            last_hidden=self.last_hidden.astype(np.float16),
            text=np.array(self.text),
            new_token_ids=np.array(self.new_token_ids),
        )


def forward_all_layers(model, tokenizer, text: str):
    """Single forward pass that returns every layer's attention and residual.

    Returns ``(attn, hidden)`` with shapes ``(layers, heads, seq, seq)`` and
    ``(layers, seq, d_model)``. Used for the single-pass head comparison.
    """
    # Equivalent to the single-pass loop in llama_vis.ipynb: embed the prompt
    # once, run it through all 28 layers, keep each layer's attention.
    # Original: model_in = self.tokenize_input(...)  ->  h here
    h = embed(model, tokenizer, text)
    attns, hiddens = [], []
    for i in range(model.args.num_hidden_layers):
        # Each layer's output is the next layer's input.
        # Original: next_out,atten_out = self.manually_run_foward(prev_out,i)
        #           ... prev_out = next_out
        h, a = layer_forward(model, h, i)
        # [0] drops the batch axis (always 1): a[0] is (24, S, S), h[0] is (S, 3072).
        attns.append(a[0])
        hiddens.append(h[0])
    # Stack per-layer lists into (28, 24, S, S) and (28, S, 3072).
    return mx.stack(attns), mx.stack(hiddens)


def generate_and_capture(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    head: int = 17,
    window: int = 100,
    kernel_size: int = 3,
    progress: bool = True,
) -> ScopeResult:
    """Greedy-generate from ``prompt`` while recording one head and the residual.

    ``head`` is 0-indexed (head 17 here is "Head 18" in 1-indexed plots).
    """
    # Equivalent to: TylerScope.gen_from_prompt(promt_path, cam_radius, lay_guess_height,
    #                                           lay_guess_width, max_gen=100, ...)
    # with the live visualizers off. Argument mapping:
    #   prompt          <- the text read from promt_path (use load_prompt() for the file)
    #   max_new_tokens  <- max_gen / max_token_gens
    #   head=17         <- the hard-coded 17 in static_cam_at_end_conv(atten_out,100,17)
    #   window=100      <- the hard-coded 100 (radius) in that same call

    # Token ids that end generation.
    # Original: eos_token_id = self.tokenizer.eos_token_id
    #           eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}
    # 28 for Llama-3.2-3B. Original: self.model.args.num_hidden_layers (in the loop header)
    n_layers = model.args.num_hidden_layers

    # Running text: starts as the prompt, one decoded token appended per step.
    # Original: curr_gen = prompt
    text = prompt
    # Per-step results, stacked at the end.
    # Original: self.atten_out, self.mlp_out_mone (reset by self.reset_trackers())
    attn_steps, hidden_steps, new_ids = [], [], []

    # Optional progress bar.
    # Original: for _ in tqdm.tqdm(range(max_token_gens),total=max_token_gens):
    steps = range(max_new_tokens)
    if progress:
        from tqdm.auto import tqdm

        steps = tqdm(steps, total=max_new_tokens)

    for _ in steps:
        # Re-tokenize and re-embed the WHOLE text every step (no KV cache), so
        # the attention maps grow by one row and column per step.
        # Original: model_in = self.tokenize_input(curr_gen)
        #           prev_out = model_in
        h = embed(model, tokenizer, text)
        # Original: atten_per_layer = [] / mlp_per_layer_mone = []
        attn_layers, hidden_layers = [], []
        for i in range(n_layers):
            # Original: next_out,atten_out = self.manually_run_foward(prev_out,i)
            #           ... prev_out = next_out   (here h is reused for both)
            h, attn = layer_forward(model, h, i)
            # attn[0, head]: drop the batch axis and pick head 17 -> (S, S); then
            # pool and keep the bottom-right 100 x 100.
            # Original: head_num,atten_slice = self.static_cam_at_end_conv(atten_out,100,17)
            #           atten_per_layer.append(atten_slice)
            attn_layers.append(end_window(attn[0, head], window, kernel_size))
            # Last token's residual stream after this layer, through the model's
            # FINAL RMSNorm (the one normally applied only after layer 28), so
            # every layer is on the same scale. (3072,)
            # Original: next_out_last = self.model.model.norm(next_out[0,-1,:])
            #           mlp_per_layer_mone.append(next_out_last)
            hidden_layers.append(model.model.norm(h[0, -1, :]))

        # Lists of 28 -> arrays: (28, 3072) and (28, 100, 100).
        # Original: mlp_per_layer_mone = mx.stack(mlp_per_layer_mone,axis=0)
        #           atten_per_layer = mx.stack(atten_per_layer,axis=0)
        #           self.atten_out.append(atten_per_layer) / self.mlp_out_mone.append(mlp_per_layer_mone)
        hidden_layers = mx.stack(hidden_layers)
        attn_steps.append(mx.stack(attn_layers))
        hidden_steps.append(hidden_layers)

        # Greedy next token from the last layer's normed hidden state.
        # hidden_layers[-1] = layer 28 output after the final norm -> vocab logits
        # -> argmax = most likely token id (no sampling, no temperature).
        # Original: gen_token_id = self.model.lm_head(mlp_per_layer_mone[-1,:]).argmax(axis=-1).item()
        #           (see unembed() in forward.py for the lm_head vs tied-embedding detail)
        next_id = unembed(model, hidden_layers[-1]).argmax(axis=-1).item()
        new_ids.append(next_id)
        # NOTE: decode-and-append, then re-tokenize the whole string next step
        # (as in the original). This can occasionally re-segment tokens.
        # clean_up_tokenization_spaces=False: Llama's BPE decode is already exact,
        # so no WordPiece-style space cleanup (newer transformers skip it for BPE
        # anyway; this just makes it explicit and silences the warning).
        # Original: curr_gen += self.tokenizer.decode(gen_token_id)
        #           self.text_gen.append(curr_gen)
        text += tokenizer.decode(next_id,clean_up_tokenization_spaces=False)
        # Stop at end-of-sequence / end-of-turn. (The original also checked
        # num_tokens_gen == max_token_gens; the for loop already stops there.)
        # Original: if gen_token_id == eos_token_id or gen_token_id == eot_token_id or num_tokens_gen == max_token_gens:
        #               break
        if next_id in eos_ids:
            break

    # Stack all steps and convert to float32 numpy:
    #   attn   (steps, 28, 100, 100)   hidden   (steps, 28, 3072)
    # Original: self.atten_out = mx.stack(self.atten_out,axis=0)
    #           self.mlp_out_mone = mx.stack(self.mlp_out_mone,axis=0)
    attn = np.array(mx.stack(attn_steps).astype(mx.float32))
    hidden = np.array(mx.stack(hidden_steps).astype(mx.float32))
    # Original: return f"''{curr_gen}''"   (the arrays stayed on the object)
    return ScopeResult(attn_window=attn, last_hidden=hidden, text=text, new_token_ids=new_ids)
