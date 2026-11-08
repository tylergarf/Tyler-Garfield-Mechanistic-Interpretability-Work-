"""Check the repo code against the original summer-2026 research code.

Run from the repo root on Apple Silicon:

    python scripts/check_against_original.py            # pooling / windows / head choice
    python scripts/check_against_original.py --model    # + full forward pass and generation (loads the 3B model)

The ``original_*`` functions below are copied from the summer notebooks
(``llama_vis.ipynb`` and ``llama_temp_vis.ipynb``) with only these changes:
the model is passed in as an argument instead of being a global/self, and
``mlx.core`` / ``mlx.nn`` are imported under the same names.
Every check requires *exact* equality, not "close enough".
"""

# How to read the comments in this file:
#   - Inside the original_* functions, plain "# ..." comments are the ORIGINAL
#     summer comments, copied word for word (spelling included) and placed
#     above the same line they sat above in TylerScope.
#   - Comments starting with "# [repo note]" were added for the repo. They say
#     which repo function each original corresponds to, or explain the check.
#   - Original comments whose code isn't in these copies (e.g. the
#     track_option == 1 branch, commented-out print()s, the live
#     TensorVisualizer calls) are left out because there's no line to attach them to.

import argparse
import sys
from pathlib import Path

import mlx
import mlx.core as mx
import mlx.nn
import numpy as np

# [repo note] Make the repo root importable so `from attnscope import ...` works
# when this file is run as `python scripts/check_against_original.py` (Python
# otherwise only looks in scripts/). Must run before the attnscope imports.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# [repo note] "noqa: E402" tells style checkers (flake8/ruff) that these imports
# are below other code on purpose (they need the sys.path line above). Python ignores it.
from attnscope import scope  # noqa: E402
from attnscope.forward import layer_forward  # noqa: E402


# ---------------------------------------------------------------------------
# Original code (verbatim apart from passing ``model`` in)
# ---------------------------------------------------------------------------
def original_manually_run_foward(model, input, layer_num):
    # [repo note] Original: TylerScope.manually_run_foward(self, input, layer_num)
    # [repo note] Repo version: attnscope.forward.layer_forward(model, h, layer_idx)

    curr_layer = model.layers[layer_num]
    num_atten_heads = model.args.num_attention_heads
    hidden_dim = model.args.hidden_size
    head_dim = hidden_dim // num_atten_heads
    swGelu = mlx.nn.SiLU()

    # We are going to the input is of size batch, seq_len, hidden_dim in this case.
    res = input

    # Next we norm the input.
    in_normed = curr_layer.input_layernorm(res)

    # Now we project and reshape.
    Q = curr_layer.self_attn.q_proj(in_normed)
    batch_len, seq_len, _ = Q.shape
    Q = Q.reshape(batch_len, seq_len, num_atten_heads, hidden_dim // num_atten_heads).transpose(0, 2, 1, 3)

    V = curr_layer.self_attn.v_proj(in_normed)
    _, _, V_hidden_dim = V.shape
    V_num_heads = V_hidden_dim // head_dim
    V = V.reshape(batch_len, seq_len, V_num_heads, head_dim)
    G_V = num_atten_heads // V.shape[2]
    V = mx.expand_dims(V, axis=-2)
    V = mx.broadcast_to(V, (batch_len, seq_len, V_num_heads, G_V, head_dim))
    V = V.reshape(batch_len, seq_len, (V_num_heads * G_V), head_dim).transpose(0, 2, 1, 3)

    K = curr_layer.self_attn.k_proj(in_normed)
    _, _, K_hidden_dim = K.shape
    K_num_heads = K_hidden_dim // head_dim
    K = K.reshape(batch_len, seq_len, K_num_heads, head_dim)
    # Now this is the number of head groups that we'll broadcast
    G_K = num_atten_heads // K.shape[2]
    # Now we actually do this expansion.
    K = mx.expand_dims(K, axis=-2)
    K = mx.broadcast_to(K, (batch_len, seq_len, K_num_heads, G_K, head_dim))
    K = K.reshape(batch_len, seq_len, (K_num_heads * G_K), head_dim).transpose(0, 2, 1, 3)

    # Now we apply the rope positional system.
    Q = curr_layer.self_attn.rope(Q)
    K = curr_layer.self_attn.rope(K)

    #Next we do the QK^T operation to get the actual attention heads.
    unmasked_att = Q @ K.transpose(0, 1, 3, 2)
    # Now we need to divide by the sqrt of the length of K
    unmasked_att = unmasked_att / mx.sqrt(head_dim)
    mask = mlx.nn.MultiHeadAttention.create_additive_causal_mask(unmasked_att.shape[-1])
    casual_att = mx.softmax((unmasked_att + mask), axis=-1)

    # Now we do the dot product with V.
    atten_out = casual_att @ V
    # Now we will get the heads back together.
    atten_out = atten_out.transpose(0, 2, 1, 3).reshape(batch_len, seq_len, hidden_dim)
    atten_out = curr_layer.self_attn.o_proj(atten_out)
    # Next we add to the residual stream 
    res = res + atten_out

    # Now we normalize attention out to get the final attention out.
    final_atten_out = curr_layer.post_attention_layernorm(res)
    # Now we are going to do the switch mlp layer.
    # First is the parralel projections.
    gt_proj = curr_layer.mlp.gate_proj(final_atten_out)
    up_proj = curr_layer.mlp.up_proj(final_atten_out)
    # Now we do the proper calculation.
    mlp_hidden = swGelu(gt_proj) * up_proj
    # Now we do the down proj layer.
    mlp_out = curr_layer.mlp.down_proj(mlp_hidden)
    # Now we add the result of the mlp to our residual stream.
    res = res + mlp_out
    # Now the final output is our residual stream and will be normed at the top of the next time this func is called.
    return res, casual_att


def original_static_cam_at_end_conv(layer_att, radius, track_head):
    # [repo note] Original: TylerScope.static_cam_at_end_conv(self, layer_att, radius, track_head)
    #             (no comments in the original). Takes the full (1, heads, seq, seq)
    #             attention tensor, picks one head, 3x3 average-pools it, keeps the
    #             bottom-right radius x radius block.
    # [repo note] Repo version: attnscope.scope.end_window(attn_head, window), which
    #             receives the already-picked (seq, seq) head and uses
    #             attnscope.scope.pool_attention for the pooling lines.
    layer_att = layer_att[0, track_head, :, :]
    seq_len, _ = layer_att.shape
    kernel_size = 3
    padding_sz = int(((kernel_size - 1) // 2))
    conv_lay = mlx.nn.AvgPool2d(kernel_size=(kernel_size, kernel_size), padding=(padding_sz), stride=1)
    conv_head = mlx.core.expand_dims(layer_att[:, :], axis=(0, 3))
    conv_head = mx.squeeze(conv_lay(conv_head))[-radius:, -radius:]
    return track_head, conv_head


def original_dynamic_camera_atten_slice(layer_att, tracking_head=None, kernel_size=3, cam_radius=30):
    # [repo note] Original: TylerScope.dynamic_camera_atten_slice(self, layer_att, orig_seq_len,
    #             tracking_head=None, kernel_size=3, cam_radius=30, track_option=0).
    #             This copy is the llama_vis.ipynb version used for the sliding-window
    #             figures: no orig_seq_len / track_option arguments, which is the same
    #             as always taking the track_option == 0 branch.
    # [repo note] Repo versions: the head-choice block -> attnscope.scope.most_active_head,
    #             the rest -> attnscope.scope.focus_window(attn_head, cam_radius).

    # First we'll find the most intresting head to slice. I am making this optional
    # so that on the dict dump side I can just use the head that was previously found
    # to be the most intresting.
    if tracking_head is None:
        median_max_att = mlx.core.squeeze(mlx.core.median(mlx.core.max(layer_att, axis=-1), axis=-1))
        tracking_head = median_max_att.argmax(axis=-1)

    padding_sz = int(((kernel_size - 1) // 2))
    conv_lay = mlx.nn.AvgPool2d(kernel_size=(kernel_size, kernel_size), padding=(padding_sz), stride=1)
    good_head = mlx.core.expand_dims(layer_att[tracking_head, :, :], axis=(0, 3))
    # Now we are going to calculate where the camera should currently center on for each head independently.
    # NOTE you could also just slice before argmaxing to get the most intresting part of the head that is not part of the input prompt.
    center_flat_ind = mlx.core.argmax(mlx.core.flatten(mlx.core.squeeze(conv_lay(good_head)), start_axis=0)).item()
    # Now get rid of the one dimensions added for the conv layer
    good_head = mlx.core.squeeze(good_head)
    good_head_height, good_head_width = good_head.shape
    # Now we figure out where the camera is going to be centered.
    camera_row_cent = int(center_flat_ind // good_head_width)
    camera_col_cent = int(center_flat_ind % good_head_width)
    # Caclulate the actual amount we'll need to subtract for the padding.
    sub_amt = (cam_radius - 1) // 2
    # Next calc the clipped row and col slices
    row_min = np.clip((camera_row_cent - sub_amt), 0, (good_head_height - 1)).astype(int)
    row_max = np.clip((camera_row_cent + sub_amt), 0, (good_head_height - 1)).astype(int)
    col_min = np.clip((camera_col_cent - sub_amt), 0, (good_head_width - 1)).astype(int)
    col_max = np.clip((camera_col_cent + sub_amt), 0, (good_head_width - 1)).astype(int)
    # Now we'll actually slice the good and bad heads to get what the camera is currently seeing and we'll 
    # pad to make each of them the same size.
    good_sliced_mat = good_head[row_min:(row_max + 1), col_min:(col_max + 1)]
    # Need to calc top bottom left right padding here
    pad_top = np.abs(camera_row_cent - sub_amt) - row_min
    pad_bottom = (camera_row_cent + sub_amt) - row_max
    pad_left = np.abs(camera_col_cent - sub_amt) - col_min
    pad_right = (camera_col_cent + sub_amt) - col_max
    # Now actually make what we'll see
    good_lens_padded = mlx.core.pad(
        good_sliced_mat, pad_width=((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0
    )
    # Return the sliced and padded matrix plus the desired trackig head.
    return tracking_head, good_lens_padded


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
# [repo note] Names of failed checks; printed at the end and used for the exit code.
FAILURES = []


def check(name, ok, detail=""):
    # [repo note] Print one PASS/FAIL line and remember failures.
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def to_np(x):
    # bfloat16 MLX arrays can't go straight to numpy; float32 holds every bf16 value exactly.
    return np.array(x.astype(mx.float32)) if isinstance(x, mx.array) else np.asarray(x)


def same(a, b):
    # [repo note] Exact equality: same dtype, same shape, every element identical.
    # No tolerance, so even a 1-bit difference fails.
    a, b = to_np(a), to_np(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def fake_attention(heads, seq, rng):
    """Random causal attention maps (rows sum to 1, upper triangle zero)."""
    # [repo note] Random scores, scaled by 3 so some rows are peaked like real heads.
    logits = rng.normal(size=(heads, seq, seq)).astype(np.float32) * 3
    # [repo note] -1e9 above the diagonal = the causal mask (no attending to the future).
    logits = logits + np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
    # [repo note] Row-wise softmax -> each row sums to 1, masked entries become 0.
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return mx.array(e / e.sum(-1, keepdims=True))


def check_windows():
    # [repo note] Compares, without loading the model:
    #   scope.most_active_head  vs  the head-choice block of original_dynamic_camera_atten_slice
    #   scope.end_window        vs  original_static_cam_at_end_conv
    #   scope.focus_window      vs  original_dynamic_camera_atten_slice (head given)
    print("Pooling, end window, focus window, head choice")
    rng = np.random.default_rng(0)
    # [repo note] Five fake layers of 24 heads. seq 5 and 12 are smaller than the
    # 100-wide end window and the 29-wide focus window, so the clipping/padding
    # edge cases get exercised; 300 is about the length of the real prompts.
    cases = [fake_attention(24, s, rng) for s in (5, 12, 40, 150, 300)]
    # Real data: the shipped windows are attention-like maps too.
    # [repo note] Every 7th layer (0, 7, 14, 21) -> (100 steps, 4, 100, 100).
    with np.load(ROOT / "data" / "cos.npz") as z:
        real = z["attn_window"][:, ::7].astype(np.float32)  # (100, 4, 100, 100)
    cases += [mx.array(real[step]) for step in (0, 50, 99)]  # treat layers as "heads"

    for i, layer_att in enumerate(cases):
        H, S, _ = layer_att.shape
        tag = f"case {i} ({H} heads, seq {S})"

        # [repo note] Head choice: call the original with tracking_head=None so it
        # picks the head itself, then compare with the repo's pick.
        orig_head, _ = original_dynamic_camera_atten_slice(layer_att)
        check(f"{tag}: most_active_head", scope.most_active_head(layer_att) == int(orig_head))

        all_end, all_focus = True, True
        for h in range(H):
            # [repo note] layer_att[None] adds the batch axis the original expects
            # (it indexes layer_att[0, track_head]); the repo gets layer_att[h] directly.
            _, o_end = original_static_cam_at_end_conv(layer_att[None], 100, h)
            all_end &= same(scope.end_window(layer_att[h], window=100), o_end)
            # [repo note] Two camera sizes: 10 (9 x 9 window) and 30 (29 x 29, the experiment's).
            for r in (10, 30):
                _, o_foc = original_dynamic_camera_atten_slice(layer_att, tracking_head=h, cam_radius=r)
                all_focus &= same(scope.focus_window(layer_att[h], cam_radius=r), o_foc)
        check(f"{tag}: end_window, every head", all_end)
        check(f"{tag}: focus_window, every head, radius 10 and 30", all_focus)


def check_model():
    # [repo note] Needs the real model (~6 GB download on first run). Compares
    #   attnscope.forward.layer_forward       vs  original_manually_run_foward  (all 28 layers)
    #   attnscope.scope.generate_and_capture  vs  the original gen_from_prompt loop  (3 tokens)
    from mlx_lm import load

    print("Full forward pass and generation (Llama-3.2-3B-Instruct)")
    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct")
    text = (ROOT / "prompts" / "cos.txt").read_text()
    # [repo note] Same as the original TylerScope.tokenize_input: encode -> add batch
    # axis -> embed. Both versions start from this identical input.
    h_new = h_old = model.model.embed_tokens(mx.expand_dims(mx.array(tokenizer.encode(text)), axis=0))
    layers_ok = True
    for i in range(model.args.num_hidden_layers):
        # [repo note] Run layer i with both versions, each fed its own previous output,
        # so any difference would compound and be caught at later layers too.
        h_new, a_new = layer_forward(model, h_new, i)
        h_old, a_old = original_manually_run_foward(model, h_old, i)
        layers_ok &= same(a_new, a_old) and same(h_new, h_old)
    check("layer_forward == manually_run_foward on all 28 layers (attention + residual)", layers_ok)

    # Three generation steps with the original loop vs generate_and_capture.
    # [repo note] The loop below is TylerScope.gen_from_prompt with the visualizers off
    # and only the trackers that were saved (atten_out, mlp_out_mone).
    prompt = scope.load_prompt(ROOT / "prompts" / "cos.txt")
    curr_gen, old_attn, old_hidden = prompt, [], []
    for _ in range(3):
        # [repo note] Original: model_in = self.tokenize_input(curr_gen); prev_out = model_in
        prev_out = model.model.embed_tokens(mx.expand_dims(mx.array(tokenizer.encode(curr_gen)), axis=0))
        # [repo note] Original: atten_per_layer = []; mlp_per_layer_mone = []
        atten_per_layer, mlp_per_layer = [], []
        for i in range(model.args.num_hidden_layers):
            next_out, atten_out = original_manually_run_foward(model, prev_out, i)
            # Now in here is where we will look at the atten out for the current layer as 
            # well as view the current layers final out.
            _, atten_slice = original_static_cam_at_end_conv(atten_out, 100, 17)
            # Now append the current run to our trackers.
            atten_per_layer.append(atten_slice)
            # [repo note] Original: next_out_last = self.model.model.norm(next_out[0,-1,:])
            #                       mlp_per_layer_mone.append(next_out_last)
            mlp_per_layer.append(model.model.norm(next_out[0, -1, :]))
            prev_out = next_out
        # [repo note] Original: mlp_per_layer_mone = mx.stack(mlp_per_layer_mone,axis=0)
        #                       self.atten_out.append(...); self.mlp_out_mone.append(...)
        mlp_per_layer = mx.stack(mlp_per_layer)
        old_attn.append(mx.stack(atten_per_layer))
        old_hidden.append(mlp_per_layer)
        # [repo note] Next token. This uses the llama_live_vis.py form (multiply by the
        # embedding matrix) because Llama-3.2-3B has no lm_head; llama_temp_vis's
        # self.model.lm_head(...) only works on models that have one (e.g. the 8B).
        gen_id = mx.matmul(mlp_per_layer[-1, :], model.model.embed_tokens.weight.T).argmax(axis=-1).item()
        # [repo note] Original: curr_gen += self.tokenizer.decode(gen_token_id)
        # The repo passes clean_up_tokenization_spaces=False. 
        # On transformers versions that ignore cleanup for BPE
        # tokenizers (the ones that print that warning) both give the same text;
        # an older version that applied cleanup could make the text check fail.
        curr_gen += tokenizer.decode(gen_id,clean_up_tokenization_spaces=False)

    # [repo note] Same 3 steps with the repo function, then compare all three outputs.
    res = scope.generate_and_capture(model, tokenizer, prompt, max_new_tokens=3, progress=False)
    check("generate_and_capture: same generated text", res.text == curr_gen)
    check("generate_and_capture: same attention windows", same(res.attn_window, mx.stack(old_attn).astype(mx.float32)))
    check("generate_and_capture: same hidden states", same(res.last_hidden, mx.stack(old_hidden).astype(mx.float32)))


if __name__ == "__main__":
    # [repo note] Command-line entry point: always run the window checks, and the
    # model checks only with --model.
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="also load the model and compare full forward passes")
    args = ap.parse_args()

    check_windows()
    if args.model:
        check_model()
    print()
    print("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    # [repo note] Exit code 1 on any failure, so the script can also be used in CI.
    sys.exit(1 if FAILURES else 0)
