"""Manual, layer-by-layer forward pass for Llama-family models in MLX.

This re-implements one ``TransformerBlock`` of ``mlx_lm``'s Llama model by hand
so the post-softmax attention probabilities of every head are available. It
mirrors the stock block exactly (pre-RMSNorm, GQA attention with RoPE, causal
mask, SwiGLU MLP, two residual adds). No KV cache is used: every call processes
the full sequence.

Requires Apple Silicon + ``mlx`` / ``mlx-lm``.
"""

# =============================================================================
# HOW THIS FILE MAPS TO THE ORIGINAL SUMMER CODE
#
#   this file                    original (TylerScope, llama_temp_vis.ipynb)
#   ---------------------------  ------------------------------------------------
#   layer_forward(model, h, i)   TylerScope.manually_run_foward(self, input, layer_num)
#   embed(model, tok, text)      TylerScope.tokenize_input(self, prompt)
#   unembed(model, h)            the next-token line in gen_from_prompt:
#                                  self.model.lm_head(mlp_per_layer_mone[-1,:])
#                                (or embed_tokens.weight.T in llama_live_vis.py)
#
# Variable names, original -> here:
#   input / res            -> h          (residual stream)
#   in_normed              -> x          (normed input to attention)
#   batch_len, seq_len     -> B, S
#   num_atten_heads        -> n_heads    (24 for Llama-3.2-3B)
#   hidden_dim             -> d_model    (3072)
#   K_num_heads / V_num_heads -> n_kv    (8)
#   G_K / G_V              -> group      (24 // 8 = 3)
#   Q, K, V                -> q, k, v
#   unmasked_att           -> scores
#   casual_att             -> attn
#   final_atten_out        -> x          (normed input to the MLP, reuses the name)
#   gt_proj * up_proj ...  -> one line: mlp = down_proj(silu(gate) * up)
#
# scripts/check_against_original.py runs both versions on the real model and
# checks that every layer's attention and residual are bit-for-bit identical.
# =============================================================================

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _head_dim(args) -> int:
    # Size of one attention head (128 for Llama-3.2-3B).
    # Original: head_dim = hidden_dim // num_atten_heads   (3072 // 24 = 128)
    # Some model configs store head_dim explicitly (llama_live_vis.py read
    # self.model.args.head_dim); use it if present, otherwise compute it the
    # original way. For Llama-3.2-3B both give 128.
    hd = getattr(args, "head_dim", None)
    return hd if hd else args.hidden_size // args.num_attention_heads


def layer_forward(model, h: mx.array, layer_idx: int) -> tuple[mx.array, mx.array]:
    """Run decoder block ``layer_idx`` on residual stream ``h``.

    Args:
        model: an ``mlx_lm`` Llama ``Model`` (as returned by ``mlx_lm.load``).
        h: residual stream, shape ``(batch, seq, d_model)``.
        layer_idx: which decoder block to run.

    Returns:
        ``(h_out, attn)`` where ``h_out`` has the same shape as ``h`` and
        ``attn`` is the causal attention probability tensor of shape
        ``(batch, n_heads, seq, seq)``.
    """
    # Equivalent to: TylerScope.manually_run_foward(self, input, layer_num)
    # Shapes below use Llama-3.2-3B numbers: B=1 (one prompt at a time),
    # S = number of tokens, d_model = 3072, 24 query heads, 8 KV heads, head_dim 128.

    # Grab this layer's weights and the model's sizes.
    # Original: curr_layer = self.model.layers[layer_num]
    layer = model.layers[layer_idx]
    # Original: num_atten_heads = self.model.args.num_attention_heads   (24)
    n_heads = model.args.num_attention_heads
    # Original: hidden_dim = self.model.args.hidden_size                (3072)
    d_model = model.args.hidden_size
    # Original: head_dim = hidden_dim // num_atten_heads                (128)
    head_dim = _head_dim(model.args)
    # (Original also built swGelu = mlx.nn.SiLU() here; this file calls nn.silu
    #  directly in the MLP line below, same function.)

    # --- attention ---------------------------------------------------------
    # Pre-attention RMSNorm. h itself is NOT changed; it is kept for the residual add.
    # Original: res = input
    #           in_normed = curr_layer.input_layernorm(res)
    # x: (B, S, 3072)
    x = layer.input_layernorm(h)
    # Original: batch_len,seq_len,_ = Q.shape   (read from Q there, from x here; same B and S)
    B, S, _ = x.shape

    # QUERIES, three steps in one line:
    #   q_proj(x)                          (B, S, 3072)       project
    #   .reshape(B, S, n_heads, head_dim)  (B, S, 24, 128)    split 3072 into 24 heads x 128 (no data moves)
    #   .transpose(0, 2, 1, 3)             (B, 24, S, 128)    heads in front of seq so matmul batches over heads
    # Original: Q = curr_layer.self_attn.q_proj(in_normed)
    #           Q = Q.reshape(batch_len,seq_len,num_atten_heads,hidden_dim//num_atten_heads).transpose(0,2,1,3)
    q = layer.self_attn.q_proj(x).reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
    # KEYS and VALUES: projected to only 1024 dims (8 heads x 128), not 3072.
    # That is grouped-query attention (GQA). Shapes here: (B, S, 1024).
    # Original: K = curr_layer.self_attn.k_proj(in_normed)
    k = layer.self_attn.k_proj(x)
    # Original: V = curr_layer.self_attn.v_proj(in_normed)
    #   (the original computed V before K; order doesn't matter, they are independent)
    v = layer.self_attn.v_proj(x)
    # How many KV heads exist: 1024 // 128 = 8.
    # Original: _,_,K_hidden_dim = K.shape
    #           K_num_heads = K_hidden_dim//head_dim      (and the same for V_num_heads)
    n_kv = k.shape[-1] // head_dim
    # How many query heads share each KV head: 24 // 8 = 3.
    # Original: G_K = num_atten_heads // K.shape[2]       (and G_V for V)
    group = n_heads // n_kv
    # Grouped-query attention: each KV head is shared by ``group`` query heads.
    # Three steps per line:
    #   .reshape(B, S, n_kv, head_dim)     (B, S, 8, 128)     split 1024 into 8 KV heads
    #   mx.repeat(..., group, axis=2)      (B, S, 24, 128)    copy each KV head 3x IN PLACE:
    #                                                         [k0,k0,k0,k1,k1,k1,...,k7,k7,k7]
    #                                                         so query heads 0-2 use k0, 3-5 use k1, ...
    #                                                         (tracked head 17 uses KV head 17 // 3 = 5)
    #   .transpose(0, 2, 1, 3)             (B, 24, S, 128)    same reorder as q
    # Original (4 lines, same result):
    #           K = K.reshape(batch_len,seq_len,K_num_heads,head_dim)          (B, S, 8, 128)
    #           K = mx.expand_dims(K,axis=-2)                                  (B, S, 8, 1, 128)
    #           K = mx.broadcast_to(K,(batch_len,seq_len,K_num_heads,G_K,head_dim))  (B, S, 8, 3, 128)
    #           K = K.reshape(batch_len,seq_len,(K_num_heads*G_K),head_dim).transpose(0,2,1,3)
    #   Flattening the (8, 3) axes in the last reshape gives the same k0,k0,k0,k1,... order
    #   that mx.repeat produces, so the values are identical.
    k = mx.repeat(k.reshape(B, S, n_kv, head_dim), group, axis=2).transpose(0, 2, 1, 3)
    # Same as the line above, for values.
    # Original: the four V lines (reshape, expand_dims, broadcast_to, reshape+transpose) with G_V.
    v = mx.repeat(v.reshape(B, S, n_kv, head_dim), group, axis=2).transpose(0, 2, 1, 3)

    # Rotary position embedding: rotate q and k by an angle that depends on each
    # token's position, so q.k depends on relative position. v is not rotated.
    # Original: # Now we apply the rope positional system.
    #           Q = curr_layer.self_attn.rope(Q)
    #           K = curr_layer.self_attn.rope(K)
    q = layer.self_attn.rope(q)
    k = layer.self_attn.rope(k)

    # NOTE: dividing by an mx.array scalar and adding the float32 causal mask
    # promotes the scores (and everything downstream, including the residual
    # stream) to float32. This matches the original research code exactly, so
    # the saved arrays are reproducible; it differs slightly from mlx_lm's own
    # bf16 fused-attention forward.
    # Attention scores for all 24 heads at once:
    #   k.transpose(0, 1, 3, 2)            (B, 24, 128, S)    K^T per head
    #   q @ K^T                            (B, 24, S, S)      row i = how much token i's query matches every key
    #   / sqrt(head_dim)                   scale by sqrt(128) so the softmax isn't too peaked
    # Original: unmasked_att = Q @ K.transpose(0,1,3,2)
    #           unmasked_att = unmasked_att / mx.sqrt(head_dim)
    scores = (q @ k.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(head_dim))
    # Causal mask, (S, S): 0 on and below the diagonal, a huge negative number
    # above it, so no token can attend to tokens after it. Broadcasts over B and heads.
    # Original: mask = mlx.nn.MultiHeadAttention.create_additive_causal_mask(unmasked_att.shape[-1])
    mask = nn.MultiHeadAttention.create_additive_causal_mask(S)
    # Softmax over the key axis: every row now sums to 1 and the masked (future)
    # entries become 0. THIS is the attention map that gets visualised.
    # attn: (B, 24, S, S)
    # Original: casual_att = mx.softmax((unmasked_att+mask),axis=-1)
    attn = mx.softmax(scores + mask, axis=-1)

    # Weighted sum of values, then undo the head split:
    #   attn @ v                           (B, 24, S, 128)    each token's mix of values, per head
    #   .transpose(0, 2, 1, 3)             (B, S, 24, 128)    seq back in front
    #   .reshape(B, S, d_model)            (B, S, 3072)       glue the 24 heads back together
    # Original: atten_out = casual_att @ V
    #           atten_out  = atten_out.transpose(0,2,1,3).reshape(batch_len,seq_len,hidden_dim)
    attn_out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, d_model)
    # Output projection (mixes the heads), then the first residual add.
    # Original: atten_out = curr_layer.self_attn.o_proj(atten_out)
    #           res = res + atten_out
    h = h + layer.self_attn.o_proj(attn_out)

    # --- SwiGLU MLP --------------------------------------------------------
    # Post-attention RMSNorm; h is kept for the second residual add.
    # Original: final_atten_out = curr_layer.post_attention_layernorm(res)
    x = layer.post_attention_layernorm(h)
    # SwiGLU MLP in one line, read inside-out:
    #   layer.mlp.gate_proj(x)             (B, S, 8192)   gate projection
    #   nn.silu(...)                       SiLU activation on the gate
    #   layer.mlp.up_proj(x)               (B, S, 8192)   up projection (parallel to the gate)
    #   silu(gate) * up                    elementwise product
    #   layer.mlp.down_proj(...)           (B, S, 3072)   back down to model width
    # Original: gt_proj = curr_layer.mlp.gate_proj(final_atten_out)
    #           up_proj = curr_layer.mlp.up_proj(final_atten_out)
    #           mlp_hidden = swGelu(gt_proj) * up_proj        (swGelu = mlx.nn.SiLU())
    #           mlp_out = curr_layer.mlp.down_proj(mlp_hidden)
    mlp = layer.mlp.down_proj(nn.silu(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x))
    # Second residual add. h is the layer's output and the next layer's input
    # (the next layer norms it first).
    # Original: res = res + mlp_out
    h = h + mlp

    # Original: return res,casual_att
    return h, attn


def embed(model, tokenizer, text: str) -> mx.array:
    """Tokenize ``text`` (tokenizer defaults, i.e. with BOS) and embed it."""
    # Equivalent to: TylerScope.tokenize_input(self, prompt)
    # String -> list of token ids -> (1, S) array. [None, :] adds the batch axis
    # of size 1 (that is the B in layer_forward).
    # Original: tokenized_in = mx.expand_dims(mx.array(self.tokenizer.encode(prompt)),axis=0)
    ids = mx.array(tokenizer.encode(text))[None, :]
    # Look up each id's embedding vector: (1, S) -> (1, S, 3072). This is the
    # residual stream fed to layer 0.
    # Original: embed_tokens = self.model.model.embed_tokens(tokenized_in)
    return model.model.embed_tokens(ids)


def unembed(model, h_normed: mx.array) -> mx.array:
    """Project final-norm hidden states to vocabulary logits.

    Llama-3.2-3B ties input/output embeddings (no ``lm_head``); larger Llamas
    have a separate ``lm_head``. Handle both.
    """
    # Equivalent to the next-token logits in gen_from_prompt. There were two
    # original versions:
    #   llama_temp_vis.ipynb: self.model.lm_head(mlp_per_layer_mone[-1,:])
    #                         (works for models WITH an lm_head, e.g. the 8B)
    #   llama_live_vis.py:    mx.matmul(mlp_per_layer_mone[-1,:], self.model.model.embed_tokens.weight.T)
    #                         (works for Llama-3.2-3B, which reuses the embedding matrix)
    # Output: logits over the vocabulary, (128256,) for one token.
    if hasattr(model, "lm_head"):
        # Model has a separate output matrix: same as the llama_temp_vis line.
        return model.lm_head(h_normed)
    # Tied embeddings (Llama-3.2-3B): multiply by the embedding matrix transposed.
    # as_linear(x) computes x @ embed_tokens.weight.T, same as the llama_live_vis line.
    return model.model.embed_tokens.as_linear(h_normed)
