# Qwen3.8-Flash-Next: the exact forward pass

This is the spec the engine implements. It was derived from the checkpoint
(`/main-storage/llms/qwen3.8-flash-next-awq`: `config.json` and safetensors
headers) and the reference implementation in
[leapdragon/vllm-rdna2-qwen](https://github.com/leapdragon/vllm-rdna2-qwen)
(`vllm/models/qwen4_exp/`, Apache-2.0), commit `4eddd61`. When this file and
the checkpoint disagree, the checkpoint wins and this file has a bug.

Checkpoint tensor names are given without the `model.language_model.` prefix.
All norms use eps = `rms_norm_eps` = 1e-6.

## Shapes

| Symbol | Value | Source |
|---|---|---|
| `H` hidden | 2560 | `hidden_size` |
| `HC` hyper-connection streams | 4 | `hc_count` |
| `R` HC low rank | 320 | `hc_lowrank` |
| layers | 48; layer `i` is QSA (full attention) when `i % 4 == 3`, else GDN | `layer_types` |
| vocab | 248320, untied `lm_head` | |
| QSA | 24 q heads, 2 kv heads, head_dim 256, output gate, rotary_dim 64 | |
| QSA indexer | 4 heads x 128, 1 key head, budget 2048 tokens, compress ratio 4 | `indexer_*` |
| GDN | 16 qk heads x 128, 48 v heads x 128, conv kernel 4 | `linear_*` |
| MoE | 512 experts, top-10, renormalized, intermediate 640; 1 shared expert (640) with a sigmoid gate | |
| PLE | at layer 1 only (`ple_layer_ids: [2]` is 1-based); 16 n-gram heads x 160; 320,001,536 table rows | |
| RoPE | NeoX style, theta 1e7, first 64 dims of each head | `rope_parameters` |

For text-only input the MRoPE sections `[11,11,10]` all see the same position,
so they reduce to plain 1-D RoPE. Vision is out of scope for now.

## Weights and quantization

- Routed experts (`layers.L.mlp.experts.E.{gate,up,down}_proj`): compressed-tensors
  `pack-quantized`, symmetric int4, group 128 along the input dim.
  - `weight_packed`: int32 `[out, in/8]`. Word `c` holds input columns `8c..8c+7`,
    element `8c+i` in bits `4i..4i+3`, stored as `q + 8` (unsigned nibble).
  - `weight_scale`: bf16 `[out, in/128]`. The weight is `(nibble - 8) * scale`.
- Everything else is bf16 (converted to fp16/fp32 at load; gfx1030 has no bf16).
  That includes the GDN, QSA, indexer, HC, router, shared expert, PLE projections,
  embeddings, `lm_head` and the whole MTP module (`model_mtp.safetensors`).
- The PLE n-gram table is not in the AWQ checkpoint. It is the int4 sidecar
  `/main-storage/llms/qwen3.8-flash-next-ple/ples_int4/`: 128 shards of 2,500,012
  rows. Each has `weight_i4` u8 `[rows, 80]` (low nibble first) and `weight_scale`
  f16 `[rows, 10]` (one per 16 values). The value is `(nibble - 8) * scale`. Row
  `r` lives in shard `r / 2500012`.

## Notation

- `rms(x)` = `x / sqrt(mean(x^2) + eps)`.
- `gemma_norm(x, w)` = `rms(x) * (1 + w)`. It is used for HC, QSA q/k, the
  indexer and PLE.
- `norm(x, w)` = `rms(x) * w`. It is used only for the GDN gated output norm.
- `grouped_gemma_norm(X, w)` for `X` of width `HC*H`: `gemma_norm` applied to
  each `H`-sized stream independently, with `w` of width `HC*H`.
- The multi-stream state `X` has width `HC*H` = 10240, with stream-major layout
  (stream `s` = elements `s*H .. s*H+H-1`).

## Top level

```
X = repeat(embed_tokens[token], HC)             # [10240]
pending = none
for L in 0..47:
    X, pending = decoder_layer(L, X, pending)
Xf = combine(X, pending)                        # materialize the last MLP output
y  = hc_mix(hyper_connection_mixer, Xf)         # [2560], final mixer has no inject
logits = lm_head @ y
```

## Hyper-connection (`attn_hyper_connection`, `mlp_hyper_connection`, `hyper_connection_mixer`)

Weights: `hc_norm.weight [10240]`, `input_mix_weight_down [320, 10240]`,
`input_mix_weight_up [10240, 320]`, and (except the final mixer)
`block_inject_weight [4, 10240]`.

```
hc_mix(X):
    xn   = grouped_gemma_norm(X, hc_norm)                  # [10240]
    d    = input_mix_weight_down @ xn                      # [320]
    inj  = block_inject_weight @ xn                        # [4] injection logits
    u    = silu(d / HC)
    g    = sigmoid(input_mix_weight_up @ u)                # [10240]
    block_in[j] = mean_s( g[s*H+j] * xn[s*H+j] )           # [2560]
    return block_in, inj

combine(X, (block_out, inj)):
    X[s*H+j] += block_out[j] * 2*sigmoid(inj[s] / HC)
```

A layer's combine is independent of any weights (it only needs `inj`), so
implementations are free to defer it and fuse it into the next `hc_mix`'s norm.

## Decoder layer

```
decoder_layer(L, X, pending):
    X = combine(X, pending)
    if L == 1:  X += ple(X, tokens)
    a_in, a_inj = hc_mix(attn_hyper_connection[L], X)
    a_out = gdn(L, a_in)  if L % 4 != 3  else  qsa(L, a_in)
    X = combine(X, (a_out, a_inj))
    m_in, m_inj = hc_mix(mlp_hyper_connection[L], X)
    m_out = moe(L, m_in)
    return X, (m_out, m_inj)
```

## GDN: gated delta net (`linear_attn`, 36 layers)

Weights: `in_proj_qkv [10240, 2560]` (rows: q 16x128 | k 16x128 | v 48x128),
`in_proj_z [6144, 2560]`, `in_proj_b [48, 2560]`, `in_proj_a [48, 2560]`,
`conv1d.weight [10240, 1, 4]`, `A_log [48]`, `dt_bias [48]`, `norm.weight [128]`,
`out_proj [2560, 6144]`.

State per sequence: a conv window (the last 3 pre-conv `qkv` rows, `[10240, 3]`)
and a recurrent `S[48][128 k][128 v]` in fp32 (the checkpoint's
`mamba_ssm_dtype`).

```
qkv = in_proj_qkv @ x;  z = in_proj_z @ x;  b = in_proj_b @ x;  a = in_proj_a @ x
qkv = silu(causal_depthwise_conv(qkv, conv1d, kernel 4))   # uses the previous 3 rows
q, k, v = split(qkv)                           # q,k: 16 heads, v: 48 heads
q = l2norm(q) / sqrt(128);  k = l2norm(k)      # per head, eps 1e-6
g    = -exp(A_log) * softplus(a + dt_bias)     # [48]; softplus threshold 20
beta = sigmoid(b)                              # [48]
for v-head h (key head kh = h / 3):
    S[h] *= exp(g[h])
    u     = (v[h] - S[h]^T k[kh]) * beta[h]    # [128]
    S[h] += k[kh] outer u                      # S[k][v]
    o[h]  = S[h]^T q[kh]                       # [128]
o[h] = norm(o[h], norm.weight) * sigmoid(z[h])   # output_gate_type = sigmoid
out  = out_proj @ o
```

## QSA: sparse attention with a learned indexer (`self_attn`, 12 layers)

Weights: `q_proj [12288, 2560]` (per head: 256 query rows then 256 gate rows),
`k_proj [512, 2560]`, `v_proj [512, 2560]`, `q_norm [256]`, `k_norm [256]`,
`o_proj [2560, 6144]`, `indexer.index_qk_proj [640, 2560]`,
`indexer.q_layernorm [128]`, `indexer.k_layernorm [128]`.

Per sequence and layer the cache holds K and V (2 heads x 256 per token), the
indexer's raw keys (128 per token) and compressed keys (128 per complete group of
4 tokens).

```
qg = q_proj @ x;  q[h] = qg[h*512 .. +256];  gate[h] = qg[h*512+256 .. +256]
k = k_proj @ x;  v = v_proj @ x
q[h] = rope(gemma_norm(q[h], q_norm), pos);  k[j] = rope(gemma_norm(k[j], k_norm), pos)

# indexer
iq = index_qk_proj @ x                          # [640] = 4 query heads | 1 key
iq_h  = rope(gemma_norm(iq[h*128..], q_layernorm), pos)     # h = 0..3
raw_k[pos] = iq[512..640]                       # stored un-normalized
when (pos+1) % 4 == 0:                           # group c = pos/4 is complete
    ck[c] = rope(gemma_norm(mean(raw_k[4c..4c+3]), k_layernorm), 4c)

# selection for a query at position p
nb = (p+1) / 4                                  # complete groups visible
score[c] = sum_h relu(iq_h . ck[c]) / sqrt(128), c < nb
blocks = top-512 of score (all of them if nb <= 512)
tokens = { 4c..4c+3 : c in blocks } U { 4*nb .. p }   # plus the open tail group

# attention over the selected tokens only; q head h uses kv head h / 12
o[h] = softmax_t( q[h] . K[t] / 16 ) . V[t]
out  = o_proj @ (o * sigmoid(gate))
```

`rope` rotates only the first 64 dims, NeoX style (pairs `i` and `i+32`),
`inv_freq[i] = theta^(-2i/64)`. It uses the same rotary table for the 128-dim
indexer heads.

With 2048 or fewer visible tokens every token is selected, and QSA is exactly
causal attention.

## MoE (`mlp`, all 48 layers)

```
logits = gate @ x                               # [512]
p = softmax(logits);  top = top10(p);  w = p[top] / sum(p[top])
y = sum_e w_e * down_e( silu(gate_e @ x) * (up_e @ x) )
y += sigmoid(shared_expert_gate @ x) * shared_down( silu(shared_gate @ x) * (shared_up @ x) )
```

## PLE: per-layer n-gram embedding (layer 1 only)

Weights: `ple.key_proj [10240, 2560]`, `ple.value_proj [2560, 2560]`,
`ple.norm_key/norm_query/norm_conv [10240]` (grouped Gemma norms, per 2560
stream), `ple.conv1d.weight [10240, 1, 4]` (depthwise, dilation 3, causal),
plus the n-gram table.

N-gram ids for token position `p` (16 heads = 2 n-gram orders x 8):

```
ctx[p-s] for s = 0,1,2, or EOS (248044) when p-s is before the start of the
sequence or before the most recent EOS at or before p-1 (EOS starts a segment).
mult[i]   = 2*(splitmix64(1234 + 10007*0 + GAMMA*(i+1)) % half) + 1,  i = 0..2
            half = max(1, ((2^63-1) / 248320) / 2)
size[h]   = the (h+1)-th prime > 20,000,000 - 1;  offset[h] = sum of earlier sizes
order n in {2,3}, heads h = (n-2)*8 .. +8:
    mixed = XOR_{i<n} ( ctx[p-i] * mult[i] )    # 64-bit
    id[h] = mixed % size[h] + offset[h]
e = concat_h table[id[h]]                        # [2560] = 16 x 160
```

Then:

```
key   = grouped_gemma_norm(key_proj @ e, norm_key)            # [4 x 2560]
query = grouped_gemma_norm(X, norm_query)
gs    = (key[s] . query[s]) / sqrt(2560)
gate  = sigmoid( sign(gs) * sqrt(max(|gs|, 1e-6)) )
gv[s] = gate[s] * (value_proj @ e)                            # [4 x 2560]
c     = silu(dilated_causal_conv(grouped_gemma_norm(gv, norm_conv)))  # taps p, p-3, p-6, p-9
ple   = gv + c                                                # added to X
```

The conv keeps the last 9 inputs as per-sequence state. The n-gram context
across chunk boundaries is the previous 2 tokens.

## MTP (one extra layer; phase 5)

`e = fc_embedding @ gemma_norm(embed(t+1), pre_fc_norm_embedding)`; the backbone's
pre-final-mixer multi-stream `X` is normed with `pre_fc_norm_hidden` (grouped),
each stream passes through `fc_hidden`, and `e` is added to every stream. Then a
QSA decoder layer (with bf16 routed experts, fused `gate_up_proj`) runs, and the
MTP's own final mixer feeds the shared `lm_head`.
