//
// Input embedding and position helper implementations ported from llama.cpp
// (src/llama-graph.cpp)
//
// Reference:
//   https://github.com/ggml-org/llama.cpp
//

#ifndef LLMPP_INP_H
#define LLMPP_INP_H

#include "types.h"

//
// Build input token embeddings with optional embedding scale.
// Ported from llm_graph_context::build_inp_embd (llama-graph.cpp:1807)
//
static inline struct ggml_tensor * llm_inp_build_embd(
        struct ggml_context * ctx0,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * tok_embd,
        struct ggml_tensor  * tokens,
        int64_t               n_embd,
        float                 f_embedding_scale = 0.0f,
        llm_ffn_cb            cb = nullptr) {

    struct ggml_tensor * cur = ggml_get_rows(ctx0, tok_embd, tokens);

    if (f_embedding_scale != 0.0f) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, f_embedding_scale);
    }

    if (cb) cb(cur, "embd", -1);

    ggml_build_forward_expand(gf, cur);

    return cur;
}

//
// Build input position tensor (1D I32 tokens).
// Ported from llm_graph_context::build_inp_pos (llama-graph.cpp:1896)
//
static inline struct ggml_tensor * llm_inp_build_pos(
        struct ggml_context * ctx0,
        int64_t               n_tokens,
        int64_t               n_pos_per_embd = 1) {

    return ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * n_pos_per_embd);
}

//
// Build position bias from bucket indices.
// Ported from llm_graph_context::build_pos_bias (llama-graph.cpp:2025)
//
static inline struct ggml_tensor * llm_inp_build_pos_bias(
        struct ggml_context * ctx0,
        struct ggml_tensor  * pos_bucket,
        struct ggml_tensor  * attn_rel_b,
        llm_ffn_cb            cb = nullptr) {

    struct ggml_tensor * pos_bucket_1d = ggml_reshape_1d(
        ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    if (cb) cb(pos_bucket_1d, "pos_bucket_1d", -1);

    struct ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute(ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont(ctx0, pos_bias);

    if (cb) cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

#endif // LLMPP_INP_H
