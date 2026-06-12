//
// Layer normalization implementation ported from llama.cpp
// (src/llama-graph.cpp:1131)
//
// Reference:
//   https://github.com/ggml-org/llama.cpp
//

#ifndef LLMPP_NORM_H
#define LLMPP_NORM_H

#include "types.h"

//
// Build normalization: LayerNorm, RMSNorm, or GroupNorm.
// Applies weight (mw) and bias (mb) after normalization.
//
// Ported from llm_graph_context::build_norm (llama-graph.cpp:1131)
//
static inline struct ggml_tensor * llm_norm_build(
        struct ggml_context * ctx0,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * mw,
        struct ggml_tensor  * mb,
        llm_norm_type         type,
        int                   il,
        float                 f_norm_eps      = 1e-5f,
        float                 f_norm_rms_eps  = 1e-5f,
        int32_t               n_norm_groups   = 0,
        float                 f_norm_group_eps = 1e-5f,
        llm_ffn_cb            cb = nullptr) {

    switch (type) {
        case LLM_NORM:
            cur = ggml_norm(ctx0, cur, f_norm_eps);
            break;
        case LLM_NORM_RMS:
            cur = ggml_rms_norm(ctx0, cur, f_norm_rms_eps);
            break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, n_norm_groups, f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        if (cb) cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            if (cb) cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}

#endif // LLMPP_NORM_H
