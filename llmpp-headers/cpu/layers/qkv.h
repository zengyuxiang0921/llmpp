//
// QKV projection implementation ported from llama.cpp
// (src/llama-graph.cpp:1167)
//
// Reference:
//   https://github.com/ggml-org/llama.cpp
//

#ifndef LLMPP_QKV_H
#define LLMPP_QKV_H

#include "types.h"

//
// Result structure for QKV projections
//
struct llm_qkv_result {
    struct ggml_tensor * Qcur;
    struct ggml_tensor * Kcur;
    struct ggml_tensor * Vcur;
};

//
// Build Q, K, V projections with optional bias and clamping.
// Supports both fused wqkv and separate wq/wk/wv paths.
//
// Ported from llm_graph_context::build_qkv (llama-graph.cpp:1167)
//
static inline struct llm_qkv_result llm_qkv_build(
        struct ggml_context * ctx0,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * wqkv,
        struct ggml_tensor  * wqkv_b,
        struct ggml_tensor  * wqkv_s,
        struct ggml_tensor  * wq,
        struct ggml_tensor  * wq_b,
        struct ggml_tensor  * wq_s,
        struct ggml_tensor  * wk,
        struct ggml_tensor  * wk_b,
        struct ggml_tensor  * wk_s,
        struct ggml_tensor  * wv,
        struct ggml_tensor  * wv_b,
        struct ggml_tensor  * wv_s,
        int64_t               n_embd_head,
        int64_t               n_head,
        int64_t               n_head_kv,
        int64_t               n_tokens,
        float                 f_clamp_kqv = 0.0f,
        llm_ffn_cb            cb = nullptr) {

    const int64_t n_embd_q  = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;

    struct ggml_tensor * Qcur, * Kcur, * Vcur;

    if (wqkv) {
        // fused QKV path
        struct ggml_tensor * qkv = llm_build_lora_mm(ctx0, wqkv, cur, wqkv_s);
        if (cb) cb(qkv, "wqkv", -1);
        if (wqkv_b) {
            qkv = ggml_add(ctx0, qkv, wqkv_b);
            if (cb) cb(qkv, "wqkv_b", -1);
        }
        if (f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -f_clamp_kqv, f_clamp_kqv);
            if (cb) cb(qkv, "wqkv_clamped", -1);
        }
        Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
        Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q));
        Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q + n_embd_kv));
    } else {
        // separate Q/K/V path
        Qcur = llm_build_lora_mm(ctx0, wq, cur, wq_s);
        if (cb) cb(Qcur, "Qcur", -1);
        if (wq_b) {
            Qcur = ggml_add(ctx0, Qcur, wq_b);
            if (cb) cb(Qcur, "Qcur", -1);
        }
        if (f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -f_clamp_kqv, f_clamp_kqv);
            if (cb) cb(Qcur, "Qcur_clamped", -1);
        }

        Kcur = llm_build_lora_mm(ctx0, wk, cur, wk_s);
        if (cb) cb(Kcur, "Kcur", -1);
        if (wk_b) {
            Kcur = ggml_add(ctx0, Kcur, wk_b);
            if (cb) cb(Kcur, "Kcur", -1);
        }
        if (f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -f_clamp_kqv, f_clamp_kqv);
            if (cb) cb(Kcur, "Kcur_clamped", -1);
        }

        Vcur = llm_build_lora_mm(ctx0, wv, cur, wv_s);
        if (cb) cb(Vcur, "Vcur", -1);
        if (wv_b) {
            Vcur = ggml_add(ctx0, Vcur, wv_b);
            if (cb) cb(Vcur, "Vcur", -1);
        }
        if (f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -f_clamp_kqv, f_clamp_kqv);
            if (cb) cb(Vcur, "Vcur_clamped", -1);
        }

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    }

    if (cb) cb(Qcur, "Qcur", -1);
    if (cb) cb(Kcur, "Kcur", -1);
    if (cb) cb(Vcur, "Vcur", -1);

    return { Qcur, Kcur, Vcur };
}

#endif // LLMPP_QKV_H
