//
// Multi-head attention implementation ported from llama.cpp
// (src/llama-graph.cpp:2040)
//
// Reference:
//   https://github.com/ggml-org/llama.cpp
//

#ifndef LLMPP_ATTENTION_H
#define LLMPP_ATTENTION_H

#include "types.h"

//
// Build multi-head attention: Q/K/V projection, softmax, weighted sum.
// Supports:
//   - Flash attention (via cparams.flash_attn)
//   - ALiBi bias
//   - Attention logit softcapping (Grok, etc.)
//   - KQ bias
//   - MLA (Multi-head Latent Attention) via v_mla
//   - Sinks (for window attention)
//
// Ported from llm_graph_context::build_attn_mha (llama-graph.cpp:2040)
//
static inline struct ggml_tensor * llm_attn_build_mha(
        struct ggml_context * ctx0,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * kq_b,
        struct ggml_tensor  * kq_mask,
        struct ggml_tensor  * sinks,
        struct ggml_tensor  * v_mla,
        float                 kq_scale,
        int                   il,
        bool                  flash_attn      = false,
        float                 f_max_alibi_bias = 0.0f,
        float                 f_attn_logit_softcapping = 0.0f,
        bool                  attn_soft_cap    = false,
        llm_arch              arch             = LLM_ARCH_UNKNOWN,
        llm_ffn_cb            cb = nullptr) {

    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream,
                     q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    struct ggml_tensor * cur;

    if (flash_attn) {
        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale,
                                  f_max_alibi_bias,
                                  attn_soft_cap ? f_attn_logit_softcapping : 0.0f);
        if (cb) cb(cur, "fattn", il);

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        if (v_mla) {
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            if (cb) cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur);
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        if (cb) cb(kq, "kq", il);

        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, f_attn_logit_softcapping != 0.0f
                ? 1.0f / f_attn_logit_softcapping : 1.0f));
            if (cb) cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, f_attn_logit_softcapping);
            if (cb) cb(kq, "kq_scaled", il);
        }

        if (attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / f_attn_logit_softcapping);
            if (cb) cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh(ctx0, kq);
            if (cb) cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, f_attn_logit_softcapping);
            if (cb) cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            if (cb) cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        if (cb) cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            if (cb) cb(v, "v_cont", il);
        }

        struct ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        if (cb) cb(kqv, "kqv", il);

        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            if (cb) cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

//
// Build full attention: MHA + output projection (wo).
// Simplified standalone version without KV cache integration.
//
// Ported from llm_graph_context::build_attn (no-cache variant, llama-graph.cpp:2200)
//
static inline struct ggml_tensor * llm_attn_build(
        struct ggml_context * ctx0,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * wo,
        struct ggml_tensor  * wo_b,
        struct ggml_tensor  * wo_s,
        struct ggml_tensor  * q_cur,
        struct ggml_tensor  * k_cur,
        struct ggml_tensor  * v_cur,
        struct ggml_tensor  * kq_b,
        struct ggml_tensor  * sinks,
        struct ggml_tensor  * v_mla,
        float                 kq_scale,
        int                   il,
        struct ggml_tensor  * kq_mask = nullptr,
        bool                  flash_attn = false,
        float                 f_max_alibi_bias = 0.0f,
        float                 f_attn_logit_softcapping = 0.0f,
        bool                  attn_soft_cap = false,
        llm_arch              arch = LLM_ARCH_UNKNOWN,
        llm_ffn_cb            cb = nullptr) {

    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    struct ggml_tensor * cur = llm_attn_build_mha(
        ctx0, gf,
        q_cur, k_cur, v_cur,
        kq_b, kq_mask, sinks, v_mla, kq_scale, il,
        flash_attn, f_max_alibi_bias, f_attn_logit_softcapping, attn_soft_cap, arch, cb);
    if (cb) cb(cur, "kqv_out", il);

    if (wo) {
        cur = llm_build_lora_mm(ctx0, wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

#endif // LLMPP_ATTENTION_H
