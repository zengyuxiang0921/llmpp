//
// FFN implementation ported from llama.cpp (src/llama-graph.cpp:1244)
//
// Reference:
//   https://github.com/ggml-org/llama.cpp
//

#ifndef LLMPP_FFN_H
#define LLMPP_FFN_H

#include "types.h"

//
// Build FFN: up / gate / down projections with optional activation and bias.
// Handles SiLU, GELU, ReLU, SwiGLU, GeGLU, ReGLU activations.
// Supports both sequential (LLM_FFN_SEQ) and parallel (LLM_FFN_PAR) gate layouts.
//
// Ported from llm_graph_context::build_ffn (llama-graph.cpp:1244)
//
static inline struct ggml_tensor * llm_ffn_build(
        struct ggml_context * ctx0,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * up,
        struct ggml_tensor  * up_b,
        struct ggml_tensor  * up_s,
        struct ggml_tensor  * gate,
        struct ggml_tensor  * gate_b,
        struct ggml_tensor  * gate_s,
        struct ggml_tensor  * down,
        struct ggml_tensor  * down_b,
        struct ggml_tensor  * down_s,
        struct ggml_tensor  * act_scales,
        llm_ffn_op_type       type_op,
        llm_ffn_gate_type     type_gate,
        int                   il,
        llm_arch              arch = LLM_ARCH_UNKNOWN,
        const llm_ffn_clamp_params * clamp = nullptr,
        llm_ffn_cb            cb = nullptr) {

    // up projection
    struct ggml_tensor * tmp = up ? llm_build_lora_mm(ctx0, up, cur) : cur;
    if (cb) cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        if (cb) cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        if (cb) cb(tmp, "ffn_up_s", il);
    }

    // gate projection
    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = llm_build_lora_mm(ctx0, gate, tmp);
                    if (cb) cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = llm_build_lora_mm(ctx0, gate, cur);
                    if (cb) cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            if (cb) cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            if (cb) cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    // activation function
    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                // Step35: clamp gate (after SiLU) and up before multiplication
                if (arch == LLM_ARCH_STEP35 && il >= 0 && clamp) {
                    const float limit = clamp->swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        struct ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                        if (cb) cb(gate_act, "ffn_silu", il);
                        gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                        if (cb) cb(gate_act, "ffn_silu_clamped", il);

                        tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                        if (cb) cb(tmp, "ffn_up_clamped", il);

                        cur = ggml_mul(ctx0, gate_act, tmp);
                        if (cb) cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                if (cb) cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                if (cb) cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                if (cb) cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                if (cb) cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    if (cb) cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                if (cb) cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                if (cb) cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                if (cb) cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                if (cb) cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                if (cb) cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                if (cb) cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                if (cb) cb(cur, "ffn_reglu", il);
            } break;
        default:
            GGML_ABORT("fatal error: unknown FFN op type");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        if (cb) cb(cur, "ffn_gate_par", il);
    }

    // down projection
    if (down) {
        cur = llm_build_lora_mm(ctx0, down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        if (cb) cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        if (cb) cb(cur, "ffn_down_s", il);
    }

    return cur;
}

#endif // LLMPP_FFN_H
