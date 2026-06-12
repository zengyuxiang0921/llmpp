//
// Mixture of Experts FFN implementation ported from llama.cpp
// (src/llama-graph.cpp:1408)
//
// Reference:
//   https://github.com/ggml-org/llama.cpp
//

#ifndef LLMPP_MOE_FFN_H
#define LLMPP_MOE_FFN_H

#include "types.h"

//
// Build MoE FFN: gating, top-k expert selection, per-expert FFN, weighted sum.
//
// Supports:
//   - Merged gate_up or separate gate/up weights
//   - Softmax, sigmoid, or sofmax-weighted gating
//   - Expert group routing (DeepSeek-style)
//   - Per-expert bias and scale
//   - Various activation functions
//   - Step35 per-layer clamping
//
// Ported from llm_graph_context::build_moe_ffn (llama-graph.cpp:1450)
//
static inline struct ggml_tensor * llm_moe_ffn_build(
        struct ggml_context * ctx0,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * gate_inp,
        struct ggml_tensor  * gate_inp_b,
        struct ggml_tensor  * up_exps,
        struct ggml_tensor  * up_exps_b,
        struct ggml_tensor  * gate_exps,
        struct ggml_tensor  * gate_exps_b,
        struct ggml_tensor  * down_exps,
        struct ggml_tensor  * down_exps_b,
        struct ggml_tensor  * exp_probs_b,
        int64_t               n_expert,
        int64_t               n_expert_used,
        llm_ffn_op_type       type_op,
        bool                  norm_w,
        float                 w_scale,
        llama_expert_gating_func_type gating_op,
        int                   il,
        struct ggml_tensor  * probs_in = nullptr,
        struct ggml_tensor  * gate_up_exps = nullptr,
        struct ggml_tensor  * gate_up_exps_b = nullptr,
        struct ggml_tensor  * up_exps_s = nullptr,
        struct ggml_tensor  * gate_exps_s = nullptr,
        struct ggml_tensor  * down_exps_s = nullptr,
        int64_t               n_expert_groups = 0,
        int64_t               n_group_used = 0,
        const llm_ffn_clamp_params * clamp = nullptr,
        llm_arch              arch = LLM_ARCH_UNKNOWN,
        llm_ffn_cb            cb = nullptr) {

    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = (arch == LLM_ARCH_LLAMA4);
    const int64_t n_group_experts = n_expert_groups > 0 ? (n_expert / n_expert_groups) : 0;

    struct ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = llm_build_lora_mm(ctx0, gate_inp, cur);
        if (cb) cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        if (cb) cb(logits, "ffn_moe_logits_biased", il);
    }

    struct ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits);
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits);
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits;
            } break;
        default:
            GGML_ABORT("fatal error: unknown gating function");
    }
    if (cb) cb(probs, "ffn_moe_probs", il);

    // add experts selection bias (DeepSeek V3 style)
    struct ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        if (cb) cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits);
        if (cb) cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups (DeepSeek V3 style)
    if (n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / n_expert_groups;

        struct ggml_tensor * selection_groups = ggml_reshape_3d(
            ctx0, selection_probs, n_exp_per_group, n_expert_groups, n_tokens);

        struct ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2);
        group_scores = ggml_get_rows(ctx0,
            ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0],
                            selection_groups->ne[1], selection_groups->ne[2]),
            group_scores);

        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(
            ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3]));
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]);

        struct ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, (int32_t)n_group_used);
        if (cb) cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups);
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups);
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens);
        if (cb) cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    struct ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, (int32_t)n_expert_used);
    if (cb) cb(selected_experts->src[0], "ffn_moe_argsort", il);
    if (cb) cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != n_expert_groups) {
        struct ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    struct ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts);
    if (cb) cb(weights, "ffn_moe_weights", il);

    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights);
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        if (cb) cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        struct ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
        if (cb) cb(weights_sum, "ffn_moe_weights_sum", il);

        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5f, INFINITY);
        if (cb) cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum);
        if (cb) cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }

    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        if (cb) cb(weights, "ffn_moe_weights_scaled", il);
    }

    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        struct ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        if (cb) cb(cur, "ffn_moe_weighted", il);
    }

    struct ggml_tensor * up = nullptr;

    if (gate_up_exps) {
        // merged gate_up path
        struct ggml_tensor * gate_up = llm_build_lora_mm_id(ctx0, gate_up_exps, cur, selected_experts);
        if (cb) cb(gate_up, "ffn_moe_gate_up", il);

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            if (cb) cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        if (up_exps_s) {
            struct ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts);
            gate_up = ggml_mul(ctx0, gate_up, s);
            if (cb) cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2],
                           gate_up->nb[1], gate_up->nb[2], 0);
        if (cb) cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2],
                           gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        if (cb) cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = llm_build_lora_mm_id(ctx0, up_exps, cur, selected_experts);
        if (cb) cb(up, "ffn_moe_up", il);

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            if (cb) cb(up, "ffn_moe_up_biased", il);
        }

        if (up_exps_s) {
            struct ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts);
            up = ggml_mul(ctx0, up, s);
            if (cb) cb(up, "ffn_moe_up_scaled", il);
        }

        if (gate_exps) {
            cur = llm_build_lora_mm_id(ctx0, gate_exps, cur, selected_experts);
            if (cb) cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            if (cb) cb(cur, "ffn_moe_gate_biased", il);
        }

        if (gate_exps_s) {
            struct ggml_tensor * s = ggml_reshape_3d(ctx0, gate_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts);
            cur = ggml_mul(ctx0, cur, s);
            if (cb) cb(cur, "ffn_moe_gate_scaled", il);
        }
    }

    const bool has_gate = (gate_exps || gate_up_exps);

    // activation
    switch (type_op) {
        case LLM_FFN_SILU:
            if (has_gate) {
                if (arch == LLM_ARCH_STEP35 && il >= 0 && clamp) {
                    const float limit = clamp->swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        struct ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                        if (cb) cb(gate_act, "ffn_moe_silu", il);
                        gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                        if (cb) cb(gate_act, "ffn_moe_silu_clamped", il);

                        up = ggml_clamp(ctx0, up, -limit, limit);
                        if (cb) cb(up, "ffn_moe_up_clamped", il);

                        cur = ggml_mul(ctx0, gate_act, up);
                        if (cb) cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
                cur = ggml_swiglu_split(ctx0, cur, up);
                if (cb) cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                if (cb) cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                if (cb) cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                if (cb) cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                if (cb) cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                if (cb) cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                if (cb) cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                if (cb) cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error: unknown MoE FFN op type");
    }

    struct ggml_tensor * experts = llm_build_lora_mm_id(ctx0, down_exps, cur, selected_experts);
    if (cb) cb(experts, "ffn_moe_down", il);

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        if (cb) cb(experts, "ffn_moe_down_biased", il);
    }

    if (down_exps_s) {
        struct ggml_tensor * s = ggml_reshape_3d(ctx0, down_exps_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, selected_experts);
        experts = ggml_mul(ctx0, experts, s);
        if (cb) cb(experts, "ffn_moe_down_scaled", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        if (cb) cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    // order the views before the adds
    const int64_t max_experts = 64; // LLAMA_MAX_EXPERTS
    struct ggml_tensor * cur_experts[64] = { nullptr };

    for (uint32_t i = 0; i < (uint32_t)n_expert_used && i < max_experts; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i * experts->nb[1]);
        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    struct ggml_tensor * moe_out = cur_experts[0];

    // note: use n_expert_used (not hparams.n_expert_used) to avoid excessive add nodes during warmup
    for (uint32_t i = 1; i < (uint32_t)n_expert_used && i < max_experts; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
        ggml_build_forward_expand(gf, moe_out);
    }

    if (n_expert_used == 1) {
        moe_out = ggml_cont(ctx0, moe_out);
    }

    if (cb) cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

//
// Simplified MoE FFN without bias tensors.
// Ported from llm_graph_context::build_moe_ffn (no-bias variant, llama-graph.cpp:1408)
//
static inline struct ggml_tensor * llm_moe_ffn_build_simple(
        struct ggml_context * ctx0,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * gate_inp,
        struct ggml_tensor  * up_exps,
        struct ggml_tensor  * gate_exps,
        struct ggml_tensor  * down_exps,
        struct ggml_tensor  * exp_probs_b,
        int64_t               n_expert,
        int64_t               n_expert_used,
        llm_ffn_op_type       type_op,
        bool                  norm_w,
        float                 w_scale,
        llama_expert_gating_func_type gating_op,
        int                   il,
        struct ggml_tensor  * probs_in = nullptr,
        struct ggml_tensor  * gate_up_exps = nullptr,
        struct ggml_tensor  * up_exps_s = nullptr,
        struct ggml_tensor  * gate_exps_s = nullptr,
        struct ggml_tensor  * down_exps_s = nullptr,
        int64_t               n_expert_groups = 0,
        int64_t               n_group_used = 0,
        const llm_ffn_clamp_params * clamp = nullptr,
        llm_arch              arch = LLM_ARCH_UNKNOWN,
        llm_ffn_cb            cb = nullptr) {

    return llm_moe_ffn_build(
        ctx0, gf,
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert, n_expert_used,
        type_op, norm_w, w_scale, gating_op, il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s,
        n_expert_groups, n_group_used,
        clamp, arch, cb
    );
}

#endif // LLMPP_MOE_FFN_H
