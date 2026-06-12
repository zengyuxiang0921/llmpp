//
// Shared types and helpers for GGML neural network layers.
// Ported from llama.cpp (src/llama-graph.h, src/llama-graph.cpp)
//
// Reference:
//   https://github.com/ggml-org/llama.cpp
//

#ifndef LLMPP_TYPES_H
#define LLMPP_TYPES_H

#include "../ggml/include/ggml.h"

#include <cstdint>
#include <functional>
#include <array>

//
// FFN operation types -- matches llama.cpp llm_ffn_op_type
//
enum llm_ffn_op_type : int {
    LLM_FFN_NONE      = 0,
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
    LLM_FFN_GEGLU,
    LLM_FFN_REGLU,
    LLM_FFN_SWIGLU_OAI_MOE,
};

//
// FFN gate type -- matches llama.cpp llm_ffn_gate_type
//
enum llm_ffn_gate_type {
    LLM_FFN_SEQ,       // gate is sequential after up
    LLM_FFN_PAR,       // gate is parallel to up (fused gate_up)
};

//
// Normalization type -- matches llama.cpp llm_norm_type
//
enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

//
// Architecture identifiers -- subset used by FFN/attention implementations
//
enum llm_arch : int {
    LLM_ARCH_UNKNOWN   = 0,
    LLM_ARCH_LLAMA4    = 1,
    LLM_ARCH_GLM4      = 2,
    LLM_ARCH_GLM4_MOE  = 3,
    LLM_ARCH_JAIS2     = 4,
    LLM_ARCH_STEP35    = 5,
    LLM_ARCH_GROK      = 6,
    LLM_ARCH_GROVEMOE  = 7,
};

//
// Expert gating function type -- matches llama.cpp llama_expert_gating_func_type
//
enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
};

//
// Clamping parameters for SwiGLU (from llama_hparams)
//
struct llm_ffn_clamp_params {
    std::array<float, 512> swiglu_clamp_exp   = {};  // expert FFN
    std::array<float, 512> swiglu_clamp_shexp = {};  // shared expert
};

//
// Callback type for naming intermediate tensors (matches llama.cpp cb())
//
using llm_ffn_cb = std::function<void(struct ggml_tensor * cur, const char * name, int il)>;

//
// LoRA-aware matrix multiply: w * cur, optionally scaled by w_s.
// Ported from llm_graph_context::build_lora_mm (llama-graph.cpp:1072)
//
// For standalone use without LoRA, this simply does ggml_mul_mat(ctx0, w, cur).
//
static inline struct ggml_tensor * llm_build_lora_mm(
        struct ggml_context * ctx0,
        struct ggml_tensor  * w,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * w_s = nullptr) {
    struct ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }
    return res;
}

//
// LoRA-aware matrix multiply with expert IDs (used by MoE).
// Ported from llm_graph_context::build_lora_mm_id (llama-graph.cpp:1103)
//
static inline struct ggml_tensor * llm_build_lora_mm_id(
        struct ggml_context * ctx0,
        struct ggml_tensor  * w,
        struct ggml_tensor  * cur,
        struct ggml_tensor  * ids) {
    struct ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    return res;
}

#endif // LLMPP_TYPES_H
