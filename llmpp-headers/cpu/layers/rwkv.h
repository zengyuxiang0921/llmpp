//
// RWKV token shift operations ported from llama.cpp
// (src/llama-graph.cpp:2811)
//
// Reference:
//   https://github.com/ggml-org/llama.cpp
//

#ifndef LLMPP_RWKV_H
#define LLMPP_RWKV_H

#include "types.h"

//
// Build recurrent state: clear + copy states.
// Ported from llm_graph_context::build_rs (llama-graph.cpp:2732)
//
static inline struct ggml_tensor * llm_rwkv_build_rs(
        struct ggml_context * ctx0,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * s,
        struct ggml_tensor  * state_copy_main,
        struct ggml_tensor  * state_copy_extra,
        int32_t               state_size,
        int32_t               n_seqs,
        uint32_t              n_rs,
        uint32_t              rs_head,
        int32_t               rs_zero,
        const std::function<struct ggml_tensor *(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *)> & get_state_rows) {

    struct ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states
    struct ggml_tensor * state_zero = ggml_view_1d(
        ctx0, states, state_size * (rs_zero >= 0), rs_zero * states->nb[1] * (rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    struct ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    struct ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs) * s->nb[1])));

    return output_states;
}

//
// Build RWKV token shift: load previous recurrent state for a layer.
// Ported from llm_graph_context::build_rwkv_token_shift_load (llama-graph.cpp:2811)
//
static inline struct ggml_tensor * llm_rwkv_token_shift_load(
        struct ggml_context * ctx0,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * token_shift_all,
        struct ggml_tensor  * state_copy_main,
        struct ggml_tensor  * state_copy_extra,
        int32_t               state_size,
        int32_t               n_seqs,
        uint32_t              n_rs,
        uint32_t              rs_head,
        int32_t               rs_zero,
        int64_t               n_embd,
        int64_t               token_shift_count,
        const std::function<struct ggml_tensor *(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *)> & get_state_rows) {

    struct ggml_tensor * token_shift = llm_rwkv_build_rs(
        ctx0, gf, token_shift_all,
        state_copy_main, state_copy_extra,
        state_size, n_seqs, n_rs, rs_head, rs_zero,
        get_state_rows);

    token_shift = ggml_reshape_3d(ctx0, token_shift, n_embd, token_shift_count, n_seqs);

    return token_shift;
}

//
// Build RWKV token shift: store current recurrent state for a layer.
// Ported from llm_graph_context::build_rwkv_token_shift_store (llama-graph.cpp:2832)
//
static inline struct ggml_tensor * llm_rwkv_token_shift_store(
        struct ggml_context * ctx0,
        struct ggml_tensor  * token_shift,
        struct ggml_tensor  * state_dst,
        int64_t               n_embd,
        int64_t               n_seqs,
        int64_t               token_shift_count,
        int64_t               kv_head,
        int64_t               n_embd_r) {

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, state_dst, n_embd_r * n_seqs, n_embd_r * kv_head * ggml_element_size(state_dst))
    );
}

#endif // LLMPP_RWKV_H
