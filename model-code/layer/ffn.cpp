// llmpp.layer.ffn
// Feed-Forward Network: out = silu(gate * in) * (up * in) * down
// Supports both dense and fused gate_up layouts.

#include "cuda/ops/silu.cu"
#include "cuda/ops/mul_mat.cu"
#include "cuda/ops/mul.cu"

void ffn_forward(float * out, const float * hidden,
                 const float * up_w, const float * gate_w, const float * down_w,
                 int n_tokens, int n_embd, int n_ff) {

    float * gate_out = /* temp buffer */ nullptr;
    float * up_out   = /* temp buffer */ nullptr;
    float * act      = /* temp buffer */ nullptr;

    // gate = silu(gate_w * hidden)
    mul_mat_kernel(gate_w, hidden, gate_out, n_embd, n_tokens, n_embd);
    silu_kernel(gate_out, gate_out, n_embd * n_tokens);

    // up = up_w * hidden
    mul_mat_kernel(up_w, hidden, up_out, n_embd, n_tokens, n_embd);

    // act = gate * up (element-wise)
    mul_kernel(act, gate_out, up_out, n_embd * n_tokens);

    // down = down_w * act
    mul_mat_kernel(down_w, act, out, n_embd, n_tokens, n_ff);
}
