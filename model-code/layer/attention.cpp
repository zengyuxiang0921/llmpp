// llmpp.layer.attention
// Multi-Head Attention: out = softmax(Q*K^T/sqrt(d)) * V * W_o

#include "cuda/ops/mul_mat.cu"
#include "cuda/ops/soft_max.cu"

void attn_forward(float * out, const float * hidden,
                  const float * wq, const float * wk, const float * wv,
                  const float * wo,
                  int n_tokens, int n_embd, int n_head, int n_head_kv) {
    int n_embd_head = n_embd / n_head;
    int n_embd_kv   = n_embd_head * n_head_kv;

    float * q = /* temp */ nullptr;
    float * k = /* temp */ nullptr;
    float * v = /* temp */ nullptr;

    // QKV projections
    mul_mat_kernel(wq, hidden, q, n_embd, n_tokens, n_embd);
    mul_mat_kernel(wk, hidden, k, n_embd_kv, n_tokens, n_embd);
    mul_mat_kernel(wv, hidden, v, n_embd_kv, n_tokens, n_embd);

    // Scaled dot-product attention (simplified)
    float * scores = /* temp */ nullptr;
    // scores = q * k^T  (per-head)
    // scores = softmax(scores / sqrt(n_embd_head))

    float * attn_out = /* temp */ nullptr;
    // attn_out = scores * v

    // Output projection
    mul_mat_kernel(wo, attn_out, out, n_embd, n_tokens, n_embd);
}
