// llmpp.pipeline.inference
// Full inference pipeline: token embedding -> N decoder blocks -> output

#include "../layer/ffn.cpp"
#include "../layer/attention.cpp"
#include "cuda/ops/rms_norm.cu"

// Forward pass for one transformer layer
void transformer_block(float * out, const float * hidden,
                       const float * attn_norm_w,
                       const float * wq, const float * wk, const float * wv, const float * wo,
                       const float * ffn_norm_w,
                       const float * ffn_up, const float * ffn_gate, const float * ffn_down,
                       int n_tokens, int n_embd, int n_head, int n_head_kv, int n_ff) {

    float * attn_in = /* temp */ nullptr;
    float * ffn_in  = /* temp */ nullptr;

    // Attention with pre-norm
    rms_norm_kernel(attn_in, hidden, attn_norm_w, n_tokens, n_embd, 1e-5f);
    attn_forward(attn_in, attn_in, wq, wk, wv, wo,
                 n_tokens, n_embd, n_head, n_head_kv);

    // Residual add: attn_in was reused as attn output, add hidden
    for (int i = 0; i < n_embd * n_tokens; i++) {
        attn_in[i] += hidden[i];
    }

    // FFN with pre-norm
    rms_norm_kernel(ffn_in, attn_in, ffn_norm_w, n_tokens, n_embd, 1e-5f);
    ffn_forward(out, ffn_in, ffn_up, ffn_gate, ffn_down,
                n_tokens, n_embd, n_ff);

    // Residual add
    for (int i = 0; i < n_embd * n_tokens; i++) {
        out[i] += attn_in[i];
    }
}
