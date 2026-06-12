#pragma once

#include <cmath>

static inline void rope(float * out, const float * in, const float * freqs, int n_tokens, int n_embd, int n_head, int n_embd_head) {
    for (int i = 0; i < n_tokens; i++) {
        for (int h = 0; h < n_head; h++) {
            for (int j = 0; j < n_embd_head; j += 2) {
                int idx = i * n_embd + h * n_embd_head + j;
                float cos = std::cos(freqs[i * n_embd_head / 2 + j / 2]);
                float sin = std::sin(freqs[i * n_embd_head / 2 + j / 2]);

                float x0 = in[idx];
                float x1 = in[idx + 1];

                out[idx]     = x0 * cos - x1 * sin;
                out[idx + 1] = x0 * sin + x1 * cos;
            }
        }
    }
}
