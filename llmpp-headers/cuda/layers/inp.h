//
// CUDA input embedding layer: get_rows from embedding table.
// Composes the get_rows op to gather token embeddings.
// GPU path (__CUDACC__): launches get_rows CUDA kernel.
// CPU fallback (#else): calls static inline get_rows directly.
// No temp buffers needed — writes directly to output.
//

#pragma once

#include "../ops/get_rows.h"

static inline void llm_inp_embd(
        float       * out,
        const float * embd_table,
        const int   * tokens,
        int           n_tokens,
        int           n_embd) {

#ifdef __CUDACC__
    const int total = n_tokens * n_embd;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    get_rows<<<grid, block>>>(out, embd_table, tokens, n_tokens, n_embd);
#else
    get_rows(out, embd_table, tokens, n_tokens, n_embd);
#endif
}
