## 2026-06-11T11:28:01.816Z Initial decisions

- Op files in llmpp-headers/cuda/ops/ use `.cu` extension (matching plan)
- Each op is a self-contained header with `#ifdef __CUDACC__` / `#else` pattern
- model-code layer/pipeline files include ops as `#include "cuda/ops/<name>.cu"` (relative to llmpp-headers/)
- New ops (gelu, relu, add, mul, rope, get_rows) follow same pattern as existing silu/mul_mat
- Existing soft_max and rms_norm in kernel/ are CPU-only; need CUDA version added for the op library