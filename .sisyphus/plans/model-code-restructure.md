# Plan: Restructure model-code and create standard CUDA operator library

## Current State

- `model-code/kernel/*.cu` — contains low-level kernel implementations (silu, mul_mat, rms_norm, soft_max)
  — These should be standard ops, part of the runtime library, NOT embedded in GGUF
- `model-code/layer/*.cpp` / `model-code/pipeline/*.cpp` — model-specific glue code
  — These SHOULD be embedded in GGUF as `llmpp.layer.*`, `llmpp.pipeline.*`
- `llmpp-headers/` has `cpu/` dir but no `cuda/` dir yet

## Target Structure

```
llmpp-headers/
  cpu/                     ← existing CPU inference headers (unchanged)
  cuda/
    ops/                   ← STANDARD OP LIBRARY (compiled from disk, not embedded)
      silu.cu
      gelu.cu
      relu.cu
      add.cu
      mul.cu
      mul_mat.cu
      soft_max.cu
      rms_norm.cu
      rope.cu
      get_rows.cu

model-code/
  layer/                   ← MODEL-SPECIFIC (embedded in GGUF)
    ffn.cpp
    attention.cpp
  pipeline/
    inference.cpp
```

Each standard op is a self-contained header with both `__CUDACC__` (GPU) and CPU fallback paths.

## Tasks

### Task 1: Create standard CUDA operator library

- [x] Create `llmpp-headers/cuda/ops/` directory
- [x] Write 10 standard op files (silu, gelu, relu, add, mul, mul_mat, soft_max, rms_norm, rope, get_rows)
- [x] Each file has `#ifdef __CUDACC__` for CUDA parallel version, `#else` for CPU scalar version

### Task 2: Rewrite model-code to be clean glue code

- [x] `model-code/layer/ffn.cpp` — include only "cuda/ops/..." ops, just wire up inputs/outputs
- [x] `model-code/layer/attention.cpp` — same, pure composition
- [x] `model-code/pipeline/inference.cpp` — same, full forward pass
- [x] Remove `model-code/kernel/` directory (kernels moved to llmpp-headers/cuda/ops/)

### Task 3: Update convert_gguf_to_llmpp.py

- [x] Keep the existing `--model-code` argument
- [x] Since standard ops are NOT embedded, the script just embeds model-code/ files as-is
- [x] No include resolution needed (include paths point to llmpp-headers at compile time)
- [x] The `load_model_code()` function already works correctly — it naturally scans only model-code/ layer & pipeline files since kernel/ was removed

## Execution

Run `/start-work` to execute tasks in sequence.
