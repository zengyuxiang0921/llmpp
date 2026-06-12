## 2026-06-11T19:44:00Z Security review findings

### M1: get_rows.cu — Missing bounds check on indices[row]
- `src_row = indices[row]` used directly as array offset into `in` with no validation
- No way to validate even if wanted — `in` row count is not passed to kernel
- If indices contains an out-of-range value (negative or >= total rows), OOB read
- Fix: pass `n_src_rows` and guard: `if (src_row < 0 || src_row >= n_src_rows) return;`

### M2: Integer overflow in n_tokens * n_embd computations
- Both are `int` (signed 32-bit); product used for bounds checks and element counts
- Affected: ffn.cpp lines 19,25; inference.cpp lines 25,35; rope.cu line 8; get_rows.cu line 9
- Overflow to small positive → bounds check passes incorrectly → OOB access
- Fix: cast to `int64_t` before multiplication in all these sites

### L1: soft_max.cu — No guard for n_cols == 0
- Line 12 reads `row_in[0]` unconditionally; OOB if n_cols == 0
- Requires caller bug, not user input; defense-in-depth
- Fix: add `if (n_cols <= 0) return;` at entry

### L2: rope.cu — Division by zero if n_embd == 0
- Line 13: `head / n_embd` UB if n_embd == 0
- Fix: add `if (n_embd <= 0) return;` at entry

### L3: mul_mat.cu — Index overflow for very large matrices
- `row * k + i` uses int32; overflows if n_embd > 46340 (not practical today, future concern)
- Fix: cast strides to `int64_t`

### Verified Safe
- silu.cu: denominator always >= 1.0
- rms_norm.cu: eps = 1e-5f guarantees positive divisor
- soft_max.cu: sum >= 1.0 (max element contributes exp(0) = 1.0)
- add, mul, relu, gelu: no divisions, bounds-protected
