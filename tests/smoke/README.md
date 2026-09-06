# tests/smoke — real-weight debug tooling (not a gate)

Ad-hoc drivers used for the 2026-09-04 real-container smoke + bisection:

- `trace_real.c` — opens the converted container (tiered, sync I/O),
  prefills a fixed id prompt with `trace_h` capture, prints per-layer
  FNV-1a digests of the last-position block stream + top-10 logits;
  optional teacher-forced decode chain (argv[3]) dumps per-step digests.
  Build: `cc -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Ic -o
  bin/trace_real tests/smoke/trace_real.c -lpthread`
- `tools/ref_check.py` — oracle (HF-semantics) prefill on the real
  container via a lazy mmap ShardSet; writes `ref/ref.npz` +
  digests.
- `tools/ref_decode.py` — oracle teacher-forced decode chain, same
  digest format as trace_real.
- `tools/ref_compare.py` — per-layer digest comparison, prints first
  divergence.

Results 2026-09-04 (real container, prompt "The capital of France is",
ids 785,6722,315,9621,374): C == oracle BITWISE — prefill logits digest
d5b773bc8ab2cb14, all 45 layers, plus 184/184 teacher-forced decode
digests (4 steps × 45 layers + logits). Reference top-2 next tokens:
279 " the" (13.8125), 12089 " Paris" (13.6875).
