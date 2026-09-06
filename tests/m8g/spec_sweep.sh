#!/bin/bash
# tests/m8g/spec_sweep.sh — M8a re-pin ENGINE acceptance sweep (post-gate7).
# Per config binary (bin/apus_spec_<tag>; "default" = bin/apus): a greedy
# 48-token --spec run on the Paris prompt, stream-compared byte-exact
# against the non-spec reference (ref48) — a mismatch under ANY config is
# an invariant break (stop and report). Logs + stats land in
# tests/m8g/spec_sweep/. Legs run SEQUENTIALLY (RAM).
set -u
cd "$(dirname "$0")/../.."
P="Write a short travel guide for Paris: three paragraphs covering landmarks, food, and getting around."
OUT=tests/m8g/spec_sweep
mkdir -p "$OUT"
export APUS_GEXPERT_CACHE_MB=2048

echo "=== reference (non-spec greedy, 48 tokens) ==="
./bin/apus run --model weights/glm-5.3-flash --tiered --prompt "$P" \
  --max-tokens 48 --temp 0 > "$OUT/ref48.out" 2> "$OUT/ref48.err"
grep -a "decode 48 tok" "$OUT/ref48.err" || tail -2 "$OUT/ref48.err"

for tag in default postnorm0 hc2_0 postnorm1 prenorm1; do
  bin="./bin/apus"
  [ "$tag" != "default" ] && bin="./bin/apus_spec_$tag"
  echo "=== $tag ($bin) ==="
  t0=$(date +%s)
  "$bin" run --model weights/glm-5.3-flash --tiered --spec --spec-k 3 \
    --prompt "$P" --max-tokens 48 --temp 0 > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  t1=$(date +%s)
  if cmp -s "$OUT/ref48.out" "$OUT/$tag.out"; then
    echo "$tag: stream BITWISE == reference; wall $((t1 - t0))s"
  else
    echo "$tag: STREAM MISMATCH vs reference — INVARIANT BROKEN" >&2
  fi
  grep -a "apus: spec:" "$OUT/$tag.err"
done
echo "=== sweep done ==="
