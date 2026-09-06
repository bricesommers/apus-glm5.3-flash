# apus — root Makefile
# Milestone M2: tokenizer (c/tok.h) + message encoding (c/encoding.h) + tests.

UNAME   := $(shell uname)
# M12a-1: clang on macOS, gcc on Linux/x86_64 (same warning set).
# M15: Windows via MinGW-w64 gcc (MSYS2 UCRT64 shell: uname is MINGW64_NT-*
# and OS=Windows_NT is always set). No MSVC support (C11 + GNU extensions).
# Windows uses -std=gnu11 (not strict c11): MinGW hides strdup/clock_gettime
# et al. behind __STRICT_ANSI__; gnu11 exposes them. FP semantics unchanged
# (-ffp-contract=off stays pinned).
ifeq ($(OS),Windows_NT)
STD     := -std=gnu11
else
STD     := -std=c11
endif
ifeq ($(UNAME),Darwin)
CC      := clang
else
CC      := gcc
endif
CFLAGS  ?= $(STD) -O2 -Wall -Wextra
# Pin FP mul+add contraction OFF on every platform: scalar kernels and the
# in-test scalar references have documented two-rounding sequences, but
# newer clangs (GitHub macos-latest) auto-contract loops to FMA where our
# dev clang 17 did not — same flags, different bits (test-m6c f32 bitwise
# gate). No-op where the compiler was not contracting anyway.
CFLAGS  += -ffp-contract=off
ifeq ($(OS),Windows_NT)
# M15: MinGW-w64. -D_GNU_SOURCE does not exist here; the POSIX surface is
# shimmed in c/compat.h. The -fno-tree-vectorize pair below mirrors the
# Linux flags (numerics no-op; keeps the x86 anchor builds consistent).
# -lpsapi: GetProcessMemoryInfo (compat.h RSS). pthreads come from
# winpthreads (MSYS2's mingw-w64 gcc), plain -lpthread.
CFLAGS  += -fno-tree-vectorize -fno-tree-slp-vectorize
else ifneq ($(UNAME),Darwin)
# Linux (M12a-1): under -std=c11 glibc hides pread/posix_memalign/strdup/
# clock_gettime/posix_fadvise behind feature-test macros; _GNU_SOURCE
# exposes them.
# -fno-tree-vectorize -fno-tree-slp-vectorize: works around a Rosetta
# linux/amd64-emulation mistranslation of gcc -O2 auto-vectorized SSE2
# code (test-m8 SIGTRAPs with "rosetta error: could not find free space
# for allocation"; ASan/UBSan clean, macOS -O2 clean, -O1 clean).
# Numerics are unaffected: FP reductions are never reassociated without
# -ffast-math and elementwise loops are per-element identical either way,
# so the scalar kernels produce the same bits with or without these flags.
# M12a-2 (AVX2) hand-writes the vector kernels and can revisit this.
CFLAGS  += -D_GNU_SOURCE -fno-tree-vectorize -fno-tree-slp-vectorize
endif
LDLIBS  := -lm
ifeq ($(OS),Windows_NT)
# -lpsapi: GetProcessMemoryInfo (compat.h RSS). -static: bundle
# winpthreads/mingw runtime — a MinGW .exe otherwise needs
# libwinpthread-1.dll on PATH (silent exit 127 when missing), and static
# linking makes the binaries portable for end users anyway.
LDLIBS  += -lpsapi -static
endif
# M9b: Accelerate.framework (system vecLib/AMX BLAS) for the batch-M prefill
# GEMM dispatch (c/blas.h). macOS system framework, ships with the OS.
ifeq ($(UNAME),Darwin)
LDLIBS  += -framework Accelerate
endif
ASAN_CFLAGS := -std=c11 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer

M2   := tests/m2
BIN  := $(M2)/bin
PY   := .venv/bin/python

M1   := tests/m1

M0   := tests/m0

M3G  := tests/m3g
BIN3G := $(M3G)/bin

M4G  := tests/m4g
BIN4G := $(M4G)/bin

M4H  := tests/m4h
BIN4H := $(M4H)/bin

M5G  := tests/m5g
BIN5G := $(M5G)/bin

M6G  := tests/m6g
BIN6G := $(M6G)/bin

M7A  := tests/m7a

M8G  := tests/m8g
BIN8G := $(M8G)/bin

M2_DEPS := c/json.h c/tok.h c/uni_tables.h
M3G_DEPS := c/num.h c/bf16.h c/fp8blk.h c/pool.h c/x86.h
M4G_DEPS := c/num.h c/bf16.h c/fp8blk.h c/gmhc.h c/gmoe.h c/pool.h c/x86.h c/compat.h
M4H_DEPS := $(M4G_DEPS) c/gkda.h c/gdsa.h
M5G_DEPS := $(M4H_DEPS) c/gmodel.h c/json.h c/st.h c/compat.h
M6G_DEPS := $(M5G_DEPS) c/gcache.h c/gpilot.h
M8G_DEPS := $(M6G_DEPS) c/gmtp.h c/sample.h
APUS_DEPS := c/json.h c/st.h c/compat.h c/sample.h c/tok.h c/encoding.h \
             c/uni_tables.h $(M8G_DEPS)

.PHONY: all test-m1 test-m2 asan-m2 ubsan-m2 golden golden-exhaustive \
        check-hf \
        test-m3g ubsan-m3g golden-m3g bench-m3g \
        test-m4g ubsan-m4g golden-m4g \
        test-m4h ubsan-m4h golden-m4h \
        test-m5g ubsan-m5g golden-m5g \
        test-m6g ubsan-m6g golden-m6g bench-gio \
        clean apus \
        test-m7a ubsan-m7a golden-m7a \
        test-m7b ubsan-m7b bench-m7b \
        test-m8g ubsan-m8g golden-m8g \
        win-cross

all: $(BIN)/test_tok $(BIN)/test_encoding

$(BIN):
	mkdir -p $(BIN)

$(BIN)/test_tok: $(M2)/test_tok.c $(M2_DEPS) | $(BIN)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN)/test_encoding: $(M2)/test_encoding.c $(M2_DEPS) c/encoding.h | $(BIN)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN)/test_tok_asan: $(M2)/test_tok.c $(M2_DEPS) | $(BIN)
	$(CC) $(ASAN_CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN)/test_encoding_asan: $(M2)/test_encoding.c $(M2_DEPS) c/encoding.h | $(BIN)
	$(CC) $(ASAN_CFLAGS) -Ic -o $@ $< $(LDLIBS)

golden:
	$(PY) $(M2)/gen_golden.py

golden-exhaustive:
	$(PY) $(M2)/gen_golden.py --exhaustive

test-m2: all golden
	./$(BIN)/test_tok
	./$(BIN)/test_encoding

# M1 is a pure-Python suite (glm5_next converter + downloader vs synthetic
# shards, fully offline); see tests/m1/README.md.
test-m1:
	$(PY) -m unittest discover -s $(M1)

# --- M0 anchor: oracle vs TRUE HuggingFace glm5_next (opt-in) ----------------
# tests/m0/check_vs_hf.py runs the numpy oracle against transformers @ git
# main (torch CPU, bf16) on a synthetic tiny checkpoint — the external
# anchor for the C==oracle verification chain (2026-09-05: it caught the
# kv_b_proj per-head split divergence, gate 7). NOT in the default battery:
# needs torch + transformers-main in .venv (CI runners don't carry torch).
check-hf:
	@$(PY) -c "import torch, transformers" 2>/dev/null || { echo "check-hf needs torch + transformers @ git main in .venv:"; echo "  .venv/bin/pip install torch \"git+https://github.com/huggingface/transformers.git\""; exit 1; }
	$(PY) $(M0)/check_vs_hf.py

asan-m2: $(BIN)/test_tok_asan $(BIN)/test_encoding_asan golden
	./$(BIN)/test_tok_asan
	./$(BIN)/test_encoding_asan

# UBSan-only variant (Apple's ASan runtime hangs in dyld init on some macOS
# versions; see tests/m2/README notes in the M2 report)
ubsan-m2: CFLAGS = -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer
ubsan-m2: $(BIN)/test_tok_ubsan $(BIN)/test_encoding_ubsan golden
	./$(BIN)/test_tok_ubsan
	./$(BIN)/test_encoding_ubsan

$(BIN)/test_tok_ubsan: $(M2)/test_tok.c $(M2_DEPS) | $(BIN)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS)

$(BIN)/test_encoding_ubsan: $(M2)/test_encoding.c $(M2_DEPS) c/encoding.h | $(BIN)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS)

# --- M3 (GLM): fp8blk dequant + bf16 GEMV/GEMM (c/fp8blk.h, c/bf16.h) -------
# E4M3+F32-scale 128x128-block dequant -> BF16 and the BF16 GEMV/GEMM
# kernels, all paths (scalar/NEON/AVX2/mt) bitwise == the numpy oracle
# goldens; thread-count independence diffed across APUS_THREADS=1/4/8.

$(BIN3G):
	mkdir -p $(BIN3G)

$(BIN3G)/test_m3g: $(M3G)/test_m3g.c $(M3G_DEPS) | $(BIN3G)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN3G)/test_m3g_ubsan: $(M3G)/test_m3g.c $(M3G_DEPS) | $(BIN3G)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -ffp-contract=off $(if $(filter-out Darwin,$(UNAME)),-D_GNU_SOURCE) -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m3g:
	$(PY) $(M3G)/gen_golden.py

test-m3g: $(BIN3G)/test_m3g golden-m3g
	APUS_THREADS=1 ./$(BIN3G)/test_m3g > $(BIN3G)/out_t1.txt
	APUS_THREADS=4 ./$(BIN3G)/test_m3g > $(BIN3G)/out_t4.txt
	APUS_THREADS=8 ./$(BIN3G)/test_m3g > $(BIN3G)/out_t8.txt
	diff $(BIN3G)/out_t1.txt $(BIN3G)/out_t4.txt
	diff $(BIN3G)/out_t1.txt $(BIN3G)/out_t8.txt
	cat $(BIN3G)/out_t4.txt

# UBSan-only, like the other milestones (-ffp-contract=off pinned: the
# sanitizer flags override CFLAGS, dropping the default flags; the gates
# are bitwise, so contraction must stay off)
ubsan-m3g: $(BIN3G)/test_m3g_ubsan golden-m3g
	APUS_THREADS=1 ./$(BIN3G)/test_m3g_ubsan > $(BIN3G)/out_u1.txt
	APUS_THREADS=4 ./$(BIN3G)/test_m3g_ubsan > $(BIN3G)/out_u4.txt
	diff $(BIN3G)/out_u1.txt $(BIN3G)/out_u4.txt
	cat $(BIN3G)/out_u4.txt

# P5: decode-GEMV effective-bandwidth bench (informational, NOT a gate) —
# scalar / single-thread SIMD / mt GB/s per real decode shape + rooflines.
$(BIN3G)/bench_bf16: $(M3G)/bench_bf16.c $(M3G_DEPS) | $(BIN3G)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

bench-m3g: $(BIN3G)/bench_bf16
	./$(BIN3G)/bench_bf16

# --- M4a (GLM): mHC + MoE sublayers (c/gmhc.h, c/gmoe.h) -------------------
# GLM mHC residual stream (norm-before-fn unweighted RMSNorm, Sinkhorn-20,
# unweighted-mean head) and GLM MoE (sigmoid router with selection-only
# bias, swiglu_limit experts, bf16-stepped ascending-expert accumulation),
# gated against the M0 oracle's own sublayer functions (f32 mode). BITWISE
# where host expf == numpy f32 exp (probed at runtime; on x86-64 the
# golden-m4g recipe pins numpy to its baseline exp kernel via
# NPY_DISABLE_CPU_FEATURES so the bitwise tier engages there too —
# macOS arm64 needs no pin); tolerance fallback + always-bitwise
# selections otherwise (tests/m4g/README.md). Thread-count independence
# diffed across APUS_THREADS=1/4/8.

$(BIN4G):
	mkdir -p $(BIN4G)

$(BIN4G)/test_m4g: $(M4G)/test_m4g.c $(M4G_DEPS) | $(BIN4G)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN4G)/test_m4g_ubsan: $(M4G)/test_m4g.c $(M4G_DEPS) | $(BIN4G)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -ffp-contract=off $(if $(filter-out Darwin,$(UNAME)),-D_GNU_SOURCE) -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m4g:
	NPY_DISABLE_CPU_FEATURES="X86_V3 X86_V4 AVX512_SKX AVX512_SPR" $(PY) $(M4G)/gen_golden.py

test-m4g: $(BIN4G)/test_m4g golden-m4g
	APUS_THREADS=1 ./$(BIN4G)/test_m4g > $(BIN4G)/out_t1.txt
	APUS_THREADS=4 ./$(BIN4G)/test_m4g > $(BIN4G)/out_t4.txt
	APUS_THREADS=8 ./$(BIN4G)/test_m4g > $(BIN4G)/out_t8.txt
	diff $(BIN4G)/out_t1.txt $(BIN4G)/out_t4.txt
	diff $(BIN4G)/out_t1.txt $(BIN4G)/out_t8.txt
	cat $(BIN4G)/out_t4.txt

# UBSan-only, like the other milestones (-ffp-contract=off pinned, same
# reason as ubsan-m3g)
ubsan-m4g: $(BIN4G)/test_m4g_ubsan golden-m4g
	APUS_THREADS=1 ./$(BIN4G)/test_m4g_ubsan > $(BIN4G)/out_u1.txt
	APUS_THREADS=4 ./$(BIN4G)/test_m4g_ubsan > $(BIN4G)/out_u4.txt
	diff $(BIN4G)/out_u1.txt $(BIN4G)/out_u4.txt
	cat $(BIN4G)/out_u4.txt

# --- M4b (GLM): KDA + DSA/indexer sublayers (c/gkda.h, c/gdsa.h) --------
# GLM KDA linear-attention sublayer in BOTH orderings (the M0 KDA ORDERING
# CONTRACT: chunked prefill chunk 64 incl. state-carrying continuations,
# recurrent decode chains) and the DSA sublayer (MLA pure-NoPE q_a/q_b,
# kv_a/kv_b, o_proj FP8 via m3g dequant+bf16 GEMM; Lightning indexer with
# kpool-4 pool rebuild from the full cache every forward, always-selected
# tail, stable-descending top-k), gated against the M0 oracle's own
# sublayer functions (f32 mode). BITWISE where host expf == numpy f32 exp
# (probed at runtime; golden-m4h pins numpy to its baseline exp kernel via
# NPY_DISABLE_CPU_FEATURES on x86-64 — macOS arm64 needs no pin);
# tolerance fallback with fragile-query-skip selections otherwise
# (tests/m4h/README.md). Thread-count independence diffed across
# APUS_THREADS=1/4/8.

$(BIN4H):
	mkdir -p $(BIN4H)

$(BIN4H)/test_m4h: $(M4H)/test_m4h.c $(M4H_DEPS) | $(BIN4H)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN4H)/test_m4h_ubsan: $(M4H)/test_m4h.c $(M4H_DEPS) | $(BIN4H)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -ffp-contract=off $(if $(filter-out Darwin,$(UNAME)),-D_GNU_SOURCE) -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m4h:
	NPY_DISABLE_CPU_FEATURES="X86_V3 X86_V4 AVX512_SKX AVX512_SPR" $(PY) $(M4H)/gen_golden.py

test-m4h: $(BIN4H)/test_m4h golden-m4h
	APUS_THREADS=1 ./$(BIN4H)/test_m4h > $(BIN4H)/out_t1.txt
	APUS_THREADS=4 ./$(BIN4H)/test_m4h > $(BIN4H)/out_t4.txt
	APUS_THREADS=8 ./$(BIN4H)/test_m4h > $(BIN4H)/out_t8.txt
	diff $(BIN4H)/out_t1.txt $(BIN4H)/out_t4.txt
	diff $(BIN4H)/out_t1.txt $(BIN4H)/out_t8.txt
	cat $(BIN4H)/out_t4.txt

# UBSan-only, like the other milestones (-ffp-contract=off pinned, same
# reason as ubsan-m3g)
ubsan-m4h: $(BIN4H)/test_m4h_ubsan golden-m4h
	APUS_THREADS=1 ./$(BIN4H)/test_m4h_ubsan > $(BIN4H)/out_u1.txt
	APUS_THREADS=4 ./$(BIN4H)/test_m4h_ubsan > $(BIN4H)/out_u4.txt
	diff $(BIN4H)/out_u1.txt $(BIN4H)/out_u4.txt
	cat $(BIN4H)/out_u4.txt

# --- M5 (GLM): full model forward (c/gmodel.h) ----------------------------
# The full GLM-5.3-Flash text-model forward on synthetic weights: M1 v2
# container loading (dense tensors via ApusStSet + one-pread-per-expert
# slabs via ApusStLazy, eager dequant — the M6 cache slots behind
# apus_gmodel_expert), the 45-layer mHC stack (KDA|DSA + dense|MoE) with
# prefill (chunked KDA) and an 8-step decode chain (recurrent KDA) per
# the §4 contract, gated BITWISE against the M0 oracle's model_forward
# (f32 mode) where host expf == numpy f32 exp (probed at runtime;
# golden-m5g pins numpy to its baseline exp kernel via
# NPY_DISABLE_CPU_FEATURES on x86-64 — macOS arm64 needs no pin). The
# tolerance tier teacher-forces the router/indexer selections (structural
# relu-clip zero ties + compounding bf16 flips; tests/m5g/README.md) and
# compares with the measured compounding classes. Thread-count
# independence diffed across APUS_THREADS=1/4/8.

$(BIN5G):
	mkdir -p $(BIN5G)

$(BIN5G)/test_m5g: $(M5G)/test_m5g.c $(M5G_DEPS) | $(BIN5G)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN5G)/test_m5g_ubsan: $(M5G)/test_m5g.c $(M5G_DEPS) | $(BIN5G)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -ffp-contract=off $(if $(filter-out Darwin,$(UNAME)),-D_GNU_SOURCE) -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m5g:
	NPY_DISABLE_CPU_FEATURES="X86_V3 X86_V4 AVX512_SKX AVX512_SPR" $(PY) $(M5G)/gen_golden.py

test-m5g: $(BIN5G)/test_m5g golden-m5g
	APUS_THREADS=1 ./$(BIN5G)/test_m5g > $(BIN5G)/out_t1.txt
	APUS_THREADS=4 ./$(BIN5G)/test_m5g > $(BIN5G)/out_t4.txt
	APUS_THREADS=8 ./$(BIN5G)/test_m5g > $(BIN5G)/out_t8.txt
	diff $(BIN5G)/out_t1.txt $(BIN5G)/out_t4.txt
	diff $(BIN5G)/out_t1.txt $(BIN5G)/out_t8.txt
	cat $(BIN5G)/out_t4.txt

# UBSan-only, like the other milestones (-ffp-contract=off pinned, same
# reason as ubsan-m3g)
ubsan-m5g: $(BIN5G)/test_m5g_ubsan golden-m5g
	APUS_THREADS=1 ./$(BIN5G)/test_m5g_ubsan > $(BIN5G)/out_u1.txt
	APUS_THREADS=4 ./$(BIN5G)/test_m5g_ubsan > $(BIN5G)/out_u4.txt
	diff $(BIN5G)/out_u1.txt $(BIN5G)/out_u4.txt
	cat $(BIN5G)/out_u4.txt

# --- M8a (GLM): MTP (classic NextN) oracle fixtures ---------------------
# numpy goldens for the oracle's MTP draft head (tools/oracle.py
# mtp_forward/mtp_chain) on tiny synthetic containers carrying the
# layers.<L>.* NextN block through the real M1 converter path (apus-mtp-*
# shard group): batched true-pair replay + the draft chain, seeded from a
# main-model decode step whose top layer is KDA (kda_top) vs DSA
# (dsa_top). The pinned hnorm input / (h,id) pairing (tools/mtp_pin.py on
# the real container) live in oracle.MTP_HNORM_INPUT / MTP_PAIR_LAG.
# The C gate (M8b, tests/m8g/test_m8g.c) covers: the model + h-out surface
# vs the goldens, the per-token-interleaved decode batch (bitwise ==
# sequential by construction), the mtp_forward replay + draft chain vs the
# oracle goldens (two-tier exp probe, m5g conventions), spec-vs-non-spec
# stream equivalence + rollback state digests (greedy + fixed-seed
# sampled, depths 1/2/3), forced-draft patterns (draft_override), and the
# tiered run (the M6 cache serving the MTP slabs at store layer n_main+0).
# test-m8g still runs the fixture integrity verify first. golden-m8g pins
# numpy to its baseline exp kernel via NPY_DISABLE_CPU_FEATURES on x86-64
# (macOS arm64 needs no pin) — same convention as golden-m4g/m4h/m5g.

golden-m8g:
	NPY_DISABLE_CPU_FEATURES="X86_V3 X86_V4 AVX512_SKX AVX512_SPR" $(PY) $(M8G)/gen_fixtures.py

$(BIN8G):
	mkdir -p $(BIN8G)

$(BIN8G)/test_m8g: $(M8G)/test_m8g.c $(M8G_DEPS) | $(BIN8G)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN8G)/test_m8g_ubsan: $(M8G)/test_m8g.c $(M8G_DEPS) | $(BIN8G)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -ffp-contract=off $(if $(filter-out Darwin,$(UNAME)),-D_GNU_SOURCE) -Ic -o $@ $< $(LDLIBS) -lpthread

test-m8g: apus $(BIN8G)/test_m8g golden-m8g
	$(PY) $(M8G)/verify_fixtures.py
	APUS_THREADS=1 ./$(BIN8G)/test_m8g > $(BIN8G)/out_t1.txt
	APUS_THREADS=4 ./$(BIN8G)/test_m8g > $(BIN8G)/out_t4.txt
	APUS_THREADS=8 ./$(BIN8G)/test_m8g > $(BIN8G)/out_t8.txt
	diff $(BIN8G)/out_t1.txt $(BIN8G)/out_t4.txt
	diff $(BIN8G)/out_t1.txt $(BIN8G)/out_t8.txt
	cat $(BIN8G)/out_t4.txt
	cp $(M8G)/golden/dsa_top/config.json $(M8G)/golden/dsa_top/container/config.json
	IDS=$$($(PY) -c "import numpy as np; print(','.join(map(str, np.fromfile('$(M8G)/golden/dsa_top/prompt_ids.bin', dtype=np.int32))))"); \
	./bin/apus run --model $(M8G)/golden/dsa_top/container --ids "$$IDS" --max-tokens 16 --greedy --quiet --no-spec > $(BIN8G)/cli_nospec.txt 2>/dev/null; \
	./bin/apus run --model $(M8G)/golden/dsa_top/container --ids "$$IDS" --max-tokens 16 --greedy --quiet --spec --spec-k 3 > $(BIN8G)/cli_spec.txt 2>/dev/null; \
	./bin/apus run --model $(M8G)/golden/dsa_top/container --ids "$$IDS" --max-tokens 16 --greedy --quiet > $(BIN8G)/cli_default.txt 2>/dev/null; \
	diff $(BIN8G)/cli_nospec.txt $(BIN8G)/cli_spec.txt
	diff $(BIN8G)/cli_default.txt $(BIN8G)/cli_spec.txt
	mkdir -p $(BIN8G)/nomtp
	if ./bin/apus run --model $(BIN8G)/nomtp --ids "1,2,3" --max-tokens 4 --quiet --spec > /dev/null 2>&1; then \
	    echo "test-m8g: --spec on an MTP-less dir must fail" >&2; exit 1; \
	fi

# UBSan-only, like the other milestones (-ffp-contract=off pinned, same
# reason as ubsan-m3g)
ubsan-m8g: $(BIN8G)/test_m8g_ubsan golden-m8g
	APUS_THREADS=1 ./$(BIN8G)/test_m8g_ubsan > $(BIN8G)/out_u1.txt
	APUS_THREADS=4 ./$(BIN8G)/test_m8g_ubsan > $(BIN8G)/out_u4.txt
	diff $(BIN8G)/out_u1.txt $(BIN8G)/out_u4.txt
	cat $(BIN8G)/out_u4.txt

# --- M6 (GLM): expert tiering/cache + pilot prefetch --------------------
# The GLM slab-streaming expert cache (c/gcache.h — per-layer LRU +
# working set with end-of-block promotion, generation-tagged miss overlap
# on a pthread I/O pool, demand/speculative job classes, dequant-on-fill
# via the m3g kernel, RSS guard, buffer recycling) behind the M5
# apus_gmodel_expert() seam, plus the GLM router-lookahead pilot
# (c/gpilot.h). Gates: eager == big-cache == 1-slot-cache == pilot
# digests (bitwise cache neutrality, host-exp independent), one pread per
# fill (instrumented), eviction correctness under a 1-slot budget, pilot
# recall + prefetch coverage on the clustered vs random locality fixtures
# (synchronous I/O for deterministic counters), the state-sizing helper.
# Thread-count independence diffed across APUS_THREADS=1/4/8.

$(BIN6G):
	mkdir -p $(BIN6G)

$(BIN6G)/test_m6g: $(M6G)/test_m6g.c $(M6G_DEPS) | $(BIN6G)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6G)/test_m6g_ubsan: $(M6G)/test_m6g.c $(M6G_DEPS) | $(BIN6G)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -ffp-contract=off $(if $(filter-out Darwin,$(UNAME)),-D_GNU_SOURCE) -Ic -o $@ $< $(LDLIBS) -lpthread

# P2: standalone slab-I/O bench (informational, NOT a gate) — raw pread
# ceiling (F_NOCACHE vs cached x thread counts) + dequant-on-fill cost on
# a real glm5_next container. Usage: make bench-gio [CONTAINER=dir]
CONTAINER ?= weights/glm-5.3-flash
$(BIN6G)/bench_gio: $(M6G)/bench_gio.c $(M6G_DEPS) | $(BIN6G)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

bench-gio: $(BIN6G)/bench_gio
	./$(BIN6G)/bench_gio $(CONTAINER)

golden-m6g:
	$(PY) $(M6G)/gen_cluster.py

test-m6g: $(BIN6G)/test_m6g golden-m5g golden-m6g
	APUS_THREADS=1 ./$(BIN6G)/test_m6g > $(BIN6G)/out_t1.txt
	APUS_THREADS=4 ./$(BIN6G)/test_m6g > $(BIN6G)/out_t4.txt
	APUS_THREADS=8 ./$(BIN6G)/test_m6g > $(BIN6G)/out_t8.txt
	diff $(BIN6G)/out_t1.txt $(BIN6G)/out_t4.txt
	diff $(BIN6G)/out_t1.txt $(BIN6G)/out_t8.txt
	cat $(BIN6G)/out_t4.txt

# UBSan-only, like the other milestones (-ffp-contract=off pinned, same
# reason as ubsan-m3g)
ubsan-m6g: $(BIN6G)/test_m6g_ubsan golden-m5g golden-m6g
	APUS_THREADS=1 ./$(BIN6G)/test_m6g_ubsan > $(BIN6G)/out_u1.txt
	APUS_THREADS=4 ./$(BIN6G)/test_m6g_ubsan > $(BIN6G)/out_u4.txt
	diff $(BIN6G)/out_u1.txt $(BIN6G)/out_u4.txt
	cat $(BIN6G)/out_u4.txt

# engine CLI binary
bin:
	mkdir -p bin

ifdef metal
apus: bin/apus_metal
else
apus: bin/apus
endif

bin/apus: c/apus.c $(APUS_DEPS) | bin
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

# --- M7b: the optional GLM Metal backend ------------------------------------------
# c/backend_gmetal.mm — BF16 GEMV/GEMM + the fused
# FP8-block-dequant GEMM, BITWISE == the pinned CPU kernels (the shaders
# reproduce the sequential-k two-rounding order exactly; the only measured
# class is the fp32-subnormal denormal flush, unreachable from normative
# data — tests/m7b/README.md). Ephemeral zero-copy wraps only: no pointer
# cache, nothing held across the M6 layer_end/RSS-guard boundary.
# `make metal=1 apus` (or `make bin/apus_metal`) links the backend;
# apus.c enables it and the GLM engine consults its hook table. CPU
# stays the default; bin/apus is behaviorally untouched. macOS-only
# (M12a-1: the targets are stubbed with a clear error on Linux).
# (The V4 backend c/backend_metal.mm + its test_kernels/bench_metal legs
# were removed 2026-09-05 with the V4 engine cleanup.)

ifeq ($(UNAME),Darwin)

M7B    := tests/m7b
BIN7B  := $(M7B)/bin
M7B_DEPS := $(APUS_DEPS) c/backend_gmetal.h

METAL_CXX      := clang++
METAL_CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
METAL_LDLIBS   := -framework Foundation -framework Metal

bin/apus_metal: c/apus.c c/backend_gmetal.mm $(M7B_DEPS) | bin
	$(CC) $(CFLAGS) -Ic -c c/apus.c -o bin/apus_metal_main.o
	$(METAL_CXX) $(METAL_CXXFLAGS) -Ic -c c/backend_gmetal.mm -o bin/backend_gmetal.o
	$(METAL_CXX) -o $@ bin/apus_metal_main.o bin/backend_gmetal.o $(LDLIBS) -lpthread $(METAL_LDLIBS)

$(BIN7B):
	mkdir -p $(BIN7B)

$(BIN7B)/backend_gmetal.o: c/backend_gmetal.mm c/backend_gmetal.h c/bf16.h c/fp8blk.h | $(BIN7B)
	$(METAL_CXX) $(METAL_CXXFLAGS) -Ic -c c/backend_gmetal.mm -o $@

$(BIN7B)/test_gkernels: $(M7B)/test_gkernels.c $(BIN7B)/backend_gmetal.o $(M7B_DEPS)
	$(CC) $(CFLAGS) -Ic -c $(M7B)/test_gkernels.c -o $(BIN7B)/test_gkernels_c.o
	$(METAL_CXX) -o $@ $(BIN7B)/test_gkernels_c.o $(BIN7B)/backend_gmetal.o $(LDLIBS) $(METAL_LDLIBS)

$(BIN7B)/test_gmodel: $(M7B)/test_gmodel.c $(BIN7B)/backend_gmetal.o $(M7B_DEPS)
	$(CC) $(CFLAGS) -Ic -c $(M7B)/test_gmodel.c -o $(BIN7B)/test_gmodel_c.o
	$(METAL_CXX) -o $@ $(BIN7B)/test_gmodel_c.o $(BIN7B)/backend_gmetal.o $(LDLIBS) -lpthread $(METAL_LDLIBS)

$(BIN7B)/bench_gmetal: $(M7B)/bench_gmetal.c $(BIN7B)/backend_gmetal.o $(M7B_DEPS)
	$(CC) $(CFLAGS) -Ic -c $(M7B)/bench_gmetal.c -o $(BIN7B)/bench_gmetal_c.o
	$(METAL_CXX) -o $@ $(BIN7B)/bench_gmetal_c.o $(BIN7B)/backend_gmetal.o $(LDLIBS) -lpthread $(METAL_LDLIBS)

# The GLM gates: test_gkernels (kernel bitwise battery) + test_gmodel
# (CPU-vs-Metal bitwise on the m5g container, eager and 1-slot tiered,
# thread-count diffed) + the m7a server suite on the Metal binary with
# the floor at 0 so the tiny parrot shapes are offloaded.
test-m7b: $(BIN7B)/test_gkernels $(BIN7B)/test_gmodel bin/apus_metal golden-m5g golden-m7a golden-m8g
	./$(BIN7B)/test_gkernels
	APUS_THREADS=1 ./$(BIN7B)/test_gmodel > $(BIN7B)/out_t1.txt
	APUS_THREADS=4 ./$(BIN7B)/test_gmodel > $(BIN7B)/out_t4.txt
	APUS_THREADS=8 ./$(BIN7B)/test_gmodel > $(BIN7B)/out_t8.txt
	diff $(BIN7B)/out_t1.txt $(BIN7B)/out_t4.txt
	diff $(BIN7B)/out_t1.txt $(BIN7B)/out_t8.txt
	cat $(BIN7B)/out_t4.txt
	APUS_BIN=$(CURDIR)/bin/apus_metal APUS_METAL=1 APUS_GMETAL_MIN_KB=0 APUS_GMETAL_MAX_M=0 $(PY) $(M7A)/test_server.py

bench-m7b: $(BIN7B)/bench_gmetal
	./$(BIN7B)/bench_gmetal

# UBSan on the C side (the .mm stays unsanitized, like ASan-less m2..m7a)
$(BIN7B)/test_gkernels_ubsan: $(M7B)/test_gkernels.c $(BIN7B)/backend_gmetal.o $(M7B_DEPS)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -ffp-contract=off -Ic -c $(M7B)/test_gkernels.c -o $(BIN7B)/test_gkernels_ubsan_c.o
	$(METAL_CXX) -fsanitize=undefined -o $@ $(BIN7B)/test_gkernels_ubsan_c.o $(BIN7B)/backend_gmetal.o $(LDLIBS) $(METAL_LDLIBS)

$(BIN7B)/test_gmodel_ubsan: $(M7B)/test_gmodel.c $(BIN7B)/backend_gmetal.o $(M7B_DEPS)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -ffp-contract=off -Ic -c $(M7B)/test_gmodel.c -o $(BIN7B)/test_gmodel_ubsan_c.o
	$(METAL_CXX) -fsanitize=undefined -o $@ $(BIN7B)/test_gmodel_ubsan_c.o $(BIN7B)/backend_gmetal.o $(LDLIBS) -lpthread $(METAL_LDLIBS)

$(BIN7B)/apus_metal_ubsan: c/apus.c $(BIN7B)/backend_gmetal.o $(M7B_DEPS)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -c c/apus.c -o $(BIN7B)/apus_metal_ubsan_main.o
	$(METAL_CXX) -fsanitize=undefined -o $@ $(BIN7B)/apus_metal_ubsan_main.o $(BIN7B)/backend_gmetal.o $(LDLIBS) -lpthread $(METAL_LDLIBS)

ubsan-m7b: $(BIN7B)/test_gkernels_ubsan $(BIN7B)/test_gmodel_ubsan $(BIN7B)/apus_metal_ubsan golden-m5g golden-m7a golden-m8g
	./$(BIN7B)/test_gkernels_ubsan
	APUS_THREADS=1 ./$(BIN7B)/test_gmodel_ubsan > $(BIN7B)/out_u1.txt
	APUS_THREADS=4 ./$(BIN7B)/test_gmodel_ubsan > $(BIN7B)/out_u4.txt
	diff $(BIN7B)/out_u1.txt $(BIN7B)/out_u4.txt
	cat $(BIN7B)/out_u4.txt
	APUS_BIN=$(CURDIR)/$(BIN7B)/apus_metal_ubsan APUS_METAL=1 APUS_GMETAL_MIN_KB=0 APUS_GMETAL_MAX_M=0 $(PY) $(M7A)/test_server.py

else  # !Darwin: Metal is macOS-only (M12a-1)

bin/apus_metal test-m7b bench-m7b ubsan-m7b:
	@echo "error: the Metal backend (m7b) is macOS-only, not available on $(UNAME)" >&2; exit 1

endif

# --- M7a: OpenAI-compatible server (apus serve NDJSON + tools/server.py) ---
# GLM-5.3-Flash serving: the engine auto-selects glm5_next containers via
# apus.index.json and serves them with the M5/M6 GLM stack (eager or
# --tiered + pilot). End-to-end on the scripted GLM parrot fixtures
# (regenerate with golden-m7a; see tests/m7a/README.md for the protocol
# and endpoint coverage). The NPY pin matches the other GLM goldens (the
# parrot's margins make the token gate host-independent anyway).
# M8d: the suite also covers serve --spec (default ON) on the tests/m8g
# dsa_top container (the parrot has no MTP block) — hence golden-m8g.

golden-m7a:
	NPY_DISABLE_CPU_FEATURES="X86_V3 X86_V4 AVX512_SKX AVX512_SPR" $(PY) $(M7A)/gen_fixtures.py

test-m7a: bin/apus golden-m7a golden-m8g
	$(PY) $(M7A)/test_server.py

bin/apus_ubsan: c/apus.c $(APUS_DEPS) | bin
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

# UBSan-only, like the other milestones (Apple ASan runtime broken here)
ubsan-m7a: bin/apus_ubsan golden-m7a golden-m8g
	APUS_BIN=$(CURDIR)/bin/apus_ubsan $(PY) $(M7A)/test_server.py

# --- M15x: Windows cross-compile gate (build-only) -------------------------
# `make win-cross` cross-compiles bin/apus + the CI windows battery's C
# test binaries with brew mingw-w64 (x86_64-w64-mingw32-gcc 16.2; thread
# model posix = winpthreads — static libwinpthread.a/libpsapi.a verified
# present in /opt/homebrew/opt/mingw-w64), via command-line overrides that
# force the OS=Windows_NT platform branches at the top of this file
# (gnu11, the -fno-tree-vectorize pair, -lpsapi -static) while
# UNAME!=Darwin skips the clang/Accelerate paths. Build-ONLY: nothing
# executes on this host (no wine) — running the battery stays with CI
# windows-latest. collect2 appends .exe to the extension-less -o names, so
# no rule changes. `-B` forces the rebuild (an earlier version rm'd the
# extension-less targets first, which DELETED the native bin/apus from the
# tree until the next native build — the .exe artifacts now simply coexist
# with the native binaries; the cross .exe never satisfies the
# extension-less target name, so native builds recompile natively).
WIN_CROSS_VARS := OS=Windows_NT UNAME=MinGW-cross CC=x86_64-w64-mingw32-gcc
WIN_CROSS_BINS := bin/apus \
     $(BIN)/test_tok $(BIN)/test_encoding \
     $(BIN3G)/test_m3g \
     $(BIN4G)/test_m4g $(BIN4H)/test_m4h \
     $(BIN5G)/test_m5g $(BIN6G)/test_m6g \
     $(BIN8G)/test_m8g

win-cross:
	$(MAKE) -B $(WIN_CROSS_VARS) $(WIN_CROSS_BINS)
	@echo "win-cross: $(words $(WIN_CROSS_BINS)) PE binaries built (build-only; execution = CI windows-latest)"

clean:
	rm -rf $(BIN) $(M2)/golden $(BIN3G) $(M3G)/golden $(BIN4G) $(M4G)/golden $(M7A)/fixtures $(BIN7B) $(BIN6G) $(M6G)/golden $(BIN4H) $(M4H)/golden $(BIN5G) $(M5G)/golden $(BIN8G) $(M8G)/golden bin
