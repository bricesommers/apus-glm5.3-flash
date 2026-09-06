/*
 * c/gmodel.h — GLM-5.3-Flash full model forward (M5): config parsing,
 * M1 container (format v2) weight loading, and the 45-layer mHC stack
 * forward — token embed (BF16) -> per-layer [mHC collapse -> weighted
 * RMSNorm -> KDA | DSA -> mHC expand -> mHC collapse -> weighted RMSNorm
 * -> dense MLP | MoE -> mHC expand] -> unweighted-mean HyperHead ->
 * final RMSNorm -> BF16 lm_head logits. M8b adds: the MTP (NextN) block
 * load (layers.<L>.*, ApusGmodelMtpW view + expert-store layer n_main+0 —
 * the forward glue lives in c/gmtp.h), the per-token-interleaved decode
 * batch (apus_gmodel_decode_batch — bitwise == one-by-one decode by
 * construction), and the h-out surface (last-layer stream + post-norm
 * rows) the MTP glue reads.
 * C11, libc only (+ the M3/M4 building blocks and c/st.h + c/json.h for
 * the container). Scalar + the m3g bitwise kernels; SIMD/mt beyond that
 * is M7 perf work and must preserve these orders bitwise.
 *
 * Normative reference: tools/oracle.py model_forward (f32-faithful mode),
 * porting reference/inference/modeling_glm5_next.py (glm5:1260-1330,
 * 1432-1495, 2180-2182). Every sublayer call is the M4a/M4b gated piece
 * (c/gmhc.h, c/gmoe.h, c/gkda.h, c/gdsa.h); the model level adds NO new
 * numerics: per-token mHC map/collapse/expand calls are bitwise ==
 * the oracle's batched rows (the maps are per-token independent), the
 * KDA ORDERING CONTRACT (docs/ARCHITECTURE.md §4) is honored by passing
 * decode=0 on the prefill path (CHUNKED KDA) and decode=1 on the decode
 * path (RECURRENT KDA) — the phase is explicit in the API, never
 * inferred from s.
 *
 * Container loading (M1 format v2, tests/m1/README.md is the spec):
 * dense tensors are resolved by name through model.safetensors.index.json
 * (c/st.h ApusStSet, lazy materialization; BF16/F32/FP8 payloads stay
 * zero-copy views into shard buffers). Routed experts go through the
 * v2 manifest's expert_slabs: ONE pread per expert slab (the M1
 * coalescing invariant) via c/st.h ApusStLazy, the 6 members carved by
 * the pinned order (gate.up.down x weight,scale) and config-derived
 * shapes, then dequantized to BF16 once (m3g apus_fp8blk_dequant).
 *
 * M6 SEAM (tiering/cache): M5 keeps every expert RESIDENT (eager
 * dequant at open into an owned arena). The ONLY access path is
 * apus_gmodel_expert() (+ the per-call eg/eu/ed pointer fill in the MoE
 * wiring) — M6 slots the slab-streaming cache (c/gcache.h) behind that
 * accessor (cache slots instead of arena pointers) via
 * apus_gmodel_open2(tiered): the wiring then resolves only the ROUTED
 * experts per call (pre-pass union -> demand hints -> just-in-time
 * resolves -> forward -> layer_end), bitwise-identical to the eager
 * path (tests/m6g). DSA KV/indexer caches and KDA states live in
 * ApusGmodelState, sized by the caller-supplied kv capacity.
 * Optional ApusGmodelHooks (post-attention + routed) are the c/gpilot.h
 * prefetch surface; hooks never touch numerics.
 *
 * P4 (both numerics-neutral, gated): (a) tiered PREFILL MoE is
 * token-chunked (APUS_GPREFILL_MOE_CHUNK, default 8; 0 = unchunked) —
 * each chunk pre-passes/resolves/forwards its tokens and layer_end
 * promotes before the next chunk, bounding the live expert union at
 * chunk*topk payloads instead of all-s (the P2 41.5 GB prefill peak);
 * bitwise-safe by per-token MoE independence (the m6g tiered==eager
 * digest gate pins it); (b) when the GLM Metal backend is enabled at
 * open, all model-owned dense weight regions are registered with
 * c/backend_gmetal.h for persistent zero-copy wraps (unregistered at
 * close; gcache expert payloads are never registered — the ephemeral
 * per-op policy and the M6 layer_end invariant are unchanged).
 *
 * Usage: #define APUS_GMODEL_IMPLEMENTATION in exactly one TU. That TU
 * (or others linked with it) must instantiate the dependencies:
 * APUS_BF16_IMPLEMENTATION, APUS_FP8BLK_IMPLEMENTATION,
 * APUS_GMHC_IMPLEMENTATION, APUS_GMOE_IMPLEMENTATION,
 * APUS_GKDA_IMPLEMENTATION, APUS_GDSA_IMPLEMENTATION,
 * APUS_ST_IMPLEMENTATION, APUS_JSON_IMPLEMENTATION, and (M6)
 * APUS_GCACHE_IMPLEMENTATION + APUS_COMPAT_IMPLEMENTATION (referenced
 * unconditionally; the eager path simply never calls them).
 */
#ifndef APUS_GMODEL_H
#define APUS_GMODEL_H

#include <stddef.h>
#include <stdint.h>

#include "gkda.h"
#include "gdsa.h"
#include "gmoe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --- config (reference/config.json text_config subset, or the flat
 * oracle fixture config) ---------------------------------------------*/

typedef struct {
    int hidden_size, vocab_size, num_hidden_layers;
    float rms_norm_eps;
    int hc_mult, hc_sinkhorn_iters;
    float hc_eps;
    int first_k_dense_replace;
    int intermediate_size, moe_intermediate_size;
    int n_routed_experts, n_shared_experts, num_experts_per_tok;
    float routed_scaling_factor;
    int norm_topk_prob;
    float swiglu_limit;
    int num_attention_heads, q_lora_rank, kv_lora_rank;
    int qk_nope_head_dim, v_head_dim;
    int index_n_heads, index_head_dim, index_topk, index_kpool;
    int index_kpool_always_select_tail;
    int linear_num_heads, linear_head_dim, linear_conv_kernel_dim;
    float linear_lower_bound;
    int num_nextn_predict_layers;   /* 0 when absent (MTP/NextN, M8b) */
    int *layer_is_dsa;      /* [L] 1 = deepseek_sparse_attention */
    int *mlp_is_sparse;     /* [L] 1 = MoE, 0 = dense */
} ApusGmodelConfig;

/* The layer-pattern rule (verified to reproduce the real config's
 * explicit 45-layer lists EXACTLY): DSA at layers i % 4 == 3, KDA
 * elsewhere; dense MLP below first_k_dense_replace, sparse after.
 * Arrays are caller-provided [L]. */
void apus_gmodel_layer_patterns(int L, int first_k_dense,
                                int *is_dsa, int *is_sparse);

/* Parse a config JSON. Accepts BOTH the real reference/config.json
 * (top-level "text_config" object + "linear_attn_config" sub-object +
 * explicit layer_types/mlp_layer_types lists) and the flat oracle
 * fixture config (tools/oracle.py generate). Explicit layer lists are
 * used when present; the rule above fills them otherwise. Returns 0 on
 * success (-1 + err otherwise). */
int apus_gmodel_config_load(const char *path, ApusGmodelConfig *cfg,
                            char *err, size_t errcap);
void apus_gmodel_config_free(ApusGmodelConfig *cfg);

/* --- model ------------------------------------------------------------*/

typedef struct ApusGmodel ApusGmodel;

/* Open the model: parse the config, read apus.index.json (format v2,
 * verified), resolve all dense tensors, and EAGERLY load+dequant every
 * routed expert through its slab (one pread per expert). Returns NULL
 * on error (err filled if given). */
ApusGmodel *apus_gmodel_open(const char *container_dir,
                             const char *config_path,
                             char *err, size_t errcap);

/* M6 tiered open (c/gcache.h behind apus_gmodel_expert): with
 * tiered != 0 the routed experts are NOT loaded at open; a slab-streaming
 * cache is built instead (LRU budget cache_bytes, 0 = APUS_GEXPERT_CACHE_MB
 * env; io_threads 0 = env/default 4, <0 = synchronous I/O; nocache/
 * rss_budget_bytes 0 = env defaults). The forward is bitwise-identical
 * to the eager path (tests/m6g). tier == NULL == apus_gmodel_open. */
typedef struct {
    int    tiered;
    size_t cache_bytes;
    int    slots_per_layer;
    int    io_threads;
    int    nocache;
    size_t rss_budget_bytes;
    int    want_mtp;        /* M8b: load the MTP (NextN) block when the
                             * container carries one (layers.<L>.*) and size
                             * the tiered cache to serve its slabs (store
                             * layer n_main+0). Eager open (tier == NULL or
                             * tiered == 0) always loads it when present —
                             * eager mode is fixture-scale only. 0 keeps the
                             * pre-M8b behavior exactly (records skipped,
                             * cache sized for the main layers only). */
} ApusGmodelTierCfg;
ApusGmodel *apus_gmodel_open2(const char *container_dir,
                              const char *config_path,
                              const ApusGmodelTierCfg *tier,
                              char *err, size_t errcap);
void apus_gmodel_close(ApusGmodel *m);

/* The M6 expert cache (NULL in eager mode) — stats/introspection. */
struct ApusGcache *apus_gmodel_cache(const ApusGmodel *m);

const ApusGmodelConfig *apus_gmodel_config(const ApusGmodel *m);
int apus_gmodel_layer_is_dsa(const ApusGmodel *m, int layer);
int apus_gmodel_layer_is_sparse(const ApusGmodel *m, int layer);

/* THE expert accessor (the M6 seam — see the file-top comment). In
 * tiered mode the returned views point into cache slots and stay valid
 * until the wiring's layer_end on that layer (callers must re-fetch per
 * forward — the wiring always does). */
typedef struct {
    const uint16_t *gate;   /* [moe_inter, dim] BF16 codes (dequantized) */
    const uint16_t *up;     /* [moe_inter, dim] */
    const uint16_t *down;   /* [dim, moe_inter] */
} ApusGmodelExpertW;
const ApusGmodelExpertW *apus_gmodel_expert(const ApusGmodel *m,
                                            int layer, int e);

/* --- M8b: MTP (NextN) block + speculative-decode surfaces ---------------
 * The MTP block (layers.<L>.* in the container, the apus-mtp-* shard
 * group) is a full DSA + sparse-MoE decoder layer with PLAIN residuals
 * (no mHC) plus the e/h glue (enorm/hnorm/eh_proj BF16, shared_head.norm),
 * sharing the main model's embed/lm_head. Detected at open from
 * num_nextn_predict_layers (config) or the eh_proj tensor's presence;
 * loaded per ApusGmodelTierCfg.want_mtp. Its routed experts sit at expert-
 * store layer apus_gmodel_mtp_layer() == num_hidden_layers (n_main+0) —
 * apus_gmodel_expert(m, L, e) serves them (eager arena or the M6 cache). */
int apus_gmodel_has_mtp(const ApusGmodel *m);
int apus_gmodel_mtp_layer(const ApusGmodel *m);     /* L, or -1 */
typedef struct {
    const ApusGdsaW *dsa;               /* block attention weights */
    const uint16_t *input_norm, *post_norm;     /* [dim] BF16 views */
    const ApusGmoeRouterW *router;
    const uint16_t *shr_g, *shr_u, *shr_d;      /* shared expert (owned) */
    const uint16_t *enorm, *hnorm;              /* [dim] BF16 views */
    const uint16_t *eh_proj;                    /* [dim, 2*dim] BF16 view */
    const uint16_t *shared_norm;                /* [dim] BF16 view */
} ApusGmodelMtpW;
int apus_gmodel_mtp(const ApusGmodel *m, ApusGmodelMtpW *out);  /* -1 none */
const uint16_t *apus_gmodel_embed(const ApusGmodel *m); /* [V, dim] view */
const uint16_t *apus_gmodel_head(const ApusGmodel *m);  /* [V, dim] view */

/* --- M6 hooks (the c/gpilot.h prefetch surface) --------------------------
 * Optional, NULL by default, never touch numerics (read-only on the
 * hidden state; tests/m6g gates digest equality hooks-ON vs OFF).
 * post_attn fires after layer l's attention-site mHC expand with the
 * block stream h [s, hc*dim] BF16 codes and the call's base position;
 * routed fires after a sparse layer's selections are known (tiered mode
 * only), idx [s*topk]. Both run on the compute thread. */
typedef struct {
    void *ctx;
    void (*post_attn)(void *ctx, int layer, const uint16_t *h, size_t s,
                      size_t pos0);
    void (*routed)(void *ctx, int layer, const int32_t *idx, size_t s,
                   size_t pos0);
} ApusGmodelHooks;
void apus_gmodel_set_hooks(ApusGmodel *m, const ApusGmodelHooks *hooks);

/* Read-only router/pilot view of a sparse layer (the c/gpilot.h attach
 * input). Returns 0 on success, -1 for dense layers. */
typedef struct {
    const float *hc_fn, *hc_base, *hc_scale;    /* ffn mHC (fn widened) */
    const uint16_t *post_norm;                  /* [dim] BF16 codes */
    const uint16_t *gate_w;                     /* [E, dim] BF16 codes */
    const float *gate_bias;                     /* [E] f32 */
} ApusGmodelPilotView;
int apus_gmodel_pilot_view(const ApusGmodel *m, int layer,
                           ApusGmodelPilotView *out);

/* --- state (decode-carried) + scratch arena ---------------------------*/

typedef struct ApusGmodelState ApusGmodelState;

/* kv_cap: total token capacity (prompt + decode) for the DSA KV/indexer
 * caches. KDA states are context-length independent. All states
 * zero-initialized (fresh prefill). */
ApusGmodelState *apus_gmodel_state_new(const ApusGmodel *m, size_t kv_cap);
void apus_gmodel_state_free(ApusGmodelState *st);
size_t apus_gmodel_pos(const ApusGmodelState *st);

/* State-cache memory for a given kv_cap (M6 memory management): the
 * exact bytes apus_gmodel_state_new allocates for the KDA conv+recurrent
 * states (context-length independent) and the DSA KV/indexer caches
 * (linear in kv_cap) — the caller sizes kv_cap to its budget. The engine
 * NEVER evicts or quantizes KV (the quality invariant): exceeding the
 * capacity fails the forward loudly (-1). Engine scratch is excluded. */
size_t apus_gmodel_state_bytes(const ApusGmodel *m, size_t kv_cap);

/* Per-layer state accessors (NULL for the other kind). */
const ApusGkdaState *apus_gmodel_kda_state(const ApusGmodelState *st,
                                           int layer);
const ApusGdsaState *apus_gmodel_dsa_state(const ApusGmodelState *st,
                                           int layer);

/* Snapshot/rollback support (M8b, c/gmtp.h ApusGspec): mutable state
 * views + pos/cap. KDA conv/rec states are flat memcpy-able arrays; DSA
 * rollback is the n-rewind (cache rows past n are dead capacity — the
 * deterministic re-feed rewrites them; attention/indexer never read
 * beyond n). */
ApusGkdaState *apus_gmodel_kda_state_mut(ApusGmodelState *st, int layer);
ApusGdsaState *apus_gmodel_dsa_state_mut(ApusGmodelState *st, int layer);
size_t apus_gmodel_state_cap(const ApusGmodelState *st);
void apus_gmodel_set_pos(ApusGmodelState *st, size_t pos);

/* --- forward ------------------------------------------------------------*/

/* TEST-ONLY teacher-forcing (tests/m5g tolerance tier; NULL in
 * production): per-layer override arrays indexed by layer (entries for
 * layers of the other kind are ignored, pass NULL). router_idx[l] is
 * [s, num_experts_per_tok]; indexer_topk[l] is [s, index_topk+kpool-1].
 * Selections are teacher-forced because structural relu-clip zero ties
 * in the indexer (and sub-margin router gaps under compounding bf16
 * flips) are host-exp sensitive — tests/m5g/README.md. */
typedef struct {
    const int32_t *const *router_idx;
    const int32_t *const *indexer_topk;
} ApusGmodelForces;

/* Prefill: s tokens from position st->pos (must be 0 for a fresh state;
 * continuations on carried states follow the §4.4 contract). KDA layers
 * take the CHUNKED path. logits: [s, vocab] BF16 codes out. trace_h
 * (TEST-ONLY, may be NULL): [L, s, hc_mult, dim] codes, filled with each
 * layer's block-output stream. Returns 0 on success. */
int apus_gmodel_prefill(ApusGmodel *m, ApusGmodelState *st,
                        const int32_t *ids, size_t s,
                        uint16_t *logits, const ApusGmodelForces *forces,
                        uint16_t *trace_h);
/* Decode: one token, RECURRENT KDA path. logits: [vocab] codes out.
 * trace_h (may be NULL): [L, 1, hc_mult, dim] codes. */
int apus_gmodel_decode_step(ApusGmodel *m, ApusGmodelState *st,
                            int32_t id, uint16_t *logits,
                            const ApusGmodelForces *forces,
                            uint16_t *trace_h);

/* M8b decode batch (the speculative verify batch): s tokens, RECURRENT
 * ordering — KDA layers step conv+recurrent state token by token
 * (c/gkda.h), DSA layers take the per-token-interleaved path
 * (c/gdsa.h apus_gdsa_decode_batch): BITWISE == s one-by-one
 * apus_gmodel_decode_step calls by construction. logits: [s, vocab] codes.
 * h_out/yn_out (the MTP h-surface, either may be NULL): h_out
 * [s, hc_mult*dim] = the LAST layer's post-block mHC stream per position;
 * yn_out [s, dim] = the post-final-norm rows (the lm_head input). The
 * c/gmtp.h source selector derives the hnorm input from these. */
int apus_gmodel_decode_batch(ApusGmodel *m, ApusGmodelState *st,
                             const int32_t *ids, size_t s,
                             uint16_t *logits, const ApusGmodelForces *forces,
                             uint16_t *h_out, uint16_t *yn_out);
/* Prefill variant with the same h-surface (the spec prefill's true-pair
 * replay input). Chunked-KDA ordering, as apus_gmodel_prefill. */
int apus_gmodel_prefill_h(ApusGmodel *m, ApusGmodelState *st,
                          const int32_t *ids, size_t s,
                          uint16_t *logits, const ApusGmodelForces *forces,
                          uint16_t *h_out, uint16_t *yn_out);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
/* Outside the include guard (the c/blas.h pattern): safe to re-include. */
#if defined(APUS_GMODEL_IMPLEMENTATION) && !defined(APUS_GMODEL_IMPL_INCLUDED)
#define APUS_GMODEL_IMPL_INCLUDED

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fp8blk.h"
#include "gcache.h"
#include "json.h"
#include "st.h"
#include "compat.h"     /* apus_env_int (APUS_GPREFILL_MOE_CHUNK) */

/* --- config -------------------------------------------------------------*/

void apus_gmodel_layer_patterns(int L, int first_k_dense,
                                int *is_dsa, int *is_sparse) {
    for (int i = 0; i < L; i++) {
        is_dsa[i] = (i % 4 == 3);
        is_sparse[i] = (i >= first_k_dense);
    }
}

void apus_gmodel_config_free(ApusGmodelConfig *cfg) {
    free(cfg->layer_is_dsa);
    free(cfg->mlp_is_sparse);
    memset(cfg, 0, sizeof *cfg);
}

static int apus_gm_cfg_num(JVal *o, const char *key, double *out) {
    JVal *v = o ? json_obj_get(o, key) : NULL;
    if (!v || json_type(v) != J_NUM) return -1;
    *out = json_num(v);
    return 0;
}

static int apus_gm_cfg_bool(JVal *o, const char *key, int *out) {
    JVal *v = o ? json_obj_get(o, key) : NULL;
    if (!v) return -1;
    if (json_type(v) == J_BOOL) { *out = json_bool(v); return 0; }
    if (json_type(v) == J_NUM) { *out = (int)json_num(v) != 0; return 0; }
    return -1;
}

int apus_gmodel_config_load(const char *path, ApusGmodelConfig *cfg,
                            char *err, size_t errcap) {
    char jerr[160];
    JVal *root = json_parse_file(path, jerr, sizeof jerr);
    if (!root) {
        if (err && errcap) snprintf(err, errcap, "gmodel cfg: %s", jerr);
        return -1;
    }
    memset(cfg, 0, sizeof *cfg);
    JVal *tc = json_obj_get(root, "text_config");
    if (!tc) tc = root;                 /* flat fixture config */
    double d;
    int fail = 0;
    struct { const char *key; int *out; } ints[] = {
        {"hidden_size", &cfg->hidden_size},
        {"vocab_size", &cfg->vocab_size},
        {"num_hidden_layers", &cfg->num_hidden_layers},
        {"hc_mult", &cfg->hc_mult},
        {"hc_sinkhorn_iters", &cfg->hc_sinkhorn_iters},
        {"first_k_dense_replace", &cfg->first_k_dense_replace},
        {"intermediate_size", &cfg->intermediate_size},
        {"moe_intermediate_size", &cfg->moe_intermediate_size},
        {"n_routed_experts", &cfg->n_routed_experts},
        {"n_shared_experts", &cfg->n_shared_experts},
        {"num_experts_per_tok", &cfg->num_experts_per_tok},
        {"num_attention_heads", &cfg->num_attention_heads},
        {"q_lora_rank", &cfg->q_lora_rank},
        {"kv_lora_rank", &cfg->kv_lora_rank},
        {"qk_nope_head_dim", &cfg->qk_nope_head_dim},
        {"v_head_dim", &cfg->v_head_dim},
        {"index_n_heads", &cfg->index_n_heads},
        {"index_head_dim", &cfg->index_head_dim},
        {"index_topk", &cfg->index_topk},
        {"index_kpool", &cfg->index_kpool},
    };
    for (size_t i = 0; i < sizeof ints / sizeof ints[0]; i++)
        if (apus_gm_cfg_num(tc, ints[i].key, &d)) fail = 1;
        else *ints[i].out = (int)d;
    struct { const char *key; float *out; } flts[] = {
        {"rms_norm_eps", &cfg->rms_norm_eps},
        {"hc_eps", &cfg->hc_eps},
        {"routed_scaling_factor", &cfg->routed_scaling_factor},
        {"swiglu_limit", &cfg->swiglu_limit},
    };
    for (size_t i = 0; i < sizeof flts / sizeof flts[0]; i++)
        if (apus_gm_cfg_num(tc, flts[i].key, &d)) fail = 1;
        else *flts[i].out = (float)d;
    if (apus_gm_cfg_bool(tc, "norm_topk_prob", &cfg->norm_topk_prob))
        fail = 1;
    if (apus_gm_cfg_bool(tc, "index_kpool_always_select_tail",
                         &cfg->index_kpool_always_select_tail))
        fail = 1;
    /* MTP/NextN (M8b): optional — the flat fixture configs predate the key
     * (detection then falls back to container tensor presence at open). */
    if (!apus_gm_cfg_num(tc, "num_nextn_predict_layers", &d))
        cfg->num_nextn_predict_layers = (int)d;
    /* KDA: real config nests linear_attn_config; the fixture is flat. */
    JVal *la = json_obj_get(tc, "linear_attn_config");
    if (la) {
        if (apus_gm_cfg_num(la, "num_heads", &d)) fail = 1;
        else cfg->linear_num_heads = (int)d;
        if (apus_gm_cfg_num(la, "head_dim", &d)) fail = 1;
        else cfg->linear_head_dim = (int)d;
        if (apus_gm_cfg_num(la, "short_conv_kernel_size", &d)) fail = 1;
        else cfg->linear_conv_kernel_dim = (int)d;
        if (apus_gm_cfg_num(la, "gate_lower_bound", &d)) fail = 1;
        else cfg->linear_lower_bound = (float)d;
    } else {
        if (apus_gm_cfg_num(tc, "linear_num_heads", &d)) fail = 1;
        else cfg->linear_num_heads = (int)d;
        if (apus_gm_cfg_num(tc, "linear_head_dim", &d)) fail = 1;
        else cfg->linear_head_dim = (int)d;
        if (apus_gm_cfg_num(tc, "linear_conv_kernel_dim", &d)) fail = 1;
        else cfg->linear_conv_kernel_dim = (int)d;
        if (apus_gm_cfg_num(tc, "linear_lower_bound", &d)) fail = 1;
        else cfg->linear_lower_bound = (float)d;
    }
    int L = cfg->num_hidden_layers;
    if (fail || L <= 0) {
        if (err && errcap)
            snprintf(err, errcap, "gmodel cfg: missing/invalid keys in %s",
                     path);
        json_free(root);
        return -1;
    }
    cfg->layer_is_dsa = (int *)malloc((size_t)L * sizeof(int));
    cfg->mlp_is_sparse = (int *)malloc((size_t)L * sizeof(int));
    apus_gmodel_layer_patterns(L, cfg->first_k_dense_replace,
                               cfg->layer_is_dsa, cfg->mlp_is_sparse);
    /* explicit lists win when present */
    JVal *lt = json_obj_get(tc, "layer_types");
    if (lt && json_type(lt) == J_ARR && json_arr_len(lt) == (size_t)L) {
        for (int i = 0; i < L; i++) {
            const char *s = json_str(json_arr_get(lt, (size_t)i));
            cfg->layer_is_dsa[i] =
                s && !strcmp(s, "deepseek_sparse_attention");
        }
    }
    JVal *mt = json_obj_get(tc, "mlp_layer_types");
    if (mt && json_type(mt) == J_ARR && json_arr_len(mt) == (size_t)L) {
        for (int i = 0; i < L; i++) {
            const char *s = json_str(json_arr_get(mt, (size_t)i));
            cfg->mlp_is_sparse[i] = s && !strcmp(s, "sparse");
        }
    }
    json_free(root);
    return 0;
}

/* --- model structs ------------------------------------------------------*/

typedef struct {
    /* mHC (fn widened to f32 at load — owned; base/scale are F32 views) */
    float *hc_attn_fn, *hc_ffn_fn;
    const float *hc_attn_base, *hc_attn_scale;
    const float *hc_ffn_base, *hc_ffn_scale;
    const uint16_t *input_norm, *post_norm;     /* BF16 views */
    ApusGkdaW kda;                              /* valid when !is_dsa */
    ApusGdsaW dsa;                              /* valid when is_dsa */
    uint16_t *kda_conv;                         /* owned [3qkv*K] concat */
    /* dense MLP (owned, dequantized) */
    uint16_t *mlp_g, *mlp_u, *mlp_d;
    /* MoE */
    ApusGmoeRouterW router;
    uint16_t *shr_g, *shr_u, *shr_d;            /* owned (dequantized) */
    ApusGmodelExpertW *experts;                 /* [E] -> expert_arena */
    uint16_t *expert_arena;                     /* owned, E*3*inter*dim */
} ApusGmodelLayer;

struct ApusGmodel {
    ApusGmodelConfig cfg;
    ApusStSet *set;             /* dense tensor views live here */
    ApusGmodelLayer *layers;    /* [L] */
    const uint16_t *embed, *norm, *head;        /* BF16 views */
    /* M6: tiered expert streaming (NULL + tiered 0 = eager arena) */
    int tiered;
    ApusGcache *cache;
    ApusGmodelHooks hooks;      /* NULL-ed by default (the pilot surface) */
    /* P4: Metal persistent wraps — model-owned weight regions registered
     * with c/backend_gmetal.h when the GLM Metal backend is enabled at
     * open (base pointers kept here for unregister at close) */
    void **mtl_reg;
    int mtl_reg_n, mtl_reg_cap;
    /* P4: token-chunked prefill MoE (tiered): APUS_GPREFILL_MOE_CHUNK
     * tokens per chunk, 0 = unchunked */
    int prefill_moe_chunk;
    /* M8b: MTP (NextN) block. has_mtp: the block was detected AND loaded
     * (eager: whenever present; tiered: tier->want_mtp). Its DSA + MoE
     * weights live in layers[L] (allocated L + has_mtp); the glue views: */
    int has_mtp;
    const uint16_t *mtp_enorm, *mtp_hnorm;      /* [dim] BF16 views */
    const uint16_t *mtp_eh_proj;                /* [dim, 2*dim] BF16 view */
    const uint16_t *mtp_shared_norm;            /* [dim] BF16 view */
};

/* P4: register a model-owned weight region with the GLM Metal backend for
 * a persistent zero-copy wrap (no-op when the backend is disabled — the
 * bf16.h TU weak stubs). The base pointer is recorded so
 * apus_gmodel_close unregisters before the memory is freed; the M6 expert
 * payloads are NEVER registered (gcache-owned, recycled — the ephemeral
 * policy stays). */
static void apus_gm_reg(ApusGmodel *m, const void *p, size_t nb) {
    if (!p || !nb || !apus_gmetal_is_enabled()) return;
    if (apus_gmetal_register_region(p, nb)) return;   /* fail-soft */
    if (m->mtl_reg_n == m->mtl_reg_cap) {
        int ncap = m->mtl_reg_cap ? 2 * m->mtl_reg_cap : 64;
        void **nr = (void **)realloc(m->mtl_reg, (size_t)ncap * sizeof *nr);
        if (!nr) { apus_gmetal_unregister_region(p); return; }
        m->mtl_reg = nr;
        m->mtl_reg_cap = ncap;
    }
    m->mtl_reg[m->mtl_reg_n++] = (void *)p;
}

struct ApusGmodelState {
    const ApusGmodel *m;
    size_t pos, cap;
    ApusGkdaState *kda;         /* [L]; fields allocated for KDA layers */
    ApusGdsaState *dsa;         /* [L]; fields allocated for DSA layers */
    /* engine-owned scratch arena (grown on demand; numerics-neutral) */
    size_t s_cap;
    uint16_t *h, *h2, *x, *sub, *yh;
    float *x4f, *pre, *post, *comb, *mixes, *xf;
    const uint16_t **eg, **eu, **ed;
    ApusGmoeScratch moe_sc;
    size_t moe_sc_s;
    int moe_sc_ready;
    ApusGdsaScratch dsa_sc;     /* P5: reusable FP8-dequant wbuf (grow-only) */
    uint16_t *xg, *xu, *xh;
    float *xxf;
    /* M6 tiered wiring scratch: pre-pass selections (routed-union) */
    int32_t *sel_idx;       /* [s*topk] (s-grown) */
    float *sel_scores, *sel_biased;     /* [E] */
    float *sel_wgt;                     /* [topk] */
    uint8_t *sel_mark;                  /* [E] union bitset (byte flags) */
};

/* --- container loading ----------------------------------------------------*/

static const ApusStTensor *apus_gm_get(ApusStSet *set, const char *name,
                                       char *err, size_t errcap) {
    const ApusStTensor *t = apus_st_set_get(set, name);
    if (!t && err && errcap)
        snprintf(err, errcap, "gmodel: tensor %s missing", name);
    return t;
}

/* dtype + 2-D/1-D shape check */
static int apus_gm_expect(const ApusStTensor *t, ApusStDtype dt,
                          int64_t d0, int64_t d1, const char *name,
                          char *err, size_t errcap) {
    int ok = t && t->dtype == dt && t->ndim == (d1 < 0 ? 1 : 2)
             && t->shape[0] == d0 && (d1 < 0 || t->shape[1] == d1);
    if (!ok && err && errcap)
        snprintf(err, errcap, "gmodel: %s dtype/shape mismatch", name);
    return ok ? 0 : -1;
}

#define APUS_GM_LOAD(...) do { \
    if (err && errcap) snprintf(err, errcap, __VA_ARGS__); \
    goto fail; \
} while (0)

/* dequant one FP8 tensor pair into a fresh bf16 buffer */
static uint16_t *apus_gm_dequant(const ApusStTensor *c,
                                 const ApusStTensor *s, size_t O, size_t K,
                                 const char *name, char *err, size_t errcap) {
    size_t sb = apus_fp8blk_nblocks(O) * apus_fp8blk_nblocks(K);
    if (!c || !s || c->dtype != APUS_ST_F8_E4M3 || s->dtype != APUS_ST_F32
        || c->nbytes != O * K || s->nbytes != sb * sizeof(float)) {
        if (err && errcap)
            snprintf(err, errcap, "gmodel: %s fp8 pair invalid", name);
        return NULL;
    }
    uint16_t *out = (uint16_t *)malloc(O * K * sizeof(uint16_t));
    /* open-time, main thread: the pool is idle here (gcache workers start
     * later), so the mt dequant is safe; elementwise => same bits. */
    apus_fp8blk_dequant_mt((const uint8_t *)c->data, (const float *)s->data,
                           out, O, K);
    return out;
}

ApusGmodel *apus_gmodel_open(const char *container_dir,
                             const char *config_path,
                             char *err, size_t errcap) {
    return apus_gmodel_open2(container_dir, config_path, NULL, err,
                             errcap);
}

ApusGmodel *apus_gmodel_open2(const char *container_dir,
                              const char *config_path,
                              const ApusGmodelTierCfg *tier,
                              char *err, size_t errcap) {
    ApusGmodel *m = (ApusGmodel *)calloc(1, sizeof *m);
    char nm[224];
    if (apus_gmodel_config_load(config_path, &m->cfg, err, errcap)) {
        free(m);
        return NULL;
    }
    m->tiered = tier && tier->tiered;
    m->prefill_moe_chunk = apus_env_int("APUS_GPREFILL_MOE_CHUNK", 8);
    const ApusGmodelConfig *c = &m->cfg;
    const int L = c->num_hidden_layers;
    const size_t dim = (size_t)c->hidden_size;
    const size_t hc = (size_t)c->hc_mult;
    const size_t mix = (2 + hc) * hc;
    const size_t inter = (size_t)c->intermediate_size;
    const size_t minter = (size_t)c->moe_intermediate_size;
    const int E = c->n_routed_experts;

    m->set = apus_st_set_open(container_dir, err, errcap);
    if (!m->set) { apus_gmodel_config_free(&m->cfg); free(m); return NULL; }
    ApusStSet *set = m->set;

    /* manifest (format v2) + expert slab records */
    snprintf(nm, sizeof nm, "%s/apus.index.json", container_dir);
    char jerr[160];
    JVal *man = json_parse_file(nm, jerr, sizeof jerr);
    if (!man) APUS_GM_LOAD("gmodel: %s", jerr);
    {
        JVal *fv = json_obj_get(man, "format_version");
        JVal *mt = json_obj_get(man, "model_type");
        if (!fv || (long)json_num(fv) != 2 || !mt
            || strcmp(json_str(mt) ? json_str(mt) : "", "glm5_next"))
            APUS_GM_LOAD("gmodel: apus.index.json not a glm5_next v2 "
                         "container");
    }
    JVal *slabs = json_obj_get(man, "expert_slabs");

    /* M8b: MTP (NextN) block detection — the config key when present
     * (num_nextn_predict_layers), else the eh_proj tensor's presence (the
     * flat fixture configs predate the key). Loaded when eager or when the
     * tiered config asks for it (want_mtp); a config that CLAIMS the block
     * but lacks its tensors fails loudly below (the loader requires them). */
    int mtp_detected = c->num_nextn_predict_layers > 0;
    if (!mtp_detected) {
        snprintf(nm, sizeof nm, "layers.%d.eh_proj.weight", L);
        mtp_detected = apus_st_set_get(set, nm) != NULL;
    }
    m->has_mtp = mtp_detected && (!m->tiered || (tier && tier->want_mtp));
    const int NL = L + m->has_mtp;

    m->layers = (ApusGmodelLayer *)calloc((size_t)NL,
                                          sizeof *m->layers);
    m->embed = m->norm = m->head = NULL;

    /* top-level */
    {
        const ApusStTensor *t;
        t = apus_gm_get(set, "embed_tokens.weight", err, errcap);
        if (apus_gm_expect(t, APUS_ST_BF16, c->vocab_size, dim,
                           "embed_tokens.weight", err, errcap))
            goto fail;
        m->embed = (const uint16_t *)t->data;
        t = apus_gm_get(set, "norm.weight", err, errcap);
        if (apus_gm_expect(t, APUS_ST_BF16, dim, -1, "norm.weight",
                           err, errcap))
            goto fail;
        m->norm = (const uint16_t *)t->data;
        t = apus_gm_get(set, "lm_head.weight", err, errcap);
        if (apus_gm_expect(t, APUS_ST_BF16, c->vocab_size, dim,
                           "lm_head.weight", err, errcap))
            goto fail;
        m->head = (const uint16_t *)t->data;
        apus_gm_reg(m, t->data, t->nbytes);
    }

    for (int l = 0; l < NL; l++) {
        const int is_mtp = (l == L);    /* M8b: the NextN block */
        ApusGmodelLayer *Lr = &m->layers[l];
        const ApusStTensor *t, *sc;
        /* norms (the MTP block has the same input/post attention norms) */
        snprintf(nm, sizeof nm, "layers.%d.input_layernorm.weight", l);
        t = apus_gm_get(set, nm, err, errcap);
        if (apus_gm_expect(t, APUS_ST_BF16, dim, -1, nm, err, errcap))
            goto fail;
        Lr->input_norm = (const uint16_t *)t->data;
        snprintf(nm, sizeof nm,
                 "layers.%d.post_attention_layernorm.weight", l);
        t = apus_gm_get(set, nm, err, errcap);
        if (apus_gm_expect(t, APUS_ST_BF16, dim, -1, nm, err, errcap))
            goto fail;
        Lr->post_norm = (const uint16_t *)t->data;
        /* mHC: fn BF16 -> widened f32 (owned); base/scale F32 views.
         * The MTP block has NO mHC (plain residuals) — skipped. */
        for (int sub = 0; !is_mtp && sub < 2; sub++) {
            const char *sn = sub ? "ffn" : "attn";
            snprintf(nm, sizeof nm, "layers.%d.hc_%s_fn", l, sn);
            t = apus_gm_get(set, nm, err, errcap);
            if (apus_gm_expect(t, APUS_ST_BF16, (int64_t)mix, hc * dim,
                               nm, err, errcap))
                goto fail;
            float *fn = (float *)malloc(mix * hc * dim * sizeof(float));
            const uint16_t *fb = (const uint16_t *)t->data;
            for (size_t i = 0; i < mix * hc * dim; i++)
                fn[i] = apus_bf16_f32(fb[i]);
            if (sub) Lr->hc_ffn_fn = fn; else Lr->hc_attn_fn = fn;
            snprintf(nm, sizeof nm, "layers.%d.hc_%s_base", l, sn);
            t = apus_gm_get(set, nm, err, errcap);
            if (apus_gm_expect(t, APUS_ST_F32, (int64_t)mix, -1, nm,
                               err, errcap))
                goto fail;
            if (sub) Lr->hc_ffn_base = (const float *)t->data;
            else Lr->hc_attn_base = (const float *)t->data;
            snprintf(nm, sizeof nm, "layers.%d.hc_%s_scale", l, sn);
            t = apus_gm_get(set, nm, err, errcap);
            if (apus_gm_expect(t, APUS_ST_F32, 3, -1, nm, err, errcap))
                goto fail;
            if (sub) Lr->hc_ffn_scale = (const float *)t->data;
            else Lr->hc_attn_scale = (const float *)t->data;
        }
        /* attention (the MTP block is always DSA — the pinned container
         * facts, M8a; a KDA-shaped MTP block fails loudly on the missing
         * DSA tensors below) */
        if (!is_mtp && !c->layer_is_dsa[l]) {
            const size_t H = (size_t)c->linear_num_heads;
            const size_t D = (size_t)c->linear_head_dim;
            const size_t qkv = H * D;
            const size_t ck = (size_t)c->linear_conv_kernel_dim;
            ApusGkdaW *w = &Lr->kda;
            w->dim = dim; w->H = H; w->D = D;
            w->lower_bound = c->linear_lower_bound;
            w->eps = c->rms_norm_eps;
            struct { const char *suf; const uint16_t **p; } bw[] = {
                {"q_proj.weight", &w->q_w}, {"k_proj.weight", &w->k_w},
                {"v_proj.weight", &w->v_w}, {"b_proj.weight", &w->b_w},
                {"o_norm.weight", &w->o_norm}, {"o_proj.weight", &w->o_w},
            };
            for (size_t i = 0; i < sizeof bw / sizeof bw[0]; i++) {
                snprintf(nm, sizeof nm, "layers.%d.self_attn.%s", l,
                         bw[i].suf);
                t = apus_gm_get(set, nm, err, errcap);
                if (!t || t->dtype != APUS_ST_BF16)
                    APUS_GM_LOAD("gmodel: %s invalid", nm);
                *bw[i].p = (const uint16_t *)t->data;
                /* P4: matmul weights (2-D) get persistent Metal wraps;
                 * 1-D norms are never matmul inputs */
                if (t->ndim == 2) apus_gm_reg(m, t->data, t->nbytes);
            }
            /* conv: concat q|k|v [qkv, 1, K] -> [3qkv, K] (owned) */
            Lr->kda_conv = (uint16_t *)malloc(3 * qkv * ck
                                              * sizeof(uint16_t));
            for (int p = 0; p < 3; p++) {
                snprintf(nm, sizeof nm, "layers.%d.self_attn.%c_conv1d.weight",
                         l, "qkv"[p]);
                t = apus_gm_get(set, nm, err, errcap);
                if (!t || t->dtype != APUS_ST_BF16
                    || t->nbytes != qkv * ck * sizeof(uint16_t))
                    APUS_GM_LOAD("gmodel: %s invalid", nm);
                memcpy(Lr->kda_conv + (size_t)p * qkv * ck, t->data,
                       qkv * ck * sizeof(uint16_t));
            }
            w->conv_w = Lr->kda_conv;
            snprintf(nm, sizeof nm, "layers.%d.self_attn.f_a_proj.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (!t || t->dtype != APUS_ST_BF16 || t->ndim != 2)
                APUS_GM_LOAD("gmodel: %s invalid", nm);
            w->Dr = (size_t)t->shape[0];
            w->f_a = (const uint16_t *)t->data;
            apus_gm_reg(m, t->data, t->nbytes);
            snprintf(nm, sizeof nm, "layers.%d.self_attn.f_b_proj.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (!t || t->dtype != APUS_ST_BF16)
                APUS_GM_LOAD("gmodel: %s invalid", nm);
            w->f_b = (const uint16_t *)t->data;
            apus_gm_reg(m, t->data, t->nbytes);
            snprintf(nm, sizeof nm, "layers.%d.self_attn.g_a_proj.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (!t || t->dtype != APUS_ST_BF16)
                APUS_GM_LOAD("gmodel: %s invalid", nm);
            w->g_a = (const uint16_t *)t->data;
            apus_gm_reg(m, t->data, t->nbytes);
            snprintf(nm, sizeof nm, "layers.%d.self_attn.g_b_proj.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (!t || t->dtype != APUS_ST_BF16)
                APUS_GM_LOAD("gmodel: %s invalid", nm);
            w->g_b = (const uint16_t *)t->data;
            apus_gm_reg(m, t->data, t->nbytes);
            snprintf(nm, sizeof nm, "layers.%d.self_attn.A_log", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (apus_gm_expect(t, APUS_ST_F32, (int64_t)H, -1, nm,
                               err, errcap))
                goto fail;
            w->A_log = (const float *)t->data;
            snprintf(nm, sizeof nm, "layers.%d.self_attn.dt_bias", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (apus_gm_expect(t, APUS_ST_F32, (int64_t)qkv, -1, nm,
                               err, errcap))
                goto fail;
            w->dt_bias = (const float *)t->data;
        } else {
            const size_t H = (size_t)c->num_attention_heads;
            const size_t qd = (size_t)c->qk_nope_head_dim;
            const size_t vd = (size_t)c->v_head_dim;
            const size_t ql = (size_t)c->q_lora_rank;
            const size_t kl = (size_t)c->kv_lora_rank;
            const size_t IH = (size_t)c->index_n_heads;
            const size_t ID = (size_t)c->index_head_dim;
            ApusGdsaW *w = &Lr->dsa;
            w->dim = dim; w->H = H; w->qd = qd; w->vd = vd;
            w->ql = ql; w->kl = kl;
            w->eps = c->rms_norm_eps;
            w->IH = IH; w->ID = ID;
            w->topk = (size_t)c->index_topk;
            w->kpool = (size_t)c->index_kpool;
            w->tail = c->index_kpool_always_select_tail;
            struct { const char *suf; const uint8_t **cp;
                     const float **sp; size_t O, K; } fw[] = {
                {"q_a_proj", &w->q_a_c, &w->q_a_s, ql, dim},
                {"q_b_proj", &w->q_b_c, &w->q_b_s, H * qd, ql},
                {"kv_a_proj_with_mqa", &w->kv_a_c, &w->kv_a_s, kl, dim},
                {"o_proj", &w->o_c, &w->o_s, dim, H * vd},
            };
            for (size_t i = 0; i < sizeof fw / sizeof fw[0]; i++) {
                snprintf(nm, sizeof nm, "layers.%d.self_attn.%s.weight", l,
                         fw[i].suf);
                t = apus_gm_get(set, nm, err, errcap);
                snprintf(nm, sizeof nm,
                         "layers.%d.self_attn.%s.weight_scale_inv", l,
                         fw[i].suf);
                sc = apus_gm_get(set, nm, err, errcap);
                size_t sb = apus_fp8blk_nblocks(fw[i].O)
                          * apus_fp8blk_nblocks(fw[i].K);
                if (!t || !sc || t->dtype != APUS_ST_F8_E4M3
                    || sc->dtype != APUS_ST_F32
                    || t->nbytes != fw[i].O * fw[i].K
                    || sc->nbytes != sb * sizeof(float))
                    APUS_GM_LOAD("gmodel: layers.%d.self_attn fp8 invalid",
                                 l);
                *fw[i].cp = (const uint8_t *)t->data;
                *fw[i].sp = (const float *)sc->data;
                apus_gm_reg(m, t->data, t->nbytes);
                apus_gm_reg(m, sc->data, sc->nbytes);
            }
            struct { const char *suf; const uint16_t **p; } bw[] = {
                {"q_a_layernorm.weight", &w->q_a_norm},
                {"kv_a_layernorm.weight", &w->kv_a_norm},
                {"kv_b_proj.weight", &w->kv_b},
                {"indexer.wq_b.weight", &w->idx_wq_b},
                {"indexer.wk.weight", &w->idx_wk},
                {"indexer.k_norm.weight", &w->idx_knorm_w},
                {"indexer.k_norm.bias", &w->idx_knorm_b},
                {"indexer.weights_proj.weight", &w->idx_wproj},
                {"indexer.index_kpool_compress_ape", &w->idx_ape},
                {"indexer.index_kpool_compress_gate", &w->idx_gate},
            };
            for (size_t i = 0; i < sizeof bw / sizeof bw[0]; i++) {
                snprintf(nm, sizeof nm, "layers.%d.self_attn.%s", l,
                         bw[i].suf);
                t = apus_gm_get(set, nm, err, errcap);
                if (!t || t->dtype != APUS_ST_BF16)
                    APUS_GM_LOAD("gmodel: %s invalid", nm);
                *bw[i].p = (const uint16_t *)t->data;
                /* P4: matmul weights (2-D) get persistent Metal wraps;
                 * 1-D norms are never matmul inputs */
                if (t->ndim == 2) apus_gm_reg(m, t->data, t->nbytes);
            }
        }
        /* M8b: MTP glue tensors (enorm/hnorm/eh_proj BF16 [dim,2*dim],
         * shared_head.norm) — the e/h concat projection + the fused
         * add+norm target of the draft head. Loaded BEFORE the MLP branch:
         * the sparse branch's tiered-mode `continue` must not skip it. */
        if (is_mtp) {
            snprintf(nm, sizeof nm, "layers.%d.enorm.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (apus_gm_expect(t, APUS_ST_BF16, dim, -1, nm, err, errcap))
                goto fail;
            m->mtp_enorm = (const uint16_t *)t->data;
            snprintf(nm, sizeof nm, "layers.%d.hnorm.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (apus_gm_expect(t, APUS_ST_BF16, dim, -1, nm, err, errcap))
                goto fail;
            m->mtp_hnorm = (const uint16_t *)t->data;
            snprintf(nm, sizeof nm, "layers.%d.eh_proj.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (apus_gm_expect(t, APUS_ST_BF16, dim, 2 * (int64_t)dim, nm,
                               err, errcap))
                goto fail;
            m->mtp_eh_proj = (const uint16_t *)t->data;
            apus_gm_reg(m, t->data, t->nbytes);
            snprintf(nm, sizeof nm, "layers.%d.shared_head.norm.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (apus_gm_expect(t, APUS_ST_BF16, dim, -1, nm, err, errcap))
                goto fail;
            m->mtp_shared_norm = (const uint16_t *)t->data;
        }
        /* MLP (the MTP block is always sparse MoE WITH a shared expert) */
        if (!is_mtp && !c->mlp_is_sparse[l]) {
            snprintf(nm, sizeof nm, "layers.%d.mlp.gate_proj.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            snprintf(nm, sizeof nm,
                     "layers.%d.mlp.gate_proj.weight_scale_inv", l);
            sc = apus_gm_get(set, nm, err, errcap);
            Lr->mlp_g = apus_gm_dequant(t, sc, inter, dim, nm, err, errcap);
            if (!Lr->mlp_g) goto fail;
            snprintf(nm, sizeof nm, "layers.%d.mlp.up_proj.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            snprintf(nm, sizeof nm,
                     "layers.%d.mlp.up_proj.weight_scale_inv", l);
            sc = apus_gm_get(set, nm, err, errcap);
            Lr->mlp_u = apus_gm_dequant(t, sc, inter, dim, nm, err, errcap);
            if (!Lr->mlp_u) goto fail;
            snprintf(nm, sizeof nm, "layers.%d.mlp.down_proj.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            snprintf(nm, sizeof nm,
                     "layers.%d.mlp.down_proj.weight_scale_inv", l);
            sc = apus_gm_get(set, nm, err, errcap);
            Lr->mlp_d = apus_gm_dequant(t, sc, dim, inter, nm, err, errcap);
            if (!Lr->mlp_d) goto fail;
            apus_gm_reg(m, Lr->mlp_g, inter * dim * sizeof(uint16_t));
            apus_gm_reg(m, Lr->mlp_u, inter * dim * sizeof(uint16_t));
            apus_gm_reg(m, Lr->mlp_d, inter * dim * sizeof(uint16_t));
        } else {
            snprintf(nm, sizeof nm, "layers.%d.mlp.gate.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (apus_gm_expect(t, APUS_ST_BF16, E, dim, nm, err, errcap))
                goto fail;
            Lr->router.gate_w = (const uint16_t *)t->data;
            apus_gm_reg(m, t->data, t->nbytes);
            snprintf(nm, sizeof nm,
                     "layers.%d.mlp.gate.e_score_correction_bias", l);
            t = apus_gm_get(set, nm, err, errcap);
            if (apus_gm_expect(t, APUS_ST_F32, E, -1, nm, err, errcap))
                goto fail;
            Lr->router.gate_bias = (const float *)t->data;
            Lr->router.E = E;
            Lr->router.topk = c->num_experts_per_tok;
            Lr->router.dim = dim;
            Lr->router.route_scale = c->routed_scaling_factor;
            /* shared expert (dequantized, owned) */
            const size_t sinter = minter
                                * (size_t)c->n_shared_experts;
            snprintf(nm, sizeof nm,
                     "layers.%d.mlp.shared_experts.gate_proj.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            snprintf(nm, sizeof nm,
                     "layers.%d.mlp.shared_experts.gate_proj.weight_scale_inv",
                     l);
            sc = apus_gm_get(set, nm, err, errcap);
            Lr->shr_g = apus_gm_dequant(t, sc, sinter, dim, nm, err,
                                        errcap);
            if (!Lr->shr_g) goto fail;
            snprintf(nm, sizeof nm,
                     "layers.%d.mlp.shared_experts.up_proj.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            snprintf(nm, sizeof nm,
                     "layers.%d.mlp.shared_experts.up_proj.weight_scale_inv",
                     l);
            sc = apus_gm_get(set, nm, err, errcap);
            Lr->shr_u = apus_gm_dequant(t, sc, sinter, dim, nm, err,
                                        errcap);
            if (!Lr->shr_u) goto fail;
            snprintf(nm, sizeof nm,
                     "layers.%d.mlp.shared_experts.down_proj.weight", l);
            t = apus_gm_get(set, nm, err, errcap);
            snprintf(nm, sizeof nm,
                     "layers.%d.mlp.shared_experts.down_proj.weight_scale_inv",
                     l);
            sc = apus_gm_get(set, nm, err, errcap);
            Lr->shr_d = apus_gm_dequant(t, sc, dim, sinter, nm, err,
                                        errcap);
            if (!Lr->shr_d) goto fail;
            apus_gm_reg(m, Lr->shr_g, sinter * dim * sizeof(uint16_t));
            apus_gm_reg(m, Lr->shr_u, sinter * dim * sizeof(uint16_t));
            apus_gm_reg(m, Lr->shr_d, sinter * dim * sizeof(uint16_t));
            /* routed experts: eager slab load (one pread per expert),
             * dequantized into the arena — SKIPPED in tiered mode, where
             * the M6 cache (below) streams slabs behind
             * apus_gmodel_expert() and Lr->experts[e] is re-filled per
             * resolve. */
            Lr->experts = (ApusGmodelExpertW *)calloc((size_t)E,
                                                      sizeof *Lr->experts);
            if (m->tiered) continue;
            Lr->expert_arena = (uint16_t *)malloc(
                (size_t)E * 3 * minter * dim * sizeof(uint16_t));
            /* slab member sizes from the config (pinned order:
             * gate.w gate.s up.w up.s down.w down.s) */
            const size_t gw_b = minter * dim;
            const size_t gs_b = apus_fp8blk_nblocks(minter)
                              * apus_fp8blk_nblocks(dim) * sizeof(float);
            const size_t dw_b = dim * minter;
            const size_t ds_b = apus_fp8blk_nblocks(dim)
                              * apus_fp8blk_nblocks(minter) * sizeof(float);
            const size_t slab_b = 2 * (gw_b + gs_b) + dw_b + ds_b;
            uint8_t *slab = (uint8_t *)malloc(slab_b ? slab_b : 1);
            ApusStLazy *lz = NULL;
            char lz_shard[160] = "";
            for (int e = 0; e < E; e++) {
                /* find the slab record */
                JVal *rec = NULL;
                char blk[32];
                snprintf(blk, sizeof blk, "layers.%d", l);
                for (size_t i = 0; slabs && i < json_arr_len(slabs); i++) {
                    JVal *r = json_arr_get(slabs, i);
                    const char *b = json_str(json_obj_get(r, "block"));
                    long ex = (long)json_num(json_obj_get(r, "expert"));
                    if (b && !strcmp(b, blk) && ex == e) { rec = r; break; }
                }
                if (!rec)
                    APUS_GM_LOAD("gmodel: no expert slab for layers.%d.%d",
                                 l, e);
                const char *shard = json_str(json_obj_get(rec, "shard"));
                uint64_t off = (uint64_t)json_num(json_obj_get(rec,
                                                               "offset"));
                uint64_t nb = (uint64_t)json_num(json_obj_get(rec,
                                                              "nbytes"));
                if (!shard || nb != slab_b)
                    APUS_GM_LOAD("gmodel: slab layers.%d.%d size %llu != "
                                 "%zu", l, e, (unsigned long long)nb,
                                 slab_b);
                if (strcmp(lz_shard, shard)) {
                    if (lz) apus_st_lazy_close(lz);
                    snprintf(nm, sizeof nm, "%s/%s", container_dir, shard);
                    lz = apus_st_lazy_open(nm, 0, err, errcap);
                    if (!lz) { free(slab); goto fail; }
                    snprintf(lz_shard, sizeof lz_shard, "%s", shard);
                }
                if (apus_st_lazy_pread(lz, off, slab, slab_b)) {
                    free(slab);
                    APUS_GM_LOAD("gmodel: slab pread failed layers.%d.%d",
                                 l, e);
                }
                uint16_t *dst = Lr->expert_arena
                              + (size_t)e * 3 * minter * dim;
                const uint8_t *gw_c = slab;
                const float *gw_s = (const float *)(slab + gw_b);
                const uint8_t *up_c = slab + gw_b + gs_b;
                const float *up_s = (const float *)(slab + gw_b + gs_b
                                                    + gw_b);
                const uint8_t *dn_c = slab + 2 * (gw_b + gs_b);
                const float *dn_s = (const float *)(slab + 2 * (gw_b + gs_b)
                                                    + dw_b);
                apus_fp8blk_dequant(gw_c, gw_s, dst, minter, dim);
                apus_fp8blk_dequant(up_c, up_s, dst + minter * dim,
                                    minter, dim);
                apus_fp8blk_dequant(dn_c, dn_s, dst + 2 * minter * dim,
                                    dim, minter);
                Lr->experts[e].gate = dst;
                Lr->experts[e].up = dst + minter * dim;
                Lr->experts[e].down = dst + 2 * minter * dim;
            }
            free(slab);
            if (lz) apus_st_lazy_close(lz);
        }
    }
    json_free(man);
    /* M6: build the slab-streaming expert cache behind the accessor. */
    if (m->tiered) {
        ApusGcacheCfg cc;
        memset(&cc, 0, sizeof cc);
        cc.n_layers = L + m->has_mtp;   /* MTP slabs: store layer n_main+0 */
        cc.n_experts = E;
        cc.moe_inter = minter;
        cc.dim = dim;
        if (tier) {
            cc.cache_bytes = tier->cache_bytes;
            cc.slots_per_layer = tier->slots_per_layer;
            cc.io_threads = tier->io_threads;
            cc.nocache = tier->nocache;
            cc.rss_budget_bytes = tier->rss_budget_bytes;
        }
        m->cache = apus_gcache_open(container_dir, &cc, err, errcap);
        if (!m->cache) {
            apus_gmodel_close(m);
            return NULL;
        }
    }
    return m;

fail:
    json_free(man);
    apus_gmodel_close(m);
    return NULL;
}

void apus_gmodel_close(ApusGmodel *m) {
    if (!m) return;
    int NL = m->cfg.num_hidden_layers + m->has_mtp;
    /* P4: release the Metal persistent wraps BEFORE the weight memory is
     * freed (the wraps borrow the pages; no deallocator) */
    for (int i = 0; i < m->mtl_reg_n; i++)
        apus_gmetal_unregister_region(m->mtl_reg[i]);
    free(m->mtl_reg);
    if (m->layers) {
        for (int l = 0; l < NL; l++) {
            ApusGmodelLayer *Lr = &m->layers[l];
            free(Lr->hc_attn_fn);
            free(Lr->hc_ffn_fn);
            free(Lr->kda_conv);
            free(Lr->mlp_g);
            free(Lr->mlp_u);
            free(Lr->mlp_d);
            free(Lr->shr_g);
            free(Lr->shr_u);
            free(Lr->shr_d);
            free(Lr->experts);
            free(Lr->expert_arena);
        }
        free(m->layers);
    }
    if (m->set) apus_st_set_close(m->set);
    if (m->cache) apus_gcache_close(m->cache);
    apus_gmodel_config_free(&m->cfg);
    free(m);
}

struct ApusGcache *apus_gmodel_cache(const ApusGmodel *m) {
    return m->cache;
}

const ApusGmodelConfig *apus_gmodel_config(const ApusGmodel *m) {
    return &m->cfg;
}

int apus_gmodel_layer_is_dsa(const ApusGmodel *m, int layer) {
    return m->cfg.layer_is_dsa[layer];
}

int apus_gmodel_layer_is_sparse(const ApusGmodel *m, int layer) {
    return m->cfg.mlp_is_sparse[layer];
}

const ApusGmodelExpertW *apus_gmodel_expert(const ApusGmodel *m,
                                            int layer, int e) {
    const int L = m->cfg.num_hidden_layers;
    const int ok_layer = (layer >= 0 && layer < L
                          && m->cfg.mlp_is_sparse[layer])
                      || (layer == L && m->has_mtp);
    if (!ok_layer || e < 0 || e >= m->cfg.n_routed_experts)
        return NULL;
    if (!m->tiered)
        return &m->layers[layer].experts[e];
    /* tiered: resolve through the cache (mutates it — the accessor is
     * compute-thread only, the cache is internally locked) and refresh
     * the per-expert view struct. Views stay valid until the wiring's
     * layer_end on this layer (c/gcache.h contract). */
    ApusGmodel *mm = (ApusGmodel *)m;
    ApusGcacheW w;
    if (apus_gcache_resolve(mm->cache, layer, e, &w))
        return NULL;
    ApusGmodelExpertW *xw = &mm->layers[layer].experts[e];
    xw->gate = w.gate;
    xw->up = w.up;
    xw->down = w.down;
    return xw;
}

void apus_gmodel_set_hooks(ApusGmodel *m, const ApusGmodelHooks *hooks) {
    if (hooks) m->hooks = *hooks;
    else memset(&m->hooks, 0, sizeof m->hooks);
}

int apus_gmodel_pilot_view(const ApusGmodel *m, int layer,
                           ApusGmodelPilotView *out) {
    if (!m || layer < 0 || layer >= m->cfg.num_hidden_layers
        || !m->cfg.mlp_is_sparse[layer])
        return -1;
    const ApusGmodelLayer *Lr = &m->layers[layer];
    out->hc_fn = Lr->hc_ffn_fn;
    out->hc_base = Lr->hc_ffn_base;
    out->hc_scale = Lr->hc_ffn_scale;
    out->post_norm = Lr->post_norm;
    out->gate_w = Lr->router.gate_w;
    out->gate_bias = Lr->router.gate_bias;
    return 0;
}

/* --- M8b MTP accessors ----------------------------------------------------*/

int apus_gmodel_has_mtp(const ApusGmodel *m) { return m->has_mtp; }

int apus_gmodel_mtp_layer(const ApusGmodel *m) {
    return m->has_mtp ? m->cfg.num_hidden_layers : -1;
}

int apus_gmodel_mtp(const ApusGmodel *m, ApusGmodelMtpW *out) {
    if (!m->has_mtp) return -1;
    const ApusGmodelLayer *Lr = &m->layers[m->cfg.num_hidden_layers];
    out->dsa = &Lr->dsa;
    out->input_norm = Lr->input_norm;
    out->post_norm = Lr->post_norm;
    out->router = &Lr->router;
    out->shr_g = Lr->shr_g;
    out->shr_u = Lr->shr_u;
    out->shr_d = Lr->shr_d;
    out->enorm = m->mtp_enorm;
    out->hnorm = m->mtp_hnorm;
    out->eh_proj = m->mtp_eh_proj;
    out->shared_norm = m->mtp_shared_norm;
    return 0;
}

const uint16_t *apus_gmodel_embed(const ApusGmodel *m) { return m->embed; }
const uint16_t *apus_gmodel_head(const ApusGmodel *m) { return m->head; }

/* --- state + scratch ------------------------------------------------------*/

ApusGmodelState *apus_gmodel_state_new(const ApusGmodel *m, size_t kv_cap) {
    const ApusGmodelConfig *c = &m->cfg;
    const int L = c->num_hidden_layers;
    ApusGmodelState *st = (ApusGmodelState *)calloc(1, sizeof *st);
    st->m = m;
    st->cap = kv_cap;
    st->kda = (ApusGkdaState *)calloc((size_t)L, sizeof *st->kda);
    st->dsa = (ApusGdsaState *)calloc((size_t)L, sizeof *st->dsa);
    const size_t qkv = (size_t)c->linear_num_heads
                     * (size_t)c->linear_head_dim;
    const size_t ck = (size_t)c->linear_conv_kernel_dim;
    const size_t H = (size_t)c->num_attention_heads;
    const size_t qd = (size_t)c->qk_nope_head_dim;
    const size_t vd = (size_t)c->v_head_dim;
    const size_t ID = (size_t)c->index_head_dim;
    for (int l = 0; l < L; l++) {
        if (!c->layer_is_dsa[l]) {
            const size_t conv_n = 3 * qkv * (ck - 1);
            st->kda[l].conv_state = (uint16_t *)calloc(
                conv_n ? conv_n : 1,
                sizeof(uint16_t));
            st->kda[l].rec_state = (float *)calloc(
                qkv * c->linear_head_dim, sizeof(float));
        } else {
            ApusGdsaState *ds = &st->dsa[l];
            ds->cap = kv_cap;
            ds->n = 0;
            ds->k_cache = (uint16_t *)calloc(H * kv_cap * qd,
                                             sizeof(uint16_t));
            ds->v_cache = (uint16_t *)calloc(H * kv_cap * vd,
                                             sizeof(uint16_t));
            ds->idx_k = (uint16_t *)calloc(kv_cap * ID,
                                           sizeof(uint16_t));
            ds->idx_gate = (uint16_t *)calloc(kv_cap * ID,
                                              sizeof(uint16_t));
        }
    }
    return st;
}

void apus_gmodel_state_free(ApusGmodelState *st) {
    if (!st) return;
    int L = st->m->cfg.num_hidden_layers;
    for (int l = 0; l < L; l++) {
        free(st->kda[l].conv_state);
        free(st->kda[l].rec_state);
        free(st->dsa[l].k_cache);
        free(st->dsa[l].v_cache);
        free(st->dsa[l].idx_k);
        free(st->dsa[l].idx_gate);
    }
    free(st->kda);
    free(st->dsa);
    free(st->h);
    free(st->h2);
    free(st->x);
    free(st->sub);
    free(st->yh);
    free(st->x4f);
    free(st->pre);
    free(st->post);
    free(st->comb);
    free(st->mixes);
    free(st->xf);
    free(st->eg);
    free(st->eu);
    free(st->ed);
    if (st->moe_sc_ready) apus_gmoe_scratch_free(&st->moe_sc);
    apus_gdsa_scratch_free(&st->dsa_sc);
    free(st->xg);
    free(st->xu);
    free(st->xh);
    free(st->xxf);
    free(st->sel_idx);
    free(st->sel_scores);
    free(st->sel_biased);
    free(st->sel_wgt);
    free(st->sel_mark);
    free(st);
}

size_t apus_gmodel_state_bytes(const ApusGmodel *m, size_t kv_cap) {
    const ApusGmodelConfig *c = &m->cfg;
    const int L = c->num_hidden_layers;
    const size_t qkv = (size_t)c->linear_num_heads
                     * (size_t)c->linear_head_dim;
    const size_t ck = (size_t)c->linear_conv_kernel_dim;
    const size_t H = (size_t)c->num_attention_heads;
    const size_t qd = (size_t)c->qk_nope_head_dim;
    const size_t vd = (size_t)c->v_head_dim;
    const size_t ID = (size_t)c->index_head_dim;
    size_t total = 0;
    for (int l = 0; l < L; l++) {
        if (!c->layer_is_dsa[l])
            total += 3 * qkv * (ck - 1) * sizeof(uint16_t)
                   + qkv * (size_t)c->linear_head_dim * sizeof(float);
        else
            total += (H * kv_cap * qd + H * kv_cap * vd
                      + 2 * kv_cap * ID) * sizeof(uint16_t);
    }
    return total;
}

size_t apus_gmodel_pos(const ApusGmodelState *st) { return st->pos; }

const ApusGkdaState *apus_gmodel_kda_state(const ApusGmodelState *st,
                                           int layer) {
    return st->m->cfg.layer_is_dsa[layer] ? NULL : &st->kda[layer];
}

const ApusGdsaState *apus_gmodel_dsa_state(const ApusGmodelState *st,
                                           int layer) {
    return st->m->cfg.layer_is_dsa[layer] ? &st->dsa[layer] : NULL;
}

ApusGkdaState *apus_gmodel_kda_state_mut(ApusGmodelState *st, int layer) {
    return st->m->cfg.layer_is_dsa[layer] ? NULL : &st->kda[layer];
}

ApusGdsaState *apus_gmodel_dsa_state_mut(ApusGmodelState *st, int layer) {
    return st->m->cfg.layer_is_dsa[layer] ? &st->dsa[layer] : NULL;
}

size_t apus_gmodel_state_cap(const ApusGmodelState *st) { return st->cap; }

void apus_gmodel_set_pos(ApusGmodelState *st, size_t pos) { st->pos = pos; }

/* grow the scratch arena for a call of s tokens (numerics-neutral) */
static int apus_gm_scratch(ApusGmodelState *st, size_t s) {
    const ApusGmodelConfig *c = &st->m->cfg;
    const size_t dim = (size_t)c->hidden_size;
    const size_t hc = (size_t)c->hc_mult;
    const size_t mix = (2 + hc) * hc;
    const int E = c->n_routed_experts;
    const size_t inter = (size_t)c->intermediate_size;
    if (s > st->s_cap) {
        free(st->h); free(st->h2); free(st->x); free(st->sub);
        free(st->yh); free(st->xf); free(st->post); free(st->comb);
        free(st->sel_idx);
        st->h = (uint16_t *)malloc(s * hc * dim * sizeof(uint16_t));
        st->h2 = (uint16_t *)malloc(s * hc * dim * sizeof(uint16_t));
        st->x = (uint16_t *)malloc(s * dim * sizeof(uint16_t));
        st->sub = (uint16_t *)malloc(s * dim * sizeof(uint16_t));
        st->yh = (uint16_t *)malloc(s * dim * sizeof(uint16_t));
        st->xf = (float *)malloc(s * dim * sizeof(float));
        st->post = (float *)malloc(s * hc * sizeof(float));
        st->comb = (float *)malloc(s * hc * hc * sizeof(float));
        st->sel_idx = (int32_t *)malloc(s * (size_t)c->num_experts_per_tok
                                        * sizeof(int32_t));
        st->s_cap = s;
    }
    if (!st->x4f) {
        st->x4f = (float *)malloc(hc * dim * sizeof(float));
        st->pre = (float *)malloc(hc * sizeof(float));
        st->mixes = (float *)malloc(mix * sizeof(float));
        st->eg = (const uint16_t **)malloc((size_t)E * sizeof(uint16_t *));
        st->eu = (const uint16_t **)malloc((size_t)E * sizeof(uint16_t *));
        st->ed = (const uint16_t **)malloc((size_t)E * sizeof(uint16_t *));
        size_t xmax = dim > inter ? dim : inter;
        st->xg = (uint16_t *)malloc(inter * sizeof(uint16_t));
        st->xu = (uint16_t *)malloc(inter * sizeof(uint16_t));
        st->xh = (uint16_t *)malloc(inter * sizeof(uint16_t));
        st->xxf = (float *)malloc(xmax * sizeof(float));
        st->sel_scores = (float *)malloc((size_t)E * sizeof(float));
        st->sel_biased = (float *)malloc((size_t)E * sizeof(float));
        st->sel_wgt = (float *)malloc((size_t)c->num_experts_per_tok
                                      * sizeof(float));
        st->sel_mark = (uint8_t *)malloc((size_t)E);
    }
    return st->h && st->h2 && st->x && st->sub && st->yh && st->xf
           && st->post && st->comb ? 0 : -1;
}

/* --- forward --------------------------------------------------------------*/

/* One mHC site (attn or ffn): maps + collapse per token (bitwise == the
 * oracle's batched rows — the maps are per-token independent). */
static void apus_gm_hc_pre(const ApusGmodel *m, ApusGmodelState *st,
                           const float *fn, const float *scale,
                           const float *base, size_t s) {
    const ApusGmodelConfig *c = &m->cfg;
    const size_t dim = (size_t)c->hidden_size;
    const size_t hc = (size_t)c->hc_mult;
    for (size_t t = 0; t < s; t++) {
        const uint16_t *hr = st->h + t * hc * dim;
        for (size_t i = 0; i < hc * dim; i++)
            st->x4f[i] = apus_bf16_f32(hr[i]);
        apus_gmhc_maps(st->x4f, dim, hc, fn, scale, base,
                       c->rms_norm_eps, c->hc_eps, c->hc_sinkhorn_iters,
                       st->pre, st->post + t * hc, st->comb + t * hc * hc,
                       st->mixes);
        apus_gmhc_collapse(st->x4f, st->pre, st->x + t * dim, dim, hc);
    }
}

/* mHC expand per token into st->h2, then swap h/h2. */
static void apus_gm_hc_post(ApusGmodelState *st, size_t s) {
    const ApusGmodelConfig *c = &st->m->cfg;
    const size_t dim = (size_t)c->hidden_size;
    const size_t hc = (size_t)c->hc_mult;
    for (size_t t = 0; t < s; t++)
        apus_gmhc_expand(st->sub + t * dim, st->h + t * hc * dim,
                         st->post + t * hc, st->comb + t * hc * hc,
                         st->h2 + t * hc * dim, dim, hc);
    uint16_t *tmp = st->h;
    st->h = st->h2;
    st->h2 = tmp;
}

static int apus_gm_run(ApusGmodel *m, ApusGmodelState *st,
                       const int32_t *ids, size_t s, int decode,
                       uint16_t *logits, const ApusGmodelForces *forces,
                       uint16_t *trace_h, uint16_t *h_out,
                       uint16_t *yn_out) {
    const ApusGmodelConfig *c = &m->cfg;
    const int L = c->num_hidden_layers;
    const size_t dim = (size_t)c->hidden_size;
    const size_t hc = (size_t)c->hc_mult;
    const size_t V = (size_t)c->vocab_size;
    if (apus_gm_scratch(st, s))
        return -1;
    /* embed: replicate the row across the hc streams (glm5:1478) */
    for (size_t t = 0; t < s; t++) {
        const uint16_t *row = m->embed + (size_t)ids[t] * dim;
        for (size_t j = 0; j < hc; j++)
            memcpy(st->h + (t * hc + j) * dim, row,
                   dim * sizeof(uint16_t));
    }
    for (int l = 0; l < L; l++) {
        ApusGmodelLayer *Lr = &m->layers[l];
        /* attn HC site */
        apus_gm_hc_pre(m, st, Lr->hc_attn_fn, Lr->hc_attn_scale,
                       Lr->hc_attn_base, s);
        for (size_t t = 0; t < s; t++)
            apus_gdsa_rmsnorm(st->x + t * dim, Lr->input_norm,
                              st->x + t * dim, dim, c->rms_norm_eps);
        if (!c->layer_is_dsa[l])
            apus_gkda_forward(&Lr->kda, st->x, s, &st->kda[l], decode,
                              st->sub, NULL);
        else if (decode)
            /* M8b: the decode batch (s >= 1) interleaves per token —
             * bitwise == s one-by-one decode steps by construction.
             * P5: the shared per-state dequant scratch is numerics-neutral
             * (decode is serial on this thread). */
            apus_gdsa_decode_batch2(&Lr->dsa, st->x, s, &st->dsa[l], st->sub,
                                    forces && forces->indexer_topk
                                        ? forces->indexer_topk[l] : NULL,
                                    &st->dsa_sc);
        else
            apus_gdsa_forward3(&Lr->dsa, st->x, s, &st->dsa[l], st->sub,
                               NULL,
                               forces && forces->indexer_topk
                                   ? forces->indexer_topk[l] : NULL,
                               &st->dsa_sc);
        apus_gm_hc_post(st, s);
        /* M6 hook surface (pilot): read-only on the post-attention stream */
        if (m->hooks.post_attn)
            m->hooks.post_attn(m->hooks.ctx, l, st->h, s, st->pos);
        /* ffn HC site */
        apus_gm_hc_pre(m, st, Lr->hc_ffn_fn, Lr->hc_ffn_scale,
                       Lr->hc_ffn_base, s);
        for (size_t t = 0; t < s; t++)
            apus_gdsa_rmsnorm(st->x + t * dim, Lr->post_norm,
                              st->x + t * dim, dim, c->rms_norm_eps);
        if (!c->mlp_is_sparse[l]) {
            const size_t inter = (size_t)c->intermediate_size;
            for (size_t t = 0; t < s; t++)
                apus_gmoe_expert(Lr->mlp_g, Lr->mlp_u, Lr->mlp_d,
                                 st->x + t * dim, dim, inter,
                                 c->swiglu_limit, st->sub + t * dim,
                                 st->xg, st->xu, st->xh, st->xxf);
        } else {
            const size_t minter = (size_t)c->moe_intermediate_size;
            const int E = c->n_routed_experts;
            const int topk = c->num_experts_per_tok;
            ApusGmoeW gw;
            gw.router = Lr->router;
            gw.inter = minter;
            gw.eg = st->eg;
            gw.eu = st->eu;
            gw.ed = st->ed;
            gw.sg = Lr->shr_g;
            gw.su = Lr->shr_u;
            gw.sd = Lr->shr_d;
            gw.sinter = minter * (size_t)c->n_shared_experts;
            gw.limit = c->swiglu_limit;
            if (!st->moe_sc_ready || st->moe_sc_s < s) {
                if (st->moe_sc_ready) apus_gmoe_scratch_free(&st->moe_sc);
                apus_gmoe_scratch_init(&st->moe_sc, &gw, s);
                st->moe_sc_s = s;
                st->moe_sc_ready = 1;
            }
            const int32_t *forced = forces && forces->router_idx
                                    ? forces->router_idx[l] : NULL;
            if (!m->tiered) {
                for (int e = 0; e < E; e++) {
                    const ApusGmodelExpertW *xw =
                        apus_gmodel_expert(m, l, e);
                    st->eg[e] = xw->gate;
                    st->eu[e] = xw->up;
                    st->ed[e] = xw->down;
                }
                apus_gmoe_forward2(&gw, st->x, s, st->sub, NULL,
                                   &st->moe_sc, forced);
            } else {
                /* M6 tiered: resolve ONLY the routed experts. Pre-pass the
                 * selections (the forced indices when teacher-forcing;
                 * otherwise the same apus_gmoe_router apus_gmoe_forward2
                 * will re-run — deterministic, bitwise-identical), demand-
                 * hint the union (loads overlap the just-in-time resolves),
                 * resolve ascending, NULL the rest (never dereferenced:
                 * forward2 touches selected only).
                 * P4: PREFILL IS TOKEN-CHUNKED (APUS_GPREFILL_MOE_CHUNK,
                 * default 8; 0 or >= s = unchunked — the decode path, s ==
                 * 1, is always a single chunk): each chunk pre-passes,
                 * resolves, and forwards its tokens, then layer_end
                 * promotes its working set before the next chunk resolves.
                 * The unchunked prefill held ALL s tokens' union live at
                 * once (~10 GiB of payloads per layer at s=26 — the P2
                 * 41.5 GB footprint peak); the chunk bounds the live union
                 * at tc*topk payloads. BITWISE-SAFE by per-token
                 * independence: forward2's per-token ascending-expert
                 * accumulation depends only on the token's own x row, and
                 * every resolve returns the same dequantized bytes
                 * regardless of cache state — the m6g digest gate (tiered,
                 * chunked) == eager (unchunked) pins this. The routed hook
                 * still fires ONCE per layer with the full [s, topk]
                 * selections. */
                size_t tc = s;
                if (s > 1 && m->prefill_moe_chunk > 0
                    && (size_t)m->prefill_moe_chunk < s)
                    tc = (size_t)m->prefill_moe_chunk;
                for (size_t t0 = 0; t0 < s; t0 += tc) {
                    size_t tn = s - t0 < tc ? s - t0 : tc;
                    memset(st->sel_mark, 0, (size_t)E);
                    for (size_t t = t0; t < t0 + tn; t++) {
                        int32_t *it = st->sel_idx + t * (size_t)topk;
                        if (forced)
                            memcpy(it, forced + t * (size_t)topk,
                                   (size_t)topk * sizeof(int32_t));
                        else
                            apus_gmoe_router(&Lr->router, st->x + t * dim,
                                             st->sel_scores, it,
                                             st->sel_wgt, st->sel_biased);
                        for (int j = 0; j < topk; j++)
                            st->sel_mark[it[j]] = 1;
                    }
                    for (int e = 0; e < E; e++) {
                        st->eg[e] = st->eu[e] = st->ed[e] = NULL;
                        if (st->sel_mark[e])
                            apus_gcache_hint_demand(m->cache, l, e);
                    }
                    for (int e = 0; e < E; e++) {
                        if (!st->sel_mark[e]) continue;
                        const ApusGmodelExpertW *xw =
                            apus_gmodel_expert(m, l, e);
                        if (!xw) return -1;
                        st->eg[e] = xw->gate;
                        st->eu[e] = xw->up;
                        st->ed[e] = xw->down;
                    }
                    apus_gmoe_forward2(&gw, st->x + t0 * dim, tn,
                                       st->sub + t0 * dim, NULL,
                                       &st->moe_sc,
                                       forced ? forced + t0 * (size_t)topk
                                              : NULL);
                    /* M6: end-of-block promotion + RSS guard — PER CHUNK
                     * (this is what bounds the live union; the resolved
                     * views are dead once forward2 returns, which is
                     * exactly the gcache pointer-stability contract). */
                    apus_gcache_layer_end(m->cache, l);
                }
                if (m->hooks.routed)
                    m->hooks.routed(m->hooks.ctx, l, st->sel_idx, s,
                                    st->pos);
            }
        }
        apus_gm_hc_post(st, s);
        if (trace_h)
            memcpy(trace_h + (size_t)l * s * hc * dim, st->h,
                   s * hc * dim * sizeof(uint16_t));
    }
    /* HyperHead (unweighted mean) + final norm + lm_head (glm5:1494,
     * 2182) */
    if (h_out)          /* M8b: last-layer post-block stream per position */
        memcpy(h_out, st->h, s * hc * dim * sizeof(uint16_t));
    for (size_t t = 0; t < s; t++) {
        apus_gmhc_head(st->h + t * hc * dim, st->yh + t * dim, dim, hc);
        apus_gdsa_rmsnorm(st->yh + t * dim, m->norm, st->yh + t * dim,
                          dim, c->rms_norm_eps);
    }
    if (yn_out)         /* M8b: post-final-norm rows (the lm_head input) */
        memcpy(yn_out, st->yh, s * dim * sizeof(uint16_t));
    apus_bf16_gemm_mt(m->head, st->yh, st->xf, logits, s, V, dim);
    st->pos += s;
    return 0;
}

int apus_gmodel_prefill(ApusGmodel *m, ApusGmodelState *st,
                        const int32_t *ids, size_t s,
                        uint16_t *logits, const ApusGmodelForces *forces,
                        uint16_t *trace_h) {
    if (st->pos + s > st->cap)
        return -1;
    return apus_gm_run(m, st, ids, s, 0, logits, forces, trace_h,
                       NULL, NULL);
}

int apus_gmodel_decode_step(ApusGmodel *m, ApusGmodelState *st,
                            int32_t id, uint16_t *logits,
                            const ApusGmodelForces *forces,
                            uint16_t *trace_h) {
    if (st->pos + 1 > st->cap)
        return -1;
    return apus_gm_run(m, st, &id, 1, 1, logits, forces, trace_h,
                       NULL, NULL);
}

int apus_gmodel_decode_batch(ApusGmodel *m, ApusGmodelState *st,
                             const int32_t *ids, size_t s,
                             uint16_t *logits, const ApusGmodelForces *forces,
                             uint16_t *h_out, uint16_t *yn_out) {
    if (st->pos + s > st->cap)
        return -1;
    return apus_gm_run(m, st, ids, s, 1, logits, forces, NULL,
                       h_out, yn_out);
}

int apus_gmodel_prefill_h(ApusGmodel *m, ApusGmodelState *st,
                          const int32_t *ids, size_t s,
                          uint16_t *logits, const ApusGmodelForces *forces,
                          uint16_t *h_out, uint16_t *yn_out) {
    if (st->pos + s > st->cap)
        return -1;
    return apus_gm_run(m, st, ids, s, 0, logits, forces, NULL,
                       h_out, yn_out);
}

#endif /* APUS_GMODEL_IMPLEMENTATION */
#endif /* APUS_GMODEL_H */
