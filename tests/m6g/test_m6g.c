/*
 * tests/m6g/test_m6g.c — M6 (GLM tiering/cache) hard gate: c/gcache.h
 * (slab-streaming expert cache behind the M5 accessor seam), c/gpilot.h
 * (router-lookahead prefetch), and the tiered c/gmodel.h wiring, against
 * the m5g fixtures (tests/m5g/golden — the converted v2 container + the
 * oracle goldens) and the m6g locality fixtures (tests/m6g/golden — the
 * clustered/random E=32 containers; tests/m6g/README.md).
 *
 * Gates:
 *   1. BITWISE NEUTRALITY (host-exp independent): eager (M5 arena) ==
 *      tiered big-cache == tiered 1-slot-cache == tiered+pilot digests of
 *      the full output stream (prefill+decode logits, per-layer trace_h)
 *      on the m5g fixture. Cache hit vs miss produces byte-identical BF16
 *      expert weights — the dequant-on-fill is the same m3g kernel on the
 *      same slab bytes. In the probe-BITWISE tier the eager run is also
 *      memcmp-anchored to the oracle goldens.
 *   2. EVICTION CORRECTNESS: slots_per_layer=1 forces an eviction churn;
 *      the digest still equals the eager one (gate 1) and evictions > 0.
 *   3. ONE PREAD PER FILL: stats.preads == stats.loads ==
 *      apus_gcache_pread_count (the ApusStLazy instrumentation) and
 *      bytes_read == preads * slab_bytes (no-pilot tiered runs — those
 *      counters are synchronous/timing-independent there).
 *   4. PREFETCH MEASUREMENT (locality fixtures, SYNCHRONOUS I/O for
 *      determinism): pilot recall on the clustered container vs the
 *      random-weights control, and prefetch coverage (demand_loads with
 *      vs without the pilot).
 *   5. apus_gmodel_state_bytes (the KDA/DSA state sizing helper) == the
 *      hand-computed byte count from the fixture dims.
 *   6. P2 SPECULATIVE DROP POLICY (sync I/O, rss_budget_bytes=1): every
 *      speculative hint is dropped at admission (hint_loads == 0,
 *      spec_dropped > 0), demand still loads, digest == pilot-OFF, and
 *      the RSS guard + EMPTY-slot resubmit path is exercised (every
 *      resolve misses, rss_drops > 0).
 *
 * Run from the repository root. Prints FNV-1a digests + synchronous
 * counters only; the Makefile diffs full output across APUS_THREADS=1/4/8
 * (timing-dependent counters — waits/wait_ns/deq_ns and the async pilot
 * run's cache stats — print only under APUS_M6G_STATS=1).
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_BF16_IMPLEMENTATION
#define APUS_FP8BLK_IMPLEMENTATION
#define APUS_GMHC_IMPLEMENTATION
#define APUS_GMOE_IMPLEMENTATION
#define APUS_GKDA_IMPLEMENTATION
#define APUS_GDSA_IMPLEMENTATION
#define APUS_GCACHE_IMPLEMENTATION
#define APUS_GMODEL_IMPLEMENTATION
#define APUS_GPILOT_IMPLEMENTATION
#include "gmodel.h"
#include "gpilot.h"
#include "gcache.h"
#include "bf16.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks = 0, failures = 0;
#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { \
        failures++; \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } \
} while (0)

/* --- file IO --------------------------------------------------------------*/

static unsigned char *read_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(sz ? (size_t)sz : 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(buf); return NULL;
    }
    fclose(f);
    *out_len = sz;
    return buf;
}

static const char *man_find(const char *man, const char *key) {
    size_t kl = strlen(key);
    const char *p = man;
    while ((p = strstr(p, key)) != NULL) {
        if ((p == man || p[-1] == '\n') && p[kl] == '=')
            return p + kl + 1;
        p += kl;
    }
    printf("manifest key %s missing\n", key);
    exit(1);
}
static long man_long(const char *man, const char *key) {
    return strtol(man_find(man, key), NULL, 10);
}

static int32_t *load_i32(const char *dir, const char *name, size_t n) {
    char path[512];
    long len;
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    unsigned char *b = read_file(path, &len);
    if ((size_t)len != n * sizeof(int32_t)) {
        printf("%s: size mismatch (%ld != %zu)\n", path, len, n);
        exit(1);
    }
    return (int32_t *)b;
}

static float *load_f32(const char *dir, const char *name, size_t n) {
    char path[512];
    long len;
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    unsigned char *b = read_file(path, &len);
    if ((size_t)len != n * sizeof(float)) {
        printf("%s: size mismatch (%ld != %zu)\n", path, len,
               n * sizeof(float));
        exit(1);
    }
    return (float *)b;
}

/* --- rolling FNV-1a 64 -----------------------------------------------------*/

static uint64_t dg_state = 14695981039346656037ull;
static void dg_add(const void *data, size_t n) {
    const unsigned char *p = data;
    for (size_t i = 0; i < n; i++) {
        dg_state ^= p[i];
        dg_state *= 1099511628211ull;
    }
}

/* --- dims (m5g fixture) ------------------------------------------------------*/

static int L, DIM, V, HC, TOPK, E5, NPRE, NSTEP, NTOT;

/* Run the m5g workload on an open model; digest logits + per-layer
 * trace_h over prefill + the decode chain. Optionally memcmp-anchor the
 * logits against the oracle goldens (probe-BITWISE tier only). */
static uint64_t run_m5g(ApusGmodel *m, const char *dir, int anchor,
                        int *anchor_fail) {
    char nm[128];
    size_t snp = (size_t)NPRE, sns = (size_t)NSTEP;
    size_t hc = (size_t)HC, dim = (size_t)DIM, v = (size_t)V;
    int32_t *pids = load_i32(dir, "prompt_ids", snp);
    int32_t *dids = load_i32(dir, "decode_ids", sns);
    uint16_t *logits = malloc(snp * v * sizeof(uint16_t));
    uint16_t *trace = malloc((size_t)L * snp * hc * dim
                             * sizeof(uint16_t));
    ApusGmodelState *st = apus_gmodel_state_new(m, (size_t)NTOT);
    dg_state = 14695981039346656037ull;
    if (apus_gmodel_prefill(m, st, pids, snp, logits, NULL, trace)) {
        printf("prefill failed\n");
        exit(1);
    }
    dg_add(logits, snp * v * sizeof(uint16_t));
    dg_add(trace, (size_t)L * snp * hc * dim * sizeof(uint16_t));
    if (anchor) {
        float *g = load_f32(dir, "prefill_logits", snp * v);
        for (size_t i = 0; i < snp * v; i++) {
            float f = apus_bf16_f32(logits[i]);
            if (memcmp(&f, g + i, 4) != 0) { *anchor_fail = 1; break; }
        }
        free(g);
    }
    for (int k = 0; k < NSTEP; k++) {
        if (apus_gmodel_decode_step(m, st, dids[k], logits, NULL, trace)) {
            printf("decode failed\n");
            exit(1);
        }
        dg_add(logits, v * sizeof(uint16_t));
        dg_add(trace, (size_t)L * hc * dim * sizeof(uint16_t));
        if (anchor) {
            snprintf(nm, sizeof nm, "decode_logits");
            float *g = load_f32(dir, nm, sns * v);
            for (size_t i = 0; i < v; i++) {
                float f = apus_bf16_f32(logits[i]);
                if (memcmp(&f, g + (size_t)k * v + i, 4) != 0) {
                    *anchor_fail = 1;
                    break;
                }
            }
            free(g);
        }
    }
    apus_gmodel_state_free(st);
    free(trace);
    free(logits);
    free(dids);
    free(pids);
    return dg_state;
}

/* --- locality workload (cluster/random fixtures) -----------------------------*/

typedef struct {
    uint64_t digest;
    uint64_t hits, misses, loads, demand_loads, hint_loads, preads,
             evictions, spec_dropped, rss_drops;
    uint64_t recall_num, recall_den;    /* pilot actual_hits/actual_experts */
    uint64_t predictions, prefill_hints;
} LocalRun;

static LocalRun run_local(const char *cdir, const char *cfgpath,
                          const int32_t *pids, size_t np,
                          const int32_t *dids, size_t nd,
                          int slots, int with_pilot, int pilot_k,
                          size_t rss_budget) {
    char err[256];
    ApusGmodelTierCfg tc;
    memset(&tc, 0, sizeof tc);
    tc.tiered = 1;
    tc.slots_per_layer = slots;
    tc.io_threads = -1;             /* synchronous I/O: deterministic */
    tc.rss_budget_bytes = rss_budget;
    ApusGmodel *m = apus_gmodel_open2(cdir, cfgpath, &tc, err, sizeof err);
    if (!m) { printf("open2 %s failed: %s\n", cdir, err); exit(1); }
    ApusGpilot *p = NULL;
    if (with_pilot) {
        const ApusGmodelConfig *c = apus_gmodel_config(m);
        ApusGpilotCfg pc;
        memset(&pc, 0, sizeof pc);
        pc.cache = apus_gmodel_cache(m);
        pc.n_layers = c->num_hidden_layers;
        pc.n_experts = c->n_routed_experts;
        pc.topk = c->num_experts_per_tok;
        pc.dim = (size_t)c->hidden_size;
        pc.hc_mult = c->hc_mult;
        pc.sinkhorn_iters = c->hc_sinkhorn_iters;
        pc.norm_eps = c->rms_norm_eps;
        pc.hc_eps = c->hc_eps;
        pc.route_scale = c->routed_scaling_factor;
        pc.enabled = 1;
        pc.pilot_k = pilot_k;
        pc.prefill_k = pilot_k;
        p = apus_gpilot_create(&pc);
        if (!p) { printf("gpilot_create failed\n"); exit(1); }
        apus_gpilot_attach(p, m);
    }
    size_t hc = (size_t)apus_gmodel_config(m)->hc_mult;
    size_t dim = (size_t)apus_gmodel_config(m)->hidden_size;
    size_t v = (size_t)apus_gmodel_config(m)->vocab_size;
    int nl = apus_gmodel_config(m)->num_hidden_layers;
    uint16_t *logits = malloc(np * v * sizeof(uint16_t));
    uint16_t *trace = malloc((size_t)nl * np * hc * dim
                             * sizeof(uint16_t));
    ApusGmodelState *st = apus_gmodel_state_new(m, np + nd);
    dg_state = 14695981039346656037ull;
    if (apus_gmodel_prefill(m, st, pids, np, logits, NULL, trace)) {
        printf("local prefill failed\n");
        exit(1);
    }
    dg_add(logits, np * v * sizeof(uint16_t));
    dg_add(trace, (size_t)nl * np * hc * dim * sizeof(uint16_t));
    for (size_t k = 0; k < nd; k++) {
        if (apus_gmodel_decode_step(m, st, dids[k], logits, NULL,
                                    trace)) {
            printf("local decode failed\n");
            exit(1);
        }
        dg_add(logits, v * sizeof(uint16_t));
        dg_add(trace, (size_t)nl * hc * dim * sizeof(uint16_t));
    }
    LocalRun r;
    memset(&r, 0, sizeof r);
    r.digest = dg_state;
    ApusGcacheStats cs;
    apus_gcache_stats(apus_gmodel_cache(m), &cs);
    r.hits = cs.hits;
    r.misses = cs.misses;
    r.loads = cs.loads;
    r.demand_loads = cs.demand_loads;
    r.hint_loads = cs.hint_loads;
    r.preads = cs.preads;
    r.evictions = cs.evictions;
    r.spec_dropped = cs.spec_dropped;
    r.rss_drops = cs.rss_drops;
    if (p) {
        ApusGpilotStats ps;
        apus_gpilot_stats(p, &ps);
        r.recall_num = ps.actual_hits;
        r.recall_den = ps.actual_experts;
        r.predictions = ps.predictions;
        r.prefill_hints = ps.prefill_hints;
    }
    apus_gmodel_state_free(st);
    free(trace);
    free(logits);
    if (p) apus_gpilot_destroy(p);
    apus_gmodel_close(m);
    return r;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "tests/m5g/golden";
    const char *ldir = argc > 2 ? argv[2] : "tests/m6g/golden";
    int verbose = getenv("APUS_M6G_STATS") != NULL;
    long mlen;
    char mpath[512];
    snprintf(mpath, sizeof mpath, "%s/manifest.txt", dir);
    char *man = (char *)read_file(mpath, &mlen);
    L = (int)man_long(man, "cfg_L");
    DIM = (int)man_long(man, "cfg_dim");
    V = (int)man_long(man, "cfg_V");
    HC = (int)man_long(man, "cfg_hc");
    TOPK = (int)man_long(man, "cfg_topk");
    E5 = 8;                     /* m5g tiny config (manifest has no E) */
    NPRE = (int)man_long(man, "prefill_len");
    NSTEP = (int)man_long(man, "decode_steps");
    NTOT = (int)man_long(man, "n_total");

    /* exp probe (golden-anchor tier decision) */
    int probe_clean;
    {
        size_t n = (size_t)man_long(man, "probe_n");
        float *px = load_f32(dir, "probe_x", n);
        float *py = load_f32(dir, "probe_y", n);
        size_t bad = 0;
        for (size_t i = 0; i < n; i++) {
            float e = expf(px[i]);
            if (memcmp(&e, py + i, 4) != 0) bad++;
        }
        probe_clean = (bad == 0);
        printf("probe: expf vs numpy f32 exp — %zu/%zu differ -> %s\n",
               bad, n, probe_clean ? "golden anchor ON" : "anchor OFF");
        free(px);
        free(py);
    }

    char err[256], cpath[512], cfgpath[512];
    snprintf(cpath, sizeof cpath, "%s/container", dir);
    snprintf(cfgpath, sizeof cfgpath, "%s/config.json", dir);

    /* --- mode A: eager (the M5 path) ------------------------------------ */
    ApusGmodel *ma = apus_gmodel_open(cpath, cfgpath, err, sizeof err);
    if (!ma) { CHECK(0, "eager open: %s", err); return 1; }
    int anchor_fail = 0;
    uint64_t da = run_m5g(ma, dir, probe_clean, &anchor_fail);
    if (probe_clean)
        CHECK(!anchor_fail, "eager logits != oracle goldens");
    printf("A eager    digest %016llx\n", (unsigned long long)da);

    /* state sizing helper (M6 memory management) */
    {
        const ApusGmodelConfig *c = apus_gmodel_config(ma);
        size_t qkv = (size_t)c->linear_num_heads
                   * (size_t)c->linear_head_dim;
        size_t ck = (size_t)c->linear_conv_kernel_dim;
        size_t H = (size_t)c->num_attention_heads;
        size_t qd = (size_t)c->qk_nope_head_dim;
        size_t vd = (size_t)c->v_head_dim;
        size_t ID = (size_t)c->index_head_dim;
        size_t want = 0;
        for (int l = 0; l < L; l++) {
            if (!c->layer_is_dsa[l])
                want += 3 * qkv * (ck - 1) * 2
                      + qkv * (size_t)c->linear_head_dim * 4;
            else
                want += (H * (size_t)NTOT * qd + H * (size_t)NTOT * vd
                         + 2 * (size_t)NTOT * ID) * 2;
        }
        size_t got = apus_gmodel_state_bytes(ma, (size_t)NTOT);
        CHECK(got == want, "state_bytes %zu != %zu", got, want);
    }

    /* --- mode B: tiered, big cache (all experts fit) ---------------------- */
    ApusGmodelTierCfg tb;
    memset(&tb, 0, sizeof tb);
    tb.tiered = 1;
    tb.slots_per_layer = E5;
    ApusGmodel *mb = apus_gmodel_open2(cpath, cfgpath, &tb, err,
                                       sizeof err);
    if (!mb) { CHECK(0, "tiered open: %s", err); return 1; }
    uint64_t db = run_m5g(mb, dir, 0, &anchor_fail);
    ApusGcacheStats sb;
    apus_gcache_stats(apus_gmodel_cache(mb), &sb);
    CHECK(db == da, "tiered big-cache digest != eager");
    CHECK(sb.preads == sb.loads && sb.loads > 0,
          "one pread per fill: preads %llu loads %llu",
          (unsigned long long)sb.preads, (unsigned long long)sb.loads);
    CHECK(apus_gcache_pread_count(apus_gmodel_cache(mb)) == sb.preads,
          "ApusStLazy pread count != stats.preads");
    CHECK(sb.bytes_read == sb.preads
          * apus_gcache_slab_bytes(apus_gmodel_cache(mb)),
          "bytes_read != preads * slab_bytes");
    CHECK(sb.misses > 0 && sb.hits > 0,
          "big cache: misses %llu hits %llu",
          (unsigned long long)sb.misses, (unsigned long long)sb.hits);
    CHECK(sb.loads == sb.demand_loads + sb.hint_loads,
          "loads != demand + speculative");
    printf("B tiered   digest %016llx hits %llu misses %llu preads %llu "
           "evictions %llu\n", (unsigned long long)db,
           (unsigned long long)sb.hits, (unsigned long long)sb.misses,
           (unsigned long long)sb.preads,
           (unsigned long long)sb.evictions);

    /* --- mode C: tiered, 1 slot per layer (eviction churn) ----------------- */
    ApusGmodelTierCfg tc1 = tb;
    tc1.slots_per_layer = 1;
    ApusGmodel *mc = apus_gmodel_open2(cpath, cfgpath, &tc1, err,
                                       sizeof err);
    if (!mc) { CHECK(0, "tiered tiny open: %s", err); return 1; }
    uint64_t dc = run_m5g(mc, dir, 0, &anchor_fail);
    ApusGcacheStats sc;
    apus_gcache_stats(apus_gmodel_cache(mc), &sc);
    CHECK(dc == da, "tiered 1-slot digest != eager");
    CHECK(sc.evictions > 0, "1-slot cache: no evictions?");
    CHECK(sc.preads == sc.loads, "1-slot: preads != loads");
    printf("C 1-slot   digest %016llx misses %llu evictions %llu\n",
           (unsigned long long)dc, (unsigned long long)sc.misses,
           (unsigned long long)sc.evictions);

    /* --- mode D: tiered + pilot (async pool; digest-only gating — the
     * pilot's cache counters have legitimately timing-dependent bits, see
     * README; the recall stats are synchronous) -------------------------- */
    ApusGmodel *md = apus_gmodel_open2(cpath, cfgpath, &tb, err,
                                       sizeof err);
    if (!md) { CHECK(0, "tiered+pilot open: %s", err); return 1; }
    ApusGpilot *pd;
    {
        const ApusGmodelConfig *c = apus_gmodel_config(md);
        ApusGpilotCfg pc;
        memset(&pc, 0, sizeof pc);
        pc.cache = apus_gmodel_cache(md);
        pc.n_layers = c->num_hidden_layers;
        pc.n_experts = c->n_routed_experts;
        pc.topk = c->num_experts_per_tok;
        pc.dim = (size_t)c->hidden_size;
        pc.hc_mult = c->hc_mult;
        pc.sinkhorn_iters = c->hc_sinkhorn_iters;
        pc.norm_eps = c->rms_norm_eps;
        pc.hc_eps = c->hc_eps;
        pc.route_scale = c->routed_scaling_factor;
        pc.enabled = 1;
        pc.pilot_k = 5;         /* < E=8: non-vacuous recall accounting */
        pc.prefill_k = 5;
        pd = apus_gpilot_create(&pc);
        if (!pd) { CHECK(0, "gpilot_create"); return 1; }
        apus_gpilot_attach(pd, md);
    }
    uint64_t dd = run_m5g(md, dir, 0, &anchor_fail);
    CHECK(dd == da, "tiered+pilot digest != eager");
    ApusGpilotStats psd;
    apus_gpilot_stats(pd, &psd);
    CHECK(psd.actual_experts > 0 && psd.predictions > 0,
          "pilot: no recall accounting");
    printf("D pilot    digest %016llx recall %llu/%llu predictions %llu\n",
           (unsigned long long)dd, (unsigned long long)psd.actual_hits,
           (unsigned long long)psd.actual_experts,
           (unsigned long long)psd.predictions);
    if (verbose) {
        ApusGcacheStats sd;
        apus_gcache_stats(apus_gmodel_cache(md), &sd);
        printf("D stats    hits %llu misses %llu demand %llu spec %llu "
               "waits %llu wait_ns %llu deq_ns %llu\n",
               (unsigned long long)sd.hits, (unsigned long long)sd.misses,
               (unsigned long long)sd.demand_loads,
               (unsigned long long)sd.hint_loads,
               (unsigned long long)sd.waits,
               (unsigned long long)sd.wait_ns,
               (unsigned long long)sd.deq_ns);
    }

    /* --- locality fixtures (synchronous I/O, deterministic counters) ----- */
    snprintf(mpath, sizeof mpath, "%s/cluster_manifest.txt", ldir);
    char *lman = (char *)read_file(mpath, &mlen);
    long np = man_long(lman, "n_prompt");
    long nd = man_long(lman, "n_decode");
    long e_loc = man_long(lman, "cfg_E");
    int32_t *pids = load_i32(ldir, "cluster_prompt", (size_t)np);
    int32_t *dids = load_i32(ldir, "cluster_decode", (size_t)nd);
    char cdir[512], rcfg[512], rdir[512];
    snprintf(cdir, sizeof cdir, "%s/cluster_container", ldir);
    snprintf(rdir, sizeof rdir, "%s/random_container", ldir);
    snprintf(rcfg, sizeof rcfg, "%s/cluster_config.json", ldir);

    LocalRun cp = run_local(cdir, rcfg, pids, (size_t)np, dids, (size_t)nd,
                            4, 1, 8, 0);
    LocalRun cn = run_local(cdir, rcfg, pids, (size_t)np, dids, (size_t)nd,
                            4, 0, 0, 0);
    LocalRun rp = run_local(rdir, rcfg, pids, (size_t)np, dids, (size_t)nd,
                            4, 1, 8, 0);
    double rec_c = cp.recall_den
                   ? (double)cp.recall_num / (double)cp.recall_den : 0.0;
    double rec_r = rp.recall_den
                   ? (double)rp.recall_num / (double)rp.recall_den : 0.0;
    CHECK(cp.digest == cn.digest, "cluster pilot-ON digest != pilot-OFF");
    CHECK(cp.recall_den > 0, "cluster pilot: no recall accounting");
    CHECK(rec_c >= 0.70, "cluster pilot recall %.3f < 0.70", rec_c);
    CHECK(rec_r < rec_c, "random recall %.3f !< cluster %.3f",
          rec_r, rec_c);
    CHECK(cp.prefill_hints > 0, "cluster pilot: no prefill union hints");
    CHECK(cp.demand_loads < cn.demand_loads,
          "prefetch coverage: demand %llu (pilot) !< %llu (none)",
          (unsigned long long)cp.demand_loads,
          (unsigned long long)cn.demand_loads);
    CHECK(cn.demand_loads > 0, "no-pilot run: no demand loads?");

    /* P2 speculative yield/drop policy (deterministic in sync I/O mode —
     * the queue/backlog rules are vacuous there, leaving the RSS-pressure
     * rule): with the RSS budget at 1 byte EVERY speculative hint must be
     * dropped at admission (never submitted), demand resolves still load,
     * and the output stream stays bitwise identical — a resolve's bytes
     * never depend on which speculative loads ran. The RSS guard also
     * frees every payload at each layer_end, so the EMPTY-slot resubmit
     * path is exercised hard. */
    LocalRun cd = run_local(cdir, rcfg, pids, (size_t)np, dids, (size_t)nd,
                            4, 1, 8, 1);
    CHECK(cd.digest == cn.digest, "drop-policy digest != pilot-OFF digest");
    CHECK(cd.hint_loads == 0,
          "drop policy: %llu speculative loads slipped through",
          (unsigned long long)cd.hint_loads);
    CHECK(cd.spec_dropped > 0, "drop policy: no speculative drops recorded");
    CHECK(cd.loads == cd.demand_loads && cd.demand_loads > 0,
          "drop policy: loads %llu != demand %llu",
          (unsigned long long)cd.loads, (unsigned long long)cd.demand_loads);
    CHECK(cd.misses == cd.loads, "drop policy: every resolve must miss "
          "(misses %llu loads %llu)", (unsigned long long)cd.misses,
          (unsigned long long)cd.loads);
    CHECK(cd.rss_drops > 0, "drop policy: RSS guard never fired");
    printf("H drop-pol digest %016llx spec_dropped %llu demand %llu "
           "rss_drops %llu\n",
           (unsigned long long)cd.digest, (unsigned long long)cd.spec_dropped,
           (unsigned long long)cd.demand_loads,
           (unsigned long long)cd.rss_drops);
    printf("E cluster  digest %016llx recall %llu/%llu (%.3f) demand %llu "
           "hits %llu misses %llu\n", (unsigned long long)cp.digest,
           (unsigned long long)cp.recall_num,
           (unsigned long long)cp.recall_den, rec_c,
           (unsigned long long)cp.demand_loads,
           (unsigned long long)cp.hits, (unsigned long long)cp.misses);
    printf("F cluster-nopilot digest %016llx demand %llu hits %llu "
           "misses %llu\n", (unsigned long long)cn.digest,
           (unsigned long long)cn.demand_loads,
           (unsigned long long)cn.hits, (unsigned long long)cn.misses);
    printf("G random   digest %016llx recall %llu/%llu (%.3f) "
           "demand %llu\n", (unsigned long long)rp.digest,
           (unsigned long long)rp.recall_num,
           (unsigned long long)rp.recall_den, rec_r,
           (unsigned long long)rp.demand_loads);
    (void)e_loc;

    printf("%s: %d checks, %d failures\n",
           failures ? "FAIL" : "PASS", checks, failures);
    apus_gpilot_destroy(pd);
    apus_gmodel_close(md);
    apus_gmodel_close(mc);
    apus_gmodel_close(mb);
    apus_gmodel_close(ma);
    free(pids);
    free(dids);
    free(lman);
    free(man);
    return failures ? 1 : 0;
}
