/*
 * tests/m6g/bench_gio.c — P2 standalone slab-I/O bench (NOT a gate;
 * informational). Establishes the machine's raw read ceiling for the GLM
 * expert-slab access pattern (ONE 24 MiB pread per expert slab, F_NOCACHE
 * streaming fd vs page-cached fd, 1..8 concurrent readers) and the cost of
 * the m3g dequant-on-fill, so the gcache I/O pool can be tuned against a
 * measured ceiling rather than a guess.
 *
 * Reads the real (or any glm5_next v2) container's apus.index.json expert
 * slab records directly — no cache, no model. Every scenario touches
 * DISTINCT slabs (slab cursor never rewinds), so cached-fd numbers are
 * cold-disk, not page-cache replays.
 *
 * Numerics are irrelevant here (no gate); dequant uses the production m3g
 * kernel so the fill scenario reproduces the worker-side fill cost.
 *
 * Env: APUS_GIO_SLABS (slabs per scenario, default 96),
 *      APUS_GIO_SKIP (leading scenarios to skip, for page-cache hygiene
 *      control), APUS_NOCACHE is NOT consulted (both fd kinds measured).
 *
 * Run from the repository root:  make bench-gio CONTAINER=weights/glm-5.3-flash
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_BF16_IMPLEMENTATION
#define APUS_FP8BLK_IMPLEMENTATION
#include "st.h"
#include "json.h"
#include "fp8blk.h"
#include "compat.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    ApusStLazy *lz;         /* cached fd + F_NOCACHE twin */
    ApusStLazy *lz_nc0;     /* same shard opened WITHOUT nocache (cached only) */
    uint64_t    off, len;
} BenchSlab;

typedef struct {
    char *name;
    ApusStLazy *lz;         /* opened with nocache=1: fd cached + fd_nc */
    ApusStLazy *lz_cached;  /* second open, nocache=0 */
} BenchShard;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* --- workload -----------------------------------------------------------*/

typedef struct {
    BenchSlab *slabs;
    size_t    n;
    size_t    slab_bytes;
    size_t    inter, dim;
    size_t    gw_b, gs_b, dw_b;
    int       use_nocache;      /* 1 = F_NOCACHE twin fd, 0 = cached fd */
    int       do_dequant;
    _Atomic size_t next;        /* slab cursor */
    double    t0, t1;           /* wall bracket of the whole scenario */
    double    deq_s;            /* summed per-thread dequant seconds */
    double    rd_s;             /* summed per-thread pread seconds */
    uint64_t  bytes;
    uint64_t  fills;
} Bench;

static void *bench_worker(void *arg) {
    Bench *b = arg;
    uint8_t *raw = apus_aligned_alloc(4096, b->slab_bytes);
    uint16_t *pl = b->do_dequant
                 ? apus_aligned_alloc(4096, 3 * b->inter * b->dim * 2)
                 : NULL;
    for (;;) {
        size_t i = atomic_fetch_add_explicit(&b->next, 1,
                                             memory_order_relaxed);
        if (i >= b->n) break;
        BenchSlab *s = &b->slabs[i];
        ApusStLazy *lz = b->use_nocache ? s->lz : s->lz_nc0;
        double r0 = now_s();
        if (apus_st_lazy_pread(lz, s->off, raw, (size_t)s->len)) {
            fprintf(stderr, "pread failed at slab %zu\n", i);
            exit(1);
        }
        double r1 = now_s();
        b->rd_s += r1 - r0;
        if (b->do_dequant) {
            const uint8_t *gw_c = raw;
            const float *gw_s = (const float *)(raw + b->gw_b);
            const uint8_t *up_c = raw + b->gw_b + b->gs_b;
            const float *up_s = (const float *)(raw + 2 * b->gw_b + b->gs_b);
            const uint8_t *dn_c = raw + 2 * (b->gw_b + b->gs_b);
            const float *dn_s = (const float *)(raw + 2 * (b->gw_b + b->gs_b)
                                                + b->dw_b);
            apus_fp8blk_dequant(gw_c, gw_s, pl, b->inter, b->dim);
            apus_fp8blk_dequant(up_c, up_s, pl + b->inter * b->dim,
                                b->inter, b->dim);
            apus_fp8blk_dequant(dn_c, dn_s, pl + 2 * b->inter * b->dim,
                                b->dim, b->inter);
            b->deq_s += now_s() - r1;
        }
        b->bytes += s->len;
        b->fills++;
    }
    apus_aligned_free(raw);
    apus_aligned_free(pl);
    return NULL;
}

static void bench_run(Bench *b, const char *tag, int threads) {
    atomic_store_explicit(&b->next, 0, memory_order_relaxed);
    b->deq_s = b->rd_s = 0.0;
    b->bytes = 0;
    b->fills = 0;
    pthread_t th[16];
    if (threads > 16) threads = 16;
    b->t0 = now_s();
    for (int i = 0; i < threads; i++)
        pthread_create(&th[i], NULL, bench_worker, b);
    for (int i = 0; i < threads; i++)
        pthread_join(th[i], NULL);
    b->t1 = now_s();
    double wall = b->t1 - b->t0;
    double mb = (double)b->bytes / (double)(1 << 20);
    printf("%-34s T=%-2d slabs=%-4llu wall %7.2fs  %8.1f MB/s  "
           "%6.1f ms/slab-wall",
           tag, threads, (unsigned long long)b->fills, wall,
           mb / wall, wall * 1e3 / (double)b->fills);
    if (b->do_dequant)
        printf("  (per-thread sums: pread %.1fs deq %.1fs, deq %.1f ms/slab)",
               b->rd_s, b->deq_s,
               b->deq_s * 1e3 / (double)b->fills);
    else
        printf("  (per-thread pread sum %.1fs, %.1f ms/slab)",
               b->rd_s, b->rd_s * 1e3 / (double)b->fills);
    printf("\n");
    fflush(stdout);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "weights/glm-5.3-flash";
    int per_scenario = apus_env_int("APUS_GIO_SLABS", 96);

    /* --- parse the manifest for slab records (gcache's parsing, trimmed) --*/
    char path[1200];
    snprintf(path, sizeof path, "%s/apus.index.json", dir);
    char jerr[160];
    JVal *man = json_parse_file(path, jerr, sizeof jerr);
    if (!man) { fprintf(stderr, "bench_gio: %s\n", jerr); return 1; }
    JVal *slabs = json_obj_get(man, "expert_slabs");
    JVal *nmv = json_obj_get(man, "n_main_layers");
    if (!slabs) { fprintf(stderr, "bench_gio: no expert_slabs\n"); return 1; }
    long n_main = nmv ? (long)json_num(nmv) : 1L << 30;

    /* fixture dims come from the real config; hard-code the glm5_next slab
     * geometry after verifying record sizes against the first record */
    BenchShard *shards = NULL;
    int shards_n = 0, shards_cap = 0;
    BenchSlab *recs = NULL;
    size_t recs_n = 0, recs_cap = 0;
    uint64_t slab_len = 0;
    for (size_t i = 0; i < json_arr_len(slabs); i++) {
        JVal *r = json_arr_get(slabs, i);
        const char *blk = json_str(json_obj_get(r, "block"));
        const char *shard = json_str(json_obj_get(r, "shard"));
        uint64_t off = (uint64_t)json_num(json_obj_get(r, "offset"));
        uint64_t nb = (uint64_t)json_num(json_obj_get(r, "nbytes"));
        int layer = -1;
        if (!blk || sscanf(blk, "layers.%d", &layer) != 1 || !shard) continue;
        if (layer >= n_main) continue;
        if (!slab_len) slab_len = nb;
        if (nb != slab_len) continue;
        if (recs_n == recs_cap) {
            recs_cap = recs_cap ? 2 * recs_cap : 4096;
            recs = realloc(recs, recs_cap * sizeof *recs);
        }
        int si = -1;
        for (int k = 0; k < shards_n; k++)
            if (!strcmp(shards[k].name, shard)) { si = k; break; }
        if (si < 0) {
            if (shards_n == shards_cap) {
                shards_cap = shards_cap ? 2 * shards_cap : 16;
                shards = realloc(shards, shards_cap * sizeof *shards);
            }
            char sp[1400];
            snprintf(sp, sizeof sp, "%s/%s", dir, shard);
            char err[256];
            ApusStLazy *lz = apus_st_lazy_open(sp, 1, err, sizeof err);
            ApusStLazy *lc = apus_st_lazy_open(sp, 0, err, sizeof err);
            if (!lz || !lc) {
                fprintf(stderr, "bench_gio: open %s: %s\n", sp, err);
                return 1;
            }
            si = shards_n;
            shards[shards_n].name = strdup(shard);
            shards[shards_n].lz = lz;
            shards[shards_n].lz_cached = lc;
            shards_n++;
        }
        recs[recs_n].lz = shards[si].lz;
        recs[recs_n].lz_nc0 = shards[si].lz_cached;
        recs[recs_n].off = off;
        recs[recs_n].len = nb;
        recs_n++;
    }
    json_free(man);
    printf("bench_gio: %zu slabs of %.3f MiB across %d shards in %s\n",
           recs_n, (double)slab_len / (double)(1 << 20), shards_n, dir);

    /* glm5_next slab geometry (checked against the manifest record size):
     * gate/up [inter,dim] + down [dim,inter] FP8 + F32 block scales */
    size_t inter = 2048, dim = 4096;
    size_t gw_b = inter * dim;
    size_t gs_b = apus_fp8blk_nblocks(inter) * apus_fp8blk_nblocks(dim) * 4;
    size_t dw_b = dim * inter;
    size_t ds_b = apus_fp8blk_nblocks(dim) * apus_fp8blk_nblocks(inter) * 4;
    size_t expect = 2 * (gw_b + gs_b) + dw_b + ds_b;
    if ((uint64_t)expect != slab_len) {
        /* tiny fixture container: derive inter/dim heuristically is
         * overkill — report and run read-only scenarios */
        printf("bench_gio: slab geometry != glm5_next (%zu != %llu) — "
               "dequant scenarios skipped\n", expect,
               (unsigned long long)slab_len);
        inter = dim = 0;
    }

    /* scenario order: interleave fd kinds and thread counts, each scenario
     * consuming the NEXT per_scenario slabs (cold — no page-cache replays) */
    size_t cursor = 0;
    int skip = apus_env_int("APUS_GIO_SKIP", 0);
    int scen = 0;
    struct { int threads; int nocache; int deq; } plan[] = {
        { 1, 1, 0 }, { 1, 0, 0 },
        { 2, 1, 0 }, { 2, 0, 0 },
        { 4, 1, 0 }, { 4, 0, 0 },
        { 6, 1, 0 }, { 8, 1, 0 }, { 8, 0, 0 },
        { 4, 1, 1 }, { 6, 1, 1 }, { 8, 1, 1 },
    };
    for (size_t pi = 0; pi < sizeof plan / sizeof plan[0]; pi++) {
        if ((size_t)per_scenario > recs_n - cursor) {
            printf("... out of slabs at scenario %zu\n", pi);
            break;
        }
        if (plan[pi].deq && !inter) continue;
        char tag[64];
        snprintf(tag, sizeof tag, "%s%s",
                 plan[pi].nocache ? "pread F_NOCACHE" : "pread cached   ",
                 plan[pi].deq ? " + dequant" : "          ");
        Bench b;
        memset(&b, 0, sizeof b);
        b.slabs = recs + cursor;
        b.n = (size_t)per_scenario;
        b.slab_bytes = slab_len;
        b.inter = inter;
        b.dim = dim;
        b.gw_b = gw_b;
        b.gs_b = gs_b;
        b.dw_b = dw_b;
        b.use_nocache = plan[pi].nocache;
        b.do_dequant = plan[pi].deq;
        cursor += (size_t)per_scenario;
        if (scen++ < skip) continue;
        bench_run(&b, tag, plan[pi].threads);
    }

    for (int i = 0; i < shards_n; i++) {
        free(shards[i].name);
        apus_st_lazy_close(shards[i].lz);
        apus_st_lazy_close(shards[i].lz_cached);
    }
    free(shards);
    free(recs);
    return 0;
}
