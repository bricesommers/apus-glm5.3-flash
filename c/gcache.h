/*
 * c/gcache.h — GLM expert tiering (M6): demand-loading of routed-expert
 * slabs from NVMe through a bounded RAM cache behind the M5 accessor seam
 * (c/gmodel.h apus_gmodel_expert). The V4 c/cache.h tiering design (per-layer
 * LRU + per-forward working set with end-of-block promotion, generation-
 * tagged miss overlap on a pthread I/O pool, demand/speculative job classes,
 * RSS guard, payload-buffer recycling) ported to the GLM glm5_next container
 * — NEW CODE, re-typed for the GLM FP8-E4M3/F32-scale slabs; the V4
 * c/cache.h is untouched (the retained V4 battery owns it). C11, libc +
 * pthreads.
 *
 * Slab fetch (the M1 coalescing invariant, tests/m1/README.md is the spec):
 * slab records come from the v2 manifest apus.index.json expert_slabs
 * ({block, expert, shard, offset, nbytes}, offset file-absolute). A cache
 * miss is ONE pread of the whole slab (instrumented: stats.preads ==
 * sum(apus_st_lazy_read_count) and bytes_read == preads * slab_bytes), the
 * six members carved by the pinned order (gate.w gate.s up.w up.s down.w
 * down.s) with config-derived sizes (slab nbytes hard-checked at open).
 *
 * Dequant-on-fill: the I/O worker dequantizes the three FP8 matrices to
 * BF16 with the m3g kernel (apus_fp8blk_dequant) into one contiguous slot
 * payload [gate | up | down] of 3*inter*dim codes — deterministic, so a
 * cache HIT and a cache MISS (and the M5 eager arena) yield byte-identical
 * BF16 expert weights. tests/m6g gates this explicitly (eager == big-cache
 * == tiny-cache digests, host-exp independent).
 *
 * Real scale: the 24.0059 MiB FP8 slab dequantizes to a 48 MiB BF16
 * payload (inter 2048 x dim 4096). The §7 budget is therefore counted in
 * PAYLOAD bytes: 12,384 experts, top-8/token, ~450 MiB/token cold demand
 * per MoE layer; ~10–14 GB of cache holds ~210–290 resident experts
 * (1.7–2.4% of 12,384). Decode speed is pilot-recall-bound (c/gpilot.h).
 *
 * Cache policy (port of the V4 design, trimmed — no pins/usage-history/
 * REPIN; see tests/m6g/README.md for the rationale):
 *   - Per-layer LRU slot arrays. Misses load into a small per-forward
 *     working set (never directly into the LRU), promoted at layer end by
 *     swapping with the coldest slots; hits bump an atomic clock.
 *   - Miss overlap: a pthread I/O pool pulls miss jobs (pread + dequant on
 *     the worker); each job is generation-tagged so a straggler completing
 *     after its slot was recycled cannot corrupt a newer generation. The
 *     compute thread never does I/O in pool mode — it waits just-in-time on
 *     a per-slot condvar. F_NOCACHE streaming reads keep expert traffic out
 *     of the page cache (macOS; Linux reads stay page-cached, c/compat.h).
 *   - Job priority: loads are demand-class (apus_gcache_hint_demand — the
 *     MoE union storm — and any LOADING slot a resolve blocks on) or
 *     speculative (apus_gcache_hint — the pilot surface). Workers pop the
 *     first demand-class job before FIFO speculative ones.
 *   - Speculative yield/drop (P2 — the hint-storm fix): a speculative hint
 *     is DROPPED (never submitted, no state changed, stats.spec_dropped++)
 *     when (a) any demand-class load is queued or in flight (speculation
 *     must yield the disk to demand), (b) the speculative queue already
 *     holds APUS_GSPEC_QUEUE jobs (default 2 x io_threads; the backlog cap),
 *     or (c) process RSS is within one payload of the RSS-guard budget
 *     (speculation must never push the guard into dropping decode-live LRU
 *     payloads). Demand loads are NEVER dropped: correctness (a resolve
 *     always returns real weights) does not depend on any speculative load
 *     ever running. Because a speculative fill's working-set entry has
 *     last == 0, end-of-block promotion can only displace empty or
 *     never-resolved (last == 0) slots — speculation cannot evict a
 *     demand-used payload through promotion either. Net effect: the I/O
 *     pool serves demand first and spends only idle capacity on
 *     speculation. Bitwise-neutral: hits and misses return the same
 *     dequantized bytes regardless of which loads ran (tests/m6g).
 *   - RSS guard: when the process memory-pressure footprint
 *     (apus_pressure_bytes — phys_footprint on macOS, which unlike
 *     resident_size charges COMPRESSED pages; the P1 run showed
 *     resident_size blind to a 32.5 GB footprint on a 32 GB machine)
 *     exceeds the budget, coldest LRU payloads are freed in place (slots
 *     keep their identity) — only runs at layer boundaries (never while a
 *     forward holds payload views).
 *   - Buffer recycling: evicted payload buffers and used raw-slab staging
 *     buffers go on bounded free lists (the V4 M6c lesson: a 25–48 MiB
 *     mmap/munmap plus zero-fill soft faults per fill is major kernel
 *     time). RSS-guard drops really free().
 *
 * Pointer-stability contract: views returned by apus_gcache_resolve stay
 * valid until the NEXT apus_gcache_layer_end (or rss_guard) call on that
 * layer — exactly the M5 wiring pattern (resolve the selected experts, run
 * the MoE sublayer, layer_end). Eviction/promotion only happens at
 * layer_end from the compute thread, so a sublayer's views are never
 * dropped under it.
 *
 * Budgets (env, sane defaults for a 32 GB Mac; all overridable via
 * ApusGcacheCfg): APUS_GEXPERT_CACHE_MB (default 6144),
 * APUS_RSS_GUARD_MB (26624 — emergency brake), APUS_IO_THREADS (4),
 * APUS_NOCACHE (1), APUS_GCACHE_BOOST (1), APUS_GSPEC_QUEUE (speculative
 * backlog cap; 0 = 2 x io_threads), APUS_GBUF_FREE (16 payloads),
 * APUS_GBUF_FREE_RAW (8 staging slabs).
 *
 * Threading contract: resolve/hint_demand/layer_end are called from the
 * compute thread (the c/gmodel.h MoE wiring); apus_gcache_hint is safe from
 * any thread (the pilot surface). rss_guard must be called between
 * forwards (layer_end calls it automatically).
 *
 * Usage: #define APUS_GCACHE_IMPLEMENTATION in exactly one TU (also needs
 * the st/json/compat/fp8blk/bf16 implementations linked — bf16 via fp8blk).
 */
#ifndef APUS_GCACHE_H
#define APUS_GCACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int    n_layers;          /* required: TOTAL model layers (records exist
                                 only for the sparse ones) */
    int    n_experts;         /* required: routed experts per sparse layer */
    size_t moe_inter;         /* required: moe_intermediate_size */
    size_t dim;               /* required: hidden_size */
    size_t cache_bytes;       /* LRU budget; 0 = APUS_GEXPERT_CACHE_MB env */
    int    slots_per_layer;   /* explicit LRU slots; 0 = derive from budget */
    size_t rss_budget_bytes;  /* 0 = APUS_RSS_GUARD_MB env */
    int    io_threads;        /* 0 = env/default (4); <0 = synchronous mode */
    int    nocache;           /* >0 = F_NOCACHE reads, <0 = cached fds,
                                 0 = APUS_NOCACHE env (default 1) */
} ApusGcacheCfg;

typedef struct ApusGcache ApusGcache;

/* Dequantized expert views (BF16 codes): gate/up [inter, dim], down
 * [dim, inter] — the same layout the M5 eager arena exposes. */
typedef struct {
    const uint16_t *gate;
    const uint16_t *up;
    const uint16_t *down;
} ApusGcacheW;

typedef struct {
    uint64_t hits;            /* resolves served from a resident LRU slot */
    uint64_t misses;          /* resolves whose slab came from disk for this
                                 block (demand loads + first consume of a
                                 hint-loaded working-set entry) */
    uint64_t loads;           /* slab fills submitted (demand + speculative) */
    uint64_t demand_loads;    /* of loads: submitted on the demand path —
                                 resolve-(re)submits AND apus_gcache_hint_
                                 demand (the MoE union storm). With the pilot
                                 on, demand_loads == the selections the pilot
                                 did NOT have present: the prefetch-coverage
                                 metric (tests/m6g) */
    uint64_t hint_loads;      /* of loads: submitted via apus_gcache_hint
                                 (the speculative pilot surface) */
    uint64_t preads;          /* apus_st_lazy_pread calls (1 per slab fill) */
    uint64_t bytes_read;
    uint64_t evictions;       /* LRU payloads replaced at promotion */
    uint64_t rss_drops;       /* LRU payloads freed by the RSS guard */
    uint64_t waits;           /* resolves that blocked on an in-flight load */
    uint64_t wait_ns;         /* total ns the compute thread blocked */
    uint64_t deq_ns;          /* total ns inside dequant (I/O workers) */
    uint64_t spec_dropped;    /* speculative hints dropped by the P2
                                 yield/drop admission control (never
                                 submitted — see the header policy block) */
    uint64_t pressure_peak;   /* P4 reporting-only: peak
                                 apus_pressure_bytes() sampled on the
                                 resolve path and in the RSS guard (0 =
                                 never sampled). Never feeds any policy. */
} ApusGcacheStats;

/* Open the store over a converted v2 container dir: reads
 * apus.index.json (format_version 2, glm5_next verified), derives the slab
 * records, hard-checks the per-expert slab size against the config-derived
 * member layout. Returns NULL on error (err filled if given). */
ApusGcache *apus_gcache_open(const char *container_dir,
                             const ApusGcacheCfg *cfg,
                             char *err, size_t errcap);
void        apus_gcache_close(ApusGcache *st);

/* Resolve expert (layer, eid): fills *w with views into a cache slot,
 * waiting just-in-time if a load is in flight. Returns 0 on success.
 * Views are valid until the next layer_end/rss_guard on `layer`. */
int  apus_gcache_resolve(ApusGcache *st, int layer, int eid,
                         ApusGcacheW *w);

/* Non-blocking prefetch hint: submits the miss job (dequant-on-fill)
 * without waiting. Safe from any thread (the pilot surface). Dedupes
 * against slots, working set, and in-flight jobs. SPECULATIVE class. */
void apus_gcache_hint(ApusGcache *st, int layer, int eid);

/* Demand-class variant: identical submission semantics, but the load is
 * served by the I/O pool ahead of queued speculative loads. Used by the
 * MoE wiring for the about-to-be-resolved union (the batch-union storm). */
void apus_gcache_hint_demand(ApusGcache *st, int layer, int eid);

/* End-of-block: promote this layer's working set into the LRU (swap with
 * the coldest slots), advance the load generation, run the RSS guard.
 * Called by the model wiring after each MoE sublayer. */
void apus_gcache_layer_end(ApusGcache *st, int layer);

/* RSS guard: if RSS > budget, free coldest LRU payloads in place (slots
 * keep eid/freq identity; working-set and in-flight loads untouched). Safe
 * to call any time no forward holds payload views. */
void apus_gcache_rss_guard(ApusGcache *st);

void     apus_gcache_stats(const ApusGcache *st, ApusGcacheStats *out);
size_t   apus_gcache_slab_bytes(const ApusGcache *st);    /* raw FP8 slab */
size_t   apus_gcache_payload_bytes(const ApusGcache *st); /* BF16 slot */
size_t   apus_gcache_resident_bytes(ApusGcache *st);      /* live payloads */
uint64_t apus_gcache_pread_count(ApusGcache *st);         /* sum over shards */

/* --- introspection / test hooks ------------------------------------------*/

/* Snapshot of one layer's cache: LRU slot eids (-1 = empty). May be NULL. */
int  apus_gcache_debug_layer(ApusGcache *st, int layer,
                             int32_t *lru_eids, int n_lru);
/* 1 iff (layer, eid) is present (resident or load in flight) — a demand
 * resolve right now would NOT submit a fresh load. */
int  apus_gcache_debug_present(ApusGcache *st, int layer, int eid);
/* 1 iff (layer, eid) is READY with a live payload. */
int  apus_gcache_debug_ready(ApusGcache *st, int layer, int eid);

#ifdef __cplusplus
}
#endif

/* =========================================================================*/
/* Outside the include guard (the c/blas.h pattern): safe to re-include. */
#if defined(APUS_GCACHE_IMPLEMENTATION) && !defined(APUS_GCACHE_IMPL_INCLUDED)
#define APUS_GCACHE_IMPL_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <pthread.h>

#include "compat.h"
#include "fp8blk.h"
#include "json.h"
#include "st.h"

/* --- records ---------------------------------------------------------------*/

typedef struct {
    ApusStLazy *lz;         /* shard reader (deduped by file name) */
    uint64_t    off;        /* absolute file offset of the slab */
    uint64_t    len;        /* slab bytes (hard-checked) */
} ApusGslabRec;

/* --- slots -----------------------------------------------------------------*/

enum { APUS_GSLOT_EMPTY = 0, APUS_GSLOT_LOADING = 1, APUS_GSLOT_READY = 2 };

typedef struct {
    int32_t   eid;          /* -1 = unassigned */
    uint16_t *payload;      /* BF16 [gate | up | down], NULL when dropped */
    uint64_t  last;         /* LRU clock of last use */
    uint64_t  freq;         /* uses since load (promotion warmth) */
    int       state;
    uint64_t  gen;          /* generation tag of the in-flight load */
    uint8_t   hot;          /* demand-class: served before speculative */
} ApusGslot;

typedef struct {
    ApusGslot  *slots;      /* LRU slots [n_slots] */
    int         n_slots;
    ApusGslot **ws;         /* per-forward working set (heap slots) */
    int         ws_n, ws_cap;
} ApusGlayerCache;

typedef struct {
    int          layer;
    ApusGslot   *slot;
    ApusGslabRec *rec;
    uint64_t     gen;
    int          hot;         /* demand-class snapshot, refreshed at pop
                                 (a queued speculative job boosted by a
                                 blocked resolve pops as demand) */
} ApusGjob;

struct ApusGcache {
    int             n_layers, E;
    size_t          inter, dim;
    size_t          slab_bytes;     /* raw FP8 slab */
    size_t          payload_elems;  /* 3 * inter * dim */
    size_t          payload_bytes;
    /* member carve layout within the raw slab (pinned order) */
    size_t          gw_b, gs_b, dw_b, ds_b;
    ApusGslabRec   *recs;           /* [n_layers * E]; len 0 = no record */
    ApusGlayerCache *lc;            /* [n_layers] */
    struct { char *name; ApusStLazy *lz; } *shards;
    int             shards_n, shards_cap;
    /* sync */
    pthread_mutex_t mu;
    pthread_cond_t  cv;             /* slot completions */
    /* I/O pool */
    pthread_t      *threads;
    int             n_threads;      /* 0 = synchronous mode */
    ApusGjob       *jobs;
    int             jq_head, jq_n, jq_cap;
    pthread_cond_t  jq_cv;
    int             stopping;
    int             boost;
    /* P2 speculative yield/drop state (under mu) */
    int             spec_cap;       /* max queued speculative jobs */
    int             spec_q;         /* currently queued speculative jobs */
    int             hot_flight;     /* demand-class fills being run now */
    /* buffer recycling (all under mu) */
    uint16_t      **pl_free;        /* payload-class buffers */
    int             pl_free_n, pl_free_cap;
    uint8_t       **raw_free;       /* raw-slab staging buffers */
    int             raw_free_n, raw_free_cap;
    /* policy state (all under mu) */
    uint64_t        clock;
    uint64_t        gen;
    size_t          rss_budget;
    ApusGcacheStats stats;
};

/* --- small utilities --------------------------------------------------------*/

static uint64_t apus_gc_tick(ApusGcache *st) { return ++st->clock; }

static uint8_t *apus_gc_raw_get(ApusGcache *st) {
    pthread_mutex_lock(&st->mu);
    if (st->raw_free_n > 0) {
        uint8_t *b = st->raw_free[--st->raw_free_n];
        pthread_mutex_unlock(&st->mu);
        return b;
    }
    pthread_mutex_unlock(&st->mu);
    return (uint8_t *)apus_aligned_alloc(4096, st->slab_bytes);
}

static void apus_gc_raw_put(ApusGcache *st, uint8_t *b) {   /* mu held */
    if (!b) return;
    if (st->raw_free_n < st->raw_free_cap)
        st->raw_free[st->raw_free_n++] = b;
    else
        apus_aligned_free(b);
}

static uint16_t *apus_gc_pl_get(ApusGcache *st) {
    pthread_mutex_lock(&st->mu);
    if (st->pl_free_n > 0) {
        uint16_t *b = st->pl_free[--st->pl_free_n];
        pthread_mutex_unlock(&st->mu);
        return b;
    }
    pthread_mutex_unlock(&st->mu);
    return (uint16_t *)apus_aligned_alloc(4096, st->payload_bytes);
}

static void apus_gc_pl_put(ApusGcache *st, uint16_t *b) {   /* mu held */
    if (!b) return;
    if (st->pl_free_n < st->pl_free_cap)
        st->pl_free[st->pl_free_n++] = b;
    else
        apus_aligned_free(b);
}

static ApusGslabRec *apus_gc_rec(ApusGcache *st, int layer, int eid) {
    return &st->recs[(size_t)layer * st->E + eid];
}

static void apus_gc_views(const ApusGcache *st, const uint16_t *payload,
                          ApusGcacheW *w) {
    w->gate = payload;
    w->up = payload + st->inter * st->dim;
    w->down = payload + 2 * st->inter * st->dim;
}

/* --- I/O pool ------------------------------------------------------------------*/

static void apus_gc_job_push(ApusGcache *st, ApusGjob j) {
    if (st->jq_n == st->jq_cap) {
        /* grow: re-lay-out the ring linearly (the V4 M6b lesson: a plain
         * realloc keeps the bytes but the modulo indexing changes, so
         * wrapped entries would read back as garbage) */
        int ncap = st->jq_cap ? 2 * st->jq_cap : 32;
        ApusGjob *nj = (ApusGjob *)malloc((size_t)ncap * sizeof *nj);
        for (int i = 0; i < st->jq_n; i++)
            nj[i] = st->jobs[(st->jq_head + i) % st->jq_cap];
        free(st->jobs);
        st->jobs = nj;
        st->jq_cap = ncap;
        st->jq_head = 0;
    }
    int tail = (st->jq_head + st->jq_n) % st->jq_cap;
    st->jobs[tail] = j;
    st->jq_n++;
    pthread_cond_signal(&st->jq_cv);
}

/* Worker-side fill: ONE pread of the raw slab into a staging buffer, carve
 * the six members by the pinned order, dequant the three FP8 matrices into
 * the slot payload (m3g kernel — deterministic, so hit == miss == the M5
 * eager arena byte for byte), then claim the slot only if the generation
 * tag still matches (straggler safety). Buffers are recycled on a failed
 * claim — they can never alias a newer generation's slot. */
static void apus_gc_run_job(ApusGcache *st, ApusGjob j) {
    uint8_t *raw = apus_gc_raw_get(st);
    uint16_t *pl = apus_gc_pl_get(st);
    if (!raw || !pl) {
        pthread_mutex_lock(&st->mu);
        apus_gc_raw_put(st, raw);
        apus_gc_pl_put(st, pl);
        j.slot->state = APUS_GSLOT_EMPTY;
        j.slot->eid = -1;
        pthread_cond_broadcast(&st->cv);
        pthread_mutex_unlock(&st->mu);
        return;
    }
    int pbad = apus_st_lazy_pread(j.rec->lz, j.rec->off, raw,
                                  (size_t)j.rec->len);
    apus_fadvise_dontneed(-1, j.rec->off, j.rec->len); /* F_NOCACHE covers */
    struct timespec d0, d1;
    clock_gettime(CLOCK_MONOTONIC, &d0);
    if (!pbad) {
        const size_t gw_b = st->gw_b, gs_b = st->gs_b;
        const size_t dw_b = st->dw_b, ds_b = st->ds_b;
        const uint8_t *gw_c = raw;
        const float *gw_s = (const float *)(raw + gw_b);
        const uint8_t *up_c = raw + gw_b + gs_b;
        const float *up_s = (const float *)(raw + gw_b + gs_b + gw_b);
        const uint8_t *dn_c = raw + 2 * (gw_b + gs_b);
        const float *dn_s = (const float *)(raw + 2 * (gw_b + gs_b) + dw_b);
        (void)ds_b;
        apus_fp8blk_dequant(gw_c, gw_s, pl, st->inter, st->dim);
        apus_fp8blk_dequant(up_c, up_s, pl + st->inter * st->dim,
                            st->inter, st->dim);
        apus_fp8blk_dequant(dn_c, dn_s, pl + 2 * st->inter * st->dim,
                            st->dim, st->inter);
    }
    clock_gettime(CLOCK_MONOTONIC, &d1);
    uint64_t dns = (uint64_t)(d1.tv_sec - d0.tv_sec) * 1000000000ull
                 + (uint64_t)(d1.tv_nsec - d0.tv_nsec);
    pthread_mutex_lock(&st->mu);
    apus_gc_raw_put(st, raw);
    st->stats.preads++;
    st->stats.bytes_read += j.rec->len;
    st->stats.deq_ns += dns;
    if (!pbad && j.slot->state == APUS_GSLOT_LOADING
        && j.slot->gen == j.gen) {
        j.slot->payload = pl;
        j.slot->state = APUS_GSLOT_READY;
        j.slot->hot = 0;                /* served; boost no longer needed */
    } else {
        /* pread failure or stale generation: drop the payload and reset
         * the slot so a waiter re-submits a fresh job */
        apus_gc_pl_put(st, pl);
        j.slot->state = APUS_GSLOT_EMPTY;
        j.slot->hot = 0;
        if (pbad) j.slot->eid = -1;
    }
    pthread_cond_broadcast(&st->cv);
    pthread_mutex_unlock(&st->mu);
}

static void *apus_gc_worker(void *arg) {
    ApusGcache *st = arg;
    for (;;) {
        pthread_mutex_lock(&st->mu);
        while (!st->jq_n && !st->stopping)
            pthread_cond_wait(&st->jq_cv, &st->mu);
        if (st->stopping && !st->jq_n) {
            pthread_mutex_unlock(&st->mu);
            return NULL;
        }
        /* hot-first pop: a job is hot while its slot is demand-class (the
         * MoE union storm or a resolve already waiting on it). Scan from
         * the head for the first hot job; fall back to plain FIFO. The scan
         * is O(queue depth) under mu — depths are a few hundred at most
         * (pilot bursts), and serving the about-to-be-resolved slab first
         * is exactly the pipeline's priority fix: speculative next-layer
         * loads must not delay this layer's demand. */
        int pick = 0;
        if (st->boost)
            for (int i = 0; i < st->jq_n; i++) {
                ApusGjob *cjob = &st->jobs[(st->jq_head + i) % st->jq_cap];
                if (cjob->slot->hot) { pick = i; break; }
            }
        ApusGjob j = st->jobs[(st->jq_head + pick) % st->jq_cap];
        for (int i = pick; i > 0; i--)
            st->jobs[(st->jq_head + i) % st->jq_cap] =
                st->jobs[(st->jq_head + i - 1) % st->jq_cap];
        st->jq_head = (st->jq_head + 1) % st->jq_cap;
        st->jq_n--;
        /* queue-accounting uses the SUBMIT-time class (a queued spec job
         * boosted by a blocked resolve still occupied a spec slot); the
         * flight accounting uses the CURRENT class */
        if (!j.hot) st->spec_q--;
        j.hot = j.slot->hot;
        if (j.hot) st->hot_flight++;
        pthread_mutex_unlock(&st->mu);
        apus_gc_run_job(st, j);
        pthread_mutex_lock(&st->mu);
        if (j.hot) st->hot_flight--;
        pthread_mutex_unlock(&st->mu);
    }
}

/* Submit a fill for slot (caller holds mu, slot marked LOADING first). */
static void apus_gc_submit(ApusGcache *st, int layer, ApusGslot *slot,
                           ApusGslabRec *rec) {
    slot->state = APUS_GSLOT_LOADING;
    slot->gen = st->gen;
    st->stats.loads++;
    ApusGjob j = { layer, slot, rec, st->gen, slot->hot };
    if (st->n_threads > 0) {
        if (!j.hot) st->spec_q++;
        apus_gc_job_push(st, j);
    } else {
        /* synchronous mode: run inline after dropping the lock */
        pthread_mutex_unlock(&st->mu);
        apus_gc_run_job(st, j);
        pthread_mutex_lock(&st->mu);
    }
}

/* Wait until slot is READY; re-submit if a stale claim reset it (mu held).
 * Resolve path only — re-submits count as demand loads. The wait is timed
 * (stats.waits/wait_ns) and a LOADING slot the compute thread is about to
 * block on is boosted to demand-class. */
static int apus_gc_wait_ready(ApusGcache *st, int layer, ApusGslot *slot,
                              ApusGslabRec *rec) {
    struct timespec t0, t1;
    int timed = 0;
    while (slot->state != APUS_GSLOT_READY) {
        if (!timed) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            timed = 1;
            if (slot->state == APUS_GSLOT_LOADING)
                slot->hot = 1;      /* resolve is blocked on this load */
        }
        if (slot->state == APUS_GSLOT_EMPTY) {
            if (slot->eid < 0) return -1;   /* alloc/pread failure earlier */
            st->stats.demand_loads++;
            slot->hot = 1;
            apus_gc_submit(st, layer, slot, rec);
        } else {
            pthread_cond_wait(&st->cv, &st->mu);
        }
    }
    if (timed) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        st->stats.waits++;
        st->stats.wait_ns += (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull
                           + (uint64_t)(t1.tv_nsec - t0.tv_nsec);
    }
    return 0;
}

/* --- resolve / hint ------------------------------------------------------------*/

static int apus_gc_slot_find(const ApusGslot *slots, int n, int32_t eid) {
    for (int i = 0; i < n; i++)
        if (slots[i].eid == eid) return i;
    return -1;
}

static ApusGslot *apus_gc_ws_find(ApusGlayerCache *lc, int32_t eid) {
    for (int i = 0; i < lc->ws_n; i++)
        if (lc->ws[i]->eid == eid) return lc->ws[i];
    return NULL;
}

/* kind_out: 2 = working set, 3 = LRU. (mu held) */
static int apus_gc_lookup(ApusGcache *st, int layer, int eid,
                          ApusGslot **slot_out, int *kind_out) {
    ApusGlayerCache *lc = &st->lc[layer];
    ApusGslot *w = apus_gc_ws_find(lc, eid);
    if (w) { *slot_out = w; *kind_out = 2; return 0; }
    int i = apus_gc_slot_find(lc->slots, lc->n_slots, eid);
    if (i >= 0 && lc->slots[i].payload) {
        *slot_out = &lc->slots[i];
        *kind_out = 3;
        return 0;
    }
    return -1;
}

static ApusGslot *apus_gc_ws_add(ApusGlayerCache *lc, int32_t eid,
                                 int freq, int hot) {
    ApusGslot *w = (ApusGslot *)calloc(1, sizeof *w);
    if (!w) return NULL;
    w->eid = eid;
    w->freq = (uint64_t)freq;
    w->hot = (uint8_t)hot;
    if (lc->ws_n == lc->ws_cap) {
        lc->ws_cap = lc->ws_cap ? 2 * lc->ws_cap : 8;
        lc->ws = (ApusGslot **)realloc(lc->ws,
                                       (size_t)lc->ws_cap * sizeof *lc->ws);
    }
    lc->ws[lc->ws_n++] = w;
    return w;
}

int apus_gcache_resolve(ApusGcache *st, int layer, int eid,
                        ApusGcacheW *w) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E)
        return -1;
    ApusGslabRec *rec = apus_gc_rec(st, layer, eid);
    if (!rec->len) return -1;               /* not a sparse-layer expert */
    /* P4 reporting-only peak sample: resolves are where the working set
     * accumulates mid-layer, so the prefill transient peaks here. */
    {
        uint64_t rss = apus_pressure_bytes();
        if (rss > st->stats.pressure_peak) st->stats.pressure_peak = rss;
    }
    pthread_mutex_lock(&st->mu);
    ApusGslot *slot = NULL;
    int kind = 0;
    if (apus_gc_lookup(st, layer, eid, &slot, &kind) == 0) {
        /* hit := the slab was already LRU-resident. A first consume of a
         * hint-loaded working-set entry is a miss — the slab came from
         * disk for this block. */
        int resident = !(kind == 2 && slot->freq == 0)
                       && slot->state == APUS_GSLOT_READY
                       && slot->payload != NULL;
        if (apus_gc_wait_ready(st, layer, slot, rec)) {
            pthread_mutex_unlock(&st->mu);
            return -1;
        }
        slot->freq++;
        slot->last = apus_gc_tick(st);
        if (resident) st->stats.hits++;
        else st->stats.misses++;
    } else {
        /* miss: load into the per-forward working set */
        ApusGlayerCache *lc = &st->lc[layer];
        ApusGslot *ws = apus_gc_ws_add(lc, eid, 1, 1);
        if (!ws) {
            pthread_mutex_unlock(&st->mu);
            return -1;
        }
        st->stats.demand_loads++;
        apus_gc_submit(st, layer, ws, rec);
        if (apus_gc_wait_ready(st, layer, ws, rec)) {
            pthread_mutex_unlock(&st->mu);
            return -1;
        }
        ws->last = apus_gc_tick(st);
        st->stats.misses++;
        slot = ws;
    }
    apus_gc_views(st, slot->payload, w);
    pthread_mutex_unlock(&st->mu);
    return 0;
}

/* P2 speculative admission control (mu held). A speculative load is a pure
 * optimization — dropping one never changes the bytes any resolve returns
 * (the demand path loads what it needs regardless), so the drop policy is
 * free to be aggressive. Drop when:
 *   (a) a demand-class fill is in flight or queued — speculation yields the
 *       disk to demand (the P1 hint storm starved demand fills 5:1);
 *   (b) the speculative backlog is already spec_cap deep — a deeper queue
 *       is stale by the time it drains (the pilot re-hints every token);
 *   (c) RSS is within one fill (raw slab + BF16 payload) of the guard
 *       budget — speculation must never push the guard into dropping
 *       decode-live LRU payloads.
 * In synchronous mode the queue/in-flight terms are always 0 (jobs run
 * inline), leaving only the RSS rule — deterministic for the m6g gate. */
static int apus_gc_spec_admit(ApusGcache *st) {
    if (st->hot_flight > 0) return 0;
    for (int i = 0; i < st->jq_n; i++)
        if (st->jobs[(st->jq_head + i) % st->jq_cap].slot->hot) return 0;
    if (st->spec_q >= st->spec_cap) return 0;
    uint64_t rss = apus_pressure_bytes();
    if (rss && rss + st->payload_bytes + st->slab_bytes > st->rss_budget)
        return 0;
    return 1;
}

/* Shared hint body (mu NOT held). demand: mark the load hot so the I/O
 * pool serves it ahead of queued speculative loads. */
static void apus_gcache_hint_impl(ApusGcache *st, int layer, int eid,
                                  int demand) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E)
        return;
    ApusGslabRec *rec = apus_gc_rec(st, layer, eid);
    if (!rec->len) return;
    pthread_mutex_lock(&st->mu);
    ApusGslot *slot;
    int kind;
    if (apus_gc_lookup(st, layer, eid, &slot, &kind) == 0) {
        if (slot->state == APUS_GSLOT_EMPTY && slot->eid >= 0) {
            if (!demand && !apus_gc_spec_admit(st)) {
                st->stats.spec_dropped++;
                pthread_mutex_unlock(&st->mu);
                return;
            }
            slot->hot = demand ? 1 : slot->hot;
            if (demand) st->stats.demand_loads++;
            else st->stats.hint_loads++;
            apus_gc_submit(st, layer, slot, rec);
        } else if (demand && slot->state == APUS_GSLOT_LOADING) {
            slot->hot = 1;      /* boost the in-flight/queued load */
        }
        pthread_mutex_unlock(&st->mu);
        return;
    }
    if (!demand && !apus_gc_spec_admit(st)) {
        st->stats.spec_dropped++;
        pthread_mutex_unlock(&st->mu);
        return;
    }
    ApusGlayerCache *lc = &st->lc[layer];
    ApusGslot *w = apus_gc_ws_add(lc, eid, 0, demand);
    if (!w) {
        pthread_mutex_unlock(&st->mu);
        return;
    }
    if (demand) st->stats.demand_loads++;
    else st->stats.hint_loads++;
    apus_gc_submit(st, layer, w, rec);
    pthread_mutex_unlock(&st->mu);
}

void apus_gcache_hint(ApusGcache *st, int layer, int eid) {
    apus_gcache_hint_impl(st, layer, eid, 0);
}

void apus_gcache_hint_demand(ApusGcache *st, int layer, int eid) {
    apus_gcache_hint_impl(st, layer, eid, 1);
}

/* --- end-of-block promotion -----------------------------------------------------*/

/* Pick the LRU victim slot: prefer a truly empty slot, else the coldest.
 * Never returns a LOADING slot. (mu held) */
static ApusGslot *apus_gc_lru_victim(ApusGlayerCache *lc) {
    ApusGslot *best = NULL;
    for (int i = 0; i < lc->n_slots; i++) {
        ApusGslot *s = &lc->slots[i];
        if (s->state == APUS_GSLOT_LOADING) continue;
        if (s->eid < 0) return s;           /* empty: free real estate */
        if (!best || s->last < best->last) best = s;
    }
    return best;
}

void apus_gcache_layer_end(ApusGcache *st, int layer) {
    if (!st || layer < 0 || layer >= st->n_layers) return;
    pthread_mutex_lock(&st->mu);
    ApusGlayerCache *lc = &st->lc[layer];
    int out = 0;
    for (int i = 0; i < lc->ws_n; i++) {
        ApusGslot *w = lc->ws[i];
        if (w->state != APUS_GSLOT_READY) {
            /* in-flight (pilot hint not yet consumed): keep in the working
             * set; promoted by a later layer_end once READY */
            lc->ws[out++] = w;
            continue;
        }
        ApusGslot *v = apus_gc_lru_victim(lc);
        if (v && v->last <= w->last) {
            if (v->eid >= 0 && v->payload) st->stats.evictions++;
            apus_gc_pl_put(st, v->payload);
            v->eid = w->eid;
            v->payload = w->payload;
            v->freq = w->freq;
            v->last = w->last;
            v->state = APUS_GSLOT_READY;
            v->hot = 0;
            free(w);
        } else {
            /* LRU full of warmer entries (union overflow): the slab served
             * this block and is dropped without entering the cache */
            apus_gc_pl_put(st, w->payload);
            free(w);
        }
    }
    lc->ws_n = out;
    st->gen++;
    pthread_mutex_unlock(&st->mu);
    apus_gcache_rss_guard(st);
}

/* --- RSS guard ------------------------------------------------------------------*/

void apus_gcache_rss_guard(ApusGcache *st) {
    uint64_t rss = apus_pressure_bytes();
    if (rss > st->stats.pressure_peak) st->stats.pressure_peak = rss;
    if (!rss || rss <= st->rss_budget) return;
    uint64_t excess = rss - st->rss_budget;
    pthread_mutex_lock(&st->mu);
    size_t total = 0;
    for (int l = 0; l < st->n_layers; l++)
        total += (size_t)st->lc[l].n_slots;
    ApusGslot **ord = (ApusGslot **)malloc((total ? total : 1)
                                           * sizeof *ord);
    size_t n = 0;
    for (int l = 0; l < st->n_layers; l++)
        for (int i = 0; i < st->lc[l].n_slots; i++) {
            ApusGslot *s = &st->lc[l].slots[i];
            if (s->state == APUS_GSLOT_READY && s->payload) ord[n++] = s;
        }
    for (size_t i = 0; i + 1 < n; i++)
        for (size_t j = i + 1; j < n; j++)
            if (ord[j]->last < ord[i]->last) {
                ApusGslot *t = ord[i]; ord[i] = ord[j]; ord[j] = t;
            }
    uint64_t freed = 0;
    for (size_t i = 0; i < n && freed < excess; i++) {
        apus_aligned_free(ord[i]->payload);     /* real relief, no recycle */
        ord[i]->payload = NULL;
        ord[i]->state = APUS_GSLOT_EMPTY;   /* slot keeps eid/freq identity */
        freed += st->payload_bytes;
        st->stats.rss_drops++;
    }
    free(ord);
    pthread_mutex_unlock(&st->mu);
}

/* --- open / close ------------------------------------------------------------------*/

static ApusStLazy *apus_gc_shard(ApusGcache *st, const char *dir,
                                 const char *name, int nocache,
                                 char *err, size_t errcap) {
    for (int i = 0; i < st->shards_n; i++)
        if (!strcmp(st->shards[i].name, name)) return st->shards[i].lz;
    if (st->shards_n == st->shards_cap) {
        st->shards_cap = st->shards_cap ? 2 * st->shards_cap : 8;
        st->shards = realloc(st->shards,
                             (size_t)st->shards_cap * sizeof *st->shards);
    }
    char path[1400];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    ApusStLazy *lz = apus_st_lazy_open(path, nocache, err, errcap);
    if (!lz) return NULL;
    st->shards[st->shards_n].name = strdup(name);
    st->shards[st->shards_n].lz = lz;
    st->shards_n++;
    return lz;
}

ApusGcache *apus_gcache_open(const char *container_dir,
                             const ApusGcacheCfg *cfg,
                             char *err, size_t errcap) {
    if (!cfg || cfg->n_layers <= 0 || cfg->n_experts <= 0
        || !cfg->moe_inter || !cfg->dim) {
        if (err && errcap)
            snprintf(err, errcap,
                     "gcache: n_layers/n_experts/moe_inter/dim required");
        return NULL;
    }
    ApusGcacheCfg c = *cfg;
    ApusGcache *st = (ApusGcache *)calloc(1, sizeof *st);
    st->n_layers = c.n_layers;
    st->E = c.n_experts;
    st->inter = c.moe_inter;
    st->dim = c.dim;
    pthread_mutex_init(&st->mu, NULL);
    pthread_cond_init(&st->cv, NULL);
    pthread_cond_init(&st->jq_cv, NULL);

    /* member layout (pinned order: gate.w gate.s up.w up.s down.w down.s) */
    st->gw_b = st->inter * st->dim;
    st->gs_b = apus_fp8blk_nblocks(st->inter) * apus_fp8blk_nblocks(st->dim)
             * sizeof(float);
    st->dw_b = st->dim * st->inter;
    st->ds_b = apus_fp8blk_nblocks(st->dim) * apus_fp8blk_nblocks(st->inter)
             * sizeof(float);
    st->slab_bytes = 2 * (st->gw_b + st->gs_b) + st->dw_b + st->ds_b;
    st->payload_elems = 3 * st->inter * st->dim;
    st->payload_bytes = st->payload_elems * sizeof(uint16_t);

    /* manifest (format v2) */
    char path[1200];
    snprintf(path, sizeof path, "%s/apus.index.json", container_dir);
    char jerr[160];
    JVal *man = json_parse_file(path, jerr, sizeof jerr);
    if (!man) {
        if (err && errcap) snprintf(err, errcap, "gcache: %s", jerr);
        apus_gcache_close(st);
        return NULL;
    }
    JVal *fv = json_obj_get(man, "format_version");
    JVal *mt = json_obj_get(man, "model_type");
    JVal *slabs = json_obj_get(man, "expert_slabs");
    JVal *nmv = json_obj_get(man, "n_main_layers");
    if (!fv || (long)json_num(fv) != 2 || !mt
        || strcmp(json_str(mt) ? json_str(mt) : "", "glm5_next") || !slabs) {
        if (err && errcap)
            snprintf(err, errcap,
                     "gcache: apus.index.json not a glm5_next v2 container");
        json_free(man);
        apus_gcache_close(st);
        return NULL;
    }
    /* Slab records past n_main_layers are the MTP block's experts
     * (apus-mtp-* shard group): served at store layer n_main only when the
     * caller sized the store past the main layers (MTP block loaded, M8b);
     * skipped otherwise. */
    long n_main = nmv ? (long)json_num(nmv) : c.n_layers;
    if (n_main > c.n_layers) n_main = c.n_layers;

    int nocache = c.nocache > 0 ? 1
                : c.nocache < 0 ? 0
                : apus_env_int("APUS_NOCACHE", 1);
    st->recs = (ApusGslabRec *)calloc((size_t)c.n_layers * c.n_experts,
                                      sizeof *st->recs);
    int fail = 0;
    for (size_t i = 0; i < json_arr_len(slabs) && !fail; i++) {
        JVal *r = json_arr_get(slabs, i);
        const char *blk = json_str(json_obj_get(r, "block"));
        long ex = (long)json_num(json_obj_get(r, "expert"));
        const char *shard = json_str(json_obj_get(r, "shard"));
        uint64_t off = (uint64_t)json_num(json_obj_get(r, "offset"));
        uint64_t nb = (uint64_t)json_num(json_obj_get(r, "nbytes"));
        int layer = -1;
        if (!blk || sscanf(blk, "layers.%d", &layer) != 1 || !shard) {
            if (err && errcap)
                snprintf(err, errcap, "gcache: bad slab record (%s, %ld)",
                         blk ? blk : "?", ex);
            fail = 1;
            break;
        }
        if (layer >= n_main
            && !(layer == n_main && c.n_layers > n_main))
            continue;      /* MTP slab group, MTP not loaded (M8b) */
        if (layer < 0 || layer >= c.n_layers || ex < 0 || ex >= c.n_experts
            || nb != st->slab_bytes) {
            if (err && errcap)
                snprintf(err, errcap,
                         "gcache: bad slab record (%s, %ld, %llu)",
                         blk, ex, (unsigned long long)nb);
            fail = 1;
            break;
        }
        ApusGslabRec *rec = apus_gc_rec(st, layer, (int)ex);
        rec->lz = apus_gc_shard(st, container_dir, shard, nocache,
                                err, errcap);
        if (!rec->lz) { fail = 1; break; }
        rec->off = off;
        rec->len = nb;
    }
    json_free(man);
    if (fail) {
        apus_gcache_close(st);
        return NULL;
    }

    /* budgets */
    if (!c.cache_bytes)
        c.cache_bytes = apus_env_mb("APUS_GEXPERT_CACHE_MB", 6144) << 20;
    if (!c.rss_budget_bytes)
        c.rss_budget_bytes = apus_env_mb("APUS_RSS_GUARD_MB", 26624) << 20;
    if (!c.io_threads)
        c.io_threads = apus_env_int("APUS_IO_THREADS", 4);
    st->boost = apus_env_int("APUS_GCACHE_BOOST", 1);
    st->spec_cap = apus_env_int("APUS_GSPEC_QUEUE", 0);
    if (st->spec_cap <= 0) {
        st->spec_cap = 2 * (c.io_threads > 0 ? c.io_threads : 4);
        if (st->spec_cap < 4) st->spec_cap = 4;
    }
    st->rss_budget = c.rss_budget_bytes;

    /* LRU slots derive from the SPARSE-layer count (layers with slab
     * records): dividing by n_layers would hand the dense layers a share
     * of the payload budget they can never use (the P1 16 GiB run got
     * floor(16384/45/48) = 7 slots/layer instead of the intended 8 — a
     * sizeable hit-rate loss at this working-set size). */
    int n_sparse = 0;
    for (int l = 0; l < c.n_layers; l++) {
        for (int e = 0; e < c.n_experts; e++)
            if (apus_gc_rec(st, l, e)->len) { n_sparse++; break; }
    }
    if (n_sparse <= 0) n_sparse = c.n_layers;
    int spl = c.slots_per_layer;
    if (spl <= 0) {
        size_t per = c.cache_bytes
                   / ((size_t)n_sparse * st->payload_bytes);
        spl = per ? (int)per : 1;
    }
    if (spl > c.n_experts) spl = c.n_experts;

    st->lc = (ApusGlayerCache *)calloc((size_t)c.n_layers, sizeof *st->lc);
    for (int l = 0; l < c.n_layers; l++) {
        ApusGlayerCache *lc = &st->lc[l];
        lc->n_slots = spl;
        lc->slots = (ApusGslot *)calloc((size_t)spl, sizeof *lc->slots);
        for (int i = 0; i < spl; i++) lc->slots[i].eid = -1;
    }

    /* I/O pool */
    if (c.io_threads > 0) {
        st->n_threads = c.io_threads;
        st->threads = (pthread_t *)calloc((size_t)st->n_threads,
                                          sizeof *st->threads);
        for (int i = 0; i < st->n_threads; i++)
            pthread_create(&st->threads[i], NULL, apus_gc_worker, st);
    }
    /* buffer-recycling free lists (the V4 M6c lesson: per-fill mmap/munmap
     * + zero-fill soft faults are major kernel time at 25–48 MiB buffers) */
    st->pl_free_cap = apus_env_int("APUS_GBUF_FREE", 16);
    if (st->pl_free_cap < 0) st->pl_free_cap = 0;
    if (st->pl_free_cap > 1024) st->pl_free_cap = 1024;
    st->pl_free = (uint16_t **)calloc((size_t)(st->pl_free_cap
                                               ? st->pl_free_cap : 1),
                                      sizeof *st->pl_free);
    st->raw_free_cap = apus_env_int("APUS_GBUF_FREE_RAW", 8);
    if (st->raw_free_cap < 0) st->raw_free_cap = 0;
    if (st->raw_free_cap > 1024) st->raw_free_cap = 1024;
    st->raw_free = (uint8_t **)calloc((size_t)(st->raw_free_cap
                                               ? st->raw_free_cap : 1),
                                      sizeof *st->raw_free);
    return st;
}

void apus_gcache_close(ApusGcache *st) {
    if (!st) return;
    pthread_mutex_lock(&st->mu);
    st->stopping = 1;
    pthread_cond_broadcast(&st->jq_cv);
    pthread_mutex_unlock(&st->mu);
    for (int i = 0; i < st->n_threads; i++)
        pthread_join(st->threads[i], NULL);
    free(st->threads);
    free(st->jobs);
    if (st->lc) {
        for (int l = 0; l < st->n_layers; l++) {
            ApusGlayerCache *lc = &st->lc[l];
            /* slot/ws payload buffers all trace back to apus_gc_pl_get →
             * apus_aligned_alloc — they MUST go to apus_aligned_free
             * (Windows _aligned_malloc storage aborts the heap in plain
             * free()). */
            for (int i = 0; i < lc->n_slots; i++)
                apus_aligned_free(lc->slots[i].payload);
            for (int i = 0; i < lc->ws_n; i++) {
                apus_aligned_free(lc->ws[i]->payload);
                free(lc->ws[i]);
            }
            free(lc->slots);
            free(lc->ws);
        }
        free(st->lc);
    }
    for (int i = 0; i < st->shards_n; i++) {
        free(st->shards[i].name);
        apus_st_lazy_close(st->shards[i].lz);
    }
    free(st->shards);
    free(st->recs);
    for (int i = 0; i < st->pl_free_n; i++) apus_aligned_free(st->pl_free[i]);
    free(st->pl_free);
    for (int i = 0; i < st->raw_free_n; i++) apus_aligned_free(st->raw_free[i]);
    free(st->raw_free);
    pthread_mutex_destroy(&st->mu);
    pthread_cond_destroy(&st->cv);
    pthread_cond_destroy(&st->jq_cv);
    free(st);
}

/* --- stats / misc ------------------------------------------------------------------*/

void apus_gcache_stats(const ApusGcache *st, ApusGcacheStats *out) {
    pthread_mutex_lock(&((ApusGcache *)st)->mu);
    *out = st->stats;
    pthread_mutex_unlock(&((ApusGcache *)st)->mu);
}

size_t apus_gcache_slab_bytes(const ApusGcache *st) { return st->slab_bytes; }
size_t apus_gcache_payload_bytes(const ApusGcache *st) {
    return st->payload_bytes;
}

size_t apus_gcache_resident_bytes(ApusGcache *st) {
    pthread_mutex_lock(&st->mu);
    size_t n = 0;
    for (int l = 0; l < st->n_layers; l++) {
        ApusGlayerCache *lc = &st->lc[l];
        for (int i = 0; i < lc->n_slots; i++)
            if (lc->slots[i].payload) n += st->payload_bytes;
        for (int i = 0; i < lc->ws_n; i++)
            if (lc->ws[i]->payload) n += st->payload_bytes;
    }
    pthread_mutex_unlock(&st->mu);
    return n;
}

uint64_t apus_gcache_pread_count(ApusGcache *st) {
    uint64_t n = 0;
    for (int i = 0; i < st->shards_n; i++)
        n += apus_st_lazy_read_count(st->shards[i].lz);
    return n;
}

/* --- introspection / test hooks -----------------------------------------------------*/

int apus_gcache_debug_layer(ApusGcache *st, int layer,
                            int32_t *lru_eids, int n_lru) {
    if (!st || layer < 0 || layer >= st->n_layers) return -1;
    pthread_mutex_lock(&st->mu);
    ApusGlayerCache *lc = &st->lc[layer];
    if (lru_eids) {
        int n = n_lru < lc->n_slots ? n_lru : lc->n_slots;
        for (int i = 0; i < n; i++)
            lru_eids[i] = lc->slots[i].payload ? lc->slots[i].eid : -1;
    }
    pthread_mutex_unlock(&st->mu);
    return 0;
}

int apus_gcache_debug_present(ApusGcache *st, int layer, int eid) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E)
        return 0;
    pthread_mutex_lock(&st->mu);
    ApusGslot *slot;
    int kind;
    int present = apus_gc_lookup(st, layer, eid, &slot, &kind) == 0
                  && (slot->payload != NULL
                      || slot->state == APUS_GSLOT_LOADING);
    pthread_mutex_unlock(&st->mu);
    return present;
}

int apus_gcache_debug_ready(ApusGcache *st, int layer, int eid) {
    if (!st || layer < 0 || layer >= st->n_layers || eid < 0 || eid >= st->E)
        return 0;
    pthread_mutex_lock(&st->mu);
    ApusGslot *slot;
    int kind;
    int ready = apus_gc_lookup(st, layer, eid, &slot, &kind) == 0
                && slot->state == APUS_GSLOT_READY && slot->payload != NULL;
    pthread_mutex_unlock(&st->mu);
    return ready;
}

#endif /* APUS_GCACHE_IMPLEMENTATION */
#endif /* APUS_GCACHE_H */
