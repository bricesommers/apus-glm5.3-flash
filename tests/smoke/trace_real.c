/* tests/smoke/trace_real.c — real-weight debug driver: prefill a fixed
 * prompt on the converted GLM-5.3-Flash container with trace_h capture,
 * print per-layer FNV-1a digests of the last-position block output and the
 * top-10 next-token logits. Compare against tools/ref_check.py output
 * (tests/smoke/ref/digests.json) to bisect C-vs-oracle divergence at real
 * scale. Not part of any gate battery; built on demand:
 *   make bin/trace_real && bin/trace_real [ids_csv] [container_dir]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* All implementation defines BEFORE any project include: the single-TU
 * headers include each other, so a late define would be a no-op. */
#define APUS_COMPAT_IMPLEMENTATION
#define APUS_JSON_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_BF16_IMPLEMENTATION
#define APUS_FP8BLK_IMPLEMENTATION
#define APUS_GMHC_IMPLEMENTATION
#define APUS_GMOE_IMPLEMENTATION
#define APUS_GKDA_IMPLEMENTATION
#define APUS_GDSA_IMPLEMENTATION
#define APUS_GCACHE_IMPLEMENTATION
#define APUS_GPILOT_IMPLEMENTATION
#define APUS_GMODEL_IMPLEMENTATION
#include "compat.h"
#include "json.h"
#include "st.h"
#include "bf16.h"
#include "fp8blk.h"
#include "gmhc.h"
#include "gmoe.h"
#include "gkda.h"
#include "gdsa.h"
#include "gcache.h"
#include "gpilot.h"
#include "gmodel.h"

static uint64_t fnv1a64(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char **argv) {
    const char *dir = argc > 2 ? argv[2] : "weights/glm-5.3-flash";
    const char *csv = argc > 1 ? argv[1] : "785,6722,315,9621,374";

    int32_t ids[4096];
    size_t s = 0;
    char *buf = strdup(csv), *tok = strtok(buf, ",");
    while (tok && s < 4096) {
        ids[s++] = (int32_t)strtol(tok, NULL, 10);
        tok = strtok(NULL, ",");
    }
    free(buf);
    if (!s) {
        fprintf(stderr, "no ids\n");
        return 1;
    }

    char err[512] = {0};
    char cfg_path[1024];
    snprintf(cfg_path, sizeof cfg_path, "%s/config.json", dir);
    ApusGmodelTierCfg tier = { .tiered = 1, .cache_bytes = 0,
                               .slots_per_layer = 0, .io_threads = -1,
                               .nocache = 0, .rss_budget_bytes = 0 };
    ApusGmodel *m = apus_gmodel_open2(dir, cfg_path, &tier, err,
                                      sizeof err);
    if (!m) {
        fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }
    const ApusGmodelConfig *c = apus_gmodel_config(m);
    const int L = c->num_hidden_layers;
    const size_t dim = (size_t)c->hidden_size, hc = (size_t)c->hc_mult;
    const size_t V = (size_t)c->vocab_size;
    fprintf(stderr, "opened %s: L=%d dim=%zu vocab=%zu\n", dir, L, dim, V);

    ApusGmodelState *st = apus_gmodel_state_new(m, s + 64);
    uint16_t *logits = malloc(s * V * sizeof(uint16_t));
    uint16_t *trace = malloc((size_t)L * s * hc * dim * sizeof(uint16_t));
    if (!st || !logits || !trace) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    if (apus_gmodel_prefill(m, st, ids, s, logits, NULL, trace)) {
        fprintf(stderr, "prefill failed\n");
        return 1;
    }

    /* per-layer digest of the LAST position's hc*dim stream */
    for (int l = 0; l < L; l++) {
        const uint16_t *row = trace + ((size_t)l * s + (s - 1)) * hc * dim;
        printf("layer%02d %016llx\n", l,
               (unsigned long long)fnv1a64(row,
                                           hc * dim * sizeof(uint16_t)));
    }
    /* top-10 logits at the last position (bf16 -> f32) */
    const uint16_t *lg = logits + (s - 1) * V;
    int32_t top[10];
    float tv[10];
    for (int i = 0; i < 10; i++) { top[i] = -1; tv[i] = -1e30f; }
    for (size_t v = 0; v < V; v++) {
        float x = apus_bf16_f32(lg[v]);
        if (x > tv[9]) {
            int j = 9;
            while (j > 0 && tv[j - 1] < x) { tv[j] = tv[j - 1];
                                             top[j] = top[j - 1]; j--; }
            tv[j] = x; top[j] = (int32_t)v;
        }
    }
    printf("logits_last %016llx\n",
           (unsigned long long)fnv1a64(lg, V * sizeof(uint16_t)));
    printf("top10");
    for (int i = 0; i < 10; i++) printf(" %d:%.4f", top[i], tv[i]);
    printf("\n");

    /* optional teacher-forced decode chain: argv[3] = comma-separated ids.
     * Per step: logits digest + per-layer last-row digests. */
    if (argc > 3) {
        char *buf2 = strdup(argv[3]), *t2 = strtok(buf2, ",");
        int step = 0;
        while (t2) {
            int32_t id = (int32_t)strtol(t2, NULL, 10);
            if (apus_gmodel_decode_step(m, st, id, logits, NULL, trace)) {
                fprintf(stderr, "decode step %d failed\n", step);
                return 1;
            }
            printf("step%d logits %016llx\n", step,
                   (unsigned long long)fnv1a64(logits,
                                               V * sizeof(uint16_t)));
            for (int l = 0; l < L; l++)
                printf("step%d layer%02d %016llx\n", step, l,
                       (unsigned long long)fnv1a64(
                           trace + (size_t)l * hc * dim,
                           hc * dim * sizeof(uint16_t)));
            step++;
            t2 = strtok(NULL, ",");
        }
        free(buf2);
    }
    return 0;
}
