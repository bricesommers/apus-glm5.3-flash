/* tests/m8g/dump_h_real.c — real-weight h capture for the M8a hnorm pin:
 * prefill the teacher-forced Paris capture (tests/m8g/paris_tf_ids.csv,
 * plen=31 chat prompt + 251 natural-text tokens) on the converted
 * GLM-5.3-Flash container TIERED and dump the M8b h-out surface (h_out
 * [s, hc*dim] — the last-layer post-block mHC stream per position) to a
 * flat binary for tests/m8g/bin2cap.py. The C engine is bitwise == the
 * numpy oracle on the real container (the M7/P1 smoke: prefill all-45-
 * layers digest + logits digest match), so this capture is the same
 * ground truth tools/mtp_pin.py's numpy prefill produces — in minutes.
 *
 * Not part of any gate battery; built on demand (same line as
 * tests/smoke/trace_real.c, plus Accelerate for the bf16 dispatch):
 *   cc -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Ic \
 *      -o bin/dump_h_real tests/m8g/dump_h_real.c \
 *      -lm -framework Accelerate -lpthread
 *   APUS_GEXPERT_CACHE_MB=1024 bin/dump_h_real [ids_csv] [container] [out]
 *
 * Output format (tests/m8g/mtp_h_c.bin): a 32-byte header
 *   magic[8] "APM8HCAP", u32 version=1, u32 s, u32 hc, u32 dim,
 *   u32 reserved x3 (zero)
 * then s x i32 token ids, then s x hc*dim f32 (the bf16-valued stream
 * widened exactly, row-major [s][hc][dim]).
 * Sanity output on stdout: FNV-1a digests of the full prefill logits and
 * h_out, plus the top-10 last-position tokens.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* All implementation defines BEFORE any project include (single-TU). */
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
    const char *csv = argc > 1 ? argv[1] : "tests/m8g/paris_tf_ids.csv";
    const char *dir = argc > 2 ? argv[2] : "weights/glm-5.3-flash";
    const char *out = argc > 3 ? argv[3] : "tests/m8g/mtp_h_c.bin";

    /* read the CSV ids */
    FILE *f = fopen(csv, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", csv);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long csz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *cbuf = malloc((size_t)csz + 1);
    if (fread(cbuf, 1, (size_t)csz, f) != (size_t)csz) return 1;
    fclose(f);
    cbuf[csz] = 0;
    int32_t *ids = malloc(65536 * sizeof(int32_t));
    size_t s = 0;
    char *tok = strtok(cbuf, ",\n\r ");
    while (tok && s < 65536) {
        ids[s++] = (int32_t)strtol(tok, NULL, 10);
        tok = strtok(NULL, ",\n\r ");
    }
    free(cbuf);
    if (!s) {
        fprintf(stderr, "no ids in %s\n", csv);
        return 1;
    }
    fprintf(stderr, "dump_h_real: %zu ids from %s\n", s, csv);

    char err[512] = {0};
    char cfg_path[1024];
    snprintf(cfg_path, sizeof cfg_path, "%s/config.json", dir);
    /* tiered, cache budget from APUS_GEXPERT_CACHE_MB (0 = env default);
     * want_mtp = 0: the capture needs the main model only (the pre-M8b
     * cache geometry). */
    ApusGmodelTierCfg tier = { .tiered = 1, .cache_bytes = 0,
                               .slots_per_layer = 0, .io_threads = 0,
                               .nocache = 0, .rss_budget_bytes = 0,
                               .want_mtp = 0 };
    ApusGmodel *m = apus_gmodel_open2(dir, cfg_path, &tier, err,
                                      sizeof err);
    if (!m) {
        fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }
    const ApusGmodelConfig *c = apus_gmodel_config(m);
    const size_t dim = (size_t)c->hidden_size, hc = (size_t)c->hc_mult;
    const size_t V = (size_t)c->vocab_size;
    fprintf(stderr, "opened %s: L=%d dim=%zu vocab=%zu (tiered)\n", dir,
            c->num_hidden_layers, dim, V);

    ApusGmodelState *st = apus_gmodel_state_new(m, s + 8);
    uint16_t *logits = malloc(s * V * sizeof(uint16_t));
    uint16_t *h_out = malloc(s * hc * dim * sizeof(uint16_t));
    uint16_t *yn_out = malloc(s * dim * sizeof(uint16_t));
    if (!st || !logits || !h_out || !yn_out) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    /* one prefill call: h_out rows are complete for ALL s positions (the
     * token-chunked prefill MoE — APUS_GPREFILL_MOE_CHUNK — only chunks
     * the expert wiring; the h stream is computed per row regardless, and
     * the m6g tiered==eager digest gate pins the neutrality) */
    if (apus_gmodel_prefill_h(m, st, ids, s, logits, NULL, h_out, yn_out)) {
        fprintf(stderr, "prefill failed\n");
        return 1;
    }
    fprintf(stderr, "prefill done: pos=%zu\n", apus_gmodel_pos(st));

    /* dump: header + ids + h_out widened to f32 */
    FILE *o = fopen(out, "wb");
    if (!o) {
        fprintf(stderr, "cannot open %s\n", out);
        return 1;
    }
    unsigned char hdr[32];
    memset(hdr, 0, sizeof hdr);
    memcpy(hdr, "APM8HCAP", 8);
    uint32_t *hw = (uint32_t *)(hdr + 8);
    hw[0] = 1;                      /* version */
    hw[1] = (uint32_t)s;
    hw[2] = (uint32_t)hc;
    hw[3] = (uint32_t)dim;
    fwrite(hdr, 1, sizeof hdr, o);
    fwrite(ids, sizeof(int32_t), s, o);
    float *row = malloc(hc * dim * sizeof(float));
    for (size_t t = 0; t < s; t++) {
        const uint16_t *hr = h_out + t * hc * dim;
        for (size_t i = 0; i < hc * dim; i++)
            row[i] = apus_bf16_f32(hr[i]);
        fwrite(row, sizeof(float), hc * dim, o);
    }
    fclose(o);
    fprintf(stderr, "wrote %s (s=%zu hc=%zu dim=%zu)\n", out, s, hc, dim);

    /* sanity: digests + top-10 last-position tokens (trace_real style) */
    printf("logits_all %016llx\n",
           (unsigned long long)fnv1a64(logits, s * V * sizeof(uint16_t)));
    printf("h_all %016llx\n",
           (unsigned long long)fnv1a64(h_out,
                                       s * hc * dim * sizeof(uint16_t)));
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
    printf("top10");
    for (int i = 0; i < 10; i++) printf(" %d:%.4f", top[i], tv[i]);
    printf("\n");

    free(row);
    free(yn_out);
    free(h_out);
    free(logits);
    apus_gmodel_state_free(st);
    apus_gmodel_close(m);
    free(ids);
    return 0;
}
