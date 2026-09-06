/*
 * tests/m2/test_encoding.c — GLM-5.3-Flash chat-template conformance.
 *
 * golden/enc_extra_N.json (message/tools/opts spec) -> glm_encode_messages
 * must byte-match golden/enc_extra_N.txt (jinja2 rendering of
 * reference/chat_template.jinja) and its token ids golden/enc_extra_N.ids.
 * golden/enc_err_N.json: specs the template rejects — encode must fail.
 * Determinism: encode twice, byte-identical.
 * Run from the repository root.
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_TOK_IMPLEMENTATION
#define APUS_ENCODING_IMPLEMENTATION
#include "json.h"
#include "tok.h"
#include "encoding.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...) do { \
    checks++; \
    if (!(cond)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } \
} while (0)

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[sz] = 0;
    *len = (size_t)sz;
    return buf;
}

static uint32_t rd_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void check_ids(Tok *t, const JVal *messages, const JVal *tools,
                      const GlmEncOpts *opts,
                      const char *ids_path, const char *what) {
    size_t glen;
    unsigned char *g = (unsigned char *)read_file(ids_path, &glen);
    if (!g) { CHECK(0, "%s: missing %s", what, ids_path); return; }
    size_t n = 0;
    char *prompt = glm_encode_messages(messages, tools, opts);
    uint32_t *ids = prompt ? tok_encode_str(t, prompt, 1, &n) : NULL;
    CHECK(ids != NULL || n == 0, "%s: encode failed: %s", what, glm_last_error());
    uint32_t gn = glen >= 4 ? rd_u32(g) : 0;
    int ok = ids && glen >= 4 && (size_t)gn * 4 + 4 == glen && gn == n;
    if (ok)
        for (uint32_t i = 0; i < gn; i++)
            if (rd_u32(g + 4 + i * 4) != ids[i]) { ok = 0; break; }
    CHECK(ok, "%s: token ids mismatch (golden %u, got %zu)", what, gn, n);
    /* glm_encode_ids convenience wrapper must agree */
    {
        size_t n3 = 0;
        uint32_t *ids3 = glm_encode_ids(t, messages, tools, opts, &n3);
        CHECK(ids3 && n3 == n && (n == 0 || memcmp(ids, ids3, n * 4) == 0),
              "%s: glm_encode_ids != tok_encode(prompt)", what);
        free(ids3);
    }
    free(ids);
    free(prompt);
    free(g);
}

int main(void) {
    Tok *t = tok_load("reference/tokenizer.json");
    if (!t) { fprintf(stderr, "tok_load failed\n"); return 1; }

    /* ---- template conformance cases ---- */
    for (int i = 1; ; i++) {
        char spath[256], tpath[256], idspath[256], what[64];
        snprintf(spath, sizeof spath, "tests/m2/golden/enc_extra_%d.json", i);
        snprintf(tpath, sizeof tpath, "tests/m2/golden/enc_extra_%d.txt", i);
        snprintf(idspath, sizeof idspath, "tests/m2/golden/enc_extra_%d.ids", i);
        snprintf(what, sizeof what, "encoding case %d", i);

        char err[256];
        JVal *spec = json_parse_file(spath, err, sizeof err);
        if (!spec) break;
        JVal *messages = json_obj_get(spec, "messages");
        CHECK(messages && json_type(messages) == J_ARR, "case %d: bad spec", i);
        JVal *tools = json_obj_get(spec, "tools");
        GlmEncOpts opts = GLM_ENC_OPTS_DEFAULT;
        JVal *v;
        if ((v = json_obj_get(spec, "reasoning_effort")) && json_type(v) == J_STR)
            opts.reasoning_effort = json_str(v);
        if ((v = json_obj_get(spec, "clear_thinking")) && json_type(v) == J_BOOL)
            opts.clear_thinking = json_bool(v);
        if ((v = json_obj_get(spec, "add_generation_prompt")) && json_type(v) == J_BOOL)
            opts.add_generation_prompt = json_bool(v);

        size_t elen;
        char *expected = read_file(tpath, &elen);
        CHECK(expected != NULL, "%s: missing %s", what, tpath);

        char *p1 = glm_encode_messages(messages, tools, &opts);
        char *p2 = glm_encode_messages(messages, tools, &opts);
        CHECK(p1 != NULL, "%s: encode failed: %s", what, glm_last_error());
        CHECK(p1 && p2 && strcmp(p1, p2) == 0, "%s: non-deterministic encode", what);
        if (p1 && expected) {
            size_t plen = strlen(p1);
            if (!(plen == elen && memcmp(p1, expected, elen) == 0)) {
                size_t k = 0, m = plen < elen ? plen : elen;
                while (k < m && p1[k] == expected[k]) k++;
                CHECK(0, "%s: prompt mismatch (len %zu vs %zu) first diff at byte %zu: "
                      "got %.40s | want %.40s", what, plen, elen, k,
                      k < plen ? p1 + k : "", k < elen ? expected + k : "");
            } else {
                CHECK(1, "%s ok", what);
            }
        }
        check_ids(t, messages, tools, &opts, idspath, what);
        free(p1);
        free(p2);
        free(expected);
        json_free(spec);
    }

    /* ---- error cases: template raises, C encoder must fail too ---- */
    for (int i = 1; ; i++) {
        char spath[256], what[64];
        snprintf(spath, sizeof spath, "tests/m2/golden/enc_err_%d.json", i);
        snprintf(what, sizeof what, "error case %d", i);
        char err[256];
        JVal *spec = json_parse_file(spath, err, sizeof err);
        if (!spec) break;
        JVal *messages = json_obj_get(spec, "messages");
        JVal *tools = json_obj_get(spec, "tools");
        GlmEncOpts opts = GLM_ENC_OPTS_DEFAULT;
        char *p = glm_encode_messages(messages, tools, &opts);
        CHECK(p == NULL, "%s: encode should have failed but produced output", what);
        CHECK(p == NULL && glm_last_error()[0] != '\0',
              "%s: failure without error message", what);
        free(p);
        json_free(spec);
    }

    tok_free(t);
    printf("test_encoding: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
