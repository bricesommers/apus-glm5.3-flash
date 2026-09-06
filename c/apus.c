/*
 * c/apus.c — apus engine driver / CLI. GLM-5.3-Flash (glm5_next) only:
 *
 *   c/gmodel.h full forward (M5) + c/gcache.h/c/gpilot.h tiering (M6) +
 *   c/gmtp.h speculative decoding (M8b) + c/backend_gmetal.* (M7b).
 *   The model dir must hold an apus.index.json manifest with model_type
 *   "glm5_next" (the M1 v2 container); anything else is a hard error.
 *   (The retained DeepSeek-V4 engine was removed 2026-09-05 — the full
 *   V4 codebase remains in the sibling repo ../Apus.)
 *
 *   apus run --model DIR [--prompt "text" | --ids "1,2,3"]
 *            [--max-tokens N] [--seed S] [--temp T] [--top-p P] [--greedy]
 *
 * --prompt tokenizes via DIR/tokenizer.json and renders the GLM chat
 * template (c/encoding.h); --ids feeds raw token ids (used for synthetic
 * fixture models without a tokenizer). Sampling defaults: temp 1.0,
 * top_p 1.0; --greedy or --temp 0 selects argmax. The RNG is splitmix64
 * seeded by --seed (c/sample.h) — runs with the same arguments are
 * bit-identical. GLM generation stops on the eos set from
 * DIR/generation_config.json (eos_token_id), falling back to the
 * reference set GLM_EOS_IDS {154820, 154827, 154829} (c/encoding.h).
 *
 * All implementation TUs live here (single-binary build).
 */
#define APUS_JSON_IMPLEMENTATION
#define APUS_ST_IMPLEMENTATION
#define APUS_SAMPLE_IMPLEMENTATION
#define APUS_TOK_IMPLEMENTATION
#define APUS_ENCODING_IMPLEMENTATION
#define APUS_COMPAT_IMPLEMENTATION
/* The GLM-5.3-Flash engine stack (c/gmodel.h + the M3/M4/M6/M8 GLM
 * headers). */
#define APUS_BF16_IMPLEMENTATION
#define APUS_FP8BLK_IMPLEMENTATION
#define APUS_GMHC_IMPLEMENTATION
#define APUS_GMOE_IMPLEMENTATION
#define APUS_GKDA_IMPLEMENTATION
#define APUS_GDSA_IMPLEMENTATION
#define APUS_GCACHE_IMPLEMENTATION
#define APUS_GPILOT_IMPLEMENTATION
#define APUS_GMODEL_IMPLEMENTATION
#define APUS_GMTP_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef _WIN32
#include <io.h>     /* _setmode (M15) */
#include <fcntl.h>  /* _O_BINARY */
#endif

#include "sample.h"
#include "tok.h"
#include "encoding.h"
#include "gmodel.h"
#include "gpilot.h"
#include "gmtp.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void usage(FILE *f) {
    fprintf(f,
        "apus run --model DIR [--prompt \"text\" | --ids \"1,2,3\"]\n"
        "         [--max-tokens N] [--seed S] [--temp T] [--top-p P]\n"
        "         [--greedy] [--quiet] [--tiered] [--measure-locality FILE]\n"
        "         [--metal] [--spec [--spec-k K] | --no-spec]\n"
        "apus serve --model DIR [--tiered] [--metal]\n"
        "         [--spec [--spec-k K] | --no-spec]\n"
        "         NDJSON request/response protocol on stdin/stdout (M7a;\n"
        "         see tests/m7a/README.md). Driven by tools/server.py.\n"
        "  DIR must be a glm5_next container (apus.index.json with\n"
        "         model_type \"glm5_next\"); anything else is an error.\n"
        "  GLM model dirs need config.json (container config) plus\n"
        "         tokenizer.json (text/chat requests) and optionally\n"
        "         generation_config.json (eos stop set; default the\n"
        "         reference {154820, 154827, 154829}).\n"
        "  --tiered: experts-on-demand through the M6 cache:\n"
        "         c/gcache.h behind apus_gmodel_expert (budget\n"
        "         APUS_GEXPERT_CACHE_MB in 48 MiB payloads; the M6 pilot\n"
        "         attaches automatically, APUS_GPILOT=0 disables,\n"
        "         APUS_GPILOT_K/APUS_GPILOT_PREFILL_K tune it; P2: speculative\n"
        "         hints yield to demand and drop under backlog/pressure —\n"
        "         APUS_GSPEC_QUEUE, APUS_RSS_GUARD_MB on phys_footprint).\n"
        "         P4: prefill MoE is token-chunked (bounds the live expert\n"
        "         union, bitwise-neutral) — APUS_GPREFILL_MOE_CHUNK\n"
        "         (default 8, 0 = unchunked).\n"
        "  --metal: Metal GPU backend for the dense compute (M7b; also\n"
        "         APUS_METAL=1): BF16\n"
        "         GEMV/GEMM + the fused FP8-block-dequant GEMM\n"
        "         (c/backend_gmetal.mm, BITWISE == the pinned CPU kernels;\n"
        "         P4: model-owned dense weights get persistent zero-copy\n"
        "         wraps, expert payloads/activations stay ephemeral;\n"
        "         APUS_GMETAL_MIN_KB, default OFF — the bf16 GEMV offload\n"
        "         is a measured net loss at real scale on 32 GB; set\n"
        "         32768/0 to re-enable; APUS_GMETAL_FP8_MIN_KB, default 0,\n"
        "         gates the fused path; APUS_GMETAL_MAX_M, default 1,\n"
        "         keeps prefill GEMMs on the CPU). Fail-soft: unsupported\n"
        "         ops and missing GPU fall back to the CPU kernels; CPU is\n"
        "         the default.\n"
        "  --measure-locality FILE: dump per-token chosen/predicted\n"
        "         expert sets (NDJSON; --tiered only):\n"
        "         the P1 dump (actual routed sets via the M6 routed hook +\n"
        "         pilot top-24 predictions) for tools/measure_glocality.py\n"
        "  --spec: GLM MTP (NextN) speculative decoding (M8b, c/gmtp.h).\n"
        "         DEFAULT ON in run mode (M8c) AND serve mode (M8d),\n"
        "         user decision 2026-09-06: the re-pin sweep measured 85.3%%\n"
        "         draft accept, 1.47x vs\n"
        "         non-spec); --no-spec / APUS_SPEC=0 opts out, --spec-k K /\n"
        "         APUS_SPEC_K = drafts per step (default 3 = the sweep\n"
        "         winner). Requires the container's MTP block\n"
        "         (layers.<L>.*): explicit --spec fails loudly when absent;\n"
        "         the default-on form falls back to non-spec with a note\n"
        "         (same for --measure-locality, which yields to it only\n"
        "         when spec was not explicitly requested). Emitted tokens\n"
        "         are bitwise identical to non-spec decoding at the same\n"
        "         seed (the tests/m8g gate). In serve mode the NDJSON event\n"
        "         shapes are unchanged — a verify batch's accepted burst is\n"
        "         emitted as ordinary per-token events (M8d).\n");
}

/* P1 GLM measure-locality dump: per token, per sparse layer, the ACTUAL
 * routed expert set (the M6 routed hook — tiered mode only, where the
 * selections are pre-passed) and the pilot's predicted top-N set for layer
 * l+1 from the post-attention stream via apus_gpilot_predict (the same
 * pure-prediction entry the runtime pilot uses). The pilot's own hooks are
 * CHAINED (called first) so the prefetch path stays live; the dump itself
 * is fprintf on read-only hook data — numerics untouched (the m6g gate
 * already proves hooks-on == hooks-off digests). */
#define APUS_GMEASURE_N 24
typedef struct {
    FILE *f;
    ApusGpilot *gpilot;     /* NULL = A sets only, no P lines */
    ApusGmodelHooks chain;  /* the pilot's hooks (zeroed when no pilot) */
    size_t hcd;             /* hc_mult * dim (post-attention stream width) */
    int topk, pn;           /* router top-k; prediction count min(N, E) */
} GMeasureDump;

static void gmeasure_post_attn(void *ctx, int layer, const uint16_t *h,
                               size_t s, size_t pos0) {
    GMeasureDump *md = ctx;
    if (md->chain.post_attn)
        md->chain.post_attn(md->chain.ctx, layer, h, s, pos0);
    if (!md->gpilot) return;
    for (size_t t = 0; t < s; t++) {
        int32_t idx[APUS_GMEASURE_N];
        if (apus_gpilot_predict(md->gpilot, layer + 1, h + t * md->hcd,
                                idx, md->pn))
            continue;   /* dense target or past the last layer */
        fprintf(md->f, "{\"type\":\"P\",\"pos\":%lld,\"layer\":%d,\"eids\":[",
                (long long)(pos0 + t), layer + 1);
        for (int j = 0; j < md->pn; j++)
            fprintf(md->f, "%s%d", j ? "," : "", idx[j]);
        fprintf(md->f, "]}\n");
    }
}

static void gmeasure_routed(void *ctx, int layer, const int32_t *idx,
                            size_t s, size_t pos0) {
    GMeasureDump *md = ctx;
    if (md->chain.routed)
        md->chain.routed(md->chain.ctx, layer, idx, s, pos0);
    for (size_t t = 0; t < s; t++) {
        fprintf(md->f, "{\"type\":\"A\",\"pos\":%lld,\"layer\":%d,\"eids\":[",
                (long long)(pos0 + t), layer);
        for (int j = 0; j < md->topk; j++)
            fprintf(md->f, "%s%d", j ? "," : "", idx[t * (size_t)md->topk + j]);
        fprintf(md->f, "]}\n");
    }
}

/* parse "1,2,3" into a malloc'd int64 array */
static int64_t *parse_ids(const char *s, int *n_out) {
    int cap = 16, n = 0;
    int64_t *ids = malloc((size_t)cap * sizeof(int64_t));
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char *end;
        long long v = strtoll(p, &end, 10);
        if (end == p) { free(ids); return NULL; }
        p = end;
        if (n == cap) { cap *= 2; ids = realloc(ids, (size_t)cap * sizeof(int64_t)); }
        ids[n++] = v;
    }
    *n_out = n;
    return ids;
}

/* M7b: enable the GLM Metal backend when --metal / APUS_METAL=1
 * (c/backend_gmetal.mm, bitwise BF16/FP8-block GEMM offload). Fail-soft:
 * on any error the hooks stay NULL and the engine runs the CPU kernels. */
static void metal_maybe_enable(int want, int quiet) {
    if (!want) return;
    char merr[256];
    int glm_ok = !apus_gmetal_enable(merr, sizeof merr);
    if (!glm_ok)
        fprintf(stderr,
                "apus: GLM metal backend unavailable (%s) — CPU fallback\n",
                merr);
    if (!quiet && glm_ok)
        fprintf(stderr, "apus: metal backend enabled (dense offload)\n");
}

/* ---- engine context shared by `run` and `serve` (M7a) ------------------ */

/* GLM container check: a model dir must hold an apus.index.json manifest
 * with model_type "glm5_next" (the M1 v2 container); anything else is a
 * hard error at engine_init (the V4 engine path was removed 2026-09-05). */
static int model_is_glm(const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/apus.index.json", model_dir);
    char perr[128];
    JVal *man = json_parse_file(path, perr, sizeof perr);
    if (!man) return 0;
    JVal *mt = json_obj_get(man, "model_type");
    int glm = mt && json_type(mt) == J_STR
              && !strcmp(json_str(mt), "glm5_next");
    json_free(man);
    return glm;
}

typedef struct {
    int glm;                /* always 1 (kept for the serve dispatch shape) */
    int tiered;
    ApusGmodel *gm;
    ApusGpilot *gpilot; /* tiered only (the M6 pilot) */
    /* M8d: resolved --spec state (serve; run passes spec/spec_k as
     * run_glm args instead). mt is bound once at startup when spec is on
     * (a pure weight-view struct, no owned memory); the per-request
     * ApusGmtpState is allocated in the generate handler. */
    int spec, spec_k;
    ApusGmtp mt;
    uint32_t eos[8];    /* stop set: generation_config.json, else the M2
                           GLM_EOS_IDS default */
    int n_eos;
    /* sampling defaults (M7a fix): generation_config.json
     * temperature/top_p (the HF shape) when the container declares them,
     * else 1.0/1.0 — applied only when the user/request did not set the
     * parameter explicitly. */
    float def_temp, def_top_p;
} Engine;

/* GLM generation_config.json parse (HF shape): eos_token_id (int or
 * array) -> the stop set, else the reference set GLM_EOS_IDS {154820,
 * 154827, 154829} (c/encoding.h); temperature/top_p (numbers, optional)
 * -> the default sampling parameters. File absent: the M2 eos set +
 * 1.0/1.0 (the synthetic m7a/m8g fixtures have no such file — their
 * behavior is unchanged). */
static void glm_load_gen_config(Engine *e, const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/generation_config.json", model_dir);
    char perr[128];
    JVal *gc = json_parse_file(path, perr, sizeof perr);
    e->n_eos = 0;
    e->def_temp = 1.0f;
    e->def_top_p = 1.0f;
    if (gc) {
        JVal *v = json_obj_get(gc, "eos_token_id");
        if (v && json_type(v) == J_NUM) {
            e->eos[e->n_eos++] = (uint32_t)json_num(v);
        } else if (v && json_type(v) == J_ARR) {
            size_t n = json_arr_len(v);
            for (size_t i = 0; i < n && e->n_eos < 8; i++) {
                JVal *x = json_arr_get(v, i);
                if (x && json_type(x) == J_NUM)
                    e->eos[e->n_eos++] = (uint32_t)json_num(x);
            }
        }
        JVal *t = json_obj_get(gc, "temperature");
        if (t && json_type(t) == J_NUM)
            e->def_temp = (float)json_num(t);
        JVal *p = json_obj_get(gc, "top_p");
        if (p && json_type(p) == J_NUM)
            e->def_top_p = (float)json_num(p);
        json_free(gc);
    }
    if (e->n_eos == 0) {
        for (int i = 0; i < 3; i++) e->eos[e->n_eos++] = GLM_EOS_IDS[i];
    }
}

static int glm_is_eos(const Engine *e, int t) {
    for (int i = 0; i < e->n_eos; i++)
        if ((uint32_t)t == e->eos[i]) return 1;
    return 0;
}

/* Load model + (tiered) expert cache + (optional) pilot. pilot_wanted:
 * create the pilot; pilot_enabled: its prefetch is active (measure mode
 * creates a disabled pilot for apus_gpilot_predict only). want_mtp
 * (M8b): load the MTP block + serve its slabs through the cache
 * (--spec). */
static int engine_init(Engine *e, const char *model_dir, int tiered,
                       int pilot_wanted, int pilot_enabled, int want_mtp,
                       char *err, size_t errcap) {
    memset(e, 0, sizeof *e);
    e->tiered = tiered;
    e->glm = model_is_glm(model_dir);
    if (!e->glm) {
        snprintf(err, errcap, "%s is not a glm5_next container "
                 "(apus.index.json model_type)", model_dir);
        return -1;
    }
    {
        char cfgpath[1024];
        snprintf(cfgpath, sizeof cfgpath, "%s/config.json", model_dir);
        ApusGmodelTierCfg tc;
        memset(&tc, 0, sizeof tc);      /* env defaults (APUS_GEXPERT_CACHE_MB
                                           etc.) fill the rest */
        tc.tiered = tiered;
        tc.want_mtp = want_mtp;
        e->gm = apus_gmodel_open2(model_dir, cfgpath, &tc, err, errcap);
        if (!e->gm) return -1;
        glm_load_gen_config(e, model_dir);
        /* M6 pilot: router-lookahead prefetch behind the model hooks.
         * Attach only when tiered (it needs the cache as a hint target);
         * APUS_GPILOT=0 disables (default on). Measure mode
         * (pilot_wanted && !pilot_enabled) forces a predict-only pilot even
         * under APUS_GPILOT=0 — apus_gpilot_predict needs the router views. */
        if (pilot_wanted && tiered
            && (apus_env_int("APUS_GPILOT", 1) || !pilot_enabled)) {
            const ApusGmodelConfig *c = apus_gmodel_config(e->gm);
            ApusGpilotCfg pc;
            memset(&pc, 0, sizeof pc);  /* pilot_k/prefill_k: env defaults */
            pc.cache = apus_gmodel_cache(e->gm);
            pc.n_layers = c->num_hidden_layers;
            pc.n_experts = c->n_routed_experts;
            pc.topk = c->num_experts_per_tok;
            pc.dim = (size_t)c->hidden_size;
            pc.hc_mult = c->hc_mult;
            pc.sinkhorn_iters = c->hc_sinkhorn_iters;
            pc.norm_eps = c->rms_norm_eps;
            pc.hc_eps = c->hc_eps;
            pc.route_scale = c->routed_scaling_factor;
            pc.enabled = pilot_enabled;
            e->gpilot = apus_gpilot_create(&pc);
            if (e->gpilot) apus_gpilot_attach(e->gpilot, e->gm);
        }
        return 0;
    }
}

static void engine_destroy(Engine *e) {
    if (e->gpilot) apus_gpilot_destroy(e->gpilot);
    if (e->gm) apus_gmodel_close(e->gm);    /* closes the tiered cache */
}

/* ================= M7a serve mode =================
 *
 * NDJSON protocol on stdin/stdout: one JSON object per line in each
 * direction (justification: the gateway tools/server.py spawns this
 * process and owns all networking; stdio keeps the engine libc-only — no
 * sockets in C — and matches the colibri gateway-drives-engine split).
 * The process stays alive across requests; the model loads once. Every
 * request gets a FRESH ApusGmodelState (KV is per-request; conversation
 * state is the gateway's job — multi-turn context is re-prefilled; KV
 * reuse across turns is a later optimization).
 *
 * Request (client -> engine):
 *   {"id": <any>, "cmd": "encode",
 *    "messages": [...], "tools": [...]|null,
 *    "clear_thinking": true|false, "reasoning_effort": "low"|"high"|null}
 *   {"id": <any>, "cmd": "generate",
 *    "messages": [...] | "text": "raw prompt" | "ids": [1,2,3],
 *    "tools": [...]|null, "clear_thinking": bool, "reasoning_effort": str|null,
 *    "max_tokens": int, "temperature": float, "top_p": float,
 *    "seed": uint, "stop": [str, ...]}
 * "tools" is a top-level template parameter (a sibling of "messages", the
 * GLM template shape — never attached to a message). The GLM template has
 * no thinking on/off switch (generation always starts <think>); everything
 * but reasoning_effort "low"/"high" maps to "Max" (c/encoding.h).
 * "text" is tokenized verbatim (no chat template, no BOS).
 *
 * Events (engine -> client):
 *   {"id","type":"encoded","text","ids"}      (encode; ids need tokenizer)
 *   {"id","type":"prompt","prompt_tokens"}    (generate, first)
 *   {"id","type":"token","token_id","text"}   (per generated token; EOS is
 *                                             never emitted; text needs a
 *                                             tokenizer; M8d: under --spec
 *                                             a verify batch's accepted
 *                                             burst arrives as several of
 *                                             these in a row — same shape)
 *   {"id","type":"done","finish_reason","prompt_tokens",
 *    "completion_tokens","text"}              finish_reason: "stop" (EOS)
 *                                             | "length" | "stop_string"
 *   {"id","type":"error","code","message"}    request failed; loop lives.
 *                                             code "bad_request" (schema/
 *                                             encoding; gateway maps to
 *                                             HTTP 400) | "engine_error"
 * Stop strings are checked against the assembled decoded text; a match
 * truncates the text at the match start (a partial last token piece is
 * emitted if it precedes the match) and finishes "stop_string".
 */

static char *serve_read_line(void) {
    size_t cap = 1 << 16, n = 0;
    char *buf = malloc(cap);
    int c;
    while ((c = fgetc(stdin)) != EOF) {
        if (c == '\n') break;
        if (n + 1 >= cap) { cap *= 2; buf = realloc(buf, cap); }
        buf[n++] = (char)c;
    }
    if (c == EOF && n == 0) { free(buf); return NULL; }
    buf[n] = 0;
    return buf;
}

static JVal *serve_resp(const JVal *id, const char *type) {
    JVal *o = json_new_obj();
    json_obj_set(o, "id", id ? json_clone(id) : json_new_null());
    json_obj_set(o, "type", json_new_str(type));
    return o;
}

static void serve_send(JVal *resp) {
    char *s = json_dumps(resp);
    fputs(s, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    free(s);
    json_free(resp);
}

static void serve_error_code(const JVal *id, const char *code,
                             const char *msg) {
    JVal *o = serve_resp(id, "error");
    json_obj_set(o, "code", json_new_str(code));
    json_obj_set(o, "message", json_new_str(msg ? msg : "error"));
    serve_send(o);
}

/* Request/schema/encoding failures: the gateway maps "bad_request" to
 * HTTP 400 (the M2 handoff: glm_encode_* failures surface as 400s).
 * Engine-internal failures use "engine_error" (-> 500/503). */
static void serve_error(const JVal *id, const char *msg) {
    serve_error_code(id, "bad_request", msg);
}

static void serve_token(const JVal *id, int token_id,
                        const char *text, size_t len) {
    JVal *o = serve_resp(id, "token");
    json_obj_set(o, "token_id", json_new_int(token_id));
    if (text) json_obj_set(o, "text", json_new_strn(text, len));
    serve_send(o);
}

/* GLM renders request tools as a top-level template parameter (c/encoding.h),
 * not attached to any message. */
static void serve_enc_opts(const JVal *req, GlmEncOpts *opts) {
    *opts = (GlmEncOpts)GLM_ENC_OPTS_DEFAULT;
    JVal *re = json_obj_get((JVal *)req, "reasoning_effort");
    if (re && json_type(re) == J_STR) opts->reasoning_effort = json_str(re);
    JVal *ct = json_obj_get((JVal *)req, "clear_thinking");
    if (ct && json_type(ct) == J_BOOL) opts->clear_thinking = json_bool(ct);
    /* note: the GLM template has no thinking on/off switch (the DeepSeek
     * "thinking" request field is gone); generation always starts <think> */
}

/* Encode request messages -> rendered prompt string (NULL on error,
 * message reported by the caller via glm_last_error()). */
static char *serve_render(const JVal *req, const JVal *id) {
    JVal *jmsgs = json_obj_get((JVal *)req, "messages");
    if (!jmsgs || json_type(jmsgs) != J_ARR) {
        serve_error(id, "messages must be a JSON array");
        return NULL;
    }
    GlmEncOpts opts;
    serve_enc_opts(req, &opts);
    char *prompt = glm_encode_messages(jmsgs, json_obj_get((JVal *)req, "tools"),
                                       &opts);
    if (!prompt) serve_error(id, glm_last_error());
    return prompt;
}

static void serve_cmd_encode(Tok *tok, const JVal *req, const JVal *id) {
    char *prompt = serve_render(req, id);
    if (!prompt) return;
    JVal *o = serve_resp(id, "encoded");
    json_obj_set(o, "text", json_new_str(prompt));
    if (tok) {
        size_t n = 0;
        uint32_t *ids = tok_encode_str(tok, prompt, 1, &n);
        JVal *arr = json_new_arr();
        for (size_t i = 0; i < n; i++)
            json_arr_push(arr, json_new_int((long long)ids[i]));
        json_obj_set(o, "ids", arr);
        free(ids);
    }
    free(prompt);
    serve_send(o);
}

/* ---- shared generate-request parsing (both engines) ---- */

/* Resolve prompt ids from the request: ids | text | messages (the GLM
 * chat template via serve_render). NULL on error (already reported). */
static int64_t *serve_prompt_ids(Tok *tok, const JVal *req, const JVal *id,
                                 size_t *n_out) {
    int64_t *ids = NULL;
    size_t n_ids = 0;
    JVal *jids = json_obj_get((JVal *)req, "ids");
    JVal *jtext = json_obj_get((JVal *)req, "text");
    JVal *jmsgs = json_obj_get((JVal *)req, "messages");
    if (jids && json_type(jids) == J_ARR) {
        n_ids = json_arr_len(jids);
        ids = malloc((n_ids ? n_ids : 1) * sizeof(int64_t));
        for (size_t i = 0; i < n_ids; i++) {
            JVal *v = json_arr_get(jids, i);
            if (!v || json_type(v) != J_NUM) {
                serve_error(id, "ids must be an array of numbers");
                free(ids);
                return NULL;
            }
            ids[i] = (int64_t)json_num(v);
        }
    } else if (jtext && json_type(jtext) == J_STR) {
        if (!tok) { serve_error(id, "no tokenizer in model dir"); return NULL; }
        uint32_t *u = tok_encode_str(tok, json_str(jtext), 1, &n_ids);
        ids = malloc((n_ids ? n_ids : 1) * sizeof(int64_t));
        for (size_t i = 0; i < n_ids; i++) ids[i] = u[i];
        free(u);
    } else if (jmsgs) {
        if (!tok) { serve_error(id, "no tokenizer in model dir"); return NULL; }
        char *prompt = serve_render(req, id);
        if (!prompt) return NULL;
        uint32_t *u = tok_encode_str(tok, prompt, 1, &n_ids);
        free(prompt);
        ids = malloc((n_ids ? n_ids : 1) * sizeof(int64_t));
        for (size_t i = 0; i < n_ids; i++) ids[i] = u[i];
        free(u);
    } else {
        serve_error(id, "generate needs messages, text, or ids");
        return NULL;
    }
    if (n_ids == 0) { serve_error(id, "empty prompt"); free(ids); return NULL; }
    *n_out = n_ids;
    return ids;
}

typedef struct {
    int max_tokens;
    int max_tokens_set;     /* request explicit; 0 = OpenAI semantics
                               (until EOS / the KV budget, see below) */
    double temperature, top_p;
    uint64_t seed;
    const char *stops[16];
    int n_stops;
} ServeSamp;

static void serve_sampling(const Engine *e, const JVal *req, ServeSamp *sp) {
    sp->max_tokens = 0;
    sp->max_tokens_set = 0;
    /* defaults: the container's generation_config.json (M7a fix) */
    sp->temperature = e->def_temp;
    sp->top_p = e->def_top_p;
    sp->seed = 0;
    sp->n_stops = 0;
    JVal *v;
    if ((v = json_obj_get((JVal *)req, "max_tokens")) && json_type(v) == J_NUM) {
        sp->max_tokens = (int)json_num(v);
        sp->max_tokens_set = 1;
    }
    if (sp->max_tokens < 0) sp->max_tokens = 0;
    if ((v = json_obj_get((JVal *)req, "temperature")) && json_type(v) == J_NUM)
        sp->temperature = json_num(v);
    if ((v = json_obj_get((JVal *)req, "top_p")) && json_type(v) == J_NUM)
        sp->top_p = json_num(v);
    if ((v = json_obj_get((JVal *)req, "seed")) && json_type(v) == J_NUM)
        sp->seed = (uint64_t)json_num(v);
    JVal *jstop = json_obj_get((JVal *)req, "stop");
    if (jstop && json_type(jstop) == J_ARR) {
        size_t ns = json_arr_len(jstop);
        for (size_t i = 0; i < ns && sp->n_stops < 16; i++) {
            JVal *s = json_arr_get(jstop, i);
            if (s && json_type(s) == J_STR && json_strlen(s))
                sp->stops[sp->n_stops++] = json_str(s);
        }
    }
}

/* Earliest stop-string match over the assembled text, or -1. */
static long stop_match(SBuf *text, const ServeSamp *sp) {
    if (!sp->n_stops) return -1;
    sb_reserve(text, 0);
    text->p[text->n] = '\0';
    long mpos = -1;
    for (int k = 0; k < sp->n_stops; k++) {
        char *hit = strstr(text->p, sp->stops[k]);
        if (hit && (mpos < 0 || hit - text->p < mpos))
            mpos = hit - text->p;
    }
    return mpos;
}

/* ================= GLM-5.3-Flash generate (M7a) =================
 * The forward is the M5/M6
 * gated engine: prefill (CHUNKED KDA) once, then the recurrent decode
 * chain; logits come out BF16 and are widened exactly (apus_bf16_f32)
 * for the shared c/sample.h sampler. KV is sized exactly to the request
 * (prompt + max_tokens; the engine never evicts — M6 policy).
 * M8d: with spec on (the default), the MTP draft/verify loop (c/gmtp.h
 * ApusGspec) replaces the decode chain — same emitted stream bitwise
 * (every emitted token is the main model's own apus_sample draw, one RNG
 * uniform per token in position order, drafts consume none), same NDJSON
 * events (one token event per emitted token; a verify batch's accepted
 * burst simply yields several in a row). */

/* One emitted token (shared by the spec and non-spec loops): EOS check
 * (never emitted), detok piece, stop-string match/truncation, the token
 * event. Returns NULL to keep going, else the finish reason ("stop" |
 * "stop_string"); *completion counts emitted (non-EOS) tokens. */
static const char *serve_emit_token(Engine *e, Tok *tok, const JVal *id,
                                    const ServeSamp *sp, SBuf *text,
                                    int *completion, int t) {
    if (glm_is_eos(e, t)) return "stop";
    (*completion)++;
    size_t plen = 0;
    char *piece = tok
        ? tok_decode(tok, (const uint32_t[]){ (uint32_t)t }, 1, &plen)
        : NULL;
    size_t oldn = text->n;
    if (piece) sb_write(text, piece, plen);
    long mpos = stop_match(text, sp);
    if (mpos >= 0) {
        if ((size_t)mpos > oldn)
            serve_token(id, t, text->p + oldn, (size_t)mpos - oldn);
        text->n = (size_t)mpos;
        free(piece);
        return "stop_string";
    }
    serve_token(id, t, piece, plen);
    free(piece);
    return NULL;
}

static void serve_cmd_generate_glm(Engine *e, Tok *tok,
                                   const JVal *req, const JVal *id) {
    size_t n_ids = 0;
    int64_t *ids = serve_prompt_ids(tok, req, id, &n_ids);
    if (!ids) return;
    ServeSamp sp;
    serve_sampling(e, req, &sp);
    /* OpenAI semantics (the base engine's serve default, max_seq 65536):
     * max_tokens omitted = generate until EOS. The cap only SIZES the
     * per-request KV (the M6 no-eviction policy needs a bound; ~12.7 KB/
     * token across the 11 DSA layers → ~0.8 GiB at 65536). */
    if (!sp.max_tokens_set)
        sp.max_tokens = apus_env_int("APUS_GSERVE_MAX_TOKENS", 65536);

    const ApusGmodelConfig *cfg = apus_gmodel_config(e->gm);
    int V = cfg->vocab_size;
    const int spec = e->spec, spec_k = e->spec_k;
    /* M8d (mirrors run_glm): --spec needs room for the verify batch
     * running ahead of the emitted count (up to spec_k positions) */
    size_t kv_cap = n_ids + (size_t)sp.max_tokens
                  + (spec ? (size_t)spec_k + 1 : 1);

    JVal *o = serve_resp(id, "prompt");
    json_obj_set(o, "prompt_tokens", json_new_int((long long)n_ids));
    serve_send(o);

    ApusGmodelState *st = apus_gmodel_state_new(e->gm, kv_cap);
    uint16_t *logits16 = malloc(n_ids * (size_t)V * sizeof(uint16_t));
    float *logits = malloc((size_t)V * sizeof(float));
    void *scratch = malloc(apus_sample_scratch_size((size_t)V));
    int32_t *ids32 = malloc(n_ids * sizeof(int32_t));
    for (size_t i = 0; i < n_ids; i++) ids32[i] = (int32_t)ids[i];
    ApusRng rng;
    apus_rng_seed(&rng, sp.seed);

    SBuf text;
    sb_init(&text);
    int completion = 0;
    const char *finish = "length";

    if (spec) {
        /* M8d: MTP draft/verify loop (c/gmtp.h ApusGspec), mirroring
         * run_glm's spec branch exactly (same call ordering, same kv_cap
         * headroom) so serve-spec == serve-non-spec at the same seed. */
        ApusGmtpState *mst = apus_gmtp_state_new(&e->mt, kv_cap
                                                 + (size_t)spec_k);
        if (!mst) {
            serve_error_code(id, "engine_error",
                             "MTP state allocation failed");
            goto out;
        }
        ApusGspec gsp;
        apus_gspec_init(&gsp, e->gm, st, &e->mt, mst, spec_k,
                        (float)sp.temperature, (float)sp.top_p,
                        &rng, scratch);
        if (apus_gspec_prefill(&gsp, ids32, n_ids)) {
            serve_error_code(id, "engine_error", "prefill failed");
            apus_gspec_free(&gsp);
            apus_gmtp_state_free(mst);
            goto out;
        }
        int step_out[64], done = 0;
        while (completion < sp.max_tokens && !done) {
            int ne = apus_gspec_step(&gsp, step_out, 64);
            if (ne <= 0) break;
            for (int i = 0; i < ne && completion < sp.max_tokens; i++) {
                const char *fin = serve_emit_token(e, tok, id, &sp, &text,
                                                   &completion, step_out[i]);
                if (fin) { finish = fin; done = 1; break; }
            }
        }
        /* per-request spec stats on stderr (the run_glm line's serve
         * twin — acceptance rates are THE spec health metric) */
        fprintf(stderr,
                "apus serve: spec: %llu emitted in %llu batches "
                "(%.2f tok/batch, %llu re-fed), draft accept %llu/%llu"
                " (%.1f%%), d1 %llu/%llu\n",
                (unsigned long long)gsp.emitted,
                (unsigned long long)gsp.batches,
                gsp.batches ? (double)gsp.emitted / (double)gsp.batches
                            : 0.0,
                (unsigned long long)gsp.refeed_tokens,
                (unsigned long long)gsp.accepted,
                (unsigned long long)gsp.offered,
                gsp.offered ? 100.0 * (double)gsp.accepted
                                   / (double)gsp.offered : 0.0,
                (unsigned long long)gsp.d1_hits,
                (unsigned long long)gsp.d1_offered);
        apus_gspec_free(&gsp);
        apus_gmtp_state_free(mst);
    } else {
        if (apus_gmodel_prefill(e->gm, st, ids32, n_ids, logits16,
                                NULL, NULL)) {
            serve_error_code(id, "engine_error", "prefill failed");
            goto out;
        }
        {
            const uint16_t *last = logits16 + (n_ids - 1) * (size_t)V;
            for (int i = 0; i < V; i++) logits[i] = apus_bf16_f32(last[i]);
        }
        for (int step = 0; step < sp.max_tokens; step++) {
            int t = apus_sample(logits, (size_t)V, (float)sp.temperature,
                                (float)sp.top_p, &rng, scratch);
            const char *fin = serve_emit_token(e, tok, id, &sp, &text,
                                               &completion, t);
            if (fin) { finish = fin; break; }
            if (apus_gmodel_decode_step(e->gm, st, t, logits16, NULL, NULL)) {
                serve_error_code(id, "engine_error", "decode failed");
                goto out;
            }
            for (int i = 0; i < V; i++) logits[i] = apus_bf16_f32(logits16[i]);
        }
    }

    {
        JVal *d = serve_resp(id, "done");
        json_obj_set(d, "finish_reason", json_new_str(finish));
        json_obj_set(d, "prompt_tokens", json_new_int((long long)n_ids));
        json_obj_set(d, "completion_tokens", json_new_int(completion));
        json_obj_set(d, "text", json_new_strn(text.p ? text.p : "", text.n));
        serve_send(d);
    }

out:
    sb_free(&text);
    free(ids32);
    free(logits16);
    free(logits);
    free(scratch);
    free(ids);
    apus_gmodel_state_free(st);
}

static void serve_cmd_generate(Engine *e, Tok *tok,
                               const JVal *req, const JVal *id) {
    serve_cmd_generate_glm(e, tok, req, id);
}

static int serve_main(int argc, char **argv) {
    const char *model_dir = NULL;
    int tiered = apus_env_int("APUS_TIERED", 0);
    int metal = apus_env_int("APUS_METAL", 0);
    int spec = apus_env_int("APUS_SPEC", 1);        /* M8d: default ON, same
                                                       as run mode (M8c) */
    int spec_explicit = getenv("APUS_SPEC") != NULL;
    int spec_k = apus_env_int("APUS_SPEC_K", 3);    /* the re-pin sweep's
                                                       winner (85.3% accept) */
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--tiered")) tiered = 1;
        else if (!strcmp(argv[i], "--metal")) metal = 1;
        else if (!strcmp(argv[i], "--spec")) { spec = 1; spec_explicit = 1; }
        else if (!strcmp(argv[i], "--no-spec")) { spec = 0; spec_explicit = 1; }
        else if (!strcmp(argv[i], "--spec-k") && i + 1 < argc) spec_k = atoi(argv[++i]);
        else { usage(stderr); return 2; }
    }
    if (!model_dir) { usage(stderr); return 2; }
    if (spec_k < 1) spec_k = 1;
    if (spec_k > 63) spec_k = 63;
    metal_maybe_enable(metal, 0);

    char err[256];
    Engine e;
    double t0 = now_s();
    int pilot_on = tiered && apus_env_int("APUS_PILOT", 1);
    if (engine_init(&e, model_dir, tiered, pilot_on, pilot_on, spec,
                    err, sizeof err)) {
        fprintf(stderr, "apus serve: %s\n", err);
        return 1;
    }
    /* M8d (the run_main M8c resolution, verbatim): an EXPLICIT --spec on
     * an MTP-less container is a loud startup error; the default-on form
     * falls back to non-spec with a note. */
    if (spec && !apus_gmodel_has_mtp(e.gm)) {
        if (spec_explicit) {
            fprintf(stderr, "apus serve: --spec requires the MTP (NextN) "
                    "block (layers.%d.*) — not present in this container\n",
                    apus_gmodel_config(e.gm)->num_hidden_layers);
            engine_destroy(&e);
            return 1;
        }
        spec = 0;
        fprintf(stderr, "apus serve: no MTP block in this container — "
                "speculative decoding off\n");
    }
    e.spec = spec;
    e.spec_k = spec_k;
    if (spec) {
        char serr[256];
        if (apus_gmtp_bind(&e.mt, e.gm, serr, sizeof serr)) {
            fprintf(stderr, "apus serve: %s\n", serr);
            engine_destroy(&e);
            return 1;
        }
    }
    fprintf(stderr,
            "apus serve: loaded %s (glm5_next, %d layers, dim %d, vocab %d%s%s) in %.2fs\n",
            model_dir, apus_gmodel_config(e.gm)->num_hidden_layers,
            apus_gmodel_config(e.gm)->hidden_size,
            apus_gmodel_config(e.gm)->vocab_size,
            tiered ? ", tiered" : "",
            spec ? ", spec" : "", now_s() - t0);

    Tok *tok = NULL;
    char tpath[1024];
    snprintf(tpath, sizeof tpath, "%s/tokenizer.json", model_dir);
    if (access(tpath, R_OK) == 0) {
        tok = tok_load(tpath);
        if (!tok) {
            fprintf(stderr, "apus serve: cannot load %s\n", tpath);
            engine_destroy(&e);
            return 1;
        }
    } else {
        fprintf(stderr, "apus serve: no %s (ids-only requests)\n", tpath);
    }

    char *line;
    while ((line = serve_read_line())) {
        char perr[128];
        JVal *req = json_parse(line, strlen(line), perr, sizeof perr);
        free(line);
        if (!req || json_type(req) != J_OBJ) {
            serve_error(NULL, "bad request line (not a JSON object)");
            json_free(req);
            continue;
        }
        JVal *id = json_obj_get(req, "id");
        JVal *cmdv = json_obj_get(req, "cmd");
        const char *cmd = json_type(cmdv) == J_STR ? json_str(cmdv) : NULL;
        if (cmd && !strcmp(cmd, "encode")) serve_cmd_encode(tok, req, id);
        else if (cmd && !strcmp(cmd, "generate"))
            serve_cmd_generate(&e, tok, req, id);
        else serve_error(id, "unknown cmd (want encode|generate)");
        json_free(req);
    }

    if (tok) tok_free(tok);
    engine_destroy(&e);
    return 0;
}

/* ================= run mode (CLI) ================= */

/* Shared CLI input resolution: --ids verbatim, else --prompt through the
 * GLM chat template (a single user message) + tokenizer. */
static int64_t *run_input_ids(const char *model_dir, const char *prompt,
                              const char *ids_str, Tok **tok_out, int *n_out) {
    *tok_out = NULL;
    if (ids_str) {
        int n = 0;
        int64_t *ids = parse_ids(ids_str, &n);
        if (!ids || n == 0) { fprintf(stderr, "apus: bad --ids\n"); exit(2); }
        *n_out = n;
        return ids;
    }
    char tpath[1024];
    snprintf(tpath, sizeof tpath, "%s/tokenizer.json", model_dir);
    Tok *tok = tok_load(tpath);
    if (!tok) { fprintf(stderr, "apus: cannot load %s\n", tpath); exit(1); }
    JVal *msgs = json_new_arr();
    JVal *msg = json_new_obj();
    json_obj_set(msg, "role", json_new_str("user"));
    json_obj_set(msg, "content", json_new_str(prompt));
    json_arr_push(msgs, msg);
    GlmEncOpts opts = GLM_ENC_OPTS_DEFAULT;
    size_t n = 0;
    uint32_t *u32 = glm_encode_ids(tok, msgs, NULL, &opts, &n);
    json_free(msgs);
    if (!u32) {
        fprintf(stderr, "apus: encoding failed: %s\n", glm_last_error());
        exit(1);
    }
    int64_t *ids = malloc(n * sizeof(int64_t));
    for (size_t i = 0; i < n; i++) ids[i] = u32[i];
    free(u32);
    *tok_out = tok;
    *n_out = (int)n;
    return ids;
}

/* GLM-5.3-Flash run path (M7a): prefill (chunked KDA) + recurrent decode
 * chain on the M5/M6 engine; BF16 logits widened exactly for the shared
 * sampler. --tiered streams experts through the M6 cache (budget
 * APUS_GEXPERT_CACHE_MB) with the pilot attached. M8b: --spec runs the
 * MTP draft/verify loop (c/gmtp.h) — emitted tokens bitwise identical to
 * the non-speculative loop for the same seed. */
static int run_glm(Engine *e, const char *model_dir, const char *prompt,
                   const char *ids_str, int max_tokens, uint64_t seed,
                   float temp, float top_p, int spec, int spec_k,
                   int quiet, double t0, const char *measure_path) {
    const ApusGmodelConfig *cfg = apus_gmodel_config(e->gm);
    if (measure_path && !e->tiered) {
        fprintf(stderr, "apus: GLM --measure-locality requires --tiered "
                "(the routed hook fires in tiered mode only)\n");
        return 2;
    }
    if (spec && measure_path) {
        fprintf(stderr, "apus: --spec and --measure-locality are mutually "
                "exclusive\n");
        return 2;
    }
    if (spec && !apus_gmodel_has_mtp(e->gm)) {
        fprintf(stderr, "apus: --spec requires the MTP (NextN) block "
                "(layers.%d.*) — not present in this container\n",
                cfg->num_hidden_layers);
        return 2;
    }
    if (!quiet)
        fprintf(stderr,
                "apus: loaded %s (glm5_next, %d layers, dim %d, vocab %d%s) "
                "in %.2fs\n",
                model_dir, cfg->num_hidden_layers, cfg->hidden_size,
                cfg->vocab_size, e->tiered ? ", tiered" : "", now_s() - t0);

    Tok *tok = NULL;
    int n_ids = 0;
    int64_t *ids = run_input_ids(model_dir, prompt, ids_str, &tok, &n_ids);

    int V = cfg->vocab_size;
    /* M8b: --spec needs room for the verify batch running ahead of the
     * emitted count (up to spec_k positions) */
    size_t kv_cap = (size_t)n_ids + (size_t)max_tokens
                  + (spec ? (size_t)spec_k + 1 : 1);
    ApusGmodelState *st = apus_gmodel_state_new(e->gm, kv_cap);
    uint16_t *logits16 = malloc((size_t)n_ids * (size_t)V * sizeof(uint16_t));
    float *logits = malloc((size_t)V * sizeof(float));
    void *scratch = malloc(apus_sample_scratch_size((size_t)V));
    int32_t *ids32 = malloc((size_t)n_ids * sizeof(int32_t));
    for (int i = 0; i < n_ids; i++) ids32[i] = (int32_t)ids[i];
    ApusRng rng;
    apus_rng_seed(&rng, seed);

    /* P1 measure-locality: chain the pilot's hooks, then dump A/P sets.
     * Read-only instrumentation (fprintf only); numerics untouched. */
    GMeasureDump md;
    FILE *mf = NULL;
    if (measure_path) {
        mf = fopen(measure_path, "w");
        if (!mf) {
            fprintf(stderr, "apus: cannot open %s\n", measure_path);
            return 1;
        }
        memset(&md, 0, sizeof md);
        md.f = mf;
        md.gpilot = e->gpilot;
        if (e->gpilot) apus_gpilot_hooks(e->gpilot, &md.chain);
        md.hcd = (size_t)cfg->hc_mult * (size_t)cfg->hidden_size;
        md.topk = cfg->num_experts_per_tok;
        md.pn = cfg->n_routed_experts < APUS_GMEASURE_N
                ? cfg->n_routed_experts : APUS_GMEASURE_N;
        ApusGmodelHooks hk = { &md, gmeasure_post_attn, gmeasure_routed };
        apus_gmodel_set_hooks(e->gm, &hk);
        /* self-contained dump: the prompt ids (generated ids follow as
         * "gen" lines in the decode loop) */
        fprintf(mf, "{\"type\":\"ids\",\"pos0\":0,\"ids\":[");
        for (int i = 0; i < n_ids; i++)
            fprintf(mf, "%s%lld", i ? "," : "", (long long)ids[i]);
        fprintf(mf, "]}\n");
    }

    t0 = now_s();
    int n_gen = 0;
    double t_prefill, t_gen;
    if (spec) {
        /* M8b: MTP draft/verify loop (c/gmtp.h ApusGspec). Emitted tokens
         * are bitwise identical to the non-speculative loop below for the
         * same seed: every emitted token is the main model's own
         * apus_sample draw from its own (interleave-bitwise) logits row,
         * one RNG uniform per token in position order; drafts consume no
         * RNG. */
        char serr[256];
        ApusGmtp mt;
        if (apus_gmtp_bind(&mt, e->gm, serr, sizeof serr)) {
            fprintf(stderr, "apus: %s\n", serr);
            return 1;
        }
        ApusGmtpState *mst = apus_gmtp_state_new(&mt, kv_cap
                                                 + (size_t)spec_k);
        if (!mst) {
            fprintf(stderr, "apus: MTP state allocation failed\n");
            return 1;
        }
        ApusGspec sp;
        apus_gspec_init(&sp, e->gm, st, &mt, mst, spec_k, temp, top_p,
                        &rng, scratch);
        if (apus_gspec_prefill(&sp, ids32, (size_t)n_ids)) {
            fprintf(stderr, "apus: spec prefill failed\n");
            return 1;
        }
        t_prefill = now_s() - t0;
        double t_gen0 = now_s();
        int step_out[64], done = 0;
        while (n_gen < max_tokens && !done) {
            int ne = apus_gspec_step(&sp, step_out, 64);
            if (ne <= 0) break;
            for (int i = 0; i < ne && n_gen < max_tokens; i++) {
                int tok_id = step_out[i];
                if (quiet) {
                    printf("%d\n", tok_id);
                } else if (tok) {
                    size_t len = 0;
                    char *text = tok_decode(tok,
                                            (const uint32_t[]){(uint32_t)tok_id},
                                            1, &len);
                    if (text) { fwrite(text, 1, len, stdout); free(text); }
                    fflush(stdout);
                } else {
                    printf("%d ", tok_id);
                    fflush(stdout);
                }
                n_gen++;
                if (glm_is_eos(e, tok_id)) { done = 1; break; }
            }
        }
        t_gen = now_s() - t_gen0;
        if (!quiet)
            fprintf(stderr,
                    "apus: spec: %llu emitted in %llu batches "
                    "(%.2f tok/batch, %llu re-fed), draft accept %llu/%llu"
                    " (%.1f%%), d1 %llu/%llu\n",
                    (unsigned long long)sp.emitted,
                    (unsigned long long)sp.batches,
                    sp.batches ? (double)sp.emitted / (double)sp.batches
                               : 0.0,
                    (unsigned long long)sp.refeed_tokens,
                    (unsigned long long)sp.accepted,
                    (unsigned long long)sp.offered,
                    sp.offered ? 100.0 * (double)sp.accepted
                                      / (double)sp.offered : 0.0,
                    (unsigned long long)sp.d1_hits,
                    (unsigned long long)sp.d1_offered);
        apus_gspec_free(&sp);
        apus_gmtp_state_free(mst);
    } else {
    if (apus_gmodel_prefill(e->gm, st, ids32, (size_t)n_ids, logits16,
                            NULL, NULL)) {
        fprintf(stderr, "apus: prefill failed\n");
        return 1;
    }
    t_prefill = now_s() - t0;
    {
        const uint16_t *last = logits16 + (size_t)(n_ids - 1) * (size_t)V;
        for (int i = 0; i < V; i++) logits[i] = apus_bf16_f32(last[i]);
    }

    double t_gen0 = now_s();
    double t_tok0 = t_gen0;
    /* P2 diagnostics (APUS_DECODE_TIMING=1, reporting only): per-token wall
     * time + gcache counter deltas (hits/misses/preads/waits/deq) to
     * decompose where decode stalls. No numerics touched. */
    int dt_on = apus_env_int("APUS_DECODE_TIMING", 0);
    ApusGcacheStats dt_prev;
    memset(&dt_prev, 0, sizeof dt_prev);
    for (int step = 0; step < max_tokens; step++) {
        int tok_id = apus_sample(logits, (size_t)V, temp, top_p, &rng, scratch);
        if (quiet) {
            printf("%d\n", tok_id);
        } else if (tok) {
            size_t len = 0;
            char *text = tok_decode(tok, (const uint32_t[]){(uint32_t)tok_id},
                                    1, &len);
            if (text) { fwrite(text, 1, len, stdout); free(text); }
            fflush(stdout);
        } else {
            printf("%d ", tok_id);
            fflush(stdout);
        }
        n_gen++;
        if (glm_is_eos(e, tok_id)) break;
        if (mf)
            fprintf(mf, "{\"type\":\"gen\",\"pos\":%lld,\"id\":%d}\n",
                    (long long)apus_gmodel_pos(st), tok_id);
        if (apus_gmodel_decode_step(e->gm, st, tok_id, logits16, NULL, NULL)) {
            fprintf(stderr, "apus: decode failed\n");
            return 1;
        }
        for (int i = 0; i < V; i++) logits[i] = apus_bf16_f32(logits16[i]);
        if (dt_on) {
            double t1 = now_s();
            if (apus_gmodel_cache(e->gm)) {
                ApusGcacheStats cs;
                apus_gcache_stats(apus_gmodel_cache(e->gm), &cs);
                fprintf(stderr,
                        "apus: tok %d dt %.2fs | +hit %llu +miss %llu "
                        "+pread %llu +wait %llu wait_ns %llu deq_ns %llu\n",
                        n_gen, t1 - t_tok0,
                        (unsigned long long)(cs.hits - dt_prev.hits),
                        (unsigned long long)(cs.misses - dt_prev.misses),
                        (unsigned long long)(cs.preads - dt_prev.preads),
                        (unsigned long long)(cs.waits - dt_prev.waits),
                        (unsigned long long)(cs.wait_ns - dt_prev.wait_ns),
                        (unsigned long long)(cs.deq_ns - dt_prev.deq_ns));
                dt_prev = cs;
            } else {
                fprintf(stderr, "apus: tok %d dt %.2fs\n", n_gen,
                        t1 - t_tok0);
            }
            t_tok0 = t1;
        }
    }
    t_gen = now_s() - t_gen0;
    }   /* spec / non-spec */
    if (mf) fclose(mf);

    if (e->gpilot) {
        ApusGpilotStats ps;
        apus_gpilot_stats(e->gpilot, &ps);
        if (!quiet)
            fprintf(stderr,
                    "apus: pilot: %llu predictions (%llu prefill), %llu hints "
                    "(%llu prefill), recall %llu/%llu\n",
                    (unsigned long long)ps.predictions,
                    (unsigned long long)ps.prefill_predictions,
                    (unsigned long long)ps.hints_issued,
                    (unsigned long long)ps.prefill_hints,
                    (unsigned long long)ps.actual_hits,
                    (unsigned long long)ps.actual_experts);
    }
    if (apus_gmodel_cache(e->gm)) {
        ApusGcacheStats ss;
        apus_gcache_stats(apus_gmodel_cache(e->gm), &ss);
        if (!quiet)
            fprintf(stderr,
                    "apus: expert cache: %llu hits %llu misses "
                    "(%llu preads, %.1f MB; %llu hint-loads %llu "
                    "demand-loads %llu spec-dropped), %llu evictions, "
                    "%llu rss drops, peak footprint %.2f GiB\n",
                    (unsigned long long)ss.hits, (unsigned long long)ss.misses,
                    (unsigned long long)ss.preads,
                    (double)ss.bytes_read / 1048576.0,
                    (unsigned long long)ss.hint_loads,
                    (unsigned long long)ss.demand_loads,
                    (unsigned long long)ss.spec_dropped,
                    (unsigned long long)ss.evictions,
                    (unsigned long long)ss.rss_drops,
                    (double)ss.pressure_peak / 1073741824.0);
    }
    if (!quiet) {
        printf("\n");
        fprintf(stderr,
                "apus: prefill %d tok in %.2fs (%.1f tok/s); "
                "decode %d tok in %.2fs (%.2f tok/s)\n",
                n_ids, t_prefill, n_ids / t_prefill,
                n_gen, t_gen, n_gen > 0 ? n_gen / t_gen : 0.0);
    }

    free(ids32);
    free(logits16);
    free(logits);
    free(scratch);
    free(ids);
    if (tok) tok_free(tok);
    apus_gmodel_state_free(st);
    return 0;
}

static int run_main(int argc, char **argv) {
    const char *model_dir = NULL, *prompt = NULL, *ids_str = NULL;
    const char *measure_path = NULL;
    int max_tokens = 32, quiet = 0;
    int tiered = apus_env_int("APUS_TIERED", 0);
    int metal = apus_env_int("APUS_METAL", 0);
    int spec = apus_env_int("APUS_SPEC", 1);        /* M8c: default ON (run
                                                       mode; serve too — M8d) */
    int spec_explicit = getenv("APUS_SPEC") != NULL;
    int spec_k = apus_env_int("APUS_SPEC_K", 3);    /* the re-pin sweep's
                                                       winner (85.3% accept) */
    uint64_t seed = 0;
    float temp = 1.0f, top_p = 1.0f;
    int temp_set = 0, topp_set = 0;     /* explicit user flags beat the
                                           container's generation_config */
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_str = argv[++i];
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) { temp = (float)atof(argv[++i]); temp_set = 1; }
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) { top_p = (float)atof(argv[++i]); topp_set = 1; }
        else if (!strcmp(argv[i], "--greedy")) { temp = 0.0f; temp_set = 1; }
        else if (!strcmp(argv[i], "--tiered")) tiered = 1;
        else if (!strcmp(argv[i], "--metal")) metal = 1;
        else if (!strcmp(argv[i], "--spec")) { spec = 1; spec_explicit = 1; }
        else if (!strcmp(argv[i], "--no-spec")) { spec = 0; spec_explicit = 1; }
        else if (!strcmp(argv[i], "--spec-k") && i + 1 < argc) spec_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--measure-locality") && i + 1 < argc) measure_path = argv[++i];
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--dump-margins")) { /* accepted; no-op
                on the GLM engine (the V4 margin dump is gone) */ }
        else { usage(stderr); return 2; }
    }
    if (!model_dir || (!prompt && !ids_str)) { usage(stderr); return 2; }
    /* M8c default-on resolution: a default-on spec yields silently to
     * --measure-locality; an EXPLICIT --spec keeps the loud mutual-
     * exclusion error in run_glm. */
    if (spec && !spec_explicit && measure_path) spec = 0;
    if (spec_k < 1) spec_k = 1;
    if (spec_k > 63) spec_k = 63;
    metal_maybe_enable(metal, quiet);

    char err[256];
    double t0 = now_s();
    int pilot_on = tiered && apus_env_int("APUS_PILOT", 1);
    Engine e;
    if (engine_init(&e, model_dir, tiered, pilot_on || measure_path != NULL,
                    pilot_on, spec, err, sizeof err)) {
        fprintf(stderr, "apus: %s\n", err);
        return 1;
    }
    /* M8c: a default-on spec on an MTP-less container falls back to the
     * non-spec loop with a note; explicit --spec keeps run_glm's loud
     * error (the m8g nomtp negative gate). */
    if (spec && !spec_explicit && !apus_gmodel_has_mtp(e.gm)) {
        spec = 0;
        if (!quiet)
            fprintf(stderr, "apus: no MTP block in this container — "
                    "speculative decoding off\n");
    }
    /* sampling defaults from the container's generation_config.json
     * (temperature/top_p; 1.0/1.0 when absent) apply only where the user
     * gave no explicit --temp/--top-p/--greedy (M7a fix) */
    if (!temp_set) temp = e.def_temp;
    if (!topp_set) top_p = e.def_top_p;
    int rc = run_glm(&e, model_dir, prompt, ids_str, max_tokens, seed,
                     temp, top_p, spec, spec_k, quiet, t0,
                     measure_path);
    engine_destroy(&e);
    return rc;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    /* M15: MSVCRT defaults stdin/stdout to text mode (\n -> \r\n), which
     * would corrupt the NDJSON serve protocol; force binary. */
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc < 2) { usage(stderr); return 2; }
    if (!strcmp(argv[1], "run")) return run_main(argc, argv);
    if (!strcmp(argv[1], "serve")) return serve_main(argc, argv);
    usage(stderr);
    return 2;
}
