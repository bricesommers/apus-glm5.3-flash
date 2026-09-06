/*
 * apus encoding.h — GLM-5.3-Flash chat template, C11.
 *
 * Faithful C port of reference/chat_template.jinja, byte-exact against the
 * jinja2 rendering in the HF transformers environment (ImmutableSandboxed-
 * Environment, trim_blocks + lstrip_blocks, loopcontrols, transformers'
 * tojson filter = Python json.dumps; tests/m2/gen_golden.py renders the
 * goldens the same way). Template semantics:
 *
 *   - output always starts with "[gMASK]<sop>"
 *   - then "<|system|>Reasoning Effort: Low|High|Max" — the effort is
 *     "low"/"high" only if given exactly so; EVERYTHING else (including
 *     unset) maps to "Max". No newline anywhere between blocks.
 *   - tools (a top-level parameter, NOT a message field) render a
 *     "<|system|>\n# Tools\n..." block: OpenAI-wrapped {"function": {...}}
 *     tools are unwrapped, defer_loading tools are skipped, the keys
 *     "defer_loading"/"strict" are dropped from the per-tool JSON, values
 *     use Python json.dumps(ensure_ascii=False) separators (", " / ": ").
 *   - "<|user|>" + visible content (NOT stripped)
 *   - "<|assistant|>" + "<think>reasoning</think>" (or "<think></think>")
 *     + content.strip() (Python str.strip) + tool calls. reasoning comes
 *     from reasoning_content (string only) or is extracted from content
 *     between <think>/</think> (split on FIRST </think> for reasoning, LAST
 *     for content). clear_thinking drops reasoning for assistant messages
 *     up to the last user message; later reasoning is always kept.
 *   - tool calls: "<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>
 *     ...</tool_call>"; arguments MUST be a JSON object (string arguments
 *     fail, like the template's UndefinedError); string values verbatim,
 *     other values json.dumps. OpenAI-wrapped {"function": {...}} calls are
 *     unwrapped (truthiness), flat {"name","arguments"} calls work too.
 *   - consecutive tool messages form a block, prefixed once with
 *     "<|observation|>"; responses are "<tool_response>...</tool_response>".
 *     If the immediately preceding assistant message has tool_calls and the
 *     block ids are complete/unique/matching on both sides, responses are
 *     re-sorted into tool_call order; otherwise message order is kept.
 *   - content lists concatenate visible items: text -> text, image ->
 *     "<|begin_of_image|><|image|><|end_of_image|>", video likewise, audio
 *     -> "<|begin_of_audio|><|end_of_audio|>" (no middle token).
 *   - tool_reference responses re-emit the referenced tool schemas from the
 *     tools parameter: "<tool_response><tools>\n{json}\n</tools></tool_response>".
 *   - unknown roles are silently skipped; add_generation_prompt appends
 *     "<|assistant|><think>".
 *
 * Divergence notes (edge cases the template renders via Python internals):
 * dict-typed message content (Python repr) is an error here; non-string
 * tool_call/result ids compare as missing (ids are strings in practice).
 *
 * Usage: #define APUS_ENCODING_IMPLEMENTATION in exactly one TU.
 */
#ifndef APUS_ENCODING_H
#define APUS_ENCODING_H

#include <stddef.h>
#include <stdint.h>

#include "json.h"
#include "tok.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *reasoning_effort; /* NULL | "low" | "high" | "max";
                                     only "low"/"high" honored, else "Max" */
    int clear_thinking;           /* default 0 */
    int add_generation_prompt;    /* default 1 */
} GlmEncOpts;

#define GLM_ENC_OPTS_DEFAULT { NULL, 0, 1 }

/* Encode a message list to the prompt string (malloc'd; NULL on error,
 * see glm_last_error()). tools may be NULL. */
char *glm_encode_messages(const JVal *messages, const JVal *tools,
                          const GlmEncOpts *opts);

/* Encode straight to token ids (combines string assembly with tok.h).
 * Special tokens in the assembled prompt are recognized. */
uint32_t *glm_encode_ids(const Tok *t, const JVal *messages, const JVal *tools,
                         const GlmEncOpts *opts, size_t *n_out);

const char *glm_last_error(void);

/* Special token ids (reference/tokenizer.json added_tokens) */
#define GLM_ID_ENDOFTEXT   154820u
#define GLM_ID_MASK        154821u  /* [MASK] */
#define GLM_ID_GMASK       154822u  /* [gMASK] */
#define GLM_ID_SMASK       154823u  /* [sMASK] */
#define GLM_ID_SOP         154824u
#define GLM_ID_EOP         154825u
#define GLM_ID_SYSTEM      154826u
#define GLM_ID_USER        154827u
#define GLM_ID_ASSISTANT   154828u
#define GLM_ID_OBSERVATION 154829u
#define GLM_ID_THINK       154841u  /* <think> */
#define GLM_ID_THINK_END   154842u  /* </think> */

/* generation_config.json eos_token_id — stop set for sampling */
extern const uint32_t GLM_EOS_IDS[3];

#ifdef __cplusplus
}
#endif

#endif /* APUS_ENCODING_H */

/* ================================================================== */
#if defined(APUS_ENCODING_IMPLEMENTATION) && !defined(APUS_ENCODING_IMPL_DONE)
#define APUS_ENCODING_IMPL_DONE

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

const uint32_t GLM_EOS_IDS[3] = { GLM_ID_ENDOFTEXT, GLM_ID_USER, GLM_ID_OBSERVATION };

/* ---------------- error reporting ---------------- */

static char glm_err[256];

const char *glm_last_error(void) { return glm_err; }

static int glm_fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(glm_err, sizeof glm_err, fmt, ap);
    va_end(ap);
    return -1;
}

/* ---------------- helpers ---------------- */

static const char *jstr(const JVal *obj, const char *key) {
    JVal *v = json_obj_get((JVal *)obj, key);
    return json_type(v) == J_STR ? json_str(v) : NULL;
}

static const char *jrole(const JVal *msg) {
    const char *r = jstr(msg, "role");
    return r ? r : "";
}

/* Python str.strip() whitespace (Py_UNICODE_ISSPACE, CPython's fixed set) */
static int py_isspace(uint32_t cp) {
    if (cp <= 0x20)
        return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x1F) || cp == 0x20;
    if (cp < 0xA0) return cp == 0x85;
    return cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

/* decode one UTF-8 codepoint (for strip scanning); *adv = length */
static uint32_t glm_utf8_decode(const unsigned char *s, size_t len, size_t *adv) {
    if (!len) { *adv = 0; return 0x110000; }
    unsigned char c = s[0];
    if (c < 0x80) { *adv = 1; return c; }
    if (c >= 0xC2 && c < 0xE0 && len >= 2 && (s[1] & 0xC0) == 0x80) {
        *adv = 2;
        return ((uint32_t)(c & 0x1F) << 6) | (s[1] & 0x3F);
    }
    if (c >= 0xE0 && c < 0xF0 && len >= 3 &&
        (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *adv = 3;
        return ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) |
               (s[2] & 0x3F);
    }
    if (c >= 0xF0 && c < 0xF5 && len >= 4 &&
        (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        *adv = 4;
        return ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
               ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    }
    *adv = 1;
    return 0x110000;
}

/* Python str.strip(): in-place, returns s */
static char *py_strip(char *s) {
    size_t n = strlen(s);
    size_t lo = 0;
    while (lo < n) {
        size_t adv;
        uint32_t cp = glm_utf8_decode((const unsigned char *)s + lo, n - lo, &adv);
        if (!py_isspace(cp)) break;
        lo += adv;
    }
    size_t hi = n;
    while (hi > lo) {
        /* scan back to the start of the last codepoint */
        size_t p = hi - 1;
        while (p > lo && ((unsigned char)s[p] & 0xC0) == 0x80) p--;
        size_t adv;
        uint32_t cp = glm_utf8_decode((const unsigned char *)s + p, hi - p, &adv);
        if (!py_isspace(cp) || p + adv != hi) break;
        hi = p;
    }
    if (lo) memmove(s, s + lo, hi - lo);
    s[hi - lo] = '\0';
    return s;
}

/* ---------------- visible_text ---------------- */

static void sb_emit_image(SBuf *b) { sb_puts(b, "<|begin_of_image|><|image|><|end_of_image|>"); }
static void sb_emit_video(SBuf *b) { sb_puts(b, "<|begin_of_video|><|video|><|end_of_video|>"); }
static void sb_emit_audio(SBuf *b) { sb_puts(b, "<|begin_of_audio|><|end_of_audio|>"); }

/* The template's visible_text macro. A MISSING content key is jinja
 * Undefined, which renders as "" — explicit JSON null renders as "None". */
static int sb_visible_text(SBuf *b, const JVal *content) {
    if (!content) return 0;
    switch (json_type(content)) {
    case J_STR:
        sb_puts(b, json_str(content));
        return 0;
    case J_ARR: {
        size_t n = json_arr_len(content);
        for (size_t i = 0; i < n; i++) {
            JVal *item = json_arr_get(content, i);
            if (json_type(item) == J_STR) {
                sb_puts(b, json_str(item));
            } else if (json_type(item) == J_OBJ) {
                const char *tp = jstr(item, "type");
                if (tp && strcmp(tp, "text") == 0) {
                    const char *tx = jstr(item, "text");
                    if (tx) sb_puts(b, tx);
                } else if (tp && (!strcmp(tp, "image") || !strcmp(tp, "image_url"))) {
                    sb_emit_image(b);
                } else if (tp && (!strcmp(tp, "video") || !strcmp(tp, "video_url"))) {
                    sb_emit_video(b);
                } else if (tp && (!strcmp(tp, "audio") || !strcmp(tp, "audio_url") ||
                                  !strcmp(tp, "input_audio"))) {
                    sb_emit_audio(b);
                }
                /* other items: no output (template has no else in the loop) */
            }
        }
        return 0;
    }
    case J_NULL: sb_puts(b, "None"); return 0;
    case J_BOOL: sb_puts(b, json_bool(content) ? "True" : "False"); return 0;
    case J_NUM: {
        /* Python str() of a JSON number == json.dumps for int; float repr */
        char *j = json_dumps(content);
        sb_puts(b, j);
        free(j);
        return 0;
    }
    default:
        return glm_fail("unsupported content type (object); "
                        "template would render a Python repr");
    }
}

/* visible_text as a malloc'd string (NULL on error) */
static char *visible_text_str(const JVal *content) {
    SBuf b;
    sb_init(&b);
    if (sb_visible_text(&b, content) < 0) {
        sb_free(&b);
        return NULL;
    }
    return sb_steal(&b);
}

/* ---------------- tool schema rendering ---------------- */

/* tool_to_json macro: "{" + '"k": json(v)}' pairs joined ", ", skipping the
 * "defer_loading"/"strict" keys; keys raw (not JSON-escaped), values via
 * Python json.dumps(ensure_ascii=False). */
static void sb_tool_to_json(SBuf *b, const JVal *tool) {
    sb_putc(b, '{');
    size_t n = json_obj_len(tool);
    int first = 1;
    for (size_t i = 0; i < n; i++) {
        const char *k = json_obj_key(tool, i);
        if (strcmp(k, "defer_loading") == 0 || strcmp(k, "strict") == 0)
            continue;
        if (!first) sb_puts(b, ", ");
        first = 0;
        sb_putc(b, '"');
        sb_puts(b, k);
        sb_puts(b, "\": ");
        json_dumps_sb(b, json_obj_val(tool, i));
    }
    sb_putc(b, '}');
}

/* unwrap OpenAI {"type":"function","function":{...}} by key presence */
static const JVal *tool_unwrap(const JVal *tool) {
    JVal *fn = json_obj_get((JVal *)tool, "function");
    return fn ? fn : tool;
}

static const char *TOOLS_BLOCK_1 =
    "<|system|>\n"
    "# Tools\n"
    "\n"
    "You may call one or more functions to assist with the user query.\n"
    "\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n";

static const char *TOOLS_BLOCK_2 =
    "</tools>\n"
    "\n"
    "For each function call, output the function name and arguments within the following XML format:\n"
    "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
    "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";

static int sb_tools_block(SBuf *b, const JVal *tools) {
    if (json_type(tools) != J_ARR)
        return glm_fail("tools must be a JSON array");
    sb_puts(b, TOOLS_BLOCK_1);
    size_t n = json_arr_len(tools);
    for (size_t i = 0; i < n; i++) {
        const JVal *tool = tool_unwrap(json_arr_get(tools, i));
        JVal *defer = json_obj_get((JVal *)tool, "defer_loading");
        if (defer && json_truthy(defer)) continue;
        if (json_type(tool) != J_OBJ)
            return glm_fail("tool %zu is not an object", i);
        sb_tool_to_json(b, tool);
        sb_putc(b, '\n');
    }
    sb_puts(b, TOOLS_BLOCK_2);
    return 0;
}

/* tool_references_to_response macro */
static int sb_tool_references_response(SBuf *b, const JVal *refs, const JVal *tools) {
    if (!tools || json_type(tools) != J_ARR)
        return glm_fail("tool_reference response needs the tools array");
    sb_puts(b, "<tool_response><tools>\n");
    size_t n = json_arr_len(refs);
    for (size_t i = 0; i < n; i++) {
        const char *name = jstr(json_arr_get(refs, i), "name");
        if (!name) continue;  /* Undefined == str is false in jinja */
        size_t m = json_arr_len(tools);
        for (size_t j = 0; j < m; j++) {
            const JVal *tool = tool_unwrap(json_arr_get(tools, j));
            const char *tn = jstr(tool, "name");
            if (tn && strcmp(tn, name) == 0) {
                sb_tool_to_json(b, tool);
                sb_putc(b, '\n');
            }
        }
    }
    sb_puts(b, "</tools></tool_response>");
    return 0;
}

/* ---------------- tool responses ---------------- */

/* id_of macro: obj.tool_call_id if truthy else obj.id if truthy else none.
 * String ids only (see header divergence note). */
static const char *id_of(const JVal *obj) {
    if (!obj || json_type(obj) != J_OBJ) return NULL;
    JVal *v = json_obj_get((JVal *)obj, "tool_call_id");
    if (v && json_truthy(v)) return json_type(v) == J_STR ? json_str(v) : NULL;
    v = json_obj_get((JVal *)obj, "id");
    if (v && json_truthy(v)) return json_type(v) == J_STR ? json_str(v) : NULL;
    return NULL;
}

/* is_list_of_outputs: content is a non-empty array whose [0] HAS an
 * "output" key (any value — jinja `is defined` is true for null too) */
static int is_list_of_outputs(const JVal *msg) {
    JVal *c = json_obj_get((JVal *)msg, "content");
    if (!c || json_type(c) != J_ARR || json_arr_len(c) == 0) return 0;
    JVal *first = json_arr_get(c, 0);
    return json_type(first) == J_OBJ && json_obj_get(first, "output") != NULL;
}

static int sb_tool_response(SBuf *b, const JVal *content) {
    char *vt = visible_text_str(content);
    if (!vt) return -1;
    sb_puts(b, "<tool_response>");
    sb_puts(b, vt);
    sb_puts(b, "</tool_response>");
    free(vt);

    return 0;
}

/* is `out` a non-empty list whose [0].type == "tool_reference"? */
static int is_tool_reference_list(const JVal *out) {
    if (!out || json_type(out) != J_ARR || json_arr_len(out) == 0) return 0;
    JVal *first = json_arr_get(out, 0);
    const char *tp = json_type(first) == J_OBJ ? jstr(first, "type") : NULL;
    return tp && strcmp(tp, "tool_reference") == 0;
}

/* render the `output` field of a list-of-outputs entry (sorted or unsorted) */
static int sb_entry_output(SBuf *b, const JVal *entry, const JVal *tools) {
    JVal *out = json_obj_get((JVal *)entry, "output");
    if (is_tool_reference_list(out))
        return sb_tool_references_response(b, out, tools);
    return sb_tool_response(b, out);
}

/* render_tool_response macro (plain tool message) */
static int sb_render_tool_response(SBuf *b, const JVal *msg, const JVal *tools) {
    JVal *c = json_obj_get((JVal *)msg, "content");
    if (c && json_type(c) == J_STR)
        return sb_tool_response(b, c);
    if (c && json_type(c) == J_ARR && json_arr_len(c) > 0 &&
        is_tool_reference_list(c))
        return sb_tool_references_response(b, c, tools);
    if (is_list_of_outputs(msg)) {
        size_t n = json_arr_len(c);
        for (size_t i = 0; i < n; i++)
            if (sb_entry_output(b, json_arr_get(c, i), tools) < 0)
                return -1;
        return 0;
    }
    return sb_tool_response(b, c);
}

/* ---------------- assistant message ---------------- */

static int sb_tool_calls(SBuf *b, const JVal *tcs) {
    if (json_type(tcs) != J_ARR)
        return glm_fail("tool_calls must be a JSON array");
    size_t n = json_arr_len(tcs);
    for (size_t i = 0; i < n; i++) {
        const JVal *tc = json_arr_get(tcs, i);
        if (json_type(tc) != J_OBJ)
            return glm_fail("tool_call %zu is not an object", i);
        JVal *fn = json_obj_get((JVal *)tc, "function");
        if (fn && json_truthy(fn)) {
            if (json_type(fn) != J_OBJ)
                return glm_fail("tool_call %zu: function is not an object", i);
            tc = fn;
        }
        const char *name = jstr(tc, "name");
        if (!name)
            return glm_fail("tool_call %zu: missing name", i);
        JVal *args = json_obj_get((JVal *)tc, "arguments");
        if (!args || json_type(args) != J_OBJ)
            return glm_fail("tool_call %zu: arguments must be an object "
                            "(the template raises on JSON strings)", i);
        sb_puts(b, "<tool_call>");
        sb_puts(b, name);
        size_t m = json_obj_len(args);
        for (size_t j = 0; j < m; j++) {
            const char *k = json_obj_key(args, j);
            JVal *v = json_obj_val(args, j);
            sb_puts(b, "<arg_key>");
            sb_puts(b, k);
            sb_puts(b, "</arg_key><arg_value>");
            if (json_type(v) == J_STR)
                sb_puts(b, json_str(v));
            else
                json_dumps_sb(b, v);
            sb_puts(b, "</arg_value>");
        }
        sb_puts(b, "</tool_call>");
    }
    return 0;
}

/* ---------------- encode_messages ---------------- */

char *glm_encode_messages(const JVal *messages, const JVal *tools,
                          const GlmEncOpts *opts) {
    glm_err[0] = '\0';
    if (!messages || json_type(messages) != J_ARR) {
        glm_fail("messages must be a JSON array");
        return NULL;
    }
    const char *effort = opts ? opts->reasoning_effort : NULL;
    int clear_thinking = opts ? opts->clear_thinking : 0;
    int add_gen = opts ? opts->add_generation_prompt : 1;

    SBuf out;
    sb_init(&out);
    sb_puts(&out, "[gMASK]<sop>");
    sb_puts(&out, "<|system|>Reasoning Effort: ");
    if (effort && strcmp(effort, "low") == 0) sb_puts(&out, "Low");
    else if (effort && strcmp(effort, "high") == 0) sb_puts(&out, "High");
    else sb_puts(&out, "Max");

    if (tools && json_truthy(tools)) {
        if (sb_tools_block(&out, tools) < 0) goto fail;
    }

    size_t n = json_arr_len(messages);
    int last_user_index = -1;
    for (size_t i = 0; i < n; i++)
        if (strcmp(jrole(json_arr_get(messages, i)), "user") == 0)
            last_user_index = (int)i;

    size_t skip_until = 0; /* tool blocks are consumed by their first message */
    for (size_t i = 0; i < n; i++) {
        if (i < skip_until) continue;
        const JVal *msg = json_arr_get(messages, i);
        const char *role = jrole(msg);

        if (strcmp(role, "user") == 0) {
            sb_puts(&out, "<|user|>");
            if (sb_visible_text(&out, json_obj_get((JVal *)msg, "content")) < 0)
                goto fail;
        } else if (strcmp(role, "assistant") == 0) {
            sb_puts(&out, "<|assistant|>");
            char *content = visible_text_str(json_obj_get((JVal *)msg, "content"));
            if (!content) goto fail;
            const char *rc = jstr(msg, "reasoning_content");
            char *rc_free = NULL;  /* set when reasoning is extracted below */
            const char *reasoning = NULL;
            if (rc) {
                reasoning = rc;
            } else if (strstr(content, "</think>")) {
                /* reasoning = content.split('</think>')[0].split('<think>')[-1]
                 * content   = content.split('</think>')[-1] */
                char *first_end = strstr(content, "</think>");
                char *last_end = first_end;
                for (char *p = strstr(first_end + 8, "</think>"); p;
                     p = strstr(p + 8, "</think>"))
                    last_end = p;
                size_t part0_n = (size_t)(first_end - content);
                char *lt = NULL;
                for (char *p = strstr(content, "<think>"); p && p < first_end;
                     p = strstr(p + 7, "<think>"))
                    lt = p;
                size_t rs = lt ? (size_t)(lt + 7 - content) : 0;
                rc_free = (char *)malloc(part0_n - rs + 1);
                if (!rc_free) { free(content); glm_fail("oom"); goto fail; }
                memcpy(rc_free, content + rs, part0_n - rs);
                rc_free[part0_n - rs] = '\0';
                reasoning = rc_free;
                memmove(content, last_end + 8, strlen(last_end + 8) + 1);
            }
            /* think block: reasoning kept unless clear_thinking drops it
             * (dropped only up to the last user message) */
            if (reasoning && (!clear_thinking || (int)i > last_user_index)) {
                sb_puts(&out, "<think>");
                sb_puts(&out, reasoning);
                sb_puts(&out, "</think>");
            } else {
                sb_puts(&out, "<think></think>");
            }
            free(rc_free);
            py_strip(content);
            if (content[0]) sb_puts(&out, content);
            free(content);
            JVal *tcs = json_obj_get((JVal *)msg, "tool_calls");
            if (tcs && json_truthy(tcs)) {
                if (sb_tool_calls(&out, tcs) < 0) goto fail;
            }
        } else if (strcmp(role, "tool") == 0) {
            if (i > 0 && strcmp(jrole(json_arr_get(messages, i - 1)), "tool") == 0)
                continue;  /* not block start (defensive; skip_until covers) */
            sb_puts(&out, "<|observation|>");
            /* block = consecutive tool messages [i, bend] */
            size_t bend = i;
            while (bend + 1 < n &&
                   strcmp(jrole(json_arr_get(messages, bend + 1)), "tool") == 0)
                bend++;
            skip_until = bend + 1;

            /* preceding assistant's tool_calls? */
            const JVal *tcs = NULL;
            if (i > 0) {
                const JVal *prev = json_arr_get(messages, i - 1);
                if (strcmp(jrole(prev), "assistant") == 0) {
                    JVal *v = json_obj_get((JVal *)prev, "tool_calls");
                    if (v && json_truthy(v) && json_type(v) == J_ARR)
                        tcs = v;
                }
            }

            /* can_sort? */
            int can_sort = 1;
            if (!tcs) {
                can_sort = 0;
            } else {
                size_t nt = json_arr_len(tcs);
                /* each block id: truthy, unique within the block, present
                 * among the tool_call ids */
                for (size_t k = i; k <= bend && can_sort; k++) {
                    const JVal *m = json_arr_get(messages, k);
                    const JVal *c = json_obj_get((JVal *)m, "content");
                    int entries = is_list_of_outputs(m);
                    size_t n_ent = entries ? json_arr_len(c) : 1;
                    for (size_t e = 0; e < n_ent && can_sort; e++) {
                        const char *eid = entries
                            ? id_of(json_arr_get(c, e)) : id_of(m);
                        if (!eid) { can_sort = 0; break; }
                        size_t dup = 0;
                        int found = 0;
                        for (size_t k2 = i; k2 <= bend; k2++) {
                            const JVal *m2 = json_arr_get(messages, k2);
                            const JVal *c2 = json_obj_get((JVal *)m2, "content");
                            int ent2 = is_list_of_outputs(m2);
                            size_t n2 = ent2 ? json_arr_len(c2) : 1;
                            for (size_t e2 = 0; e2 < n2; e2++) {
                                const char *id2 = ent2
                                    ? id_of(json_arr_get(c2, e2)) : id_of(m2);
                                if (id2 && strcmp(id2, eid) == 0) dup++;
                            }
                        }
                        if (dup > 1) { can_sort = 0; break; }
                        for (size_t q = 0; q < nt; q++) {
                            const char *tid = id_of(json_arr_get(tcs, q));
                            if (tid && strcmp(tid, eid) == 0) { found = 1; break; }
                        }
                        if (!found) can_sort = 0;
                    }
                }
                /* every tool_call id truthy + unique */
                for (size_t a = 0; a < nt && can_sort; a++) {
                    const char *ida = id_of(json_arr_get(tcs, a));
                    if (!ida) { can_sort = 0; break; }
                    for (size_t b = a + 1; b < nt && can_sort; b++) {
                        const char *idb = id_of(json_arr_get(tcs, b));
                        if (idb && strcmp(idb, ida) == 0) can_sort = 0;
                    }
                }
            }

            if (can_sort) {
                size_t nt = json_arr_len(tcs);
                for (size_t q = 0; q < nt; q++) {
                    const char *tc_id = id_of(json_arr_get(tcs, q));
                    for (size_t k = i; k <= bend; k++) {
                        const JVal *m = json_arr_get(messages, k);
                        if (is_list_of_outputs(m)) {
                            const JVal *c = json_obj_get((JVal *)m, "content");
                            size_t ne = json_arr_len(c);
                            for (size_t e = 0; e < ne; e++) {
                                const JVal *entry = json_arr_get(c, e);
                                const char *eid = id_of(entry);
                                if (eid && tc_id && strcmp(eid, tc_id) == 0) {
                                    if (sb_entry_output(&out, entry, tools) < 0)
                                        goto fail;
                                }
                            }
                        } else {
                            const char *mid = id_of(m);
                            if (mid && tc_id && strcmp(mid, tc_id) == 0) {
                                if (sb_render_tool_response(&out, m, tools) < 0)
                                    goto fail;
                            }
                        }
                    }
                }
            } else {
                for (size_t k = i; k <= bend; k++) {
                    if (sb_render_tool_response(&out, json_arr_get(messages, k),
                                                tools) < 0)
                        goto fail;
                }
            }
        } else if (strcmp(role, "system") == 0) {
            sb_puts(&out, "<|system|>");
            if (sb_visible_text(&out, json_obj_get((JVal *)msg, "content")) < 0)
                goto fail;
        }
        /* unknown roles: silently skipped (template has no else branch) */
    }

    if (add_gen)
        sb_puts(&out, "<|assistant|><think>");

    return sb_steal(&out);

fail:
    sb_free(&out);
    return NULL;
}

uint32_t *glm_encode_ids(const Tok *t, const JVal *messages, const JVal *tools,
                         const GlmEncOpts *opts, size_t *n_out) {
    char *prompt = glm_encode_messages(messages, tools, opts);
    if (!prompt) return NULL;
    uint32_t *ids = tok_encode_str(t, prompt, 1, n_out);
    free(prompt);
    return ids;
}

#endif /* APUS_ENCODING_IMPL_DONE */
