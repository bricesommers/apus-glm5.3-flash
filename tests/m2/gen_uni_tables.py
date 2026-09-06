#!/usr/bin/env python3
"""Generate c/uni_tables.h: Unicode class range tables for the apus tokenizer.

The classes are DERIVED FROM THE TOKENIZER'S OWN PRE-TOKENIZER BEHAVIOR
(reference/tokenizer.json via the HF `tokenizers` lib), not from unicodedata,
so they exactly match whatever Unicode version the tokenizers regex engine
uses (no Unicode-version drift).

The GLM-5.3-Flash pre-tokenizer is a single Split with the Qwen2-style regex

  (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3}
  | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+

so the only classes the C port needs are:
  L  -> \p{L}   (letters; CJK included — there is no CJK stage here)
  N  -> \p{N}   (numbers; digit runs chopped to <=3)
  WS -> regex \s (White_Space)
everything else falls into the "symbol" class [^\s\p{L}\p{N}] (punctuation,
symbols, controls, unassigned — all behave identically in this regex).

Probe structure per codepoint cp (>= 0x80, outside surrogates). Each probe is
wrapped as "7" + body + "b": the leading ASCII digit and trailing ASCII
letter make probe boundaries uncrossable (a digit can never join a letter
run, a letter never a digit run, and neither joins a symbol/whitespace run
across the boundary), so pieces never span two probes. Probes are
concatenated and classified from the pieces' char offsets:

  g1 = "7" + cp*4 + "b"   N  <=> first piece covers "7"+cp+cp (3 chars:
                                  the digit run chunks as 7cc | cc | b;
                                  for non-N the leading 7 is a lone chunk)
  g2 = "7a" + cp + "bb"   L  <=> exactly 2 pieces ("7" and "a"+cp+"bb";
                                  non-L breaks the letter run: N->4, else 3)
  g3 = "7" + cp + " " + cp + "b"
                          WS <=> exactly 3 pieces for non-L non-N
                                (A6 gives cp+" ", then " ?"+cp+"b" merges via
                                the [^\r\n\p{L}\p{N}]?\p{L}+ prefix rule);
                                symbols give 4 (cp, " "+cp, "b" split by A4)

ASCII (< 0x80) is hardcoded in tok.h. Surrogates cannot appear in UTF-8 and
are left unclassified (the C side treats lone invalid bytes as symbols).
"""

import sys
from tokenizers import Tokenizer

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
TOK_PATH = ROOT + "/reference/tokenizer.json"
OUT_PATH = ROOT + "/c/uni_tables.h"

CLS_OTHER, CLS_N, CLS_L, CLS_WS = 0, 1, 2, 3
NAMES = {CLS_N: "N", CLS_L: "L", CLS_WS: "WS"}


def main():
    tok = Tokenizer.from_file(TOK_PATH)
    pt = tok.pre_tokenizer

    cls = {}
    cps = [cp for cp in range(0x80, 0x110000) if not (0xD800 <= cp <= 0xDFFF)]
    BATCH = 4096
    done = 0
    for i in range(0, len(cps), BATCH):
        batch = cps[i:i + BATCH]
        probes = []
        for cp in batch:
            c = chr(cp)
            probes.append("7" + c * 4 + "b")     # g1: N probe
            probes.append("7a" + c + "bb")       # g2: L probe
            probes.append("7" + c + " " + c + "b")  # g3: WS probe
        big = "".join(probes)
        spans = []  # (start, end) char span of each probe in `big`
        pos = 0
        for p in probes:
            spans.append((pos, pos + len(p)))
            pos += len(p)
        pieces = pt.pre_tokenize_str(big)
        # assign pieces to probes by offset containment; the wrapper must
        # make boundaries uncrossable
        per = [[] for _ in probes]
        k = 0  # probes are in order; advance as piece start passes span end
        covered = 0
        for s, off in pieces:
            lo, hi = off
            if lo != covered:
                raise RuntimeError(f"gap in piece coverage at char {covered}")
            covered = hi
            while k < len(spans) and lo >= spans[k][1]:
                k += 1
            if k >= len(spans) or lo < spans[k][0] or hi > spans[k][1]:
                raise RuntimeError(
                    f"piece {s!r}@{off} crosses probe boundary near {spans[min(k, len(spans)-1)]}")
            per[k].append((lo, hi))
        if covered != len(big):
            raise RuntimeError("pieces do not cover the whole probe string")
        for j, cp in enumerate(batch):
            g1, g2, g3 = per[3 * j:3 * j + 3]
            s1 = spans[3 * j][0]
            # g1: N <=> first piece covers "7"+cp+cp
            if g1[0] == (s1, s1 + 3):
                c = CLS_N
            elif g1[0] != (s1, s1 + 1):
                raise RuntimeError(f"bad g1 for U+{cp:04X}: {g1}")
            elif len(g2) == 2:
                c = CLS_L
            elif len(g2) not in (3, 4):
                raise RuntimeError(f"bad g2 for U+{cp:04X}: {len(g2)}")
            elif len(g3) == 3:
                c = CLS_WS
            elif len(g3) == 4:
                c = CLS_OTHER
            else:
                raise RuntimeError(f"bad g3 for U+{cp:04X}: {len(g3)}")
            # cross-consistency: N must give 4 g3 pieces starting with the
            # "7N" digit chunk ("7N", " ", "N", "b" — the A2 letter prefix
            # cannot absorb N), L must give 3 g3 pieces ("7", "L", " Lb")
            s3 = spans[3 * j + 2][0]
            if c == CLS_N and not (len(g3) == 4 and g3[0] == (s3, s3 + 2)):
                raise RuntimeError(f"inconsistent N probe for U+{cp:04X}: g3={g3}")
            if c == CLS_L and len(g3) != 3:
                raise RuntimeError(f"inconsistent L probe for U+{cp:04X}: g3={len(g3)}")
            cls[cp] = c
        done += len(batch)
        print(f"\r{done}/{len(cps)}", end="", flush=True)
    print()

    # build ranges per class (sorted by cp; surrogates stay unassigned -> OTHER)
    out = {CLS_N: [], CLS_L: [], CLS_WS: []}
    sorted_cps = sorted(cls)
    for target in (CLS_N, CLS_L, CLS_WS):
        start = None
        prev = None
        for cp in sorted_cps:
            if cls[cp] == target:
                if start is None:
                    start = cp
                prev = cp
            else:
                if start is not None:
                    out[target].append((start, prev))
                    start = None
        if start is not None:
            out[target].append((start, prev))

    with open(OUT_PATH, "w") as f:
        f.write("/* Generated by tests/m2/gen_uni_tables.py — do not edit.\n")
        f.write("   Unicode class ranges derived from the reference tokenizer's own\n")
        f.write("   pre-tokenizer behavior (reference/tokenizer.json, GLM-5.3-Flash). */\n")
        f.write("#ifndef APUS_UNI_TABLES_H\n#define APUS_UNI_TABLES_H\n\n")
        f.write("#include <stdint.h>\n\n")
        for target in (CLS_N, CLS_L, CLS_WS):
            name = NAMES[target]
            rng = out[target]
            f.write(f"static const uint32_t uni_{name.lower()}_ranges[][2] = {{\n")
            for lo, hi in rng:
                f.write(f"    {{0x{lo:06X}u, 0x{hi:06X}u}},\n")
            f.write("};\n")
            f.write(f"#define UNI_{name}_NRANGES {len(rng)}u\n\n")
        f.write("#endif\n")

    for target in (CLS_N, CLS_L, CLS_WS):
        rng = out[target]
        total = sum(hi - lo + 1 for lo, hi in rng)
        print(f"{NAMES[target]}: {len(rng)} ranges, {total} codepoints")
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
