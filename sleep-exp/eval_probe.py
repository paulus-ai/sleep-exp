#!/usr/bin/env python3
"""Frozen probe evaluator for the sleep experiment (stdlib only at import time).

transformers/torch/peft are imported lazily inside run_model() only.
See README.md for schemas, template, grading and freeze procedure.
"""
import argparse
import datetime
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
PROBE_FILE = "forgetting_probe.jsonl"
FACTS_FILE = "facts.jsonl"
HELDOUT_FILE = "heldout_questions.jsonl"
PROMPT_TEMPLATE = "Question: {prompt}\nAnswer:"  # FIXED; changing it invalidates the freeze
MATCH_MODES = ("exact", "numeric", "regex")
N_PROBE, N_PER_BUCKET = 100, 25
N_FACTS, N_REV = 50, 10
N_HELDOUT = 40
BUCKETS = ("arithmetic", "factual_recall", "instruction_following", "reasoning")
REL_TOL, ABS_TOL = 1e-6, 1e-9
NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------- grading
def norm_text(s):
    """Lowercase, collapse whitespace, strip, drop trailing periods."""
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    return s.rstrip(".").strip()


def parse_number(s):
    """Parse the WHOLE response as one number (after trim/trailing period).
    Allows thousands commas. Does not accept '%'. Returns float or None."""
    t = re.sub(r"\s+", " ", str(s)).strip().rstrip(".").strip()
    if not NUM_RE.fullmatch(t):
        return None
    try:
        return float(t.replace(",", ""))
    except ValueError:
        return None


def grade(item, response):
    """Return 'correct' | 'wrong' | 'unparseable' (unparseable only for numeric)."""
    m = item["match"]
    if m == "exact":
        return "correct" if norm_text(response) == norm_text(item["answer"]) else "wrong"
    if m == "numeric":
        v = parse_number(response)
        if v is None:
            return "unparseable"
        want = float(item["answer"].replace(",", ""))
        ok = abs(v - want) <= max(ABS_TOL, REL_TOL * abs(want))
        return "correct" if ok else "wrong"
    if m == "regex":
        # applied to the normalized response (lowercased); patterns should be written lowercase
        return "correct" if re.fullmatch(item["answer"], norm_text(response)) else "wrong"
    raise ValueError("unknown match mode " + repr(m))


# ---------------------------------------------------------------- data
def load_jsonl(path):
    rows, errs = [], []
    try:
        fh = open(path, encoding="utf-8")
    except OSError as e:
        return [], ["cannot open %s: %s" % (path, e)]
    try:
      with fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, RecursionError, ValueError) as e:
                errs.append("%s:%d invalid JSON: %s" % (os.path.basename(path), n, e))
                continue
            if not isinstance(obj, dict):
                errs.append("%s:%d not an object" % (os.path.basename(path), n))
                continue
            rows.append(obj)
    except (UnicodeError, OSError) as e:
        errs.append("%s: cannot read: %s" % (path, e))
    return rows, errs


def _check_ids(rows, name, errs, prefix, count):
    seen = set()
    for i, r in enumerate(rows):
        rid = r.get("id")
        if not isinstance(rid, str) or not rid:
            errs.append("%s row %d: missing/non-string id" % (name, i + 1))
        elif rid in seen:
            errs.append("%s: duplicate id %s" % (name, rid))
        else:
            seen.add(rid)
    expected = {"%s-%03d" % (prefix, i) for i in range(1, count + 1)}
    missing, unexpected = expected - seen, seen - expected
    if missing:
        errs.append("%s: missing ids: %s" % (name, ", ".join(sorted(missing))))
    if unexpected:
        errs.append("%s: unexpected ids: %s" % (name, ", ".join(sorted(unexpected))))
    return seen


def _req_str(r, key, name, errs):
    if not isinstance(r.get(key), str) or not r[key].strip():
        errs.append("%s %s: field '%s' must be non-empty string" % (name, r.get("id"), key))


def validate_probe(rows, errs):
    _check_ids(rows, PROBE_FILE, errs, "fp", N_PROBE)
    if len(rows) != N_PROBE:
        errs.append("%s: expected %d items, got %d" % (PROBE_FILE, N_PROBE, len(rows)))
    buckets = {}
    for r in rows:
        _req_str(r, "bucket", PROBE_FILE, errs)
        _req_str(r, "prompt", PROBE_FILE, errs)
        _req_str(r, "answer", PROBE_FILE, errs)
        b = r.get("bucket")
        if isinstance(b, str) and b.strip():
            buckets[b] = buckets.get(b, 0) + 1
        m = r.get("match")
        if m not in MATCH_MODES:
            errs.append("%s %s: match must be one of %s" % (PROBE_FILE, r.get("id"), MATCH_MODES))
        elif m == "numeric" and isinstance(r.get("answer"), str):
            if parse_number(r["answer"]) is None:
                errs.append("%s %s: numeric answer %r is not a valid number" % (PROBE_FILE, r.get("id"), r["answer"]))
        elif m == "regex" and isinstance(r.get("answer"), str):
            try:
                re.compile(r["answer"])
            except (re.error, OverflowError, RecursionError) as e:
                errs.append("%s %s: regex does not compile: %s" % (PROBE_FILE, r.get("id"), e))
    for b in sorted(set(buckets) - set(BUCKETS)):
        errs.append("%s: unexpected bucket %r" % (PROBE_FILE, b))
    for b in BUCKETS:
        cnt = buckets.get(b, 0)
        if cnt != N_PER_BUCKET:
            errs.append("%s: bucket %r has %d items, expected %d" % (PROBE_FILE, b, cnt, N_PER_BUCKET))
    return buckets


def validate_facts(rows, errs):
    ids = _check_ids(rows, FACTS_FILE, errs, "fact", N_FACTS)
    if len(rows) != N_FACTS:
        errs.append("%s: expected %d facts, got %d" % (FACTS_FILE, N_FACTS, len(rows)))
    by_id = {r["id"]: r for r in rows if isinstance(r.get("id"), str)}
    n_rev = 0
    for r in rows:
        _req_str(r, "subject", FACTS_FILE, errs)
        _req_str(r, "statement", FACTS_FILE, errs)
        is_rev = r.get("is_revision")
        if not isinstance(is_rev, bool):
            errs.append("%s %s: 'is_revision' must be a boolean" % (FACTS_FILE, r.get("id")))
            continue
        sup = r.get("supersedes")
        if "supersedes" not in r:
            errs.append("%s %s: missing required field supersedes" % (FACTS_FILE, r.get("id")))
        if is_rev:
            n_rev += 1
            if not isinstance(sup, str) or sup not in ids:
                errs.append("%s %s: 'supersedes' %r is not a valid fact id" % (FACTS_FILE, r.get("id"), sup))
            elif sup == r.get("id"):
                errs.append("%s %s: supersedes itself" % (FACTS_FILE, r.get("id")))
            else:
                target = by_id[sup]
                if target.get("is_revision") is not False:
                    errs.append("%s %s: supersedes must reference a non-revision fact" % (FACTS_FILE, r.get("id")))
                if not isinstance(r.get("subject"), str) or r.get("subject") != target.get("subject"):
                    errs.append("%s %s: supersedes must reference the same subject" % (FACTS_FILE, r.get("id")))
        elif sup is not None:
            errs.append("%s %s: non-revision fact must have 'supersedes': null" % (FACTS_FILE, r.get("id")))
    if n_rev != N_REV:
        errs.append("%s: expected %d revisions, got %d" % (FACTS_FILE, N_REV, n_rev))
    return ids


def validate_heldout(rows, fact_ids, errs):
    _check_ids(rows, HELDOUT_FILE, errs, "q", N_HELDOUT)
    if len(rows) != N_HELDOUT:
        errs.append("%s: expected %d questions, got %d" % (HELDOUT_FILE, N_HELDOUT, len(rows)))
    for r in rows:
        _req_str(r, "question", HELDOUT_FILE, errs)
        _req_str(r, "answer", HELDOUT_FILE, errs)
        fid = r.get("fact_id")
        if not isinstance(fid, str) or fid not in fact_ids:
            errs.append("%s %s: fact_id %r not in facts" % (HELDOUT_FILE, r.get("id"), fid))


def validate_all(data_dir):
    errs = []
    probe, e1 = load_jsonl(os.path.join(data_dir, PROBE_FILE))
    facts, e2 = load_jsonl(os.path.join(data_dir, FACTS_FILE))
    heldout, e3 = load_jsonl(os.path.join(data_dir, HELDOUT_FILE))
    errs += e1 + e2 + e3
    fact_ids = validate_facts(facts, errs)
    buckets = validate_probe(probe, errs)
    validate_heldout(heldout, fact_ids, errs)
    return probe, errs, buckets, (len(probe), len(facts), len(heldout))


# ---------------------------------------------------------------- run
def select_items(items, limit, seed):
    items = list(items)
    if limit is not None and limit < len(items):
        rng = random.Random(seed)
        picked = set(rng.sample(range(len(items)), limit))
        items = [it for i, it in enumerate(items) if i in picked]
    return items


def run_model(args, items):
    """Real inference. Imports heavy deps here only."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        sys.exit("Missing required package for inference (%s). Install torch and transformers." % e)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError:
            sys.exit("--adapter given but peft is not installed")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device).eval()
    outs = []
    for it in items:
        text = PROMPT_TEMPLATE.format(prompt=it["prompt"])
        enc = tok(text, return_tensors="pt")  # single sequence: mask is all ones, passed explicitly
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id)
        resp = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        outs.append(resp)
    return outs, device


def summarize(items, results):
    def blank():
        return {"correct": 0, "wrong": 0, "unparseable": 0, "total": 0}
    per, overall = {}, blank()
    for it, r in zip(items, results):
        for d in (per.setdefault(it["bucket"], blank()), overall):
            d[r["verdict"]] += 1
            d["total"] += 1
    def fin(d):
        d["accuracy_all"] = d["correct"] / d["total"] if d["total"] else None  # unparseable counted in denominator
        parsed = d["total"] - d["unparseable"]
        d["parsed_total"] = parsed
        d["accuracy_parsed"] = d["correct"] / parsed if parsed else None  # unparseable excluded
        return d
    return {"overall": fin(overall), "buckets": {b: fin(d) for b, d in sorted(per.items())}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="HF model id/path (required unless --dry-run)")
    ap.add_argument("--adapter", help="optional PEFT adapter path")
    ap.add_argument("--limit", type=int, help="evaluate a seeded random subset of N items")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true", help="validate datasets only; no model")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--results-json", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")
    if not args.dry_run and not args.model:
        ap.error("--model is required unless --dry-run")

    probe, errs, buckets, counts = validate_all(args.data_dir)
    if errs:
        print("VALIDATION FAILED (%d errors):" % len(errs), file=sys.stderr)
        for e in errs[:50]:
            print("  - " + e, file=sys.stderr)
        return 1
    print("OK: probe=%d (buckets=%d) facts=%d heldout=%d" % (counts[0], len(buckets), counts[1], counts[2]))
    if args.dry_run:
        return 0

    items = select_items(probe, args.limit, args.seed)
    responses, device = run_model(args, items)
    per_item = []
    for it, resp in zip(items, responses):
        per_item.append({"id": it["id"], "bucket": it["bucket"], "match": it["match"],
                         "response": resp, "verdict": grade(it, resp)})
    out = {
        "metadata": {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seed": args.seed, "model": args.model, "adapter": args.adapter,
            "device": device, "prompt_template": PROMPT_TEMPLATE,
            "settings": {"limit": args.limit, "max_new_tokens": args.max_new_tokens,
                         "do_sample": False, "numeric_rel_tol": REL_TOL,
                         "numeric_abs_tol": ABS_TOL, "items_evaluated": len(items),
                         "items_in_probe": len(probe)},
        },
        "summary": summarize(items, per_item),
        "items": per_item,
    }
    with open(args.results_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    o = out["summary"]["overall"]
    print("overall: %d/%d correct, %d wrong, %d unparseable -> %s" % (
        o["correct"], o["total"], o["wrong"], o["unparseable"], args.results_json))
    for b, d in out["summary"]["buckets"].items():
        print("  %s: %d/%d correct (%d wrong, %d unparseable)" % (
            b, d["correct"], d["total"], d["wrong"], d["unparseable"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
