# sleep-exp probe evaluator

`eval_probe.py` — stdlib-only at import; `torch`/`transformers` (and optional `peft`) are imported only inside `run_model()`.

## Data files (default dir `data/`, override `--data-dir`)
All JSONL, one object per line, unique string `id` per file.

**forgetting_probe.jsonl** — 100 items, 25 in each named `bucket`: `arithmetic`, `factual_recall`, `instruction_following`, `reasoning`. IDs are `fp-001` through `fp-100`.
All fields are strings: `{"id","bucket","prompt","answer","match"}`
- `match`: `"exact"` | `"numeric"` | `"regex"`
- `exact`: `answer` is the expected string
- `numeric`: `answer` is a numeric string
- `regex`: `answer` is a regex pattern string (must compile; matched with `re.fullmatch` against the *normalized* response, so write lowercase patterns)

**facts.jsonl** — 50 facts (10 revisions), IDs `fact-001` through `fact-050`.
`{"id","subject","statement","is_revision","supersedes"}`
- `is_revision`: boolean (`true` for revisions, `false` otherwise)
- `supersedes`: `null` if `is_revision` is `false`, or a valid non-revision fact `id` with the same subject if `is_revision` is `true`; the key is required even when null

**heldout_questions.jsonl** — 40 questions, IDs `q-001` through `q-040`.
`{"id","fact_id","question","answer"}`
- `fact_id` must exist in `facts.jsonl`.

## Prompt template (fixed)
```
Question: {prompt}
Answer:
```
This is raw text, **not** the model's chat template; instruction/chat models may respond differently. Greedy decoding (`do_sample=False`), batch size 1, explicit `attention_mask`, pad = EOS if missing. CUDA if available else CPU (tiny GPT-2 works on CPU).

## Execution
```
python eval_probe.py --dry-run                      # validate only, no model
python eval_probe.py --model gpt2 [--adapter DIR] [--limit N] [--seed S] \
    [--max-new-tokens 32] [--results-json PATH]
```
CPU smoke command (uses a locally available model; no download is required if the path is local):
```
CUDA_VISIBLE_DEVICES="" python eval_probe.py --model /path/to/local/model --limit 2 --results-json /tmp/sleep-exp-smoke.json
```

`--limit N` takes a seeded random subset. `--model` is required unless `--dry-run`. Exit code 1 on validation failure (evaluation never runs on invalid data). Missing `torch`/`transformers` exits with a clear error message.

## Grading
Normalization (`exact`/`regex`): lowercase, collapse whitespace, strip, remove trailing periods.
Grading is strict after normalization: explanatory text, extra punctuation (other than trailing periods), and synonyms do not match. Numeric: the **entire** response (trimmed, trailing period removed, thousands commas allowed; `%` is not accepted as bare numeric) must be a single number; otherwise `unparseable`. Correct if `|x-a| <= max(1e-9, 1e-6*|a|)`. Unparseable is reported separately from wrong. Verdicts: `correct | wrong | unparseable`.

## Results JSON
`metadata` (UTC timestamp, seed, model, adapter, device, prompt_template, settings), `summary` (overall + per bucket: correct, wrong, unparseable, total, `accuracy_all` = correct/total, `parsed_total`, `accuracy_parsed` = correct/(total−unparseable)), and per-item `items` with response and verdict.

## Freeze (before any training)
```
sha256sum eval_probe.py data/forgetting_probe.jsonl data/facts.jsonl data/heldout_questions.jsonl > FREEZE.sha256
sha256sum -c FREEZE.sha256     # verify before every evaluation
```
Commit the hashes before training. **Never train on forgetting_probe.jsonl** (nor on probe prompts/answers); any edit after freezing invalidates results.
