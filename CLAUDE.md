# Instructions for Claude Code

## User Instructions

Do not frame large tasks as reasons to stop. Do not suggest wrapping up, deferring, or moving work to a later session unless the user asks to pause. If a task is substantial, state the scope/risks briefly, then continue methodically. Only stop automatically when a genuine safety, data-loss, architectural, or ambiguity threshold requires user approval.

Do not ask “want me to continue?” after completing an intermediate phase when the user has already given an ongoing implementation direction. Continue to the next logical step unless a real decision point requires input.

Commit working tools and validated changes when they reach a working state; don't hold commits hostage to a larger milestone. A tool that works today belongs in history today — waiting for "all the year templates to be finished first" left 2,795 lines of calibration UI untracked for a month (discovered 2026-09-02). Commit at proven checkpoints, with the evidence in the message.

## Check docs/CODE_MAP.md before re-deriving how something already works

Before exploring the codebase to find an existing helper, schema, or
pattern (e.g. "how do I crop a field image", "how is a model loaded",
"how is ground truth scored", "how do the tkinter tools in this repo
launch a subprocess") - check `docs/CODE_MAP.md` first. It's a
symbol-level index of reusable primitives, built specifically because
sessions were burning significant exploration effort (agent dispatches,
many file reads) rediscovering functions that already existed. It is
NOT a how-to-run guide (`README.md`/`PIPELINE_WORKFLOW.md` cover that)
or a directory layout (`README.md`'s "Project layout" covers that).

**Keep it current.** If a session does real exploration (not "I recall
roughly") to find how something already works, add or correct an entry
in `docs/CODE_MAP.md` before finishing - a stale entry is worse than no
entry, since it gets trusted without re-verification. See that file's
own header for the expected entry format.

## Check CUDA/GPU usage before editing code or launching an inference run

**Before** editing loader code (`core/loaders/*.py`, `core/row_extraction.py`,
`core/loaders/subprocess_loader_base.py`, etc.) **or** running/launching any
model inference (loader smoke tests, `run_row_extraction.py`,
`run_two_stage_extraction.py`, `model_assessment.py`, or anything that calls
`initialize_model_and_tokenizer()`), check whether a GPU process is already
running:

```bash
nvidia-smi
```

Look at the process list at the bottom of the output. If a real inference
process is already running (not just background driver/OS processes):

- **Don't launch a second inference run.** Running two models against the
  same GPU concurrently has already caused a real, confirmed failure in
  this project once (moondream2's process got stuck / produced garbage
  output when a second inference was started while the first was still
  running - see the README's moondream2 section for the incident this rule
  comes from).
- **Ask before editing loader code while a run is in progress**, since a
  mid-run edit to a loader file could affect that running process depending
  on what's still cached in memory/what gets re-imported, and there's no
  reason to take that risk without knowing first.

If nothing is running, proceed normally - this isn't a gate on every single
action, just the two categories above.

## Archive exclusion rule (Jon, 2026-09-02)

For any NEW inference, benchmark, extraction, sidecar generation, or
derived artifact, inputs must originate from one of the two designated
active project directories:

1. `J:\Genealogy\genealogy_pipeline\` (the repo)
2. `J:\Genealogy\genealogy_workspace\` (the workspace)

Anything resolved OUTSIDE those two folders is presumed historical/
archive material and MUST NOT be used unless the task explicitly
requests historical reproduction, comparison, recovery, or other
archived-artifact use. Additionally, the explicitly-archival subtrees
INSIDE the workspace carry the same presumption by name:
`genealogy_workspace\research\baselines\` (reference_pipeline_prerefactor,
reference_pipeline_v1-v4) and completed dated `research\experiments\`
artifacts. **Always report the resolved input paths before using an
archive source.**

Why this rule exists (2026-09-02, learned the expensive way): the
549-cell detector study unknowingly ran on prerefactor-archive sidecars
whose geometry predates the field-bbox rework (uniform band grid, every
adjacent row pair overlapping 10px, ~26% neighbor ink per crop on the
worst page) - a provenance investigation after the fact had to establish
that all its absolute accuracy/coverage numbers are lower bounds under
obsolete crop geometry, not representative of the current pipeline. See
docs/AUTO_ACCEPT_DETECTOR_STUDY.md's provenance addendum. Archives are
older than they look and are NOT representative of the current pipeline.
