# Instructions for Claude Code

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
