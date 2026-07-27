Prompts moved here (2026-07-25) were written for a workflow this project no longer uses: a single stage-1 call against a combined header-row + full-data-row image, describing multiple numbered census columns at once (e.g. "Only extract column 3, 4, 4a, 4b, 4c, 11, 12, 14 and 15").

Stage 1 has run per-FIELD since 2026-07-24 (one tightly-cropped, upscaled image per selected column, per row - see core/row_extraction.py's run_two_stage_extraction docstring) - these prompts no longer match what the model is actually shown and are not wired to anything active.

Kept rather than deleted because ocr_stage1_field_aware_v3.txt in particular carries real per-column domain knowledge (DLS grid address parsing, ditto-mark handling for Name, M/F-only constraint for Sex) that would be worth reusing if this project ever splits it into proper per-field prompts alongside the field_hint metadata used by structuring_stage2_*.txt.
