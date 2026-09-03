# Document taxonomy (2026-08-04)

**What changed and why**: `debug_tools/review_uncertain.py` used to
hardcode its assignable-bucket list (`ASSIGNABLE_BUCKETS`) and button
colours (`BUCKET_COLOURS`) directly in the file. When `core/schema.py`'s
`DocumentCategory` gained `WEBSITE_SCREENSHOT` (2026-08-01), nobody
updated that list - the review UI had no way to assign the one bucket
that had just been added specifically because reviewers kept hitting
images that didn't fit anything else. This refactor makes that class of
gap structurally impossible: the UI has no bucket list of its own
anymore, and a taxonomy/`DocumentCategory` mismatch now raises loudly at
load time instead of silently producing a stale button set.

## The pieces

```
config/taxonomy.yaml   <- the actual data (categories, subtypes, colours, ...)
        |
core/taxonomy.py       <- loads it, validates it against DocumentCategory
        |
core/review_modes.py   <- three modes (production/research/subtype), each
        |                  reading buttons FROM the taxonomy, never a
        |                  local list
        v
debug_tools/review_uncertain.py  <- the Tk shell. Builds buttons from
                                     whatever mode.get_options(row) returns.
                                     Contains no bucket names anywhere.
```

## Adding a new bucket - the whole procedure

1. Add a category entry to `config/taxonomy.yaml`:
   ```yaml
   - id: my_new_bucket
     display_name: My New Bucket
     enabled: true
     sort_order: 90
     description: "..."
     color: "#c0c0c0"
     supports_subtypes: false
     subtypes: []
   ```
2. Add `MY_NEW_BUCKET = "my_new_bucket"` to `core/schema.py`'s
   `DocumentCategory` enum (this part hasn't changed - it's still the
   real, pydantic-validated source of truth for what Gemma is allowed
   to output, and every bucket CSV/pipeline_db.py write still keys off
   it).

That's it. No change to `debug_tools/review_uncertain.py`,
`core/review_modes.py`, or anything else. The next time the review tool
runs (any of the three `--source` modes), `my_new_bucket` appears as a
button automatically - `core/taxonomy.py`'s `load_taxonomy()` reads the
new entry, `ReviewMode.get_options()` includes it in
`assignable_categories()`, and the UI's `_rebuild_option_buttons()`
renders whatever that list contains.

**If you forget step 2** (or step 1): `load_taxonomy()` raises a
`ValueError` naming exactly which id is missing from which side, the
moment anything tries to load the taxonomy - not a silently stale UI
discovered weeks later.

## The three modes

| Mode | `--source` value | Reads | On assign | On close |
|---|---|---|---|---|
| Production Review | `uncertain` (default) | `data/buckets/uncertain_review.csv` (a live queue) | Writes the real bucket CSV row, removes from queue, logs `reviewed_uncertain.csv`, syncs `pipeline_db.py` (real correction) | Rewrites the queue with whatever's left |
| Research Ground Truth | `misclassifications` | `data/misclassifications.csv` (a flat sample) | Fills in `correct_category` in place - no file move, no DB write | Rewrites the file with every row, labeled or not |
| Subtype Annotation | `subtype` | `data/subtype_review_queue.csv` (file_path, bucket, subtype) | Fills in `subtype` in place, using `core/taxonomy.py`'s `subtypes_for(row["bucket"])` for its button set - the ONE place options come from something other than `assignable_categories()` | Rewrites the file with every row |

All three share the exact same `ReviewApp` Tk shell - "modes change
behaviour, not layout." Adding Subtype Annotation required zero changes
to that shell; it required one new class in `core/review_modes.py`
(`SubtypeAnnotationMode`) implementing the same `ReviewMode` interface
the other two already implement. That's the concrete proof this
architecture supports a mode the UI doesn't need to change for.

Subtype Annotation is not yet fed by production - no code writes
`data/subtype_review_queue.csv` yet, since the second Gemma pass it
supports doesn't exist. The mode itself is fully functional today
(verified against synthetic input), waiting only for a queue-file
generator once that pass is built.

## Taxonomy metadata reference

Each category (and subtype) in `config/taxonomy.yaml` carries:

| Field | Used today by | Notes |
|---|---|---|
| `id` | Everything | Must exactly match a `DocumentCategory` value (or `uncertain_review`, the one disabled routing sentinel) |
| `display_name` | Button labels | |
| `enabled` | `assignable_categories()` | `false` = never shown as a button (only `uncertain_review` today) |
| `sort_order` | Button ordering | |
| `description` | Nothing yet | Reserved for a future tooltip/help panel |
| `color` | Button background | `null` falls back to the UI's default grey |
| `keyboard_shortcut` | Nothing yet | Reserved for a future fast-review key binding |
| `supports_subtypes` | `subtypes_for()` | Whether this category has a real subtype list below it |
| `subtypes` | Subtype Annotation mode | List of `{id, display_name, sort_order, description}` |

Unused fields are preserved, not stripped - the UI ignores metadata it
doesn't currently need rather than requiring the schema to shrink to
exactly what's used today.

## What this refactor deliberately did NOT touch

- `core/schema.py`'s `DocumentCategory` enum remains the real production
  validation boundary for Gemma's output - this taxonomy layer sits on
  top of it, consistency-checked against it, never replacing it.
- `ui/classifier_validation_ui.py` still has its own local bucket
  handling, untouched - it shares the `flagged_needs_new_bucket.csv` log
  with `core/review_modes.py` (same path, verified identical), but has
  not been migrated onto the taxonomy loader. A natural follow-up, not
  done here since it wasn't in scope for this task.
- Every existing production workflow (`--source uncertain`,
  `--source misclassifications`) behaves identically to before this
  refactor - verified directly against the real, in-progress
  `data/misclassifications.csv` and the real (empty) `uncertain_review.csv`,
  not just in isolated tests.
