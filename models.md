# Candidate model evaluation log

Tracks every model considered for the Stage 1/2 extraction role (alongside
pixtral_12b / moondream2) so we don't re-evaluate the same dead ends later
without remembering why. Add an entry whenever a new candidate is checked,
regardless of verdict. Most recent first.

Verdicts:
- **Loader built** - passed initial checks, a loader/config exists under
  `core/loaders/` + `config/models/`, not wired into `config/pipeline.yaml`
  (prep only - see each model's own config comment for smoke-test results).
- **Not viable** - hard blocker found, no loader built.
- **Research only, not viable** / **Research only, viable with modification** -
  evaluated by research alone (no download/load attempt), see reasoning below.

**Two-tier evaluation methodology (added 2026-07-28):** entries in this log
distinguish two different, deliberately different-purpose evaluation
stages. Don't conflate what a Candidate Qualification entry says with what
only a Production Evaluation can actually establish.

- **Candidate Qualification (smoke test)** - what most entries below are,
  unless stated otherwise. Purpose: verify the model loads correctly,
  measure VRAM/load time, confirm basic prompt-following, detect
  catastrophic failures (crashes, garbage output, incompatible
  architecture), and estimate integration effort. Run against a whole
  document image with a document-level prompt for speed/simplicity. **Not
  intended to predict production OCR accuracy** - a model that looks bad
  here may still be a strong candidate once run through the real per-field
  pipeline, and a model that looks fine here has not yet been shown to
  perform well in production either way.
- **Production Evaluation (benchmark)** - the authoritative comparison
  between models: run through the complete real OCR pipeline
  (`core/row_extraction.py`) - same table detector, same row detector, same
  column detector, same per-field crops, same parser, same prompts, same
  ground truth - with only the OCR model swapped. No entry below has gone
  through this yet; it's the real next step before choosing between
  candidates, not this log.

Fabrication seen during a Candidate Qualification run (e.g. a "confident
invented list" at whole-page scale) reflects a known, already-established
overload effect for ANY model asked to read a whole dense page in one call
(see `core/row_extraction.py`'s own docstring: built after olmOCR fabricated
an entire household this exact way) - it answers "does this model have a
catastrophic failure mode," not "how accurate is this model." Write entries
below as "X happened during the whole-page smoke test," never as a bare
claim like "this model hallucinates" - those are different claims, and only
a Production Evaluation run can support the second one. See
[[feedback_whole_page_dumps_overload_models]] (memory) for the fuller
reasoning.

---

## nanonets/Nanonets-OCR2-3B
**Checked:** 2026-07-28 | **Verdict:** Loader built - `core/loaders/nanonets_ocr2_loader.py`, `config/models/nanonets_ocr2_3b.yaml`

Real, well-established release (807K downloads, 512 likes, published
benchmarks - DocVQA 89.4%, ChartQA 78.6%, olmOCR-bench 69.5%) - a fine-tune
of `Qwen/Qwen2.5-VL-3B-Instruct`, same architecture
(`Qwen2_5_VLForConditionalGeneration`, no custom code) already supported by
this project's existing `QwenLoader`. Looked at first as a pure config-only
reuse of that loader - turned out to need a small fix instead.

**Real, confirmed bug found and fixed, not worked around blindly:** the
official base model's `config.json` sets `tie_word_embeddings: true` at the
top level. Nanonets' fine-tuned re-export restructured the config into a
nested `text_config` sub-object and only set it there - this project's
installed transformers (5.12.1) doesn't look in that nested location for
this class, so the plain `QwenLoader` path loaded the model with a
**randomly initialized, untrained output layer** (confirmed via the load
report: `lm_head.weight | MISSING... newly initialized`) and produced
complete incoherent gibberish on every test image. Root cause confirmed by
diffing this checkpoint's config against the pristine base model's - not a
guess. Fix: load the config explicitly, force `tie_word_embeddings=True` at
the top level (the checkpoint's own `model.embed_tokens.weight` is
correctly trained; only the wiring to reuse it as the output layer was
broken), pass the corrected config into `from_pretrained()`. Implemented as
a thin `QwenLoader` subclass (`NanonetsOcr2Loader`) - `QwenLoader` itself is
untouched.

**With the fix applied, this produced the strongest raw output quality of
every candidate checked this session** (Candidate Qualification, whole-page
scale - see the methodology note above): a faithful, well-structured
markdown/HTML table reconstruction of a real 1920s Canadian Pacific
passenger manifest - correct ship name, date, port, company, and real
two-tier column header structure (rowspan/colspan matching the actual
historical form). Minor run-to-run variance observed in some fine details
(ship name/destination port differed slightly between two greedy-decoded
runs - worth noting, not fully deterministic despite `do_sample: false`).
Does NOT follow this project's kv-block extraction schema on the current
prompt - it outputs markdown tables instead, matching its own
pdf2markdown training focus - a prompt-adaptation question worth solving
given the underlying quality, not a reason to rule it out.

**License not stated anywhere** in the repo (no frontmatter tag, no LICENSE
file, no README mention) despite the popularity - the base model
(Qwen2.5-VL-3B-Instruct) is Apache 2.0, but Nanonets hasn't published their
own terms for the fine-tune. Flagged, not resolved - Jon's call before any
real use.

## ibm-granite/granite-vision-4.1-4b
**Checked:** 2026-07-28 | **Verdict:** Loader built - `core/loaders/granite_vision_4_1_loader.py`, `config/models/granite_vision_4_1_4b.yaml`

This exact model was already evaluated once before, on 2026-07-11, and
deliberately skipped for needing `trust_remote_code` (see the older
`granite_vision_loader.py`'s own docstring, which still references that
decision). Re-checked now on Jon's request because IBM's card claims native
`transformers>=5.8.0` support - confirmed true against this project's
current main venv (5.12.1): `AutoConfig.from_pretrained()` resolves
`Granite4VisionForConditionalGeneration` with zero flags, no custom code
executed. Whatever blocked it in July no longer applies - the project's own
transformers version moved past the point where it mattered. Runs fully
in-process, no separate venv. 4B params, ~7.6GB VRAM for weights, ~9.6GB
peak during generation - heaviest candidate this round, still comfortable
on the 16GB card. Apache 2.0.

**Candidate Qualification result** (see the methodology note at the top of
this file - this is a smoke test, not a Production Evaluation): loads
cleanly, no catastrophic failure, comfortable VRAM/load time - qualifies to
move forward. The notes below describe what happened during the whole-page
prompt run itself; they're evidence of behavior AT THAT SCALE, not a
verdict on real per-field accuracy, which requires a Production Evaluation
run through `core/row_extraction.py` (not done for this model yet). Also:
the JSON-wrapping noted below is not a real Production pipeline problem per
Jon - granite-vision-2b does the same thing and the real per-field parser
already handles it; only this file's own whole-page test harness (calling
`parse_kv_block()` directly) doesn't tolerate that shape.

On the printed manifest, it produced dense,
genuinely grounded content (real names/addresses/dates, corroborated by
other models' independent reads of the same image) but wrapped in broken
JSON syntax instead of the requested plain key:value format - a real but
minor problem. On the OTHER TWO samples (1911 census, handwritten
manifest), it did something much more serious: **fabricated an entire
39-person fictional family** (every invented name surnamed "Janssen") on
the census page, pairing each one with an increasingly elaborate, novel,
made-up excuse for why it "couldn't read" a name it had just confidently
produced - none of these excuse-phrases are this project's real confidence
vocabulary. On the handwritten manifest, it listed ~46 alphabetically-
patterned common surnames and **every country and Canadian province on
Earth** as "confirmed" `place_names` - not plausible content for any real
document. This is exactly the "confident invented list" pattern
`config/pipeline.yaml`'s own `no_ngram_overlap` anomaly check was built to
catch (its own comment cites InternVL3-8b generating alphabetically-ordered
surnames as the motivating example). The country-list sample happened to
use valid confidence vocabulary throughout and **passed schema validation
cleanly** at whole-page scale - if this same pattern showed up in a real
Production Evaluation run, it would be exactly the fabrication failure this
project's design philosophy treats as most dangerous (a false positive that
looks like a legitimate record), not a formatting nuisance. Candidate
Qualification is done and passed; flag this specific failure mode as the
first thing to check when this model gets a real Production Evaluation
pass.

## LiquidAI/LFM2-VL-1.6B
**Checked:** 2026-07-28 | **Verdict:** Loader built - `core/loaders/lfm2_vl_loader.py`, `config/models/lfm2_vl_1_6b.yaml`

Real, legitimate Liquid AI release. This project's own `base_loader.py`
docstring already name-dropped "LFM2.5-VL" as a planned-but-never-built
loader family - this fills that gap. Genuinely native `transformers`
architecture (`Lfm2VlForConditionalGeneration`, no `trust_remote_code`, no
custom code) and the only candidate this round that runs fully **in-process
in the main venv** - transformers 5.12.1 already satisfies its declared
`>=4.57` requirement, no separate isolated venv needed at all. 2.4B params
(1.2B LFM2 language backbone + SigLIP2 vision encoder), ~3GB VRAM for
weights, ~6.3GB peak during generation - very light.

Smoke test across 3 real images: **on 2 of 3, it echoed/mutated
`extractor_printed_v2.txt`'s own "Example of CORRECT output" block**
(Marriage Certificate/John Doe/Toronto/1942) instead of reading the actual
image - the *exact same failure mode* independently seen from
deepseek-vl2-tiny on this same prompt. Two different small models converging
on the identical template-echo behavior against one specific prompt is a
real signal worth taking seriously: it may be the prompt's explicit
contrastive example itself that's the problem for smaller models, not
either model individually. On the one image that didn't get contaminated
(printed manifest), a different but still real issue: correct document
type/date/column headers, then a confidently invented tail of fields that
don't exist on the document at all (Ticket Price, Boarding Time,
Confidentiality Notice, etc.).

## shubh303/smol-vlm-base-document-extraction
**Checked:** 2026-07-28 | **Verdict:** Not viable (research only, no download/load attempt)

Not a real fine-tune despite the name. `config.json`'s `_name_or_path` field
points straight at `HuggingFaceTB/SmolVLM-Base` - the stock upstream
checkpoint, unmodified as far as any evidence shows (this project's own
roster already has the newer SmolVLM2 via `smolvlm2_loader.py`/
`smolvlm2_2b.yaml`, so this repo isn't even offering a newer base). Model
card is entirely boilerplate - "[More Information Needed]" in every section,
the default `arxiv:1910.09700` citation placeholder never replaced with
anything real. 3 downloads, 0 likes, created and last-modified 90 seconds
apart on 2024-12-31 and never touched since. No dataset, no eval numbers, no
license stated. Nothing here suggests real document-extraction fine-tuning
happened - ruled out on the metadata alone, not worth a download/load test.

## tencent/HunyuanOCR
**Checked:** 2026-07-28 | **Verdict:** Loader built - `core/loaders/hunyuan_ocr_loader.py`, `config/models/hunyuan_ocr.yaml`

Real, native `transformers` architecture (`HunYuanVLForConditionalGeneration`,
transformers >=5.13.0) - no `trust_remote_code` needed, no custom `.py` files
in the repo at all. Dense 1.1B params, ~1.9GB VRAM for weights, ~4.2GB peak
during generation - lightest of everything tested. Isolated in
`.venv_hunyuan_ocr` purely because main env is pinned to transformers 5.12.1
for pixtral/gemma/qwen, not because of any compatibility problem.

Smoke test: strong, well-grounded OCR on printed/typed sources (real names,
addresses, correct census column headers). Ignores this project's structured
kv-block extraction prompt and free-form dumps transcribed text instead -
needs a transcription-oriented prompt or a two-pass structuring approach.
**Hallucinated a full fake person (name, dates, demographics) on the one
handwritten manifest sample tried** - confident fabrication, not an "unclear"
hedge. Needs more testing specifically on handwritten sources before trusting
it - open question for Jon's own pass.

## microsoft/Mage-VL
**Checked:** 2026-07-28 | **Verdict:** Not viable

Real Microsoft repo (3 days old at check time), 4.74B params, Qwen3-4B-Instruct
backbone + custom vision encoder, Apache 2.0. Genuinely requires
`trust_remote_code=True` (confirmed by direct test - transformers refuses to
load without it, unlike HunyuanOCR). Custom code itself loads fine under this
project's main transformers 5.12.1. **Blocked on `mamba_ssm`** (a hard,
unconditional import in `modeling_mage_vl.py` - it's a Mamba/state-space
streaming architecture, not a pure transformer). No prebuilt wheel exists for
`mamba_ssm` on any platform - it's source-only and compiles CUDA C++ kernels,
and this machine has **no CUDA Toolkit installed** (`nvcc` not found at all -
confirmed, not assumed). Fixing this needs a full CUDA Toolkit + Windows C++
build tooling install, a system-level change out of scope for a candidate-model
check. Also primarily a video-understanding model with image support added,
which may matter for fit even if the build issue were solved.

## deepseek-ai/deepseek-vl2-tiny
**Checked:** 2026-07-28 | **Verdict:** Loader built - `core/loaders/deepseek_vl2_loader.py`, `config/models/deepseek_vl2_tiny.yaml`

Real, actively-used DeepSeek release (not the debug/toy repos also checked
this round). 3B total / ~1B active MoE, ~6.5GB VRAM for weights, ~8.6GB peak
during generation. `transformers` 5.12.1 (main env) doesn't recognize its
`deepseek_vl_v2` architecture at all (`KeyError`), and the HF checkpoint repo
ships no custom `.py` files itself - the real load path is DeepSeek's own
un-pip-installable GitHub package (`deepseek_vl2`), cloned into
`.venv_deepseek_vl2/DeepSeek-VL2-src/`. Needed three separate workarounds,
all documented in the loader/worker docstrings: a pinned `transformers==4.38.2`
in an isolated venv, a `tokenizers` ABI/segfault fix (this Python 3.14
environment is newer than any tokenizers wheel that version range supports),
and an `xformers` bypass (no working CUDA backend exists yet for this GPU's
compute capability 12.0 - patched the vision attention to use native PyTorch
SDPA instead, verified math-equivalent for this checkpoint).

Smoke test: real image grounding confirmed (a plain descriptive prompt got an
accurate, detailed description of an actual passenger manifest). But on this
project's real `extractor_printed_v2.txt` prompt, it **echoed the prompt's own
"example of correct output" block verbatim**, 3/3 reproducible across varied
document types (census/printed/handwritten) - didn't perform real per-image
extraction at all with that prompt. Likely a capacity/prompt-complexity
mismatch (1B active params vs. an elaborate contrastive-example prompt), not
fundamental brokenness. Needs a simpler prompt variant tested before trusting
it on the real schema.

## amd-quark/Kimi-K2.5-2-layers-tiny
**Checked:** 2026-07-28 | **Verdict:** Not viable (research only)

Not a real usable model despite the misleading name. Confirmed via its own
model card: literally the first two transformer layers of the real (huge,
100B+ class) Kimi-K2.5, published by AMD Quark as an internal debug/test
fixture for their own quantization tooling - exists to validate tensor
shapes/dtypes load correctly, not to generate coherent output. Two layers
cannot produce usable text or image understanding under any circumstance.
Also steers away from standard `from_pretrained()` loading. No further
investigation warranted.

## Intel/Qwen3.6-35B-A3B-int2-mixed-AutoRound-LLMC
**Checked:** 2026-07-28 | **Verdict:** Not viable (research only)

Real repo, real base model (Qwen3.6-35B-A3B, genuine 35B-total/3B-active MoE
VLM), real INT2/INT4 mixed quantization via AutoRound, ~13.5GB on disk.
Two disqualifiers: (1) the model card's own recommended inference path is
vLLM - specifically a not-yet-merged PR branch of vLLM, not even mainline -
and this project has zero vLLM infrastructure anywhere, every loader here is
plain `transformers.from_pretrained`. (2) Even if that were solved, ~13.5GB
of weights on this machine's 16GB card leaves almost no headroom for KV
cache/other processes, tighter than every other model in this project's
roster. Not pursued further.

## yuanzhoulvpi/llava_qwen15-4b-chat_openai-clip-vit-large-patch14-336
**Checked:** 2026-07-28 | **Verdict:** Research only, viable with modification (not pursued)

Real, public repo, standard `LlavaForConditionalGeneration` + `AutoProcessor`
loading (no `trust_remote_code`) - architecturally the closest match of
anything checked to this project's existing `pixtral_loader.py` pattern, and
would quantize to a couple GB trivially. But it's a ~2-year-old (Sept 2024,
unmaintained) personal tutorial/portfolio project by an individual HF user,
built on Qwen1.5-4B-Chat (three generations behind current Qwen), trained
*only* on LLaVA-CC3M-Pretrain-595K - a small image-caption pretraining set,
not instruction-tuned for structured extraction. Expected to badly
underperform every model already in the roster on the real task. Lowest
priority of everything checked this round - not built, no strong reason to
revisit unless higher-priority candidates all fail out.
