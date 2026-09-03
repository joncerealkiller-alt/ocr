# Gemma Internal Instrumentation & Vision Sensor Expansion — Research Report

**Status: research only, no implementation, no prompt changes, no Decision
Engine design.** Written against the `automated-sidecar-generation` branch,
2026-08-04. Extends the existing sensor-layer framing
(`project_sensor_layer_architecture`, `project_stage_terminology_and_reference_v1`
memories) and builds directly on `docs/VISION_IR_RESEARCH.md` rather than
re-deriving what that document already settled — see the "What this document
does NOT re-derive" note at the end of §0.

---

## 0. Grounding: what was actually checked, and how

Every claim about Gemma's internals below was checked directly against the
**installed `transformers==5.12.1`** package on this machine
(`C:\Users\jonny\AppData\Roaming\Python\Python314\site-packages\transformers`),
specifically `transformers/models/gemma4/modeling_gemma4.py` and
`configuration_gemma4.py`, plus a live fetch of the actual
`google/gemma-4-E2B-it` `config.json` from Hugging Face — not inferred from
Gemma 3 documentation or general VLM knowledge. Where Gemma 3's architecture
differs (checked against `transformers/models/gemma3/modeling_gemma3.py`
directly, same install), that's called out explicitly in §1.8, because a
prior research pass in `VISION_IR_RESEARCH.md` (§13) flagged this exact
gap — "whether the vision tower and projector are still cleanly separable
the same way \[as Gemma 3] isn't confirmed by anything gathered here" — as
unresolved. It's resolved now, directly, by reading the real class.

This project's own loader code (`core/loaders/gemma_loader.py`) and its
comments were also verified against the installed package rather than
trusted at face value — in every case they turned out to be accurate
(the `vision_tower`/`embed_vision`/`audio_tower`/`embed_audio` module names
cited in that file's `llm_int8_skip_modules` comment are exactly the real
attribute names on `Gemma4Model`).

**A separately-run external research pass (Gemini) was supplied mid-session
and is addressed explicitly in §1.9**, since it disagrees with the verified
findings here on two concrete points (a module name, and an implicit
assumption about where pooling happens) and is a useful example of exactly
the kind of generic-VLM-knowledge error this section's "verify, don't
assume" discipline is meant to catch.

**What this document does NOT re-derive**: `docs/VISION_IR_RESEARCH.md`
already contains ~3,000 lines of prior research directly relevant to this
task — the projector/encoder coupling problem (§3), the case for treating
Gemma's own frozen features as a linear-probe input rather than replacing
the vision tower (§§12–16), the "route embeddings as text hints, not raw
tensors" finding (§18, Route 1), the machine-consumed-vs-text-hint
consumption-channel split (§28), the evidence-contract shape for CPU-side
analysis of frozen embeddings (§26), and a full experimental battery
qualifying DINOv2/ConvNeXt/SigLIP-NaFlex/EVA-02/BEiT/Swin/MAE as
*standalone, external* vision sensors (Experiments 001, 3, 3.1–3.3). None of
that is repeated here except where directly relevant to a fact-check. This
document's job is the three things that document's own §13/§16 identified
as still open: (1) confirm Gemma's actual internal module structure instead
of assuming Gemma 3's, (2) survey OCR/document-specific vision backbones —
a genuinely different candidate list from the general-purpose
DINOv2/ConvNeXt/SigLIP family already qualified — and (3) name the sensor
families still missing before Decision Engine work starts.

---

## Part 1 — Gemma Internal Instrumentation

### 1.1 Confirmed: which HF class, and the exact call path

`config/models/gemma.yaml`'s `repo_id: google/gemma-4-E2B-it` and
`config/models/gemma_e4b.yaml`'s own comment ("Verified before creating this
file... E2B and the base E4B checkpoint share the identical architecture
class `Gemma4ForConditionalGeneration`, model_type: `gemma4`") both check
out. Confirmed independently three ways:

1. `google/gemma-4-E2B-it`'s live `config.json` (fetched directly) reports
   vision config `hidden_size=768, num_hidden_layers=16, patch_size=16,
   intermediate_size=3072, pooling_kernel_size=3, soft tokens per image=280`
   and text config `hidden_size=1536, num_hidden_layers=35,
   vocab_size=262144` — these match `Gemma4VisionConfig`'s and
   `Gemma4TextConfig`'s dataclass field names in the installed
   `configuration_gemma4.py` exactly (not a different config schema).
2. `transformers/models/auto/modeling_auto.py`'s
   `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` (the table `AutoModelForCausalLM`
   consults) has the literal entry `("gemma4", "Gemma4ForConditionalGeneration")`
   at line 700 of the installed file — confirming
   `AutoModelForCausalLM.from_pretrained()`, exactly what `gemma_loader.py`
   calls, really does instantiate the multimodal
   `Gemma4ForConditionalGeneration` class for a `model_type: gemma4`
   checkpoint, not a text-only `Gemma4ForCausalLM` (a *separate* class,
   registered under `gemma4_text`, that also exists in this install and
   would NOT have a vision path at all — worth remembering as a real
   footgun if a future Gemma config's `model_type` ever drifts).
3. `gemma_loader.py`'s own `llm_int8_skip_modules=["vision_tower",
   "embed_vision", "audio_tower", "embed_audio", "lm_head"]` — module names
   that only resolve correctly if they're real dotted-path suffixes on the
   loaded model — match `Gemma4Model.__init__`'s actual attribute
   assignments exactly (§1.2). If this weren't true, 8-bit quantization
   would silently either error or quantize the wrong modules; it doesn't,
   so this is corroborating evidence the loader code and the installed
   package genuinely agree, not just two independent guesses.

**Call path, confirmed by reading the actual `forward()`/`generate()` code,
not assumed**: `GemmaLoader._run_generate()` calls
`self.model.generate(**inputs, **gen_kwargs)` (`gemma_loader.py:218`).
`Gemma4ForConditionalGeneration` inherits `GenerationMixin`, whose
`generate()` drives the standard HF autoregressive loop — each decode step
calls `prepare_inputs_for_generation()` then the model's own `forward()`
(`modeling_gemma4.py:2472`), which calls `self.model(...)`
(`Gemma4Model.forward`, not shown fully above but confirmed to be the
`Gemma4Model` instance at `self.model`), which internally calls
`self.vision_tower(...)` and `self.embed_vision(...)` **only when
`pixel_values` is not `None`**. Confirmed directly in
`Gemma4ForConditionalGeneration.prepare_inputs_for_generation()`
(`modeling_gemma4.py:2552–2594`):

```python
# If we're in cached decoding stage, multimodal inputs are already cached and can be dropped
if is_first_iteration or not use_cache:
    model_inputs["pixel_values"] = pixel_values
    ...
else:
    # Don't pass to not apply bidirectional mask on top
    model_inputs["mm_token_type_ids"] = None
```

`pixel_values` is only forwarded on `is_first_iteration` (or when
`use_cache=False`, which this pipeline never sets — `gen_kwargs` doesn't
touch `use_cache`, so it defaults on). Every subsequent decode step reuses
the KV cache and passes `pixel_values=None`. **This confirms the
operational note's claim exactly**: vision hooks fire once per
`classify()` call, on the first forward pass only, regardless of
`max_new_tokens=256`. No duplicate vision compute across the ~256-token
classification response, and — because `classify()` calls `.generate()`
directly rather than a bare `model(**inputs)` forward — there is no
separate "logits-only forward pass" happening anywhere in this pipeline's
actual classify path (relevant to §1.9's fact-check of the Gemini example,
which sketched hooking a bare `model(**inputs)` call).

`generate()` internally routes through exactly the same `forward()` method
a bare `model(**inputs)` call would use — HF's `GenerationMixin` doesn't
maintain a separate code path for the vision-encoding step, it's ordinary
submodule dispatch inside `Gemma4Model.forward()` either way. So the
distinction the mid-session Gemini note raised ("this repo calls
`generate()`, the example calls bare `forward()` — does that change
anything?") does **not** change hook-firing behavior; both routes reach the
identical `self.vision_tower(...)` / `self.embed_vision(...)` calls inside
`Gemma4Model.forward()`. What *does* change between the two routes is
timing/count, already covered above: `generate()` calls `forward()`
repeatedly (once per output token) but only the first call carries
`pixel_values`, where a single bare `model(**inputs)` call is inherently
one-shot. Net effect on hook design: identical either way — a forward hook
on `vision_tower`/`embed_vision` fires exactly once per image either way,
you just can't assume "hook fired" implies "this was the only forward call
of the whole `generate()` invocation" the way you could for a bare
`model()` call.

### 1.2 Module hierarchy — the real one, not Gemma 3's

Reading `Gemma4Model.__init__` (`modeling_gemma4.py:2123–2148`) directly:

```python
class Gemma4Model(Gemma4PreTrainedModel):
    def __init__(self, config: Gemma4Config):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config) ...
        self.language_model = AutoModel.from_config(config=config.text_config)
        self.audio_tower = AutoModel.from_config(config.audio_config) ...
        self.embed_vision = Gemma4MultimodalEmbedder(config.vision_config, config.text_config) ...
        self.embed_audio = Gemma4MultimodalEmbedder(config.audio_config, config.text_config) ...
```

And `Gemma4ForConditionalGeneration.__init__` (`modeling_gemma4.py:2450–2454`):

```python
class Gemma4ForConditionalGeneration(Gemma4PreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    def __init__(self, config: Gemma4Config):
        self.model = Gemma4Model(config)
        self.lm_head = nn.Linear(...)
```

**Full attribute path from `loader.model` (the object `GemmaLoader` actually
holds) down to each candidate observation point**:

```
loader.model                                         Gemma4ForConditionalGeneration
└── .model                                            Gemma4Model
    ├── .vision_tower                                 Gemma4VisionModel
    │   ├── .patch_embedder                           Gemma4VisionPatchEmbedder
    │   ├── .encoder                                  Gemma4VisionEncoder
    │   │   └── .layers[0..15]                        Gemma4VisionEncoderLayer × 16
    │   └── .pooler                                   Gemma4VisionPooler
    ├── .embed_vision                                 Gemma4MultimodalEmbedder
    │   ├── .embedding_pre_projection_norm             Gemma4RMSNorm
    │   └── .embedding_projection                      nn.Linear
    ├── .audio_tower                                   Gemma4AudioModel   (not relevant here)
    ├── .embed_audio                                    Gemma4MultimodalEmbedder  (not relevant here)
    └── .language_model                                 Gemma4TextModel  (35 decoder layers on E2B)
```

**No convenience top-level properties exist** — grepped `modeling_gemma4.py`
for `@property` near these names and found none, so `loader.model.vision_tower`
(missing the `.model.` indirection) raises `AttributeError`. This matters
concretely for §1.9's fact-check: it's the second place the generic example
needed correcting.

### 1.3 The three candidate observation points, mapped to real modules

Reading `Gemma4VisionModel.forward()` (`modeling_gemma4.py:2034–2075`) and
`Gemma4Model.get_image_features()` (`modeling_gemma4.py:2150–2167`) end to
end, the pipeline inside a single `classify()` call is:

```
pixel_values (patchified, pre-projected pixels)
  │
  ▼
Gemma4VisionPatchEmbedder            — linear projection of raw patches + 2D position embedding
  │  (inputs_embeds)
  ▼
Gemma4VisionEncoder                  ◄── CANDIDATE 1: "vision encoder output"
  │  16 transformer layers, self-attention over patches, no pooling yet
  │  (BaseModelOutputWithPast.last_hidden_state — shape (batch, num_patches, 768))
  ▼
Gemma4VisionPooler                   ◄── CANDIDATE 2: "spatial pooling / resampling output"
  │  averages patches in a 3×3 grid (pooling_kernel_size=3) down to a fixed
  │  soft-token count; float32 sqrt(hidden_size) scaling
  │  (shape (batch, num_soft_tokens, 768) — num_soft_tokens = num_patches // 9,
  │   e.g. 140 for this pipeline's image_token_budget=140)
  ▼
[back in Gemma4VisionModel.forward: optional standardize (mean/var norm),
 cast to working dtype — this is vision_tower's actual RETURNED value]
  ▼
Gemma4MultimodalEmbedder (= embed_vision)   ◄── CANDIDATE 3: "vision projection"
  │  RMSNorm (no learned scale) + linear projection 768 → 1536 (E2B's text
  │  hidden_size)
  │  (shape (batch, num_soft_tokens, 1536))
  ▼
merged into inputs_embeds at the image-token placeholder positions,
consumed by Gemma4TextModel (35 decoder layers)
```

**All three candidates ARE genuinely distinct, capturable, real submodule
boundaries on Gemma 4** — this is the key finding that resolves the open
question `VISION_IR_RESEARCH.md` §13 left unconfirmed. Unlike Gemma 3
(§1.8), pooling and projection are two *separate* modules here
(`Gemma4VisionPooler` inside `vision_tower`, `Gemma4MultimodalEmbedder` as
its own top-level child), so a forward hook on each module boundary cleanly
isolates one candidate without needing to hook an inner sub-call.

One nuance worth being precise about: `Gemma4VisionModel.forward()` calls
`self.pooler(...)` and then, if `config.standardize` is true, applies an
additional mean/variance normalization **after** the pooler returns, before
`vision_tower.forward()`'s own return statement. So there are actually two
slightly different tensors you could call "candidate 2":

- **hook on `vision_tower.pooler`** — captures the pooler's raw output,
  pre-standardization, in float32 (per the pooler's own docstring: "the
  scaling can push activations past the float16 range... returned in
  float32; the caller standardizes and casts back").
- **hook on `vision_tower` itself (the whole module)** — captures the final
  post-standardization, dtype-cast tensor, i.e. exactly what
  `get_image_features()` reads as `vision_outputs.last_hidden_state` and
  feeds into `embed_vision`.

Recommend capturing at the **`vision_tower`-level** hook for candidate 2 (not
`pooler`), specifically because it's the tensor that's actually consumed
downstream — hooking `pooler` gets you a pre-standardization variant that
no other part of the model ever sees in that form, which would misrepresent
"what Gemma's own decision path looked at" if used as an evidence signal.

### 1.4 Hook attachment — recommended mechanism and placement

Standard PyTorch forward hooks (`register_forward_hook`), attached once at
loader-init time (inside `initialize_model_and_tokenizer()`, right after
`self.model = model` is set — `gemma_loader.py:100`), not per-call. Three
hooks, one per candidate:

```python
hooks = {}
hooks["vision_encoder"] = model.model.vision_tower.encoder.register_forward_hook(
    lambda module, args, output: captured.update(vision_encoder=output.last_hidden_state.detach())
)
hooks["vision_pooled"] = model.model.vision_tower.register_forward_hook(
    lambda module, args, output: captured.update(vision_pooled=output.last_hidden_state.detach())
)
hooks["vision_projected"] = model.model.embed_vision.register_forward_hook(
    lambda module, args, output: captured.update(vision_projected=output.detach())
)
```

Design notes, not yet implemented:

- **`.detach()` in every hook body is load-bearing, not optional.** This
  pipeline runs entirely under `torch.inference_mode()`
  (`gemma_loader.py:217`), so gradients are already disabled globally and a
  hook can't accidentally enable backprop through a captured tensor — but
  `.detach()` still matters for a second reason: without it, the captured
  tensor keeps a live reference into the same autograd-graph-adjacent
  storage the forward pass is using, which can interact with CUDA memory
  reuse across calls in ways that are easy to get subtly wrong. Detaching
  to an independent tensor (and, per §1.6, moving to CPU before storing) is
  the safe default.
- **Where the captured dict lives**: NOT a module-level global (this
  pipeline processes one image per `classify()` call in a loop over the
  whole manifest — `core/classifier.py:223` — a global dict would leak
  across images unless explicitly cleared every iteration, which is a real
  footgun). Recommend an instance attribute on the loader
  (`self._captured_vision_tensors: dict`), reset at the *start* of
  `classify()` before calling `_run_generate()`, read immediately after
  `_run_generate()` returns, before the loop's next iteration begins.
- **Hooks are registered once per model load, not once per image** — this
  matches the existing load-once-release-fully lifecycle convention
  (`CODE_MAP.md`'s "Load-once-per-model lifecycle" section) exactly.
  `.release()` should call `.remove()` on all three handles alongside the
  existing model/processor/tokenizer teardown, so a released loader leaves
  no dangling hooks if the same Python process ever re-initializes a model
  later in the same run (it currently doesn't, per the existing
  load-once/release-once/load-next-stage discipline, but a stale hook
  surviving past `.release()` would be a real bug if that discipline ever
  changed).

### 1.5 Tensor shapes (grounded in this pipeline's actual config, not a generic example)

Using `config/models/gemma.yaml`'s real `image_token_budget: 140` and the
live-fetched `google/gemma-4-E2B-it` config values (§1.1):

| Candidate | Shape (batch=1) | Dtype | Approx. size |
|---|---|---|---|
| 1 — vision encoder output (`vision_tower.encoder`) | `(1, num_patches, 768)` | bf16 (model dtype) | num_patches is processor-determined, not directly `image_token_budget` — see caveat below. At `pooling_kernel_size=3`, `num_patches ≈ 140 × 9 = 1260` for this pipeline's budget → **(1, 1260, 768)**, ≈ 1.9 MB |
| 2 — spatial pooling output (`vision_tower`, post-standardize) | `(1, 140, 768)` | bf16 | ≈ 215 KB |
| 3 — vision projection (`embed_vision`) | `(1, 140, 1536)` | bf16 | ≈ 430 KB |

**Caveat, stated explicitly rather than glossed over**: `num_patches` (candidate
1's sequence length) is determined by the image processor's own patchify
logic in response to `image_seq_length=image_token_budget` passed at
`processor(...)` call time (`gemma_loader.py:184–189`), not read directly off
a config constant. The `1260 = 140 × 9` figure above follows deterministically
from `Gemma4VisionModel.forward()`'s own math
(`output_length = pixel_values.shape[-2] // (pooling_kernel_size**2)`, i.e.
`num_patches = output_length × pooling_kernel_size²`), so it should be
exactly right, but this was derived from reading the formula, not from a live
forward pass with real image dimensions dumped — **recommend confirming
with one live shape-print smoke test before writing any capture code**, the
same "Stage 0: interface inspection before extracting anything" discipline
`VISION_IR_RESEARCH.md` §13/§16 already established for this exact kind of
unknown.

All three are trivially small relative to the ~3.5–16 GB of resident model
weights (§1.6) — this is not a memory-constrained instrumentation, the
actual costs are elsewhere (§1.6).

**Correction, 2026-08-05**: the `num_patches ≈ 1260`/`(1, 140, 768)`
figures above assumed the pipeline's call-site actually applied
`image_token_budget=140`. `docs/GEMMA_TOKEN_BUDGET_INVESTIGATION.md`
found this call was silently ignored (wrong kwarg name for the installed
processor version) — the pipeline was actually running at the
processor's default 280 cap the whole time. That bug is now fixed, and
the config values were raised to 280 to match already-real behavior
(see that doc's "Fix applied" addendum) rather than forcing the
long-assumed-but-never-actually-applied 140 into effect. Live-verified
shapes for a real image: encoder `(1, 2520, 768)` (2520 = 280×9),
pooled/projected `(273, 768)`/`(273, 1536)` — matches the token-budget
doc's own Phase 3/4 derivation exactly, not the `1260`/`140` figures
this section originally predicted.

### 1.6 Expected memory and runtime overhead

**Runtime**: effectively zero additional compute. The hooks don't add any
forward-pass work — `vision_tower`, `vision_tower.encoder`, and
`embed_vision` already run as part of the existing classification call
(§1.1's whole point: zero duplicate vision computation, by construction of
where the hooks attach). The only added cost is the hook's own Python-level
overhead (`.detach()`, dict assignment, optionally `.cpu()`) — sub-millisecond
per call, unmeasurable against a ~256-token generation call that already
takes on the order of seconds.

**Memory, two very different components**:

- **Per-call tensor memory** (§1.5): ~2.5 MB total across all three
  candidates, per image. Negligible against resident model VRAM
  (`config/models/gemma_e4b.yaml`'s own comment: 16.02 GB bf16 for the full
  E4B checkpoint, 3.56 GB for the QAT-quantized variant currently in use).
- **Accumulation risk across a full-corpus run — this is the real cost to
  plan for.** This pipeline classifies on the order of 1,600–2,300 images
  per corpus run (per `docs/CODE_MAP.md`'s bucket-count findings: 385 + 12
  + 220 + 32 + 87 + 103 + 1430 + 16 ≈ 2,285 rows in the most recent full
  run). If every image's three captured tensors are **kept resident in GPU
  memory** across the whole run (e.g. appended to a list without moving to
  CPU), that's ~2,285 × 2.5 MB ≈ 5.7 GB of VRAM by the end of the run —
  not catastrophic on its own, but directly competing with the resident
  model weights on a card this project's own `debug_tools/workflow_gui.py`
  identifies as an RTX 5060 Ti (16 GB class), and stacking on top of
  whatever headroom `vram_headroom_gb: 1.5` was already budgeting for.
  **Recommend `.cpu()` (or immediate serialization to disk) inside the hook
  or immediately after `_run_generate()` returns, every single image** —
  never accumulate GPU-resident tensors across the classify loop. On CPU/disk,
  the same 5.7 GB for the full corpus is a complete non-issue (this
  project already accepts a 34.6 MB baseline-embeddings file for 1,677
  images across 8 encoders combined — three per-image tensors at a few
  hundred KB each is a comparable order of magnitude, not a new problem
  class).
- **Whether to persist candidate 1 (the largest, ~1.9 MB/image) at all is a
  real design decision, not just a storage-budget one.** Candidates 2 and 3
  are the two that most directly answer this task's actual research
  question (does Gemma's own pooled/projected representation separate the
  8 routing buckets — `VISION_IR_RESEARCH.md` §§12–16's whole premise).
  Candidate 1 (pre-pooling, per-patch) is the heaviest and the one whose
  practical use (spatial/patch-level analysis) was already flagged in that
  document (§30) as "a materially heavier-weight approach... closer to
  train-or-adapt-a-spatial-detector than run-cosine-similarity" — i.e., a
  separately-scoped follow-on, not part of the near-term ask. Recommend
  capturing candidate 1 only for a small diagnostic subset during Stage 0
  interface confirmation, not as a permanent per-image capture, until a
  concrete spatial-analysis use case is scoped.

### 1.7 Cleanup strategy

- **Hook handles**: `.remove()` every registered `RemovableHandle` inside
  `.release()`, alongside the existing `self.model = None` /
  `self.processor = None` / `gc.collect()` / `torch.cuda.empty_cache()`
  sequence `core/row_extraction.py`'s `_release_model()` already performs
  for every loader. `GemmaLoader` doesn't currently override `.release()`
  (`BaseLoader`'s default is a no-op per `CODE_MAP.md`) — adding hooks would
  be the first real reason for `GemmaLoader` to need its own `.release()`
  override, since hook handles are loader-instance state with no analog in
  any other loader today.
- **Captured-tensor dict**: reset (`self._captured_vision_tensors = {}`) at
  the start of every `classify()` call, not just at load/release time —
  otherwise a failed or skipped classify call could leave stale tensors from
  the *previous* image silently attached to the next image's result if a
  caller isn't careful about checking freshness.
- **No new subprocess/venv boundary concerns**: unlike Moondream2/DeepSeek-VL2/
  Hunyuan-OCR (which run in separate subprocess venvs per
  `core/loaders/subprocess_loader_base.py`), `GemmaLoader` runs in-process on
  the main interpreter, so captured tensors need no serialization across a
  process boundary to reach the caller — a plain Python dict return from
  `classify()` (or a new sidecar method) is sufficient.

### 1.8 Compatibility risks / model-version-specific differences

**Gemma 3 vs Gemma 4 — a real, load-bearing architectural difference, not
just a naming difference.** Read directly from the installed
`modeling_gemma3.py`:

- Gemma 3's projector, `Gemma3MultiModalProjector`
  (`modeling_gemma3.py:665–698`), does **both pooling and projection inside
  one `forward()` call** — `self.avg_pool` (an `nn.AvgPool2d`) runs first,
  then `self.mm_soft_emb_norm` (RMSNorm) and the projection matmul, all in
  the same method, on the same module.
- Gemma 3's `vision_tower` (a plain SigLIP-style encoder, no
  `Gemma4VisionPooler` equivalent) returns **raw, unpooled patch tokens
  directly** — `Gemma3Model.get_image_features()`
  (`modeling_gemma3.py:733–738`) passes `vision_tower`'s raw
  `last_hidden_state` straight into `multi_modal_projector`.

Consequence: **on Gemma 3, "candidate 2" (spatial pooling output) is not a
separate module boundary you can hook cleanly the way it is on Gemma 4.**
To isolate it on Gemma 3 you'd need to hook the `avg_pool` *submodule*
specifically (`model.model.multi_modal_projector.avg_pool`, which is a real,
independently-hookable `nn.AvgPool2d` instance — it exists, it's just nested
one level deeper than the natural "module boundary" a first read of the
architecture suggests), not the projector's input or output as a whole. This
is exactly the mistake in the Gemini research pasted mid-session — see
§1.9's item 3.

**Practical implication for this pipeline specifically**: since
`config/models/gemma.yaml` and `gemma_e4b.yaml` both target `model_type:
gemma4` checkpoints (confirmed, §1.1), the clean three-way module split in
§1.2/§1.3 applies as-is today. The Gemma-3-specific caveat above matters
only if this pipeline ever adds a Gemma 3 checkpoint as a comparison config
(plausible — the project already runs E2B-vs-E4B comparisons via
byte-for-byte-mirrored configs, per `gemma_e4b.yaml`'s own header) or if a
future Gemma release reverts to a Gemma-3-shaped architecture. **Recommend
the abstraction (§1.10) branch on `model.config.model_type` /
`type(model.model).__name__` at hook-registration time rather than assuming
one fixed module path** — this is the same "config owns behavior, don't
hardcode which variant you're talking to" principle `CODE_MAP.md`'s
architectural-principle section already establishes for this codebase, just
applied to hook placement instead of prompt/generation config.

**Two other real variants observed in the installed package, not yet
relevant but worth flagging**: `gemma4_unified` (`Gemma4UnifiedForConditionalGeneration`,
its own separate module directory in this install) and `gemma4_assistant`
(`Gemma4AssistantForCausalLM`) are both distinct `model_type` values
registered in the same Auto-mapping table as plain `gemma4`. Neither is
used by any config in `config/models/*.yaml` today, but a hook-registration
helper that hardcodes `Gemma4ForConditionalGeneration`-shaped module paths
would silently break (or silently attach to nothing) if a config ever
pointed at one of these instead — another argument for runtime introspection
over a hardcoded module path.

### 1.9 Fact-checking the mid-session Gemini research against the verified findings above

The pasted Gemini output is broadly right on the *mechanism* (forward hooks
on submodules fire during a normal `generate()`/`forward()` call, zero
duplicate compute, `.detach()` for safety) and wrong on specifics in ways
that matter for actually writing the code. Concretely, checked point by
point:

1. **`model.multi_modal_projector` doesn't exist on this project's real
   model object.** Confirmed no such attribute anywhere in
   `modeling_gemma4.py` (grepped for `multi_modal_projector` and
   `@property` — zero matches). This project's checkpoint uses `embed_vision`
   (`Gemma4MultimodalEmbedder`), a differently-named, differently-shaped
   module (§1.2/§1.8). Even restricted to a genuine Gemma 3 checkpoint (where
   `multi_modal_projector` *is* the real name), the attribute still needs the
   `.model.` indirection the example omits (`model.model.multi_modal_projector`,
   not `model.multi_modal_projector` — confirmed, §1.2's "no convenience
   top-level properties" finding applies identically to Gemma 3's class
   shape).
2. **The example's `hook_candidate_2` mislabels what it captures.** It reads
   `args[0]` — the *input* to `multi_modal_projector` — and comments
   "post-pooling spatial tokens." On the real Gemma 3 architecture (§1.8),
   pooling happens **inside** `multi_modal_projector.forward()` via
   `self.avg_pool`, so the projector's *input* is the same raw, unpooled
   `vision_tower` output already captured by `hook_candidate_1` — redundant,
   not a distinct "pooled" signal. To actually capture pooling output
   distinctly on Gemma 3 you'd need a fourth hook, on `avg_pool` itself, not
   on the projector's input or output. **On this project's actual Gemma 4
   checkpoint this problem doesn't arise** — pooling and projection are
   already two separate modules (§1.3), so a hook on `vision_tower` (for
   pooling output) and a hook on `embed_vision` (for projection output)
   cleanly gives you both, no need to reach inside either module for a
   sub-call.
3. **The call-path diagram is directionally right but the worked example
   uses the wrong call for this pipeline.** The example calls bare
   `model(**inputs)` and reads `outputs.logits` for "classification
   logits." This pipeline's real `classify()` path never does that — it
   calls `.generate()` (§1.1). As established in §1.1, this doesn't change
   whether the hooks fire or how many vision-encoder calls happen (both
   routes hit the identical submodule forward calls exactly once), so the
   *mechanism* claim survives the correction intact; the code example
   itself would need to be adapted to hook around a `.generate()` call
   rather than a bare forward call to match how this pipeline actually
   invokes the model.
4. **What Gemini got right, worth crediting rather than only critiquing**:
   the "zero duplicate compute" and "hooks don't alter autoregressive
   state, `.detach()` keeps them read-only" claims are both accurate and
   match §1.1/§1.4's independently-verified findings. The three-stage
   mental model (encode → pool → project) is the right shape, generalizes
   correctly to Gemma 4 once you swap in the real module names — the value
   of the cross-check was in the specifics, not the overall architecture
   diagnosis.

### 1.10 Recommended abstraction

A single, narrow helper — not a general-purpose "vision hook framework" —
matching this codebase's existing preference for config-driven, minimally
abstracted primitives (`CODE_MAP.md`'s architectural principle section):

```python
class GemmaVisionInstrumentation:
    """Attaches/detaches forward hooks on a loaded Gemma4-family model's
    vision submodules and buffers the three candidate tensors for the
    most recent classify() call."""

    def __init__(self, model):
        self._captured: dict[str, torch.Tensor] = {}
        self._handles: list = []
        # Introspect rather than hardcode — see §1.8.
        inner = model.model  # Gemma4Model / Gemma3Model, whichever it is
        self._handles.append(
            inner.vision_tower.encoder.register_forward_hook(self._make_hook("encoder"))
        )
        self._handles.append(
            inner.vision_tower.register_forward_hook(self._make_hook("pooled"))
        )
        projector = getattr(inner, "embed_vision", None) or getattr(inner, "multi_modal_projector", None)
        self._handles.append(projector.register_forward_hook(self._make_hook("projected")))

    def _make_hook(self, name):
        def _hook(module, args, output):
            tensor = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
            self._captured[name] = tensor.detach().to("cpu")
        return _hook

    def reset(self):
        self._captured.clear()

    def release(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
```

Wiring into `GemmaLoader`: instantiate once in `initialize_model_and_tokenizer()`
after `self.model = model`; call `.reset()` at the top of `classify()`
(would need a small override or a hook into `BaseLoader.classify()` — not
scoped here, per the "no implementation" constraint); read
`self._vision_instrumentation._captured` immediately after `_run_generate()`
returns; call `.release()` from a new `GemmaLoader.release()` override. This
sketch is illustrative of the design, not a diff to apply.

**Implemented, 2026-08-05**: hooks (`enable_vision_debug_instrumentation()`),
validation (`_validate_vision_capture()`), and now persistence
(`_persist_vision_capture()`) all built directly as `GemmaLoader` methods
rather than the standalone class sketched above — this loader is the
class's only caller, so a separate class would be an abstraction with one
user. Every design point above carried through as specified: `--debug`-
gated (`self._debug_mode`, set by `build_classifier_loader(debug=...)`),
`.detach().to("cpu")` in the hook itself (not deferred), reset at the top
of `classify()`, hook removal in `release()`. One addition beyond the
sketch: `_persist_vision_capture()` writes the captured tensors plus that
call's classification result to a per-image `.pt` file under
`data/outputs/gemma_vision_debug_captures/` (`torch.save`, one file per
image stem, overwritten on repeat debug runs against the same image) —
this is what actually makes the capture usable for the "does this
separate the 8 routing buckets" research question, versus the live-only
validation the original sketch stopped at. Verified live: a real
`classify()` call under `--debug` produced a 5.1 MB `.pt` file (dominated
by the pre-pool encoder tensor, `(1, 2520, 768)` bf16) containing all
three tensors plus `{category, confidence, reason}`, correctly
reproducing the shapes this document's §1.5 table + the token-budget
investigation's Phase 4 math predicted. Stays `--debug`-only by explicit
direction — not a candidate for a production default; if the corpus-scale
research question is ever pursued, that's a deliberate future decision,
not something this persistence step defaults into.

---

## Part 2 — Vision Tower Survey

**Framing, per the task's explicit constraint**: the bar is "contributes
genuinely new information," not "exists and is popular." Two things already
in this project's own loader roster (`config/models/*.yaml`, 24 profiles,
per `VISION_IR_RESEARCH.md` §0) change the survey's shape versus a
green-field question: (1) several of the candidate names below are **already
running in this pipeline** as generation loaders — their vision towers are
not new dependencies, just not yet used as *standalone sensors*; (2) a
separate, already-executed research track
(`VISION_IR_RESEARCH.md`'s Sixth addendum onward) already qualified
DINOv2/ConvNeXt/SigLIP-NaFlex/EVA-02/BEiT/Swin/MAE as general-purpose
standalone sensors — this survey deliberately does not re-propose any of
those; it's scoped to document/OCR-specific backbones, a different
candidate pool.

### 2.1 Per-model findings

| Model | Vision backbone | Unique or reused | Independently extractable | Genuinely new info vs. this project's existing towers? | Inference cost (rough) | License | Verdict |
|---|---|---|---|---|---|---|---|
| **Florence-2** | DaViT (Davit hierarchical vision transformer) | Already in this project's loader roster (`florence_loader.py`, per `VISION_IR_RESEARCH.md` §2) | Yes — the DaViT encoder is a standard HF submodule (`model.vision_tower`), same "hook it, don't retrain it" mechanism as §1 | **No new dependency, but genuinely reused for a new purpose**: currently only consumed for generation; never run as a standalone embedding source. Florence-2 is unusual among this list for being jointly trained across detection/captioning/OCR/grounding — its representation is more task-general than a single-purpose OCR encoder | Already paid for (already loaded when the Florence-2 generation loader runs) | MIT | **Zero-cost re-use candidate**, not a new sensor to acquire — extracting its already-loaded encoder output as an additional signal alongside its existing generation role costs nothing new to add |
| **InternViT** (InternVL3) | InternViT-300M or 6B depending on config | Already in this project's loader roster (`internvl_loader.py`, two sizes configured) | Yes, same reasoning as Florence-2 | Trained at large scale with strong document/chart/OCR representation specifically called out in InternVL's own reporting (already noted in `VISION_IR_RESEARCH.md` §2) — genuinely different training distribution from DINOv2/ConvNeXt's natural-image self-supervision | Already paid for | MIT (InternViT-300M) / more permissive OSS for the smaller variants — verify per-checkpoint | **Zero-cost re-use candidate**, same reasoning as Florence-2 |
| **Qwen2.5-VL / Qwen3-VL** | Native dynamic-resolution ViT (Qwen3-VL: SigLIP2-based, SO-400M or Large-300M depending on model size, 2D-RoPE, DeepStack multi-level feature injection) | Already in this project's loader roster | Yes | Native dynamic-resolution handling is architecturally distinct from every fixed-resolution encoder already qualified in `VISION_IR_RESEARCH.md`'s battery — closer in spirit to SigLIP-NaFlex (already qualified) than to anything genuinely new, so the *novelty* here is smaller than Florence-2/InternViT's OCR-specific training | Already paid for | Apache 2.0 (Qwen family) | **Low-priority re-use candidate** — real signal, but closest in kind to an already-qualified encoder (NaFlex) |
| **GOT-OCR2** | ~80M-param ViT initialized from SAM's ViT-H, high-compression (1024×1024 → 256 tokens) | Already in this project's loader roster (`config/models/*.yaml` lists GOT-OCR2 per `VISION_IR_RESEARCH.md` §0) | Yes — `stepfun-ai/GOT-OCR2_0`'s encoder is a standard, separately loadable HF submodule | SAM-initialized encoder is trained for *segmentation-quality spatial precision*, then fine-tuned toward extreme compression for OCR — a genuinely different training lineage from every other candidate on this list (natural-image segmentation, not classification or contrastive alignment) | Already paid for; standalone (not via the full generation pipeline) the encoder-only forward is cheap — 580M total, encoder is the smaller ~80M piece | Apache 2.0 | **Zero-cost re-use candidate** — arguably the single most differentiated already-in-roster encoder, given SAM lineage |
| **SmolVLM2** | SigLIP-family (SigLIP-SO400M for the 2.2B variant, SigLIP-B/16 93M for the smaller 256M/500M variants) | Already in this project's loader roster | Yes | SigLIP-family — closely related to Gemma's own vision tower lineage (`VISION_IR_RESEARCH.md` §2 already notes "Gemma's own vision tower descends from \[SigLIP]") and to the already-qualified SigLIP-NaFlex candidate. Weak novelty | Already paid for | Apache 2.0 | **Not recommended** — most redundant already-in-roster candidate; adds little beyond what SigLIP-NaFlex and Gemma's own vision tower (Part 1) already cover |
| **LayoutLMv3** | ViT-style patch embeddings (no CNN backbone), image tokenizer initialized from **DiT**'s discrete VAE | New dependency (not in this project's loader roster) | **Not cleanly independent of OCR** — LayoutLMv3 is a *joint* text+layout+image model; its useful representation requires OCR-derived word tokens and bounding boxes as an additional input alongside pixels. Extracting *just* the vision branch reduces to extracting DiT (below) with extra steps | If used only for its image tower, this is DiT again — no added value; if used as the full joint model, it needs OCR output as an upstream dependency, which is a chicken-and-egg problem for a *pre-routing* sensor (this pipeline's Stage 1 classification happens before any OCR runs) | N/A given the dependency problem above | CC-BY-NC-4.0 (Microsoft UniLM models are typically research-only) — **verify per-checkpoint before any commercial-adjacent use** | **Not recommended as a standalone pre-OCR sensor** — the real value (joint text-layout-image reasoning) isn't accessible without already having OCR text, which doesn't exist yet at classification time in this pipeline |
| **DiT (Document Image Transformer)** | BEiT-style self-supervised ViT, pretrained specifically on document images (not natural images) via masked image modeling with a dVAE tokenizer (DALL-E's) | New dependency | Yes — `microsoft/dit-base` is a standalone HF checkpoint, no OCR/text dependency, pure image-in-embedding-out | **Genuinely new**: this is the one candidate in this whole survey pretrained specifically on the *document-image domain* using self-supervision, as opposed to natural-image self-supervision (DINOv2/ConvNeXt/MAE, already qualified) or contrastive text-image alignment (SigLIP/CLIP). Directly relevant to this project's own repeatedly-measured finding (`VISION_IR_RESEARCH.md` §15) that document pages are a very different distribution from the natural images most encoders are validated against (extreme aspect ratios, tiny degraded text) | Small (ViT-Base scale, ~86M params) — cheap, standalone forward pass | Likely research-only per Microsoft's UniLM licensing pattern (same family as LayoutLMv3) — **verify explicitly before relying on it**, not assumed permissive by default | **Recommended candidate** — the strongest document-domain-specific, non-redundant addition on this list, contingent on license verification |
| **Donut** | Swin Transformer encoder (`DonutSwinModel`, separately loadable) + BART decoder | New dependency | Yes — `DonutSwinModel` is a standard, independently loadable HF class; the encoder alone (no decoder) is a legitimate standalone use | Genuinely different pretraining objective from anything else surveyed: trained end-to-end for OCR-free document-to-markup generation, meaning the encoder's representation is shaped by "what does the decoder need to read out structured text," not classification or contrastive alignment. Distinct axis of novelty from DiT's (self-supervised, no generation objective) | Swin-Base scale, moderate — comparable to other mid-size ViT/Swin encoders already benchmarked in this project's own battery (Swin-base was Round 8 in `VISION_IR_RESEARCH.md`) | MIT (NAVER Corp) | **Recommended candidate** — clean license, standalone-extractable, genuinely different training objective from every other qualified/surveyed candidate |
| **Nougat** | Swin Transformer encoder (Donut-lineage, a Nougat-specific variant/config), academic-paper-specific training data | New dependency | Yes, same reasoning as Donut — same `VisionEncoderDecoder` family | Same architecture family as Donut but trained on a very different document distribution (academic papers with dense math/citations vs. Donut's receipts/forms training mix) — marginal novelty *over Donut specifically*, more novelty relative to everything else surveyed | Similar to Donut | **Code MIT (Meta/FAIR); model weights commonly distributed under CC-BY-NC 4.0 — non-commercial only. Must be verified explicitly against the actual checkpoint's license file before any use beyond pure research, given this is unconfirmed from search results alone** | **Conditional candidate** — architecturally redundant with Donut once Donut is adopted; only worth adding separately if the NC license is confirmed acceptable AND the academic-paper-specific training turns out to matter for this project's actual corpus (unlikely — this corpus is genealogical records, not academic papers) |
| **Pix2Struct** | ViT encoder with variable-resolution patch grids (aspect-ratio-preserving, not a fixed square resize) | New dependency | Yes — Google's `Pix2StructVisionModel` is a standalone HF class | Genuinely distinct pretraining objective: screenshot/document-to-simplified-HTML parsing, meaning the encoder is trained to preserve *structural* layout information specifically, not just visual similarity. The variable-resolution patch grid is also architecturally close to what this project already validated as valuable in SigLIP-NaFlex (already qualified) — meaningful overlap in *mechanism*, but a different *training objective* | ViT-Base/Large scale depending on variant — moderate | Apache 2.0 | **Recommended candidate**, with the caveat that its value-add over already-qualified SigLIP-NaFlex is the training objective (structure-aware) more than the resolution mechanism (already covered) — worth qualifying specifically for the "does this separate the 8 routing buckets by *layout structure* better than a generic encoder" question, not for variable-resolution handling per se |
| **PaddleOCR-VL** | NaViT-style dynamic-resolution encoder, initialized from Kwai's Keye-VL vision model, paired via a 2-layer MLP projector to ERNIE-4.5-0.3B (whole VLM is 0.9B) | New dependency | Partially — the vision encoder is architecturally separable (standard LLaVA-style: encoder → MLP projector → LM), but as of this research there's no confirmation a HF `AutoModel`-compatible standalone vision-only class ships separately from the full VLM checkpoint; likely extractable via the same forward-hook mechanism as §1, not via a clean pre-split checkpoint | NaViT-style native-resolution handling, most recently reported at 96.3% on OmniDocBench v1.6 (PaddleOCR-VL-1.6, 2026-05-28) — strong document-parsing-specific signal, but via a training lineage (Keye-VL) not yet represented anywhere in this project's roster or research | Small for a full VLM (0.9B total) but the vision-only forward cost specifically isn't separately documented | Apache 2.0 (PaddlePaddle license pattern) — verify per release | **Conditional candidate** — promising on paper (best-in-class OmniDocBench score) but the "not yet confirmed cleanly separable without a hook, same as Gemma itself" caveat means this needs its own Part-1-style interface-inspection pass before it's actually a low-friction sensor, not a documented API |
| **MinerU / MinerU2.5** | Not a single encoder — a **decoupled two-stage pipeline**: Stage I layout analysis on downsampled (1036×1036) images using `doclayout_yolo` (a YOLO-family object detector, not a transformer encoder), Stage II high-res content recognition via a Qwen2-VL-lineage 0.5B decoder on cropped regions | New dependency (the layout detector specifically) | Yes — `doclayout_yolo` is architecturally a standard object detector, trivially separable from the Stage II recognition model, and doesn't require running the whole MinerU pipeline to get layout boxes | **The single most differentiated candidate in this entire survey.** Everything else on this list (and everything already qualified in `VISION_IR_RESEARCH.md`) is an embedding-producing encoder. `doclayout_yolo` produces **structured region detections** (text/title/figure/table/formula bounding boxes) — a categorically different output shape, directly comparable to (and a plausible upgrade path for) this project's own classical-CV `page_boundary`/`table_boundary` ROI detectors in `core/image_analysis.py`, which have multiple documented failure modes (LAC microfilm: 174/218 spurious near-global boxes; `bright_region` fallback failing on aged mid-grey microfilm paper) | Small, YOLO-family detector — this project's own note that it's "10x faster... while maintaining comparable accuracy" vs. the earlier LayoutLMv3-based approach suggests genuinely cheap inference | Apache 2.0 (MinerU/OpenDataLab license pattern) — verify per release | **Strongly recommended candidate** — the only surveyed option that produces genuinely new *information type* (structured layout regions) rather than another embedding vector, and it directly targets a documented, real weakness in this project's existing ROI-detection code |
| **Qwen2.5-VL / Qwen3-VL** (revisit) | — see above — | already covered | — | — | — | — | (listed once, above, to avoid duplication with the "OCR/document foundation models" framing of the prompt) |

### 2.2 Explicitly rejected: generic-redundancy examples, per the task's own instruction

Per the task's explicit "not: another generic CLIP variant with highly
redundant embeddings" instruction, worth naming a few that were considered
and dropped without a full row above, so the omission reads as a decision
rather than an oversight:

- **Any bare CLIP/OpenCLIP checkpoint** — superseded by SigLIP2 as a
  general-purpose backbone per `VISION_IR_RESEARCH.md` §2's own finding,
  and SigLIP2/SigLIP-NaFlex are already qualified in that document's
  experimental battery. Adding CLIP alongside would be exactly the
  "another generic contrastive encoder" case the task asks to avoid.
- **A second InternViT size, or a second Qwen-VL size** — already covered
  by the roster (§2.1); a different parameter count of an architecture
  already available adds compute cost without adding a new information
  axis.
- **BLIP-2 / InstructBLIP-family Q-Former encoders** — considered and
  excluded: the Q-Former's whole design purpose is producing a *fixed,
  small* set of query-conditioned tokens for downstream instruction-following,
  not a general embedding — closer to "another projector variant" than "a
  new vision backbone," and the base vision encoders these typically wrap
  (ViT-g/14, EVA-CLIP) are natural-image contrastive encoders already
  represented in spirit by DINOv2/EVA-02 (already qualified — EVA-02 was
  Round 6 in `VISION_IR_RESEARCH.md`, and was specifically noted as "a real
  falsification, not a confirmation" of the inversion-sensitivity
  hypothesis, i.e. already measured, not a new unknown).

### 2.3 Summary ranking

Ordered by "genuinely new information, cheap to obtain, licensed cleanly" —
not a recommendation to build all of these, just the priority order if any
of Part 2 is pursued:

1. **`doclayout_yolo` (from MinerU2.5)** — new information *type*
   (structured regions, not embeddings), directly addresses a documented
   weakness in existing CV code, cheap, clean license.
2. **Already-in-roster encoders (Florence-2/InternViT/GOT-OCR2) as
   standalone sensors** — zero new dependency cost, real training-lineage
   diversity, purely a matter of adding a hook/extraction call to loaders
   that already run.
3. **DiT** — the one genuinely document-domain self-supervised encoder
   surveyed, contingent on license verification.
4. **Donut** — clean MIT license, genuinely different (generation-shaped)
   pretraining objective, standalone-extractable encoder.
5. **Pix2Struct** — structure-aware training objective, Apache 2.0, real
   but partially-overlapping novelty with already-qualified SigLIP-NaFlex.
6. **PaddleOCR-VL** — best raw benchmark numbers of anything surveyed, but
   needs its own Part-1-style interface-inspection pass before it's a
   known quantity rather than a promising unknown.
7. **Nougat** — conditional on license and on this project's corpus
   actually containing academic-paper-like material (currently doesn't,
   per the genealogical-records framing throughout this project's other
   docs).
8. **SmolVLM2, a second Qwen-VL size** — not recommended; redundant with
   already-qualified or already-in-roster candidates.

---

## Part 3 — Sensor Roadmap: gaps in observability before Decision Engine work

**Explicitly not decision-logic design** — this section names what's still
unmeasured, not how a future Decision Engine should weigh any of it.

### 3.1 What's characterized today (recap, for gap-analysis purposes only)

- Physical/deterministic image sensors — `core/image_analysis.py`'s Stage A
  (geometry, tone, ROI detectors), `core/baseline_embeddings.py`'s
  pre/post-preprocessing captures.
- External vision-tower embeddings — the qualified battery
  (DINOv2/ConvNeXt/SigLIP-NaFlex/EVA-02/BEiT/Swin/MAE) plus, per Part 2
  above, a scoped set of document-specific candidates not yet built.
- Tower consensus — `benchmark2_3_multi_tower_routing_audit.py`.
- Gemma category output, Gemma reasoning (chain-of-thought when
  `reasoning_enabled`), and now, per Part 1, Gemma's internal vision
  representations (encoder/pooled/projected) as a candidate fourth signal
  from the same model call.
- Decoder logits — explicitly "planned" per this task's own framing, not
  yet built.

### 3.2 Gap 1 — Structured layout/region detection (highest-priority new sensor family)

Nothing currently characterized produces **bounding-box-shaped** output.
`core/image_analysis.py`'s `page_boundary`/`table_boundary` are the closest
existing thing and are classical CV with documented, measured failure modes
(§2.1's MinerU row cites the specific LAC-microfilm findings). Every vision
tower qualified or surveyed so far (Parts 1–2, and the existing DINOv2/
ConvNeXt battery) produces an embedding vector or a spatial feature *map*
at best — not a discrete, labeled region list (text block / title / figure
/ table / formula). A learned layout detector (`doclayout_yolo` being the
concrete, cheap, cleanly-licensed candidate identified in Part 2) is a
genuinely missing *information type*, not just another encoder to add to
the pile.

### 3.3 Gap 2 — A confidence/margin signal that isn't self-reported

`VISION_IR_RESEARCH.md` §15 already flagged this precisely: "replacing
today's self-reported LLM confidence number... with a classifier-native
margin/distance signal... is a real, well-precedented calibration
improvement." Nothing in this project currently computes a margin, entropy,
or centroid-distance confidence for the classification decision — the only
confidence in the pipeline today is Gemma's own self-reported number in its
structured output (`ClassificationResult.confidence`), a known-weak signal
class. Part 1's Gemma-internal capture and the "planned" decoder logits are
both *inputs* this signal could eventually be computed from — but the
signal itself (a calibrated margin over the 8-way decision) doesn't exist
yet as a characterized sensor.

### 3.4 Gap 3 — Text-derived signal (nothing downstream of OCR feeds back upstream)

Every sensor characterized so far — physical, learned-vision, Gemma-internal —
operates on pixels, before any text has been read. Once OCR/extraction runs
(downstream of routing in this pipeline's current shape), its own output
carries signal that nothing currently treats as a sensor: character-level
confidence distributions, script/language identification, or even a crude
"did this produce plausible-looking text at all" check. This is explicitly
a *later* pipeline stage's output feeding back as evidence, not a
pre-routing signal — worth naming as a gap now specifically because
`VISION_IR_RESEARCH.md` §29 already identified "OCR / extraction model
selection" as "a genuinely new idea... but currently ungrounded" for
exactly this reason (no reference dataset linking image clusters to
measured extraction accuracy exists yet). The text-confidence signal itself
is a prerequisite building block for that later idea, not the idea itself.

### 3.5 Gap 4 — Ground truth / verified-label sensor (a process gap, not a model gap)

Repeated across both this document's sources and this project's own memory
record (`project_no_comprehensive_ground_truth`): there is no comprehensive,
human-verified ground truth for the 8-bucket routing decision, and per
`VISION_IR_RESEARCH.md` §14, the bucket CSVs are Gemma's own predictions,
not verified labels — every sensor characterized so far (including
everything in Parts 1–2 of this document) can be *measured* for internal
consistency (stability under transforms, cluster agreement) but not
*validated against correctness* until this gap closes. `ui/
classifier_validation_ui.py`'s `self.future_frame` (a real, empty,
positioned placeholder for Correct/Incorrect capture) is the concrete,
already-scaffolded next step — this is a UI/process gap, not a missing
model family, but it blocks evaluating everything else on this list, so it
belongs on the roadmap regardless.

### 3.6 Gap 5 — Duplicate/near-duplicate and sequence-position signal

Flagged as "trivial to add, not yet added" in `VISION_IR_RESEARCH.md` §1
(perceptual + cryptographic image hashing) and never built. Related and
also unbuilt: nothing in this pipeline currently treats **sequence
position** (this page is N of M from the same source batch, adjacent to
specific other pages) as a signal at all — every image is processed as an
independent unit. For genealogical record sets (multi-page census
schedules, passenger manifests spanning several photographed pages), the
fact that page N+1 was classified as `dense_tabular_rows` is real evidence
about what page N+1's *neighbor* likely is, and nothing currently captures
or uses that adjacency.

### 3.7 Gap 6 — Calibration/audit-sampling signal

`config/pipeline.yaml`'s `audit_sampling` is currently a flat, uniform
`rate: 0.05`. `VISION_IR_RESEARCH.md` §29 already identified
anomaly/low-confidence-prioritized sampling as "arguably the single
strongest non-Gemma consumer identified across this whole research thread" —
directly serving this project's own stated success metric
(`feedback_reduce_manual_touches_is_the_metric`) better than random
sampling. This is a consumer of other sensors (anomaly/novelty scoring)
rather than a new sensor family itself, but it's an identified, concrete,
currently-unbuilt gap worth keeping on the same list since it's one of the
most immediately actionable items here.

### 3.8 Roadmap ordering (observability gaps only, not a build plan)

In order of "most distinct new information per unit of effort," combining
Parts 1–3's findings:

1. **Ground truth capture (Gap 4)** — blocks validating everything else;
   the review UI's placeholder already exists, this is the cheapest,
   highest-leverage gap to close first, and every other item on this list
   becomes easier to evaluate once it exists.
2. **Structured layout detection (Gap 1)** — the one genuinely new
   *information type* (not another embedding) identified across this
   entire research pass, cheap and cleanly licensed (`doclayout_yolo`).
3. **Gemma-internal representation capture (Part 1)** — mechanically ready
   to build (module hierarchy fully confirmed, zero duplicate compute,
   negligible per-image cost); the open question is evaluative (does it
   separate the 8 buckets), not architectural.
4. **Duplicate/sequence signal (Gap 5)** — cheap, well-precedented,
   explicitly flagged as trivial and still not done.
5. **Document-domain encoder addition from Part 2 (DiT/Donut/`doclayout_yolo`
   already covers the highest-value case in #2 above)** — lower priority
   than #2 specifically because the layout-detector candidate already
   captures the most differentiated new signal from this survey.
6. **Confidence/margin signal (Gap 2)** and **calibration-sampling
   consumer (Gap 6)** — both real and well-motivated, but both are
   naturally *downstream* of having ground truth (#1) and a working
   embedding/logit source (#3) to calibrate against, so they sequence
   after the sensors they'd consume.
7. **Text-derived / OCR-feedback signal (Gap 3)** — explicitly the most
   speculative gap on this list (no reference dataset exists, per
   `VISION_IR_RESEARCH.md` §29) — real, but the last thing to invest in
   without a concrete, grounded next step already defined.

---

## Sources consulted for Part 2 (web research, current as of 2026-08-04)

- [PaddleOCR-VL usage docs](https://github.com/paddlepaddle/paddleocr/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md), [HF model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL), [arXiv 2510.14528](https://arxiv.org/pdf/2510.14528)
- [GOT-OCR2 HF docs](https://huggingface.co/docs/transformers/en/model_doc/got_ocr2), [arXiv 2409.01704](https://arxiv.org/pdf/2409.01704), [stepfun-ai/GOT-OCR2_0](https://huggingface.co/stepfun-ai/GOT-OCR2_0)
- [MinerU2.5 arXiv](https://arxiv.org/html/2509.22186v2), [MinerU DeepWiki layout analysis](https://deepwiki.com/chukonu-team/MinerU_2.5/6.6-layout-analysis), [MinerU pipeline backend](https://deepwiki.com/opendatalab/MinerU/2.1-pipeline-backend)
- [Qwen2.5-VL Technical Report](https://arxiv.org/pdf/2502.13923), [Qwen3.5 vision encoders overview](https://www.emergentmind.com/topics/qwen3-5-vision-encoders)
- [LayoutLMv3 HF docs](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/layoutlmv3.md), [Microsoft Research publication](https://www.microsoft.com/en-us/research/publication/layoutlmv3-pre-training-for-document-ai-with-unified-text-and-image-masking/)
- [Donut GitHub (clovaai)](https://github.com/clovaai/donut), [Donut HF docs](https://huggingface.co/docs/transformers/model_doc/donut)
- [SmolVLM HF blog](https://huggingface.co/blog/smolvlm)

## Sources consulted for Part 1 (local, direct code inspection — no external fetch except one)

- `transformers==5.12.1` installed at `C:\Users\jonny\AppData\Roaming\Python\Python314\site-packages\transformers` — `models/gemma4/modeling_gemma4.py`, `models/gemma4/configuration_gemma4.py`, `models/gemma3/modeling_gemma3.py`, `models/auto/modeling_auto.py`.
- Live fetch: `https://huggingface.co/google/gemma-4-E2B-it/raw/main/config.json`.
- `core/loaders/gemma_loader.py`, `core/loaders/base_loader.py`, `config/models/gemma.yaml`, `config/models/gemma_e4b.yaml`, `core/classifier.py`, `core/schema.py` (all read directly from this repo).
