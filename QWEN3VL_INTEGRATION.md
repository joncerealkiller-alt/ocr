# Qwen3-VL Integration — Model Assessment Tool

**Date:** 2026-07-10  
**Status:** Ready for assessment testing

---

## What was built

Five changes to integrate Qwen3-VL-2B-Instruct into the Model Assessment tool:

### 1. New loader: `core/loaders/qwen3vl_loader.py`

- Class: `Qwen3VLLoader(BaseLoader)`
- Uses `Qwen3VLForConditionalGeneration` (Qwen3 series, not Qwen2.5)
- Implements all BaseLoader abstract methods
- Per-field conditional kwarg passing (presence_penalty, no_repeat_ngram_size only if set)
- Single-image PIL Image input (matches assessment workflow)
- Output trimmed to new tokens only (batch_decode then [0])
- Execution metadata logging (device placement, max_memory, VRAM headroom)

### 2. Loader registration: `core/loader_registry.py`

Added to LOADER_REGISTRY:
```python
"Qwen3VLLoader": Qwen3VLLoader
```

### 3. Model config: `config/models/qwen3vl2b.yaml`

Configuration with VL task hyperparameters from model card:
- `repo_id: Qwen/Qwen3-VL-2B-Instruct`
- `loader_class: Qwen3VLLoader`
- VL defaults: `do_sample=False`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.0`
- `min_pixels/max_pixels`: null (using processor defaults)
- `presence_penalty`: null (awaiting confirmation on behavior)

### 4. GenerationConfig update: `core/loaders/base_loader.py`

Added field:
```python
presence_penalty: Optional[float] = None
```

Updated `content_hash()` to include presence_penalty in the deterministic config fingerprint.

### 5. Assessment UI: `model_assessment.py`

Added to OVERRIDABLE_FIELDS:
```python
"presence_penalty"
```

Now tunable in the UI for testing when/if Qwen3-VL supports the parameter.

---

## Testing this now

1. Start the assessment tool:
   ```bash
   python model_assessment.py
   ```

2. In the Model profile dropdown, select **qwen3vl2b**

3. Pick an image, prompt, and bucket type (matching the extraction prompt)

4. Run — the tool will:
   - Load Qwen3-VL-2B-Instruct
   - Apply chat template with the extraction prompt
   - Generate with greedy decoding (do_sample=False)
   - Parse output and validate against schema
   - Run anomaly checks (repeated_entry)
   - Save JSON report to `data/outputs/model_assessments/`

---

## Known open questions

**presence_penalty behavior:** Does Qwen3-VL's `model.generate()` accept this parameter directly? If not, the field in the loader should skip passing it (already implemented with the conditional check). Awaiting confirmation from other instance.

**Pixel range tuning:** `min_pixels/max_pixels` are left null to use processor defaults. If VRAM or quality issues arise, can be tuned in the YAML without changing code.

---

## Next steps (after assessment testing)

1. Test against the core validation document set (2x census, register excerpt, passenger manifest)
2. Compare output quality vs. Qwen2.5-VL-3B baseline
3. Investigate Qwen3-VL-specific failure modes (if any)
4. If stable: can add to `config/pipeline.yaml` bucket routing for production extraction

---

## Files changed

- ✓ Created: `core/loaders/qwen3vl_loader.py` (273 lines)
- ✓ Modified: `core/loader_registry.py` (+1 import, +1 registry entry)
- ✓ Created: `config/models/qwen3vl2b.yaml` (35 lines)
- ✓ Modified: `core/loaders/base_loader.py` (+1 field, +1 hash entry)
- ✓ Modified: `model_assessment.py` (+1 override field)
