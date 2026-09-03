"""
Generic YOLO-based document-layout detection - a new sensor tower
alongside core/vision_embeddings.py's QUALIFIED_ENCODERS, identified in
docs/GEMMA_INSTRUMENTATION_AND_SENSOR_SURVEY.md Part 2/Part 3 as the
single most differentiated candidate surveyed there. Unlike every
embedding encoder already qualified, this produces DISCRETE, LABELED
REGION DETECTIONS (title/text/table/figure/... bounding boxes), not a
fixed-length vector - a genuinely different output shape, and a direct
complement to core/image_analysis.py's classical-CV page_boundary/
table_boundary ROI detectors and to core/auto_sidecar.py's ruling-line-
based table/column boundary search (locate_table_boundary(),
locate_columns()). Those consumers are NOT modified here - this module
only detects and records; using the detections to inform those searches
is a separate, later decision (same "Stage 1 measures, doesn't decide"
discipline core/image_analysis.py and core/baseline_embeddings.py
already follow).

Model: DocLayout-YOLO (github.com/opendatalab/DocLayout-YOLO, a
YOLOv10-based document layout detector), DocStructBench-finetuned
checkpoint (huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench).
10 classes: title, plain text, abandon, figure, figure_caption, table,
table_caption, table_footnote, isolate_formula, formula_caption -
confirmed directly against the loaded checkpoint's own model.names
(2026-08-04), not copied from the paper. class_name is always read from
the loaded model at call time (see detect_layout()), never hardcoded, so
a different future checkpoint's class set can't silently drift out of
sync with what's actually returned.

LICENSING NOTE - real, not decorative: the `doclayout_yolo` PACKAGE
(pip install doclayout-yolo, the actual code this module imports) is
AGPL-3.0 licensed - confirmed directly against opendatalab/DocLayout-
YOLO's own LICENSE file. The DocStructBench CHECKPOINT WEIGHTS are
SEPARATELY Apache-2.0 licensed (the HF model card). This corrects
docs/GEMMA_INSTRUMENTATION_AND_SENSOR_SURVEY.md Part 2's assumption
("Apache 2.0 (MinerU/OpenDataLab license pattern)") for this specific
package - that assumption was wrong, verified here against the real
LICENSE file rather than left uncorrected. Flagged so this is a known,
deliberate dependency choice, not a silent addition of AGPL-licensed
code to the project.
"""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

DOCSTRUCTBENCH_REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
DOCSTRUCTBENCH_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"

# Fallback only - used when a Results object somehow has no .names of
# its own (not observed in testing, but detect_layout() prefers the
# LIVE model/result names over this in every normal case). Recorded here
# for documentation/reference, not as the source of truth.
DOCSTRUCTBENCH_CLASS_NAMES = {
    0: "title", 1: "plain text", 2: "abandon", 3: "figure",
    4: "figure_caption", 5: "table", 6: "table_caption",
    7: "table_footnote", 8: "isolate_formula", 9: "formula_caption",
}

DEFAULT_IMGSZ = 1024
DEFAULT_CONF = 0.2


def build_layout_model(
    repo_id: str = DOCSTRUCTBENCH_REPO_ID,
    filename: str = DOCSTRUCTBENCH_FILENAME,
):
    """
    Downloads (or reuses the huggingface_hub local cache for) the
    checkpoint and loads it.

    Deliberately NOT YOLOv10.from_pretrained(repo_id) - confirmed broken
    against the real installed package (doclayout-yolo 0.0.4, 2026-08-04):
    it instantiates the model against a hardcoded generic 'yolov10n.pt'
    architecture default instead of resolving the repo's actual
    checkpoint file, raising FileNotFoundError: 'yolov10n.pt'.
    hf_hub_download() + the README's OTHER documented constructor form
    (YOLOv10(local_path)) is the form verified working by direct test
    against a real image in this project's own data/working/.
    """
    from doclayout_yolo import YOLOv10

    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLOv10(weights_path)


def detect_layout(
    model,
    pil_image: Image.Image,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONF,
    device: str | None = None,
) -> list[dict]:
    """
    Runs one image through the layout detector and returns its
    detections as plain, JSON-serializable dicts - no model-specific
    objects leak out, matching core/vision_embeddings.py's embed_pooled()/
    embed_patch_mean() contract (plain numpy/python return types, no
    torch/model objects).

    Returns [{"class_id": int, "class_name": str, "confidence": float,
    "bbox_xyxy": [x0, y0, x1, y1]}, ...], most-confident first (the
    model's own detection order) - empty list for a page with no
    detections above `conf`, never None.

    bbox_xyxy is in the ORIGINAL image's pixel coordinates - confirmed
    directly (not assumed): the library internally resizes to `imgsz`
    for inference but rescales detections back to the source image's own
    resolution before returning, so this is directly usable against the
    same pixel space core/image_analysis.py and core/auto_sidecar.py
    already operate in, with no separate rescaling step needed by any
    caller.

    device defaults to "cuda:0" if available, else "cpu" - same
    auto-detection this project's other GPU-capable sensor code
    (core/baseline_embeddings.py's capture_baseline_embeddings()) already
    does, not a hardcoded assumption either way.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    results = model.predict(pil_image, imgsz=imgsz, conf=conf, device=device, verbose=False)
    result = results[0]
    names = getattr(result, "names", None) or getattr(model, "names", None) or DOCSTRUCTBENCH_CLASS_NAMES

    detections = []
    boxes = result.boxes
    if boxes is None:
        return detections
    for box in boxes:
        class_id = int(box.cls.item())
        detections.append({
            "class_id": class_id,
            "class_name": names.get(class_id, str(class_id)),
            "confidence": round(float(box.conf.item()), 4),
            "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })
    return detections
