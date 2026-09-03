"""
SECOND, EXPERIMENTAL document-layout detector - NOT wired into the
pipeline. Built 2026-08-05 to test-compare against core/layout_
detector.py's DocLayout-YOLO/DocStructBench sensor (the one actually
capturing data into data/baseline_embeddings.json as of this date),
per direct instruction: "keep the current loader, create a new one
test against but dont wire it into the pipeline, have it save to a
seperate file for now."

Model: YOLOv26-small, github.com/Armaggheddon/yolo_doc_layout (MIT
code license, Apache-2.0 project license - both more permissive than
core/layout_detector.py's AGPL-3.0 doclayout_yolo package dependency,
the original reason that sensor's capture defaulted off before it was
flipped to default-on 2026-08-05). Checkpoint hosted at HF repo
Armaggheddon/yolo26-document-layout, file yolo26s_doc_layout.pt ("s" =
small - user's explicit choice over nano/medium, 2026-08-05).

Trained on DocLayNet v1.2, NOT DocStructBench - a different source
dataset with a different 11-class label set (Caption, Footnote,
Formula, List-item, Page-footer, Page-header, Picture, Section-header,
Table, Text, Title - vs DocStructBench's 10 classes: title, plain text,
abandon, figure, figure_caption, table, table_caption, table_footnote,
isolate_formula, formula_caption). Do not assume class names match
between the two sensors when comparing output - they don't, by design
of being different training sources.

Uses the standard `ultralytics` package (installed 2026-08-05,
ultralytics==8.4.115, pip check clean against the existing environment)
rather than core/layout_detector.py's bespoke `doclayout_yolo` package -
these are two independently-loadable model objects, no shared state,
safe to have both installed side by side.

DOMAIN CAVEAT, stated up front rather than assumed away: DocLayNet is
modern/academic-style documents (reports, papers) - not aged microfilm,
handwritten ledgers, or scanned genealogical records, arguably an even
bigger domain gap than DocStructBench's own general/academic training
distribution. The published DocLayNet benchmark numbers (mAP50-95
0.835 for the small variant) say nothing about performance on THIS
corpus - that's exactly what this module exists to let a real test
determine, not assume from the spec sheet.
"""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

YOLO26_REPO_ID = "Armaggheddon/yolo26-document-layout"
YOLO26_SMALL_FILENAME = "yolo26s_doc_layout.pt"

# Fallback only - detect_layout_v26() prefers the live model/result names
# in every normal case, same discipline as core/layout_detector.py.
# DocLayNet's 11 categories, per github.com/Armaggheddon/yolo_doc_layout's
# own README - not yet confirmed against a loaded checkpoint's real
# model.names the way core/layout_detector.py's DOCSTRUCTBENCH_CLASS_NAMES
# was; treat this as documentation, confirm live before trusting it as
# ground truth (build_layout_model_v26()'s own first real call already
# does this via the same names-from-result pattern).
DOCLAYNET_CLASS_NAMES = {
    0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item",
    4: "Page-footer", 5: "Page-header", 6: "Picture", 7: "Section-header",
    8: "Table", 9: "Text", 10: "Title",
}

DEFAULT_IMGSZ = 1280  # per the repo's README: "All models were trained at 1280x1280"
DEFAULT_CONF = 0.2  # matches core/layout_detector.py's default, for a fair side-by-side


def build_layout_model_v26(
    repo_id: str = YOLO26_REPO_ID,
    filename: str = YOLO26_SMALL_FILENAME,
):
    """
    Downloads (or reuses the huggingface_hub local cache for) the
    YOLOv26-small document-layout checkpoint and loads it via the
    standard ultralytics.YOLO class - unlike core/layout_detector.py's
    build_layout_model(), no bespoke constructor workaround needed here;
    this is exactly the pattern the model card's own README documents.
    """
    from ultralytics import YOLO

    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return YOLO(weights_path)


def detect_layout_v26(
    model,
    pil_image: Image.Image,
    imgsz: int = DEFAULT_IMGSZ,
    conf: float = DEFAULT_CONF,
    device: str | None = None,
) -> list[dict]:
    """
    Same return contract as core/layout_detector.py's detect_layout():
    [{"class_id": int, "class_name": str, "confidence": float,
    "bbox_xyxy": [x0, y0, x1, y1]}, ...], most-confident-first, empty
    list (never None) for zero detections. bbox_xyxy in the ORIGINAL
    image's pixel coordinates (ultralytics rescales internally, same as
    doclayout_yolo does) - directly comparable/overlay-able against the
    same images core/layout_detector.py's output was drawn on.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    results = model.predict(pil_image, imgsz=imgsz, conf=conf, device=device, verbose=False)
    result = results[0]
    names = getattr(result, "names", None) or getattr(model, "names", None) or DOCLAYNET_CLASS_NAMES

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
