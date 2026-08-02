"""
Experiment 3 (deferred since Round 2): can ConvNeXt's genuine
multi-scale spatial feature map localize structure on real images
where classical CV's own boundary detector failed?

See docs/VISION_IR_RESEARCH.md's original Experiment 3 design: this is
explicitly the softest of the qualification experiments (no numeric
pass/fail threshold by design) - the practical criterion is whether a
human looking at the heatmap overlay agrees it picks out real
structure on a known CV-failure image, not a score. Produces a PNG
overlay for direct visual inspection.

Usage:
    python -m benchmark.experiment3_convnext_localization <image_path>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image

CANDIDATE = "convnext_tiny.fb_in22k"


def compute_heatmap(image_path: str) -> Image.Image:
    """
    Side-by-side (original | heatmap-alone), NOT a color-tint overlay -
    real bug caught by direct visual inspection: a red-channel-boost
    overlay is indistinguishable from a document's own inherent color
    cast (this specific test image is a naturally pink/rose-toned
    photograph, not a grayscale scan), making the first version of this
    script uninterpretable on exactly the kind of real archival image
    it needs to work on. A separate heatmap panel, using a perceptually
    uniform colormap-free grayscale intensity map, avoids that
    confound entirely.
    """
    model = timm.create_model(CANDIDATE, pretrained=True, features_only=True)
    model.eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)

    orig = Image.open(image_path).convert("RGB")
    x = transform(orig).unsqueeze(0)
    with torch.no_grad():
        feats = model(x)  # list of 4 stages, coarsest = feats[-1], shape (1, C, h, w)

    coarse = feats[-1][0]  # (C, h, w) - coarsest stage, largest receptive field
    heat = coarse.mean(dim=0).numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(orig.size, Image.NEAREST)

    # Downscale the original for a manageable side-by-side panel size.
    display_w = 900
    scale = display_w / orig.width
    display_size = (display_w, int(orig.height * scale))
    orig_small = orig.resize(display_size, Image.LANCZOS)
    heat_small = heat_img.convert("L").resize(display_size, Image.NEAREST)

    canvas = Image.new("RGB", (display_size[0] * 2 + 20, display_size[1]), "white")
    canvas.paste(orig_small, (0, 0))
    canvas.paste(heat_small.convert("RGB"), (display_size[0] + 20, 0))
    return canvas


if __name__ == "__main__":
    image_path = sys.argv[1]
    out_path = Path(image_path).stem + "_convnext_heatmap.png"
    result = compute_heatmap(image_path)
    result.save(out_path)
    print(f"Saved heatmap overlay to {out_path}")
