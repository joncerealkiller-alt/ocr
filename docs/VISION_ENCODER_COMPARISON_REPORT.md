# Vision Encoder Comparison Report

**Auto-generated from `data/outputs/vision_encoder_qualification_log.jsonl` by `benchmark/vision_encoder_comparison_report.py`. 8 candidates qualified under Vision Qualification Battery v1.0 (frozen - transforms/metrics/pass-fail unchanged since Round 1). Regenerate after every completed round; do not hand-edit the tables below - edit `ARCHITECTURE_FACTS` or the generator script instead. Interpretive analysis lives in `docs/VISION_IR_RESEARCH.md`, not here.**

## Operational characteristics

| Candidate | Family | Params (M) | Embed dim | Patches | Prefix tokens | Pooling | CPU ms | GPU ms | GPU peak MB | Native res | Dynamic res | Feature pyramid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vit_small_patch14_dinov2.lvd142m | DINOv2 | 22.06 | 384 | 1370 | 1 | cls_token | 188.3 | 14.3 | 151.2 | False | opt-in flag, patch-aligned input only | same-resolution depth stack (not true multi-scale) |
| convnext_tiny.fb_in22k | ConvNeXt | 27.82 | 768 | 49 | 0 | spatial_mean (conv/hierarchical) | 31.8 | 6.6 | 162.3 | True | native, no flag needed | true multi-scale (56->28->14->7) |
| naflexvit_base_patch16_siglip.v2_webli | SigLIP (NaFlex) | 92.93 | 768 | 576 | 0 | attention_pooling (no prefix token) | 204.8 | 12.5 | 429.9 | True | native (NaFlex design point) - but only via the dedicated patchify/patch_coord call path, NOT timm's default create_transform (see Round 4) | same-resolution depth stack (not true multi-scale) |
| vit_base_patch16_siglip_224.v2_webli | SigLIP (fixed-res) | 92.88 | 768 | 196 | 0 | attention_pooling (no prefix token) | 82.7 | 6.5 | 412.9 | False | opt-in flag, patch-aligned input only | same-resolution depth stack (not true multi-scale) |
| eva02_base_patch14_224.mim_in22k | EVA-02 | 85.76 | 768 | 257 | 1 | cls_token | 105.8 | 15.4 | 390.2 | False | opt-in flag, patch-aligned input only | same-resolution depth stack (not true multi-scale) |
| beit_base_patch16_224.in22k_ft_in22k | BEiT | 85.76 | 768 | 197 | 1 | cls_token | 84.2 | 7.9 | 389.1 | False | NOT implemented in timm's BEiT class | same-resolution depth stack (not true multi-scale) |
| swin_base_patch4_window7_224.ms_in22k | Swin | 86.74 | 1024 | 49 | 0 | spatial_mean (conv/hierarchical, NHWC) | 98.7 | 27.0 | 410.7 | False | NOT supported (window-partitioning constraint) | true multi-scale (56->28->14->7) |
| vit_base_patch16_224.mae | ? | 85.8 | 768 | 197 | 1 | cls_token | 79.0 | 6.2 | 384.6 | ? | ? | ? |

## Qualification metrics (Vision Qualification Battery v1.0)

### Aggregate summary (mean across all 10 transforms)

| Candidate | Mean recall@1 | Mean drift | Mean Jaccard@5 | Cross-repr. consistency |
|---|---|---|---|---|
| vit_small_patch14_dinov2.lvd142m | 0.971 | 0.0475 | 0.876 | 0.862 |
| convnext_tiny.fb_in22k | 1.000 | 0.0478 | 0.868 | 0.894 |
| naflexvit_base_patch16_siglip.v2_webli | 0.996 | 0.0345 | 0.867 | 0.761 |
| vit_base_patch16_siglip_224.v2_webli | 0.992 | 0.0290 | 0.876 | 0.779 |
| eva02_base_patch14_224.mim_in22k | 0.992 | 0.0434 | 0.846 | 0.969 |
| beit_base_patch16_224.in22k_ft_in22k | 0.996 | 0.0418 | 0.896 | 0.938 |
| swin_base_patch4_window7_224.ms_in22k | 0.988 | 0.0480 | 0.907 | 1.000 |
| vit_base_patch16_224.mae | 0.904 | 0.0056 | 0.839 | 0.696 |

**An aggregate mean can hide a single-transform outlier** (this is exactly how DINOv2's inversion result would look diluted into a mean) - the full per-transform tables below are the primary evidence, not this summary. Read the summary as an index into the detail, not a replacement for it.

### Full per-transform breakdown — Recall@1

| Candidate | border_crop_5pct | contrast_autocontrast | contrast_enhance_1.5x | denoise_median3 | deskew_+0.5deg | deskew_+2.0deg | deskew_-2.0deg | downscale_0.5x | invert | upscale_2x |
|---|---|---|---|---|---|---|---|---|---|---|
| vit_small_patch14_dinov2.lvd142m | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.875 | 0.917 | 1.000 | 0.917 | 1.000 |
| convnext_tiny.fb_in22k | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| naflexvit_base_patch16_siglip.v2_webli | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 1.000 | 1.000 | 1.000 | 1.000 |
| vit_base_patch16_siglip_224.v2_webli | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.958 | 1.000 | 1.000 | 1.000 |
| eva02_base_patch14_224.mim_in22k | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 |
| beit_base_patch16_224.in22k_ft_in22k | 0.958 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| swin_base_patch4_window7_224.ms_in22k | 0.958 | 1.000 | 1.000 | 1.000 | 1.000 | 0.958 | 0.958 | 1.000 | 1.000 | 1.000 |
| vit_base_patch16_224.mae | 0.958 | 1.000 | 1.000 | 1.000 | 1.000 | 0.917 | 0.958 | 1.000 | 0.208 | 1.000 |

### Full per-transform breakdown — Mean cosine drift

| Candidate | border_crop_5pct | contrast_autocontrast | contrast_enhance_1.5x | denoise_median3 | deskew_+0.5deg | deskew_+2.0deg | deskew_-2.0deg | downscale_0.5x | invert | upscale_2x |
|---|---|---|---|---|---|---|---|---|---|---|
| vit_small_patch14_dinov2.lvd142m | 0.032 | 0.000 | 0.037 | 0.031 | 0.029 | 0.075 | 0.071 | 0.033 | 0.163 | 0.002 |
| convnext_tiny.fb_in22k | 0.075 | 0.000 | 0.063 | 0.029 | 0.023 | 0.086 | 0.085 | 0.034 | 0.081 | 0.002 |
| naflexvit_base_patch16_siglip.v2_webli | 0.053 | 0.000 | 0.028 | 0.031 | 0.035 | 0.055 | 0.055 | 0.027 | 0.057 | 0.003 |
| vit_base_patch16_siglip_224.v2_webli | 0.048 | 0.000 | 0.023 | 0.025 | 0.025 | 0.045 | 0.043 | 0.021 | 0.059 | 0.002 |
| eva02_base_patch14_224.mim_in22k | 0.052 | 0.000 | 0.033 | 0.026 | 0.023 | 0.057 | 0.052 | 0.020 | 0.170 | 0.000 |
| beit_base_patch16_224.in22k_ft_in22k | 0.081 | 0.000 | 0.053 | 0.016 | 0.027 | 0.076 | 0.076 | 0.023 | 0.064 | 0.001 |
| swin_base_patch4_window7_224.ms_in22k | 0.082 | 0.000 | 0.054 | 0.022 | 0.030 | 0.084 | 0.097 | 0.034 | 0.073 | 0.003 |
| vit_base_patch16_224.mae | 0.001 | 0.000 | 0.002 | 0.000 | 0.001 | 0.002 | 0.002 | 0.001 | 0.047 | 0.000 |

### Full per-transform breakdown — Mean neighborhood Jaccard@5

| Candidate | border_crop_5pct | contrast_autocontrast | contrast_enhance_1.5x | denoise_median3 | deskew_+0.5deg | deskew_+2.0deg | deskew_-2.0deg | downscale_0.5x | invert | upscale_2x |
|---|---|---|---|---|---|---|---|---|---|---|
| vit_small_patch14_dinov2.lvd142m | 0.879 | 1.000 | 0.848 | 0.879 | 0.855 | 0.834 | 0.841 | 0.869 | 0.786 | 0.972 |
| convnext_tiny.fb_in22k | 0.804 | 1.000 | 0.786 | 0.907 | 0.935 | 0.790 | 0.813 | 0.885 | 0.813 | 0.948 |
| naflexvit_base_patch16_siglip.v2_webli | 0.778 | 1.000 | 0.917 | 0.851 | 0.833 | 0.833 | 0.861 | 0.845 | 0.796 | 0.958 |
| vit_base_patch16_siglip_224.v2_webli | 0.841 | 1.000 | 0.879 | 0.897 | 0.903 | 0.828 | 0.804 | 0.917 | 0.745 | 0.944 |
| eva02_base_patch14_224.mim_in22k | 0.786 | 0.986 | 0.792 | 0.903 | 0.875 | 0.786 | 0.796 | 0.903 | 0.662 | 0.972 |
| beit_base_patch16_224.in22k_ft_in22k | 0.851 | 1.000 | 0.861 | 0.921 | 0.907 | 0.833 | 0.833 | 0.958 | 0.792 | 1.000 |
| swin_base_patch4_window7_224.ms_in22k | 0.800 | 1.000 | 0.889 | 0.944 | 0.948 | 0.845 | 0.873 | 0.931 | 0.879 | 0.958 |
| vit_base_patch16_224.mae | 0.790 | 1.000 | 0.827 | 0.944 | 0.944 | 0.848 | 0.872 | 0.972 | 0.195 | 1.000 |

## Observed behavioral notes

(Interpretive - written by hand in `docs/VISION_IR_RESEARCH.md` after each round, not auto-generated here. This report is the quantitative substrate for that analysis, not a replacement for it.)
