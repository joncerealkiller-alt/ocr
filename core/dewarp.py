"""
4-point perspective correction ("dewarping") for raw census page scans,
built 2026-07-25 after real evidence that some of the LoRA training
crops exported from warped source pages were themselves illegible even
to a human reviewer - a distorted source image is a ceiling no amount
of downstream model/prompt work can fix, since the ground truth being
labeled against is already unrecoverable in places.

Pure PIL + numpy, no OpenCV - same dependency discipline as
core/image_preprocessing.py (see that module's docstring; this project
deliberately hand-implements things like Otsu threshold and CLAHE
rather than adding OpenCV as a dependency for one feature). Homography
solved via SVD (plain numpy linear algebra, no scipy), warp applied via
INVERSE mapping + vectorized bilinear sampling (every output pixel asks
"where does this come from in the source", not the forward/scatter
approach, which would leave holes in the output wherever the forward
mapping doesn't land on an exact output pixel).

Every function here takes a PIL Image and returns a NEW PIL Image, same
contract as core/image_preprocessing.py - none of them mutate the
input, and the caller's original image object is safe to keep using
afterward.

Corner sidecars (2026-07-26, real incident this fixes): a dewarped
output file got deleted (an overly-broad cleanup command run against
the wrong directory at the wrong time - a real, confirmed mistake, not
a tool bug), and the only way to recover it would have been re-dragging
all 4 corners on every affected page from scratch - real manual human
effort, thrown away. save_dewarp_sidecar()/regenerate_from_sidecar()
elevate the 4 dragged corners to durable metadata, the same way a row-
segmentation sidecar elevates manually-confirmed row geometry (see
core/row_segmentation.py's build_sidecar) - even if the OUTPUT jpg is
lost, the sidecar alone is enough to regenerate byte-for-byte identical
output from the original source image, without asking a human to redo
anything. This does NOT recover output files that were already deleted
BEFORE this feature existed (no sidecar was ever written for those) -
it only protects work done from here forward.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def compute_homography(
    src_points: list[tuple[float, float]], dst_points: list[tuple[float, float]],
) -> np.ndarray:
    """
    Solves for the 3x3 homography matrix H mapping each src_points[i]
    to dst_points[i] (both length-4, homogeneous coordinates implied) -
    the standard Direct Linear Transform (DLT) formulation, solved via
    SVD rather than a hand-rolled 8x8 linear solve: each point
    correspondence contributes 2 rows to a 8x9 matrix A such that
    H (as a 9-vector) is the null space of A; the right singular
    vector corresponding to the smallest singular value IS that null
    space (up to scale), which np.linalg.svd gives directly as the
    last row of Vt. Normalized so H[2,2] == 1, the conventional
    representation (a homography is only defined up to scale).

    Exactly 4 point pairs in, both length-4 - a single quad has exactly
    8 degrees of freedom (4 corners x 2 coords), matching a
    homography's 8 free parameters exactly, no least-squares
    overdetermination needed.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError(
            f"compute_homography needs exactly 4 (x, y) point pairs on each "
            f"side, got src={src.shape}, dst={dst.shape}."
        )

    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(rows, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-12:
        raise ValueError(
            "Degenerate homography (H[2,2] ~ 0) - the 4 source points are "
            "likely collinear or otherwise not a real quadrilateral."
        )
    return H / H[2, 2]


def _bilinear_sample(arr: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Vectorized bilinear sampling of `arr` (h, w, 3) at fractional
    coordinates (x, y) (each an (out_h, out_w) array) - no per-pixel
    Python loop, same vectorization discipline as core/image_
    preprocessing.py's clahe(). Coordinates that fall outside the
    source image's bounds sample as white (255) rather than clamping
    to the nearest edge pixel - a dewarped quad rarely maps to a
    perfect rectangle of the source's own aspect ratio, so SOME output
    corner region falling outside the source is expected (a small
    triangular gap at a corner where the quad wasn't perfectly
    rectangular) and should read as blank page, not smeared edge
    pixels stretched into that gap.
    """
    h, w = arr.shape[:2]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    in_bounds = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]

    top = arr[y0c, x0c] * (1 - wx) + arr[y0c, x1c] * wx
    bottom = arr[y1c, x0c] * (1 - wx) + arr[y1c, x1c] * wx
    result = top * (1 - wy) + bottom * wy

    result[~in_bounds] = 255.0
    return result


def warp_perspective(
    image: Image.Image,
    src_quad: list[tuple[float, float]],
    output_size: tuple[int, int],
) -> Image.Image:
    """
    Flattens the quadrilateral region bounded by src_quad (in `image`'s
    own pixel coordinates, ordered [top_left, top_right, bottom_right,
    bottom_left] - same winding convention as every other bbox/corner
    list in this project) into a squared output_size=(width, height)
    rectangle.

    Computed as an INVERSE warp: solves the homography mapping
    src_quad -> the output rectangle's own 4 corners, then inverts it
    and applies it to every OUTPUT pixel to find where that pixel
    should be sampled FROM in the source - the standard technique for
    resampling without holes (a forward/scatter warp - mapping each
    SOURCE pixel to wherever it lands in the output - leaves gaps
    anywhere the mapping doesn't land exactly on an integer output
    pixel, worse the more the source is being enlarged).
    """
    out_w, out_h = output_size
    if out_w <= 0 or out_h <= 0:
        raise ValueError(f"output_size must be positive, got {output_size!r}")
    dst_quad = [(0, 0), (out_w - 1, 0), (out_w - 1, out_h - 1), (0, out_h - 1)]
    H = compute_homography(src_quad, dst_quad)
    H_inv = np.linalg.inv(H)

    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float64)

    ys, xs = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    ones = np.ones(xs.size, dtype=np.float64)
    dst_coords = np.stack([xs.ravel().astype(np.float64), ys.ravel().astype(np.float64), ones])

    src_coords = H_inv @ dst_coords
    src_coords = src_coords / src_coords[2:3, :]
    src_x = src_coords[0].reshape(out_h, out_w)
    src_y = src_coords[1].reshape(out_h, out_w)

    sampled = _bilinear_sample(arr, src_x, src_y)
    return Image.fromarray(np.clip(sampled, 0, 255).astype(np.uint8))


def estimate_output_size(src_quad: list[tuple[float, float]]) -> tuple[int, int]:
    """
    Picks a reasonable output (width, height) from the quad's own edge
    lengths, rather than forcing a caller to guess one - width is the
    average of the top and bottom edge lengths, height the average of
    the left and right edge lengths, so a slightly trapezoidal
    (imperfectly-dragged) quad doesn't get arbitrarily stretched or
    squashed to some unrelated fixed size. Always at least 1x1 (a
    degenerate zero-area quad still returns something warp_perspective
    can act on rather than dividing by zero downstream).
    """
    tl, tr, br, bl = (np.asarray(p, dtype=np.float64) for p in src_quad)
    top_w = np.linalg.norm(tr - tl)
    bottom_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)
    width = max(1, round((top_w + bottom_w) / 2))
    height = max(1, round((left_h + right_h) / 2))
    return (width, height)


def dewarp_quad(
    image: Image.Image,
    corners: list[tuple[float, float]],
    output_size: tuple[int, int] | None = None,
) -> Image.Image:
    """
    Top-level convenience wrapper: flattens `image` using the 4 corners
    a human dragged into place (ui/dewarp_preprocessor_ui.py's actual
    entry point into this module). output_size defaults to
    estimate_output_size(corners) when not given explicitly.
    """
    if output_size is None:
        output_size = estimate_output_size(corners)
    return warp_perspective(image, corners, output_size)


def dewarp_sidecar_path(output_path: str | Path) -> Path:
    """
    Sidecar JSON path for a dewarped output file - lives right next to
    it, e.g. <name>_dewarped.jpg -> <name>_dewarped.json. Deliberately
    a sibling file, not a separate directory/naming scheme, so it's
    obviously associated with its output file to anyone browsing
    data/outputs/dewarped/ directly.
    """
    return Path(output_path).with_suffix(".json")


def save_dewarp_sidecar(
    source_path: str, output_path: str | Path, corners: list[tuple[float, float]],
) -> Path:
    """
    Persists the 4 corners used to produce output_path, next to it -
    see this module's docstring for the real incident this fixes.
    Overwrites any existing sidecar for this output path (matches
    save_sidecar()'s full-overwrite convention in core/row_segmentation.
    py - a fresh dewarp of the same output path really does supersede
    the prior corner placement, same as a fresh 'Refine rows' does for
    row geometry).
    """
    sidecar_path = dewarp_sidecar_path(output_path)
    record = {
        "source_file_path": str(source_path),
        "output_file_path": str(output_path),
        "corners": [[float(x), float(y)] for x, y in corners],
    }
    sidecar_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return sidecar_path


def load_dewarp_sidecar(sidecar_path: str | Path) -> dict:
    return json.loads(Path(sidecar_path).read_text(encoding="utf-8"))


def regenerate_from_sidecar(sidecar_path: str | Path) -> Path:
    """
    Rebuilds a dewarped output file from its sidecar's saved corners
    and the ORIGINAL source image - the actual self-healing operation
    this module's corner-sidecar feature exists for. Raises
    FileNotFoundError with a clear, specific message (naming which
    file is actually missing) if the sidecar itself is absent, or if
    the original source image is ALSO gone - there's nothing left to
    regenerate FROM in that case, and the caller (ui/dewarp_
    preprocessor_ui.py's worklist self-heal check) needs to know that
    distinction to correctly fall back to re-offering the image for a
    human to redo, rather than silently skipping it.
    """
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"No corner sidecar found at {sidecar_path}")
    record = load_dewarp_sidecar(sidecar_path)
    source_path = Path(record["source_file_path"])
    if not source_path.exists():
        raise FileNotFoundError(
            f"Cannot regenerate {record['output_file_path']!r} - its sidecar exists "
            f"but the original source image is also missing: {source_path}")

    image = Image.open(source_path).convert("RGB")
    flattened = dewarp_quad(image, [tuple(c) for c in record["corners"]])
    output_path = Path(record["output_file_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flattened.save(output_path)
    return output_path
