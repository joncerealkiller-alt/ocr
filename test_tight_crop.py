"""
Regression test for core.row_segmentation.tight_crop_to_ranges() and its
wiring into crop_region_from_source() - added 2026-07-22 after discovering
that a masked single-column crop was still the FULL row width (~3800px on
a real census page) with real content in as little as ~9% of it, because
apply_column_mask() only PAINTS outside the kept range white, it never
narrows the image. Confirmed via a real export where two different
columns' exported training images for the same row were byte-identical
(export_lora_dataset.py had a separate, worse bug: it was reading the
sidecar's legacy top-level mask fields, which are empty under the
per-column architecture, so no masking was ever applied to exported
images at all - fixed alongside this).

No pytest dependency, matching this project's other test_*.py files -
plain asserts, run directly:

    python test_tight_crop.py

Exits non-zero (via AssertionError) on any failure, prints PASS lines
for each check on success.
"""

from __future__ import annotations

from PIL import Image

from core.row_segmentation import tight_crop_to_ranges, crop_region_from_source


def test_basic_fixed_padding():
    img = Image.new("RGB", (3790, 40), "white")
    out = tight_crop_to_ranges(img, [(312, 668)], crop_x0=0, padding_px=20)
    expected_width = (668 - 312) + 2 * 20
    assert out.size == (expected_width, 40), out.size
    print(f"PASS: basic fixed padding -> {out.size}")


def test_percentage_padding():
    img = Image.new("RGB", (3790, 40), "white")
    kept_width = 668 - 312
    out = tight_crop_to_ranges(img, [(312, 668)], crop_x0=0, padding_pct=0.1)
    expected_width = kept_width + 2 * round(kept_width * 0.1)
    assert out.width == expected_width, out.size
    print(f"PASS: percentage padding -> {out.size}")


def test_crop_x0_offset():
    """keep_ranges are in FULL-IMAGE coordinates; the crop itself may
    start partway through the image (e.g. a row bbox that doesn't start
    at x=0) - this must translate correctly, not just work when
    crop_x0 happens to be 0."""
    img = Image.new("RGB", (3790, 40), "white")
    out = tight_crop_to_ranges(img, [(1500, 1600)], crop_x0=1400, padding_px=10)
    assert out.size == (1600 - 1500 + 20, 40), out.size
    print(f"PASS: crop_x0 offset -> {out.size}")


def test_clamping_negative_and_overflow():
    """A keep_range that falls partially outside this particular crop
    (stale mask from a differently-sized row, or padding that would push
    past an edge) must clamp, never request a crop outside image bounds
    (which PIL would raise on)."""
    img = Image.new("RGB", (3790, 40), "white")

    out = tight_crop_to_ranges(img, [(-500, 100)], crop_x0=0, padding_px=20)
    assert out.width == 120, out.size  # clamped left edge to 0

    out = tight_crop_to_ranges(img, [(3700, 5000)], crop_x0=0, padding_px=20)
    assert out.width == 3790 - 3680, out.size  # clamped right edge to image width

    print("PASS: clamping (negative range, overflow range)")


def test_empty_keep_ranges_is_noop():
    img = Image.new("RGB", (3790, 40), "white")
    out = tight_crop_to_ranges(img, [], crop_x0=0)
    assert out.size == (3790, 40), out.size
    print("PASS: empty keep_ranges leaves image unchanged")


def test_exported_width_close_to_column_not_page():
    """THE core regression this test exists for: exported/extracted
    width must be close to the ACTIVE COLUMN's width, not the full page/
    row width. Guards against exactly the bug found 2026-07-22."""
    img = Image.new("RGB", (3790, 40), "white")
    active_column_width = 668 - 312  # 356
    out = tight_crop_to_ranges(img, [(312, 668)], crop_x0=0, padding_px=20)

    assert out.width < active_column_width * 3, (
        f"exported width {out.width} is not close to column width "
        f"{active_column_width} - padding logic may be wrong")
    assert out.width != 3790, (
        "exported width still equals the full page width - tight "
        "crop did not apply at all")
    print(f"PASS: exported width {out.width} close to column width "
          f"{active_column_width} (full page was 3790)")


def test_crop_region_from_source_backward_compatible(tmp_path_str="/tmp"):
    """Existing callers that don't pass tight_crop_keep_ranges must see
    IDENTICAL output to before this parameter existed - the legacy
    multi-column extraction path depends on this."""
    import os
    img_path = os.path.join(tmp_path_str, "test_tight_crop_source.png")
    Image.new("RGB", (500, 200), "white").save(img_path)

    # no tight_crop_keep_ranges passed - must behave exactly as the
    # pre-2026-07-22 signature did
    result = crop_region_from_source(img_path, [10, 10, 400, 100], deskew_angle=0.0)
    assert result.size == (390, 90), result.size
    print(f"PASS: crop_region_from_source with no tight_crop args -> {result.size} "
          f"(unchanged from pre-fix behavior)")

    os.remove(img_path)


def test_crop_region_from_source_with_tight_crop(tmp_path_str="/tmp"):
    import os
    img_path = os.path.join(tmp_path_str, "test_tight_crop_source2.png")
    Image.new("RGB", (4000, 200), "white").save(img_path)

    bbox = [0, 0, 4000, 40]
    result = crop_region_from_source(
        img_path, bbox, deskew_angle=0.0,
        mask_ranges=[(0, 300), (700, 4000)],  # everything except 300-700 painted white
        tight_crop_keep_ranges=[(300, 700)],
        tight_crop_padding_px=20,
    )
    expected_width = (700 - 300) + 2 * 20
    assert result.size == (expected_width, 40), result.size
    print(f"PASS: crop_region_from_source with tight_crop_keep_ranges -> {result.size}")

    os.remove(img_path)


if __name__ == "__main__":
    test_basic_fixed_padding()
    test_percentage_padding()
    test_crop_x0_offset()
    test_clamping_negative_and_overflow()
    test_empty_keep_ranges_is_noop()
    test_exported_width_close_to_column_not_page()
    test_crop_region_from_source_backward_compatible()
    test_crop_region_from_source_with_tight_crop()
    print("\nALL TESTS PASSED")
