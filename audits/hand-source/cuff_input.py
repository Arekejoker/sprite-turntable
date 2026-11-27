"""Refine only the backdrop visible between the white cuffs and thighs in the design photo."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "provenance/half/inputs"
# Source RGB coordinates bound the narrow, light background between two dark edge lines.
# Reversed limits mark places where cuff and thigh touch in the photograph.
LEFT = [
    (821, 342, 340),
    (832, 340, 338),
    (834, 338.5, 339.7),
    (837, 337.5, 339.6),
    (840, 336.1, 339.2),
    (844, 334.4, 338.5),
    (848, 333.2, 337.8),
    (852, 332, 337.2),
    (858, 329.8, 336.8),
    (865, 326.2, 336.8),
    (870, 323.9, 336.8),
    (877, 320.7, 336.9),
]
RIGHT = [
    (814, 587.4, 598.1),
    (818, 588.7, 598.3),
    (823, 590.6, 600.3),
    (828, 591.6, 597.7),
    (832, 592.3, 595.0),
    (834, 593, 594.1),
    (836, 595, 593),
    (840, 596, 594),
    (842, 594.4, 595.7),
    (844, 594.8, 596.5),
    (847, 595.4, 597.6),
    (850, 595.9, 598.6),
    (854, 596.7, 600.0),
    (858, 597.2, 601.3),
    (863, 597.8, 603.4),
    (868, 598.4, 604.8),
    (876, 599.1, 607.4),
]


def refine(alpha, rgb, knots, bg_green):
    output = alpha.copy()
    guides = np.asarray(knots)
    xs = np.arange(rgb.shape[1])
    for y in range(knots[0][0], knots[-1][0] + 1):
        lo, hi = [np.interp(y, guides[:, 0], guides[:, k]) for k in (1, 2)]
        # Restore foreground where the old matte had a notch. Replace only the small
        # interstitial strip; every outer arm edge, fingertip and the rest of the photo are untouched.
        span = (xs >= min(lo, hi) - 4) & (xs <= max(lo, hi) + 4)
        possible = (xs >= lo - 0.4) & (xs <= hi + 0.4) & (lo < hi)
        fraction_bg = np.clip((rgb[y, :, 1].astype(float) - 70) / (bg_green - 70), 0, 1)
        corrected = 255 * (1 - fraction_bg * possible)
        mix = np.clip(min(y - knots[0][0], knots[-1][0] - y) / 4, 0, 1)
        output[y, span] = np.rint(
            (1 - mix) * alpha[y, span] + mix * corrected[span]
        ).astype("u1")
    return output


def condition(rgb, alpha):
    subject = Image.fromarray(rgb).convert("RGBA")
    subject.putalpha(Image.fromarray(alpha))
    s = 1.1137452711223204
    tx, ty = -202.3465321563682, 36.70517023959644
    plate = (
        subject.convert("RGBa")
        .transform(
            (640, 1920),
            Image.Transform.AFFINE,
            (1 / s, 0, -tx / s, 0, 1 / s, -ty / s),
            Image.Resampling.BICUBIC,
        )
        .convert("RGBA")
    )
    gray = Image.new("RGB", plate.size, (128, 128, 128))
    gray.paste(plate, mask=plate.getchannel("A"))
    return gray


def instep_condition():
    """Build the combined photographic reference while preserving the input of the recorded take."""
    rgb = np.asarray(Image.open(SOURCE / "01_front.png").convert("RGB"))
    alpha = np.asarray(Image.open(SOURCE / "01_front_rgba.png").getchannel("A"))
    repaired = np.asarray(
        Image.open(HERE.parents[1] / "assets/正面定妆_鞋口补全.png").convert("RGBA")
    )
    shoe = repaired[:, :, 3] != alpha
    ys, xs = np.where(shoe)
    if (
        len(xs) != 261
        or (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) != (613, 1535, 626, 1564)
        or not np.all(repaired[:, :, 3][shoe] == 255)
        or not np.array_equal(rgb, repaired[:, :, :3])
    ):
        raise ValueError(
            "The verified photographic instep or its original RGB has changed."
        )
    cuffs = refine(refine(alpha, rgb, LEFT, 199), rgb, RIGHT, 205)
    cuffs[shoe] = repaired[:, :, 3][shoe]
    return condition(rgb, cuffs)


def main():
    rgb = np.array(Image.open(SOURCE / "01_front.png").convert("RGB"))
    alpha = np.array(Image.open(SOURCE / "01_front_rgba.png").getchannel("A"))
    old = np.array(condition(rgb, alpha))
    if not np.array_equal(old, np.array(Image.open(HERE / "inputs/fullbody-grey.png"))):
        raise RuntimeError("The original H3 reference could not be reproduced")
    cleaned = refine(refine(alpha, rgb, LEFT, 199), rgb, RIGHT, 205)
    corrected = condition(rgb, cleaned)
    new = np.array(corrected)
    heights, widths = np.where(np.any(old != new, axis=2))
    # Both fingertip sets begin below this local correction. Check their full image rows and the head.
    if min(heights) < 929 or max(heights) >= 1020 or not (0 < len(widths) < 3500):
        raise RuntimeError("Unexpected spatial extent of the input change")
    for x0, y0, x1, y1 in ((99, 1016, 178, 1107), (465, 1024, 553, 1121)):
        if not np.array_equal(old[y0:y1, x0:x1], new[y0:y1, x0:x1]):
            raise RuntimeError("The fingertip evidence was modified")
    corrected.save(HERE / "inputs/fullbody-cuffs-grey.png")
    instep_condition().save(HERE / "inputs/fullbody-cuffs-instep-grey.png")
    (HERE / "cuff-input.json").write_text(
        json.dumps(
            {
                "design_alpha_pixels_changed": int((alpha != cleaned).sum()),
                "condition_rgb_pixels_changed": len(widths),
                "condition_change_bounds_xyxy": [
                    int(widths.min()),
                    int(heights.min()),
                    int(widths.max() + 1),
                    int(heights.max() + 1),
                ],
                "front_fingertip_boxes_identical": [
                    [99, 1016, 178, 1107],
                    [465, 1024, 553, 1121],
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
