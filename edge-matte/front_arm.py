"""Register the uncut design photograph to the arm contours in the opening pose."""

import io
import json
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.special import ndtr
from track import HERE, ROOT, file_sha

SOURCE = ROOT / "release/全身环绕_抗锯齿透明PNG_72帧.zip"
OUTPUT = ROOT / "release/全身环绕_正面手臂精修_72帧.zip"
REPORT = HERE / "front-arm.json"
ROWS = np.arange(565, 942)
WEIGHTS = {
    0: 1,
    1: 0.84,
    2: 0.55,
    3: 0.29,
    4: 0.10,
    63: 0.12,
    64: 0.32,
    65: 0.55,
    66: 0.78,
    67: 0.92,
    68: 0.98,
    69: 1,
    70: 1,
    71: 1,
}


def ease(x):
    t = np.clip(x, 0, 1)
    return t * t * (3 - 2 * t)


def sample(image, x):
    y = np.broadcast_to(ROWS[:, None], x.shape)
    return cv2.remap(image, x.astype("f4"), y.astype("f4"), cv2.INTER_LINEAR)


def crossing(image):
    a = image[ROWS, :, 3].astype("f4") / 255
    ix = (a > 0.5).argmax(1)
    k = np.arange(len(ix))
    a0 = a[k, ix - 1]
    return ix - 1 + (0.5 - a0) / np.maximum(a[k, ix] - a0, 1e-5)


def photo_profile(ref, front):
    guide = crossing(front)[:, None]
    steps = np.arange(-8, 8.1, 0.25)[None, :]
    px = guide + steps
    bg = sample(ref, guide + np.arange(-12, -7)).mean(1)
    skin = sample(ref, guide + np.arange(2, 6)).mean(1)
    axis = skin - bg
    contrast = np.sum(
        (sample(ref, px + 0.7) - sample(ref, px - 0.7)) * axis[:, None], axis=2
    ) / np.maximum(np.sum(axis**2, axis=1, keepdims=True), 12)
    best = (contrast - 0.002 * (steps**2)).argmax(1)
    edge = px[np.arange(len(ROWS)), best]
    # Measuring across the original, uncut skin/background transition retains the arm's
    # shallow anatomical curvature. Reject local pixel noise before evaluating coverage.
    return gaussian_filter1d(median_filter(edge, size=5), 5.2)


def restore_side(image, front, ref, profile, strength):
    guide = crossing(image)
    reference = crossing(front)
    shift = gaussian_filter1d(guide - reference, 7)
    edge = profile + shift
    edge = guide + np.clip(edge - guide, -4, 4)
    # Stay clear of the cuff and the uppermost shoulder where the contour is horizontal.
    fade = strength * ease((ROWS - 582) / 24) * ease((930 - ROWS) / 24)
    x = np.arange(image.shape[1])[None, :]
    d = x - edge[:, None]
    weight = fade[:, None] * ease((15 - np.abs(d)) / 5)
    # Work in coordinates following the photographed edge. Remove gray inherited
    # from the H3 backing, while retaining the real falloff just inside the skin.
    offsets = np.arange(-16, 18)
    tex = sample(ref, profile[:, None] + np.maximum(offsets, 1.8)[None, :])
    tex = gaussian_filter1d(tex, 1.75, axis=0)
    swatch = np.arange(9, 14)
    delta = (
        sample(image[:, :, :3].astype("f4"), edge[:, None] + swatch)
        - sample(ref, profile[:, None] + swatch)
    ).mean(1)
    tex += gaussian_filter1d(delta, 4, axis=0)[:, None]
    lookup = (np.clip(d, offsets[0], offsets[-1]) - offsets[0]).astype("f4")
    color = cv2.remap(
        tex,
        lookup,
        np.broadcast_to(np.arange(len(ROWS), dtype="f4")[:, None], lookup.shape).copy(),
        cv2.INTER_LINEAR,
    )
    slope = np.gradient(edge)
    coverage = ndtr(d / (0.93 * np.sqrt(1 + slope[:, None] ** 2)))
    base = image[ROWS].astype("f4")
    a = base[:, :, 3] / 255
    new_a = a + weight * (coverage - a)
    premul = (
        base[:, :, :3] * a[:, :, None] * (1 - weight[:, :, None])
        + color * (weight * coverage)[:, :, None]
    )
    new = np.dstack((premul / np.maximum(new_a[:, :, None], 1e-7), 255 * new_a))
    new = np.rint(new).clip(0, 255).astype("u1")
    new[new[:, :, 3] == 0] = 0
    current = image[ROWS].copy()
    current[weight > 0] = new[weight > 0]
    image[ROWS] = current


def main():
    verified = json.loads((HERE / "contour-verification.json").read_text())
    assert file_sha(SOURCE) == verified["output_sha256"]
    original = Image.open(ROOT / "provenance/half/inputs/01_front.png").convert("RGB")
    scale = 1.1137452711223204
    photo = np.asarray(
        original.transform(
            (640, 1920),
            Image.Transform.AFFINE,
            (
                1 / scale,
                0,
                202.3465321563682 / scale,
                0,
                1 / scale,
                -36.70517023959644 / scale,
            ),
            Image.Resampling.BICUBIC,
        )
    ).astype("f4")
    work = ROOT / "work/front-arm"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as tmp, zipfile.ZipFile(SOURCE) as src:
        staged = Path(tmp) / OUTPUT.name
        front = np.array(Image.open(io.BytesIO(src.read("orbit_000.png"))))
        sides = [slice(None), slice(None, None, -1)]
        profiles = [photo_profile(photo[:, s], front[:, s]) for s in sides]
        with zipfile.ZipFile(staged, "w") as dst:
            for i, entry in enumerate(src.infolist()):
                data = src.read(entry)
                if i not in WEIGHTS:
                    dst.writestr(entry, data)
                    continue
                old = np.array(Image.open(io.BytesIO(data)))
                new = old.copy()
                for side, profile in zip(sides, profiles):
                    restore_side(
                        new[:, side],
                        front[:, side],
                        photo[:, side],
                        profile,
                        WEIGHTS[i],
                    )
                changed = np.any(new != old, 2)
                assert (
                    changed.any()
                    and not changed[:583].any()
                    and not changed[930:].any()
                )
                assert changed.sum() < 14000
                with dst.open(entry, "w") as f:
                    Image.fromarray(new).save(f, format="PNG", compress_level=6)
        assert len(src.infolist()) == 72
        report = {
            "source_sha256": verified["output_sha256"],
            "output_sha256": file_sha(staged),
            "front_view_weights": WEIGHTS,
        }
        staged.replace(OUTPUT)
        REPORT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
