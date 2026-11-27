"""Antialias the boundary of native RGBA plates in premultiplied colour space."""

import json
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from track import COUNT, HERE, ROOT, file_sha

BASE = ROOT / "release/全身环绕_透明PNG_72帧.zip"
OUTPUT = ROOT / "release/全身环绕_抗锯齿透明PNG_72帧.zip"
RECORD = HERE / "contour-verification.json"


def ramp(x, low, high):
    return np.clip((x - low) / (high - low), 0, 1)


def refine(original):
    a8 = original[:, :, 3]
    bounds = cv2.boundingRect(cv2.findNonZero((a8 > 0).astype("u1")))
    x, y, w, h = bounds
    region = np.s_[max(0, y - 10) : y + h + 10, max(0, x - 10) : x + w + 10]
    src = original[region]
    a = src[:, :, 3].astype("f4") / 255
    hard = (a >= 0.5).astype("u1")
    near = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, np.ones((7, 7), "u1")) > 0
    # Continuous edge normal from a coarse coverage field. Only edit where the
    # foreground and air on opposite sides are individually well established.
    smooth = cv2.GaussianBlur(a, (0, 0), 2.4)
    dy, dx = np.gradient(smooth)
    length = np.hypot(dx, dy)
    nx, ny = dx / np.maximum(length, 0.001), dy / np.maximum(length, 0.001)
    Y, X = np.where(near)
    px, py = X.astype("f4")[:, None], Y.astype("f4")[:, None]
    nx, ny = nx[Y, X, None], ny[Y, X, None]

    def sample(image, x, y):
        return np.concatenate(
            [
                cv2.remap(
                    image,
                    x[k : k + 24000],
                    y[k : k + 24000],
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )[:, 0]
                for k in range(0, len(x), 24000)
            ]
        )

    inside = sample(a, px + 4 * nx, py + 4 * ny)
    outside = sample(a, px - 4 * nx, py - 4 * ny)
    trust = (
        ramp(inside, 0.86, 0.995)
        * ramp(1 - outside, 0.86, 0.995)
        * ramp(length[Y, X], 0.065, 0.125)
    )
    # Where the tangent curves sharply (tips, thin shafts, tight gaps), retain the
    # original shape. Along broad contours integrate subpixel samples on both sides.
    xx = cv2.GaussianBlur(dx * dx, (0, 0), 2.8)[Y, X]
    yy = cv2.GaussianBlur(dy * dy, (0, 0), 2.8)[Y, X]
    xy = cv2.GaussianBlur(dx * dy, (0, 0), 2.8)[Y, X]
    coherence = np.hypot(xx - yy, 2 * xy) / np.maximum(xx + yy, 1e-8)
    trust *= ramp(coherence, 0.55, 0.93)
    pre = np.dstack((src[:, :, :3] * a[:, :, None], a))
    total = np.zeros((len(X), 4), "f4")
    weight = 0.0
    for t in range(-7, 8):
        for n, normal_weight in ((-0.65, 0.24), (0, 0.52), (0.65, 0.24)):
            k = float(np.exp(-0.5 * (t / 3.0) ** 2) * normal_weight)
            total += k * sample(pre, px - ny * t + nx * n, py + nx * t + ny * n)
            weight += k
    filtered = pre[Y, X] + trust[:, None] * (total / weight - pre[Y, X])
    result = original.copy()
    target = result[region]
    new_a = np.rint(255 * np.clip(filtered[:, 3], 0, 1)).astype("u1")
    changed = (new_a != src[Y, X, 3]) & (trust > 0)
    Y, X, filtered, new_a = Y[changed], X[changed], filtered[changed], new_a[changed]
    color = filtered[:, :3] / np.maximum(filtered[:, 3, None], 1e-8)
    color[new_a == 0] = 0
    target[Y, X, :3] = np.rint(np.clip(color, 0, 255)).astype("u1")
    target[Y, X, 3] = new_a
    assert np.array_equal(src[~near], target[~near])
    return result


def arm_roughness(old, new):
    # Measure short-period lateral motion of the continuous half-coverage edge
    # on the outer two arm contours. Use the same uninterrupted runs in both images.
    def traces(a):
        a = a[630:930].astype(float) / 255
        m = a >= 0.5
        i = m.argmax(1)
        v = np.arange(len(i))
        p = a[v, i - 1]
        return i - 1 + (0.5 - p) / np.maximum(a[v, i] - p, 1e-6)

    scores = np.zeros(3)
    for side in (slice(None), slice(None, None, -1)):
        p, q = (traces(a[:, side]) for a in (old, new))
        ok = (np.abs(np.diff(p)) < 1.6) & (np.abs(np.diff(q)) < 1.6)
        ok = ok[:-1] & ok[1:]
        scores += [
            np.sum(np.diff(p, 2)[ok] ** 2),
            np.sum(np.diff(q, 2)[ok] ** 2),
            ok.sum(),
        ]
    return scores


def main():
    cv2.setNumThreads(2)
    sha = file_sha(BASE)
    assert (
        sha
        == json.loads((HERE / "verification.json").read_text())["png_archive_sha256"]
    )
    work = ROOT / "work/edge-aa"
    work.mkdir(exist_ok=True, parents=True)
    scores = np.zeros(3)
    counts = []
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        stage = Path(tmp) / OUTPUT.name
        with zipfile.ZipFile(BASE) as src, zipfile.ZipFile(stage, "w") as dst:
            for i in range(COUNT):
                entry = src.getinfo(f"orbit_{i:03}.png")
                with src.open(entry) as f:
                    old = np.array(Image.open(f))
                new = refine(old)
                scores += arm_roughness(old[:, :, 3], new[:, :, 3])
                counts.append(int(np.any(new != old, axis=2).sum()))
                assert (
                    abs(new[:, :, 3].sum(dtype=float) - old[:, :, 3].sum(dtype=float))
                    / old[:, :, 3].sum()
                    < 0.002
                )
                with dst.open(entry, "w") as f:
                    Image.fromarray(new).save(f, format="PNG", compress_level=6)
        assert scores[1] < scores[0] * 0.80 and min(counts) > 0
        report = {
            "source_sha256": sha,
            "output_sha256": file_sha(stage),
            "frames": len(counts),
            "changed_boundary_pixels_by_frame": counts,
            "arm_contour_samples": int(scores[2]),
            "arm_second_difference_rms_before_after_px": np.sqrt(
                scores[:2] / scores[2]
            ).tolist(),
        }
        stage.replace(OUTPUT)
        RECORD.write_text(json.dumps(report, indent=2) + "\n")
        print(report["arm_second_difference_rms_before_after_px"], flush=True)


if __name__ == "__main__":
    main()
