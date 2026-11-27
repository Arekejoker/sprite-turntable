"""Measure the stocking–leather seam from the archived native take and design photo."""

import json

import cv2
import numpy as np
from track import HERE, SIZE, TAKE, decoder

# Bounds reviewed against the native frames. Each triple gives the first and last
# scanlines around the mark and its projected width towards the inside of the foot.
WINDOWS = {
    0: (1745, 1780, 13),
    1: (1748, 1782, 13),
    2: (1750, 1784, 13),
    3: (1753, 1786, 13),
    4: (1755, 1789, 13),
    5: (1759, 1791, 13),
    6: (1762, 1795, 13),
    7: (1765, 1799, 13),
    8: (1768, 1800, 13),
    9: (1769, 1802, 13),
    10: (1771, 1804, 13),
    11: (1772, 1805, 13),
    12: (1775, 1807, 13),
    13: (1775, 1809, 13),
    14: (1776, 1810, 13),
    15: (1776, 1809, 12),
    16: (1776, 1808, 10),
    17: (1776, 1806, 9),
    18: (1778, 1803, 6),
    19: (1769, 1794, 4),
    54: (1704, 1731, 4),
    55: (1708, 1736, 5),
    56: (1708, 1739, 6),
    57: (1711, 1743, 7),
    58: (1714, 1748, 8),
    59: (1718, 1752, 9),
    60: (1723, 1756, 10),
    61: (1729, 1762, 11),
    62: (1733, 1766, 12),
    63: (1736, 1770, 12),
    64: (1740, 1775, 13),
    65: (1742, 1777, 13),
    66: (1744, 1779, 13),
    **{i: (1745, 1780, 13) for i in range(67, 72)},
}
BAND_TOP = 1580


def locate_seam(green, ys, guide, search):
    smooth = cv2.GaussianBlur(green.astype(np.float32), (3, 5), 0.65)
    descent = np.clip(-np.gradient(smooth, axis=1), -80, 80)
    xs = np.rint(guide).astype(int)[:, None] + np.arange(-search, search + 1)
    score = descent[ys[:, None], xs] - 0.58 * (xs - guide[:, None]) ** 2
    # Follow a continuous darkening edge. Specular lines on the leather run beside it.
    step = xs[1:, :, None] - xs[:-1, None, :]
    cost = 5 * (step - np.diff(guide)[:, None, None]) ** 2
    total = score[0]
    path = []
    for i in range(1, len(ys)):
        scores = total[None, :] - cost[i - 1]
        path.append(scores.argmax(1))
        total = score[i] + scores.max(1)
    indices = [total.argmax()]
    for row in reversed(path):
        indices.append(row[indices[-1]])
    peak = xs[np.arange(len(ys)), indices[::-1]]
    a, b, c = (descent[ys, peak + dx] for dx in (-1, 0, 1))
    return peak + np.clip(0.5 * (a - c) / np.minimum(a - 2 * b + c, -0.01), -0.5, 0.5)


def main():
    from cuff_input import instep_condition

    cv2.setNumThreads(2)
    strips = [np.asarray(im)[BAND_TOP:1880].copy() for im in decoder(TAKE, SIZE)]
    gray = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in strips]
    first = np.array([[494, 1747], [483, 1779]], dtype=np.float32)
    seeds = {0: first}
    offset = np.array([0, BAND_TOP], np.float32)
    for traversal in (range(1, 20), range(71, 53, -1)):
        previous = 0
        points = (first - offset).reshape(-1, 1, 2)
        for i in traversal:
            points, status, _ = cv2.calcOpticalFlowPyrLK(
                gray[previous],
                gray[i],
                points,
                None,
                winSize=(27, 31),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 0.007),
            )
            if not status.all():
                raise RuntimeError(f"Lost the instep seam at native view {i}")
            seeds[i] = points.reshape(-1, 2) + offset
            previous = i

    # At these grazing angles the vamp occludes the landmarks used at the front.
    grazing = {18: [[241, 1778], [227, 1803]], 19: [[226, 1769], [214, 1794]]}
    views = {}
    for i, (top, bottom, width) in WINDOWS.items():
        ys = np.arange(top - 5, bottom + 6)
        anchor = np.asarray(grazing.get(i, seeds[i]))
        guide = np.polyval(np.polyfit(anchor[:, 1], anchor[:, 0], 1), ys)
        if i not in grazing:
            guide += 2
        seam = locate_seam(strips[i][:, :, 1], ys - BAND_TOP, guide, 6)
        views[i] = {
            "top": top,
            "bottom": bottom,
            "width": width,
            "rim_x_quarters": np.rint(4 * seam).astype(int).tolist(),
        }
    # Measure the same seam in the clean photographic reference for texture registration.
    ref = np.asarray(instep_condition())
    ys = np.arange(1738, 1788)
    guide = np.interp(
        ys, np.arange(1740, 1786), np.array(views[0]["rim_x_quarters"]) / 4
    )
    seam = locate_seam(ref[:, :, 1], ys, guide, 3)
    record = {
        "photo_rim_first_y": int(ys[0]),
        "photo_rim_x_quarters": np.rint(4 * seam).astype(int).tolist(),
        "views": views,
    }
    (HERE / "instep-guides.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
