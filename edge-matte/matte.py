"""Refine the tracked H3 plate at native resolution and return straight RGBA."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d


def ramp(value, low, high):
    return np.clip((value - low) / (high - low), 0, 1)


def distance(binary):
    return cv2.distanceTransform(
        binary.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )


def nearest(binary):
    return tuple(
        distance_transform_edt(~binary, return_distances=False, return_indices=True)
    )


def background(rgb):
    margins = np.concatenate((rgb[:, :28], rgb[:, -28:]), axis=1)
    return gaussian_filter1d(np.median(margins, axis=1), 4, axis=0)[:, None].astype(
        np.float32
    )


def supported_shape(tracked, delta, rows):
    # A grey connected to the studio can continue deep into a finger gap or a shoe arch.
    # Enclosed near-grey highlights on the ears stay part of the subject.
    n, ids, stats, _ = cv2.connectedComponentsWithStats((delta < 8).astype(np.uint8))
    exterior = np.bincount(ids[~tracked], minlength=n)
    size = stats[:, cv2.CC_STAT_AREA]
    field = ((exterior >= 3) & (size >= 7)) | (
        (size > 70) & (stats[:, cv2.CC_STAT_TOP] + rows[0, 0] > 650)
    )
    field[0] = False
    from_track = distance(~tracked)
    reach = from_track < 32
    shape = (tracked & ~field[ids]) | ((delta > 11) & reach)
    # When a glossy surface crosses plate grey, the two observed sides of the reflection
    # bound a solid patch. Use the track to distinguish that patch from a real shoe arch.
    signal = (delta > 10).astype(np.uint8)
    bracketed = np.zeros_like(tracked)
    diagonal = np.eye(25, dtype=np.uint8)
    for line in (
        np.ones((1, 25), np.uint8),
        np.ones((25, 1), np.uint8),
        diagonal,
        np.fliplr(diagonal).copy(),
    ):
        bracketed |= cv2.morphologyEx(signal, cv2.MORPH_CLOSE, line).astype(bool)
    track_depth = distance(tracked)
    shape |= (rows > 1540) & (track_depth > 1.45) & bracketed
    # At a near edge-on ear the bright outer ridge can be several pixels outside the SAM contour.
    # The fill must reach that ridge across the otherwise almost perfectly grey lacquer.
    ear_rows = np.count_nonzero(rows[:, 0] < 395)
    signed = (track_depth - from_track)[:ear_rows]
    gy, gx = np.gradient(cv2.GaussianBlur(signed, (9, 9), 2.0))
    norm = np.maximum(np.hypot(gx, gy), 0.15)
    gy, gx = gy / norm, gx / norm
    py, px = np.indices(signed.shape, dtype=np.float32)
    witness = np.zeros_like(signed)
    for step in range(1, 9):
        sx, sy = px - step * gx, py - step * gy
        sample = cv2.remap(delta[:ear_rows], sx, sy, cv2.INTER_LINEAR)
        sample_depth = cv2.remap(
            signed, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        witness = np.maximum(
            witness, np.where(sample_depth < signed - 0.45 * step, sample, 0)
        )
    shape[:ear_rows] |= (
        (from_track[:ear_rows] < 5.2)
        & bracketed[:ear_rows]
        & ((witness > 11) | (signed > 6.0))
    )
    short_gap = cv2.morphologyEx(
        shape.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    shape |= (rows > 1540) & (from_track < 1.6) & short_gap
    n, ids, stats, _ = cv2.connectedComponentsWithStats(shape.astype(np.uint8))
    main = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return ids == main


def plate(rgb8, tracked, joints):
    full_h, full_w = tracked.shape
    ys, xs = np.nonzero(tracked)
    y0, y1 = max(0, ys.min() - 45), min(full_h, ys.max() + 46)
    x0, x1 = max(0, xs.min() - 45), min(full_w, xs.max() + 46)
    region = np.s_[y0:y1, x0:x1]
    rgb = rgb8[region].astype(np.float32)
    bg = background(rgb8)[y0:y1]
    rows = np.arange(y0, y1)[:, None]
    diff = rgb - bg
    delta = np.max(np.abs(diff), axis=2)
    shape = supported_shape(tracked[region], delta, rows)
    luma = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    cx = int(np.median(np.nonzero(shape & (rows >= 405) & (rows < 468))[1]))
    central = np.abs(np.arange(rgb.shape[1]) - cx) <= 50
    dark_width = np.count_nonzero((shape & (luma < 104))[:, central], axis=1)
    crown_rows = (np.convolve(dark_width > 55, np.ones(9), "same") == 9) & (
        rows[:, 0] >= 314
    )
    crown = max(318, int(rows[np.flatnonzero(crown_rows)[0], 0]) - 4)
    from_center = np.abs(np.arange(rgb.shape[1]) - cx)
    skull_radius = 10 + 105 * np.sqrt(ramp(rows, crown - 6, crown + 71))
    glossy = (rows > 1540) | (
        (rows < 395) & ((rows < crown + 9) | (from_center > skull_radius + 9))
    )
    crest = (np.abs(rows - crown) < 17) & (np.abs(np.arange(rgb.shape[1]) - cx) < 74)
    closed = cv2.morphologyEx(
        shape.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    shape |= crest & closed.astype(bool)
    # A narrow colour crossing along a shaded cuff can momentarily equal the grey plate.
    # Bridge single-pixel cracks only if the surrounding material still has local texture.
    narrow_close = cv2.morphologyEx(
        shape.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    shape |= (
        narrow_close & (cv2.blur(delta, (3, 3)) > 17) & (rows > 590) & (rows < 1015)
    )
    inside, outside = distance(shape), distance(~shape)
    # A ridge provides a donor along thin heel shafts or ear points, where an erosion would have no seed.
    ridges = (inside >= cv2.dilate(inside, np.ones((5, 5), np.uint8)) - 0.01) & (
        delta > 30
    )
    donors = ((inside >= 4.8) | (ridges & (inside >= 1))) & shape & (delta > 17)
    donor = cv2.GaussianBlur(rgb[nearest(donors)], (5, 5), 1.05)
    direction = donor - bg
    projection = np.sum(diff * direction, axis=2) / np.maximum(
        np.sum(direction**2, axis=2), 64
    )
    weight = ramp(projection, 0.035, 0.975)
    donor_y = donor @ np.array([0.299, 0.587, 0.114], np.float32)
    warm = rgb[:, :, 0] - rgb[:, :, 2]
    warm_ref = donor[:, :, 0] - donor[:, :, 2]
    warm_bg = bg[:, :, 0] - bg[:, :, 2]
    # Rim lighting on skin is much paler than the adjacent material. Chroma helps locate its coverage.
    skin = ramp(warm_ref, 17, 32) * ramp(donor_y, 68, 105)
    chroma = ramp((warm - warm_bg) / np.maximum(warm_ref - warm_bg, 8), 0.015, 0.94)
    weight = weight * (1 - 0.57 * skin) + np.minimum(weight, chroma) * 0.57 * skin
    # Protect genuine highlights at the tips of the glossy ears when their RGB crosses the plate grey.
    weight = np.maximum(
        weight, (rows < crown + 15) * ramp(delta, 13, 52) * ramp(inside, 0.4, 2.0)
    )
    weight = np.maximum(weight, 0.90 * ramp(inside, 0.6, 2.7) * ramp(delta, 12, 34))
    solid = ramp(inside, 1.0, 3.6)
    alpha = solid + (1 - solid) * weight
    alpha *= 1 - ramp(outside, 0.9, 2.9)
    alpha[(delta < 4) & ~shape] = 0
    geometry = cv2.GaussianBlur(shape.astype(np.float32), (5, 5), 0.8)
    alpha = np.where(
        glossy,
        np.clip(
            alpha, np.maximum(ramp(inside, 0.42, 2.5), geometry - 0.07), geometry + 0.08
        ),
        alpha,
    )
    alpha = np.minimum(alpha, 1.0)
    expanded = alpha[:, :, None]
    unmixed = np.clip((rgb - (1 - expanded) * bg) / np.maximum(expanded, 0.09), 0, 255)
    donor_blend = np.maximum(1 - ramp(inside, 1.2, 5.7), 1 - ramp(alpha, 0.48, 0.98))
    straight = unmixed + donor_blend[:, :, None] * (donor - unmixed)
    highlight = crest * ramp(luma, 92, 136) * ramp(inside, 0.8, 2.2)
    straight += highlight[:, :, None] * (rgb - straight)
    straight += (glossy * ramp(inside, 1.35, 3.4))[:, :, None] * (rgb - straight)

    # Work only at the perimeter of the bob. The ear tips, face, collar and neck each have their own donors.
    head = (rows >= 303) & (rows <= 587)
    dark = head & shape & (luma < 86) & (warm < 34)
    dark_density = cv2.blur(dark.astype(np.float32), (7, 7))
    hair_donor = dark & (dark_density > 0.74) & (inside > 3.7)
    if hair_donor.any():
        hair = cv2.GaussianBlur(rgb[nearest(hair_donor)], (5, 5), 1.0)
        hair_y = hair @ np.array([0.299, 0.587, 0.114], np.float32)
        dist_dark = distance(~hair_donor)
        skin_seed = head & (warm > 32) & (luma > 76) & (inside > 3.5)
        skin_seed &= cv2.blur(skin_seed.astype(np.float32), (5, 5)) > 0.72
        white = head & (rows > 485) & (luma > 170) & (warm < 33) & (inside > 4)
        white_seed = distance(white) > 5.5
        other = distance(~(skin_seed | white_seed))
        perimeter = (
            (1 - ramp(dist_dark, 13, 22))
            * ramp(rows, crown + 3, crown + 19)
            * (1 - ramp(rows, 561, 586))
            * (1 - ramp(inside, 10, 18))
            * (1 - ramp(from_center, 114, 133))
            * (1 - ramp(from_center, skull_radius, skull_radius + 12))
        )
        locality = perimeter * ramp(other - dist_dark, -0.8, 3.0)
        # The generated rim is itself brighter than the observed grey backing. Its positive
        # luminance cannot represent a partially covered dark strand on that backing.
        hair_core = shape & head & (luma < 102) & (warm < 38)
        space = distance(~hair_core)
        depth = distance(hair_core)
        coverage = np.clip((165 - luma) / np.maximum(165 - hair_y, 40), 0, 1)
        coverage *= 1 - ramp(space, 1.0, 2.5)
        coverage = np.maximum(coverage, ramp(depth, 1.0, 3.0))
        hair_blend = np.maximum(
            1 - ramp(depth, 1.2, 4.5), 1 - ramp(coverage, 0.62, 0.99)
        )
        hair_rgb = rgb + hair_blend[:, :, None] * (hair - rgb)
        alpha += locality * (coverage - alpha)
        straight += locality[:, :, None] * (hair_rgb - straight)
        # Where a lock ends on a bare shoulder, compare both real materials to the
        # neutral halo. A distance-only choice of hair or skin would leave pale triangular islands.
        if skin_seed.any():
            skin_dist = distance(~skin_seed)
            skin_rgb = cv2.GaussianBlur(rgb[nearest(skin_seed)], (5, 5), 1.15)
            skin_warm = skin_rgb[:, :, 0] - skin_rgb[:, :, 2]
            skin_alpha = ramp(
                (warm - warm_bg - 6) / np.maximum(skin_warm - warm_bg - 6, 16), 0, 0.94
            )
            skin_alpha *= (1 - ramp(skin_dist, 2.0, 6.6)) * ramp(luma, 52, 90)
            total = coverage + skin_alpha
            mixed = (
                hair_rgb * coverage[:, :, None] + skin_rgb * skin_alpha[:, :, None]
            ) / np.maximum(total[:, :, None], 0.001)
            join = (
                perimeter
                * ramp(rows, 482, 504)
                * ramp(skin_dist, 1.5, 3.7)
                * ramp(distance(~white_seed), 3, 9)
            )
            alpha += join * (np.maximum(coverage, skin_alpha) - alpha)
            straight += join[:, :, None] * (mixed - straight)

    # A beige pixel trapped between two pink fingertips has high luminance but little red-blue
    # separation. Keep the native pink nail highlights and use nearby skin for fractional RGB.
    hand_zone = (rows > 988) & (rows < 1150)
    pink = hand_zone & shape & (warm > 30) & (luma > 100) & (inside >= 1.4)
    pink &= cv2.blur(pink.astype(np.float32), (3, 3)) > 0.45
    if pink.any():
        hand_dist = distance(~pink)
        hand_donor = cv2.GaussianBlur(rgb[nearest(pink)], (3, 3), 0.8)
        pink_warm = hand_donor[:, :, 0] - hand_donor[:, :, 2]
        pink_alpha = ramp(
            (warm - warm_bg - 9) / np.maximum(pink_warm - warm_bg - 9, 16), 0.05, 0.96
        )
        pink_alpha *= 1 - ramp(hand_dist, 1.4, 4.4)
        cuff = (
            (rows > 915)
            & (rows < 1050)
            & shape
            & (luma > 161)
            & (warm < 25)
            & (inside >= 3.0)
        )
        cuff_dist = distance(~(distance(cuff) > 3.5))
        pale = (
            (1 - ramp(inside, 4, 9))
            * (1 - ramp(hand_dist, 7, 14))
            * ramp(cuff_dist, 5, 10)
        )
        pale *= (
            ramp(rows, 992, 1016)
            * (1 - ramp(rows, 1134, 1148))
            * (1 - ramp(warm, 20, 34))
            * ramp(luma, 105, 145)
        )
        alpha += pale * (np.minimum(alpha, pink_alpha) - alpha)
        straight += pale[:, :, None] * (hand_donor - straight)
        # During the turn the thumb and index form a closed ring. The video model paints
        # its backing slightly blue or beige, so this pocket need not connect to field grey.
        finger_banks = (
            shape & (warm > 27) & (luma > 107) & (rgb[:, :, 0] > 145) & hand_zone
        )
        enclosed = cv2.morphologyEx(
            finger_banks.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 39), np.uint8)
        ).astype(bool)
        wrist_region = np.zeros_like(shape)
        columns = np.arange(x0, x1)[None, :]
        for side in ("left", "right"):
            wx, wy, visibility = joints[side + "_wrist"]
            ex = joints[side + "_elbow"][0]
            if visibility >= 0.42:
                cx_hand = wx + 0.30 * (wx - ex)
                wrist_region |= ((columns - cx_hand) / 58) ** 2 + (
                    (rows - wy - 57) / 83
                ) ** 2 <= 1
        pocket = (
            enclosed
            & wrist_region
            & (warm < 11)
            & (luma > 87)
            & (luma < 175)
            & (hand_dist < 16)
            & (rows > 1006)
        )
        # Exclude isolated low-chroma skin highlights; an opening has a coherent centre.
        _n, labels, stats, _ = cv2.connectedComponentsWithStats(pocket.astype(np.uint8))
        valid = stats[:, cv2.CC_STAT_AREA] >= 5
        valid[0] = False
        pocket = valid[labels]
        if pocket.any():
            # When the hand passes across the thigh, the same opening shows fabric.
            # Trace the two exposed stocking edges just below the fingers and extend their
            # local slopes to the opening. Retain the observed thigh at the crossover.
            thigh_band = (rows[:, 0] >= 1124) & (rows[:, 0] < 1189)
            leg_color = (
                (warm > 11)
                & (rgb[:, :, 0] - rgb[:, :, 1] > 5)
                & (rgb[:, :, 0] < 167)
                & (rgb[:, :, 1] < 135)
            )
            fabric = tracked[region][thigh_band] & leg_color[thigh_band]
            left = np.min(np.where(fabric, columns, full_w), axis=1)
            right = np.max(np.where(fabric, columns, 0), axis=1)
            good = (right - left) > 32
            fit = np.polyfit(
                rows[thigh_band, 0][good], np.column_stack((left, right))[good], 1
            )
            left_edge = rows * fit[0, 0] + fit[1, 0]
            right_edge = rows * fit[0, 1] + fit[1, 1]
            clear_of_leg = ramp(
                np.maximum(left_edge - columns, columns - right_edge), -1, 7
            )
            native_depth = np.minimum(distance(delta > 9), distance(tracked[region]))
            pocket_weight = (
                (1 - ramp(distance(~pocket), 0.8, 4.0))
                * (1 - ramp(warm, 19, 34))
                * ramp(luma, 77, 100)
                * clear_of_leg
                * (1 - ramp(native_depth, 14, 20))
            )
            interior_skin = ramp(
                (warm - warm_bg - 8) / np.maximum(pink_warm - warm_bg - 8, 16),
                0.08,
                0.97,
            )
            alpha += pocket_weight * (np.minimum(alpha, interior_skin) - alpha)
            straight += pocket_weight[:, :, None] * (hand_donor - straight)

    # A small subpixel reconstruction filter only affects fractional coverage and its direct neighbours.
    alpha = 0.79 * alpha + 0.21 * cv2.GaussianBlur(alpha, (3, 3), 0.65)
    exterior_flat = ramp(distance(delta < 5), 1.8, 5.0) * ramp(
        distance(~tracked[region]), 0.85, 3.0
    )
    alpha *= 1 - exterior_flat
    final_a = np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    final_a[final_a < 5] = 0
    final_a[final_a > 252] = 255
    colors = np.rint(np.clip(straight, 0, 255)).astype(np.uint8)
    colors[final_a == 0] = 0
    out = np.zeros((full_h, full_w, 4), np.uint8)
    out[region] = np.dstack((colors, final_a))
    return out
