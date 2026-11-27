"""Restore the photographed stocking inside the visible shoe opening of the selected take."""

import hashlib
import json
from functools import cached_property

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, median_filter
from track import HERE, ROOT

PHOTO = ROOT / "audits/hand-source/inputs/fullbody-cuffs-instep-grey.png"


def ease(value, start, end):
    t = np.clip((value - start) / (end - start), 0, 1)
    return t * t * (3 - 2 * t)


class Instep:
    def __init__(self):
        data = json.loads((HERE / "instep-guides.json").read_text())
        self.views = {int(i): v for i, v in data["views"].items()}
        self.photo_rim = np.array(data["photo_rim_x_quarters"]) / 4
        self.photo_rows = data["photo_rim_first_y"] + np.arange(len(self.photo_rim))

    @cached_property
    def photo(self):
        from cuff_input import instep_condition

        source = np.asarray(Image.open(PHOTO).convert("RGB"))
        if not np.array_equal(source, np.asarray(instep_condition())):
            raise ValueError(
                "The combined reference differs from the original photograph and checked matte."
            )
        return source

    def strip(self, index):
        view = self.views[index]
        top, bottom, width = view["top"], view["bottom"], view["width"]
        rim = np.array(view["rim_x_quarters"]) / 4
        ys = np.arange(top - 5, bottom + 6)
        left, right = int(np.floor(rim.min() - width - 15)), int(np.ceil(rim.max() + 4))
        region = np.s_[ys[0] : ys[-1] + 1, left:right]
        inward = rim[:, None] - np.arange(left, right)
        return region, ys[:, None], inward, top, bottom, width

    def footprint(self, index):
        """The declared edit envelope, also used for exact prior-release comparisons."""
        region, ys, inward, top, bottom, width = self.strip(index)
        allowed = (
            (ys > top - 2) & (ys < bottom + 1) & (inward > -2) & (inward < width + 1.5)
        )
        return region, allowed

    def outside_digest(self, rgba, index):
        if index in self.views:
            region, allowed = self.footprint(index)
            rgba = rgba.copy()
            rgba[region][allowed] = 0
        return hashlib.sha256(rgba.tobytes()).hexdigest()

    def restore(self, rgba, raw, index):
        if index not in self.views:
            return rgba
        region, ys, inward, top, bottom, width = self.strip(index)
        cut = rgba[region]
        native = raw[region].astype(np.float32)
        alpha = cut[:, :, 3].astype(np.float32)
        # Register the photo by height along the opening and distance inward from its seam.
        source_y = 1745 + (ys - top) * 35 / (bottom - top)
        source_rim = np.interp(source_y, self.photo_rows, self.photo_rim)
        source_x = source_rim - np.maximum(inward * 13 / width, 3)
        texture = cv2.remap(
            self.photo,
            source_x.astype("f4"),
            np.broadcast_to(source_y, inward.shape).astype("f4"),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)
        # Match the local lighting on the same foot. The last two grazing views can have
        # only a sliver of cloth visible, with no usable horizontal swatch on some rows.
        light = (inward > width + 2) & (inward < width + 13) & (alpha == 255)
        light &= (
            (native[:, :, 0] - native[:, :, 2] > 16)
            & (native[:, :, 1] > 52)
            & (native[:, :, 1] < 113)
        )
        offsets = native - texture
        mean_offset = (
            np.median(offsets[light], axis=0)
            if light.sum() >= 9
            else np.array([-2, -4, -2])
        )
        lighting = np.array(
            [
                np.median(delta[valid], axis=0) if valid.sum() >= 2 else mean_offset
                for delta, valid in zip(offsets, light)
            ]
        )
        lighting = gaussian_filter1d(median_filter(lighting, size=(3, 1)), 2, axis=0)
        target = texture + lighting[:, None]
        # Retain the observed falloff into the dark leather. Gray contamination lifts
        # the green channel and reduces the red–blue difference of the photographed hosiery.
        shade = np.minimum(1, native[:, :, 1] / np.maximum(target[:, :, 1], 1))
        sock = target * shade[:, :, None]
        brighter = native[:, :, 1] - target[:, :, 1]
        less_warm = (target[:, :, 0] - target[:, :, 2]) - (
            native[:, :, 0] - native[:, :, 2]
        )
        evidence = ease(np.maximum(brighter, 0.75 * less_warm), 1, 8)

        area = (
            ease(ys, top - 2, top + 2)
            * ease(bottom - ys, -1, 3)
            * ease(width - inward, -1.5, 2)
        )
        ink = area * ease(inward, -0.25, 1.4)
        cover = area * ease(inward, -2, -0.4)
        if index in (18, 19) or 54 <= index <= 59:
            # Where cloth meets open air, keep the approved subpixel silhouette.
            opaque = (rgba[:, :, 3] > 248).astype(np.uint8)
            yslice, xslice = region
            margin = opaque[
                yslice.start - 5 : yslice.stop + 5, xslice.start - 5 : xslice.stop + 5
            ]
            depth = cv2.distanceTransform(margin, cv2.DIST_L2, 5)[5:-5, 5:-5]
            ink *= ease(depth, 1, 2.3)
            cover = np.zeros_like(cover)

        lost = ease(255 - alpha, 0, 18)
        base = cut[:, :, :3] + (cover * lost)[:, :, None] * (native - cut[:, :, :3])
        blend = ink * np.maximum(evidence, lost)
        rgb = base + blend[:, :, None] * (sock - base)
        cut[:, :, :3] = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
        cut[:, :, 3] = np.rint(alpha + cover * (255 - alpha)).astype(np.uint8)
        return rgba
