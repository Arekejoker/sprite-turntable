"""Open every shipping asset and measure its pixels against the native take."""

import hashlib
import json
import subprocess
import zipfile
from itertools import zip_longest

import cv2
import numpy as np
from build import DARK, FILM, LIGHT, PNGS, RELEASE, VIEWER, pair, pngname
from instep import PHOTO, Instep
from matte import background, distance
from PIL import Image
from track import COUNT, HERE, SIZE, TAKE, decoder, file_sha


def verify(folder, masks):
    film = folder / FILM
    stream = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,width,height,avg_frame_rate,duration",
                "-of",
                "json",
                str(film),
            ]
        )
    )["streams"][0]
    assert (
        int(stream["nb_frames"]),
        stream["width"],
        stream["height"],
        stream["avg_frame_rate"],
    ) == (COUNT, 2 * SIZE[0], SIZE[1], "72/5")
    assert abs(float(stream["duration"]) - 5) < 1e-5
    frames = []
    fore_hashes = set()
    instep = Instep()
    baseline = json.loads((HERE / "instep-baseline.json").read_text())
    gray_reference = np.asarray(
        Image.open(PHOTO.with_name("fullbody-cuffs-grey.png"))
    ).astype(np.int16)
    photo = instep.photo.astype(np.int16)
    photo_mark = (gray_reference - photo).max(axis=2) > 12
    assert np.count_nonzero(photo_mark) == 255
    front_instep = {}
    with zipfile.ZipFile(folder / PNGS) as archive:
        assert archive.namelist() == [pngname(i) for i in range(COUNT)]
        video = decoder(film, (2 * SIZE[0], SIZE[1]))
        for i, (raw, encoded) in enumerate(zip_longest(decoder(TAKE, SIZE), video)):
            if i == COUNT:
                assert raw is not None and encoded is None
                continue
            assert i < COUNT and encoded is not None and raw is not None
            with archive.open(pngname(i)) as f:
                rgba = Image.open(f)
                assert rgba.mode == "RGBA" and rgba.size == SIZE
                out = np.array(rgba)
            alpha = out[:, :, 3]
            assert (
                instep.outside_digest(out, i)
                == baseline["outside_footprint_rgba_sha256_by_frame"][i]
            ), f"Pixels outside the shoe correction changed in native frame {i}."
            if i == 12:
                assert alpha[1069, 465] == 0 and alpha[1064, 464] == 0
            # Native coordinates on an open shoe arch, a neutral shoe reflection and a folded ear.
            if i == 17:
                assert alpha[1755, 375] == 0
            if i == 25:
                assert alpha[1718, 154] == 255
            if i == 55:
                assert np.min(alpha[325:332, 500]) >= 245
            fore_hashes.add(hashlib.sha256(out.tobytes()).hexdigest())
            original = np.array(raw)
            bg = background(original)
            delta = np.abs(original - bg).max(axis=2)
            dark_interior = (distance(original.max(axis=2) < 83) > 3.25) & (
                distance(masks[i]) > 3.0
            )
            assert int(alpha[dark_interior].min()) >= 236, (
                i,
                int(alpha[dark_interior].min()),
            )
            deep = (distance(masks[i]) > 24) & (distance(delta > 9) > 22)
            if i in instep.views:
                region, footprint = instep.footprint(i)
                deep[region] &= ~footprint
                x, y = baseline["stocking_probes_xy_by_frame"][str(i)]
                swatch = out[y, x].astype(int)
                assert swatch[3] == 255 and swatch[0] - swatch[2] >= 14, (
                    i,
                    x,
                    y,
                    swatch,
                )
            if i == 0:
                assert np.all(alpha[photo_mark] == 255)
                photo_error = np.abs(
                    out[:, :, :3][photo_mark] - photo[photo_mark]
                ).mean()
                panels = np.asarray(encoded, dtype=np.int16)
                dark_mark = panels[:, : SIZE[0]][photo_mark]
                light_mark = panels[:, SIZE[0] :][photo_mark]
                panel_error = float(np.abs(dark_mark - light_mark).mean())
                assert photo_error < 4 and panel_error < 4, (photo_error, panel_error)
                front_instep = {
                    "photo_defined_core_pixels": int(photo_mark.sum()),
                    "minimum_straight_alpha": int(alpha[photo_mark].min()),
                    "mean_rgb_error_vs_design_photo": round(float(photo_error), 3),
                    "decoded_dark_vs_light_panel_mean_rgb_error": round(panel_error, 3),
                }
            assert (
                deep.sum() > 90000
                and np.all(alpha[deep] == 255)
                and np.array_equal(out[:, :, :3][deep], original[deep])
            )
            shoulder_core = deep[570:800]
            assert shoulder_core.sum() > 10000
            assert not (
                alpha[:32].any()
                or alpha[-32:].any()
                or alpha[:, :32].any()
                or alpha[:, -32:].any()
            )
            assert np.all(out[:, :, :3][alpha == 0] == 0)
            half = alpha >= 128
            n, _, s, _ = cv2.connectedComponentsWithStats(half.astype(np.uint8))
            areas = np.sort(s[1:, cv2.CC_STAT_AREA])[::-1]
            stray = int(areas[1]) if n > 2 else 0
            # Test the opaque skin in the fingers independent of the tracked outline.
            fingers = original[1010:1117].astype(np.int16)
            warm = (fingers[:, :, 0] - fingers[:, :, 2] > 29) & (fingers[:, :, 1] > 110)
            fp = distance(warm) > 5.5
            skin_kept = (
                float(np.mean(alpha[1010:1117][fp] >= 248)) if fp.any() else None
            )
            film_mae = float(
                np.abs(
                    np.asarray(pair(rgba), dtype=np.int16)
                    - np.asarray(encoded, dtype=np.int16)
                ).mean()
            )
            assert film_mae < 2.0, (i, film_mae)
            assert skin_kept is None or skin_kept > 0.98, (i, skin_kept)
            # Pick studio patches with a uniform 11px neighbourhood next to observable outlines.
            # This includes naturally open areas between limbs and under the raised shoe arch.
            reliable_gray = (distance(delta < 5) > 5) & (distance(~masks[i]) > 3)
            leaked = int(np.count_nonzero(alpha[reliable_gray]))
            assert leaked == 0
            frames.append(
                {
                    "frame": i,
                    "alpha_half_area": int(half.sum()),
                    "tracked_area": int(masks[i].sum()),
                    "isolated_max_area_px": stray,
                    "unchanged_interior_pixels": int(deep.sum()),
                    "unchanged_shoulder_band_pixels": int(shoulder_core.sum()),
                    "fractional_alpha_pixels": int(((alpha > 0) & (alpha < 255)).sum()),
                    "opaque_finger_core_fraction": None
                    if skin_kept is None
                    else round(skin_kept, 6),
                    "studio_leak_pixels": leaked,
                    "film_mean_rgb_error": round(film_mae, 4),
                }
            )
            if i == 32:
                # The grey strip to the left of the bob, at the rear three-quarter view.
                box = np.s_[402:490, 200:226]
                averaged = out[:, :, :3] * (alpha[:, :, None] / 255) + bg * (
                    1 - alpha[:, :, None] / 255
                )
                before = original[box].mean(2) > 153
                after = averaged[box].mean(2) > 153
                example = {
                    "frame": i,
                    "box_xyxy": [200, 402, 226, 490],
                    "threshold_rgb_mean": 153,
                    "original_bright_pixels": int(before.sum()),
                    "new_bright_pixels": int(after.sum()),
                }
                assert before.sum() > 90 and after.sum() < 0.1 * before.sum()
    assert len(frames) == COUNT and len(fore_hashes) == COUNT
    ratios = np.array([f["alpha_half_area"] / f["tracked_area"] for f in frames])
    assert np.max(np.abs(ratios - np.roll(ratios, 1))) < 0.04
    return {
        "native_take_sha256": file_sha(TAKE),
        "tracked_support_sha256": file_sha(HERE / "support.npz"),
        "png_archive_sha256": file_sha(folder / PNGS),
        "film_sha256": file_sha(film),
        "viewer_sha256": file_sha(folder / VIEWER),
        "frame_range_zero_based": [0, 71],
        "native_size": list(SIZE),
        "unique_rgba_frames": len(fore_hashes),
        "alpha": "straight / unassociated, 8-bit sRGB channels; transparent outside is RGBA 0",
        "film": stream,
        "film_backgrounds_srgb": [DARK, LIGHT],
        "instep": {
            "previous_png_archive_sha256": baseline["previous_png_archive_sha256"],
            "corrected_photo_reference_sha256": file_sha(PHOTO),
            "corrected_views_zero_based": sorted(instep.views),
            "frames_identical_to_prior_release_outside_footprint": len(frames),
            "front": front_instep,
        },
        "representative_rear_hair": example,
        "max_neighbor_change_in_area_ratio_including_loop": round(
            float(np.abs(ratios - np.roll(ratios, 1)).max()), 6
        ),
        "frames": frames,
    }


if __name__ == "__main__":
    cv2.setNumThreads(2)
    with np.load(HERE / "support.npz") as f:
        m = np.unpackbits(f["masks"], axis=2, count=SIZE[0]).astype(bool)
    result = verify(RELEASE, m)
    if json.loads(json.dumps(result)) != json.loads(
        (HERE / "verification.json").read_text()
    ):
        raise ValueError("The shipping files differ from the verification record")
    print(f"Verified {COUNT} PNGs and the full five-second film.")
