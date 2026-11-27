"""Inspect the full orbit, both visible hands, the loop and the native first-frame hand silhouettes."""

import argparse
import json
from pathlib import Path

import numpy as np
from native import HERE, SIZE, decoder, file_sha
from PIL import Image, ImageDraw

JOINTS = {
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_ankle": 27,
    "right_ankle": 28,
}
WORK = HERE.parents[1] / "work/hand-source/inspection"
FRONT_HANDS = {
    "screen_left": (99, 995, 178, 1107),
    "screen_right": (465, 1012, 553, 1121),
}


def skin(image, box):
    a = np.int16(image.crop(box))
    r, g, b = a.transpose(2, 0, 1)
    return (r - g > 10) & (r - b > 17) & (g > 76)


def visible_wrist(joints, side, threshold):
    wrist = joints.get(side + "_wrist")
    hip = joints.get(side + "_hip")
    # Both arms rest at the sides. A tracker can attach an occluded far wrist to the head.
    return (
        wrist is not None
        and hip is not None
        and wrist[2] >= threshold
        and 0.40 * SIZE[1] <= wrist[1] <= 0.63 * SIZE[1]
        and abs(wrist[1] - hip[1]) < 0.12 * SIZE[1]
    )


def card(image, joints, index):
    board = Image.new("RGB", (170, 197), "#e4e7e3")
    draw = ImageDraw.Draw(board)
    draw.text((4, 2), str(index), fill="black")
    for n, side in enumerate(("left", "right")):
        if not visible_wrist(joints, side, 0.55):
            continue
        x, y, _ = joints[side + "_wrist"]
        cut = image.crop((round(x) - 65, round(y) - 38, round(x) + 65, round(y) + 122))
        board.paste(cut.resize((81, 100), Image.Resampling.LANCZOS), (n * 84, 17))
    # A fixed crop still shows both sleeves when the tracker loses one at occlusion.
    board.paste(
        image.crop((5, 817, 635, 1113)).resize((168, 79), Image.Resampling.LANCZOS),
        (0, 118),
    )
    return board


def main():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("take", choices=["original", "hand", "cuff"])
    ap.add_argument("pose_model", type=Path)
    args = ap.parse_args()
    source = HERE / "inputs" / (args.take + "-take.mkv")
    WORK.mkdir(parents=True, exist_ok=True)
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.pose_model)),
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
    )
    samples, tiles, body = [], [], []
    first = last = None
    with vision.PoseLandmarker.create_from_options(opts) as detector:
        for i, im in enumerate(decoder(source, SIZE)):
            if i == 0:
                first = im
            if i == 72:
                last = im
                continue
            result = detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(im))
            )
            joints = {}
            if result.pose_landmarks:
                for name, j in JOINTS.items():
                    p = result.pose_landmarks[0][j]
                    joints[name] = [
                        round(p.x * im.width, 2),
                        round(p.y * im.height, 2),
                        round(p.visibility, 3),
                    ]
            tiles.append(card(im, joints, i))
            body.append(im.resize((90, 270), Image.Resampling.LANCZOS))
            samples.append(joints)
    if len(samples) != 72 or last is None:
        raise ValueError(
            "Both closure and all 72 unique view positions must be inspected"
        )
    for k in range(3):
        sheet = Image.new("RGB", (170 * 6, 197 * 4), "white")
        for j, tile in enumerate(tiles[24 * k : 24 * (k + 1)]):
            sheet.paste(tile, (j % 6 * 170, j // 6 * 197))
        sheet.save(WORK / f"{args.take}-hands-{k + 1}.jpg", quality=94)
    sheet = Image.new("RGB", (96 * 12, 290 * 2), "#e4e7e3")
    draw = ImageDraw.Draw(sheet)
    for j, image in enumerate(body[::3]):
        x, y = j % 12 * 96, j // 12 * 290
        sheet.paste(image, (x, y + 16))
        draw.text((x + 3, y), str(j * 3), fill="black")
    sheet.save(WORK / f"{args.take}-body.jpg", quality=94)
    condition = (
        "fullbody-cuffs-grey.png" if args.take == "cuff" else "fullbody-grey.png"
    )
    input_image = Image.open(HERE / "inputs" / condition).convert("RGB")
    metrics = {}
    for name, box in FRONT_HANDS.items():
        ref, actual, end = (skin(im, box) for im in (input_image, first, last))
        metrics[name] = {
            "box_xyxy": box,
            "input_skin_pixels": int(ref.sum()),
            "first_frame_skin_iou_to_input": round(
                float((ref & actual).sum() / (ref | actual).sum()), 4
            ),
            "return_frame_skin_iou_to_input": round(
                float((ref & end).sum() / (ref | end).sum()), 4
            ),
        }
    offsets = {}
    for side in ("left", "right"):
        heights = []
        for joints in samples:
            if not visible_wrist(joints, side, 0.7):
                continue
            wrist, hip = joints[side + "_wrist"], joints[side + "_hip"]
            if min(wrist[2], hip[2]) >= 0.7:
                heights.append(wrist[1] - hip[1])
        if heights:
            q10, q90 = np.percentile(heights, [10, 90])
            offsets[side] = {
                "visible_views": len(heights),
                "p10_to_p90_span_px": round(float(q90 - q10), 2),
            }
    result = {
        "source_sha256": file_sha(source),
        "pose_model_sha256": file_sha(args.pose_model),
        "hand_regions_on_front_condition": metrics,
        "wrist_height_relative_to_same_hip": offsets,
        "wrist_note": "Image-space spread includes perspective, occlusions, detection and actual pose changes.",
        "visible_joints_by_frame": samples,
    }
    (HERE / f"{args.take}-landmarks.json").write_text(
        json.dumps(result, separators=(",", ":")) + "\n"
    )
    print(json.dumps({"front": metrics, "wrist_height": offsets}))


if __name__ == "__main__":
    main()
