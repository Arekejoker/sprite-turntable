"""Build the native PNG sequence, two-background film, and offline inspector."""

import base64
import io
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from instep import Instep
from matte import plate
from PIL import Image
from track import CACHE, COUNT, HERE, ROOT, SIZE, TAKE, decoder, file_sha

RELEASE = ROOT / "release"
PNGS = "全身环绕_透明PNG_72帧.zip"
FILM = "全身环绕_深浅底净边_5秒.mp4"
VIEWER = "人物边缘_逐帧换底对照.html"
DARK = (28, 35, 45)
LIGHT = (243, 240, 233)


def pngname(index):
    return f"orbit_{index:03}.png"


def composite(rgba, color):
    image = Image.new("RGB", SIZE, color)
    image.paste(rgba, mask=rgba.getchannel("A"))
    return image


def pair(rgba):
    board = Image.new("RGB", (2 * SIZE[0], SIZE[1]))
    board.paste(composite(rgba, DARK))
    board.paste(composite(rgba, LIGHT), (SIZE[0], 0))
    return board


def boxes(rgba, joints):
    a = np.asarray(rgba.getchannel("A")) > 128

    def center(top, bottom):
        x = np.nonzero(a[top:bottom])[1]
        return (int(x.min()) + int(x.max())) // 2

    def crop(cx, top, w, h):
        return [
            max(0, min(SIZE[0] - w, int(cx - w / 2))),
            max(0, min(SIZE[1] - h, int(top))),
            w,
            h,
        ]

    hand = []
    for side in ("left", "right"):
        x, y, score = joints[side + "_wrist"]
        ex, _ey, _ = joints[side + "_elbow"]
        if score >= 0.48:
            hand.append(crop(x + 0.25 * (x - ex), y - 66, 148, 210))
    if not hand:
        hand = [crop(center(1000, 1080), 914, 480, 218)]
    if len(hand) == 2 and abs(hand[0][0] - hand[1][0]) < 85:
        hand = [
            crop((hand[0][0] + hand[1][0]) / 2 + 74, min(b[1] for b in hand), 210, 220)
        ]
    return {
        "hair": [crop(center(408, 461), 303, 310, 284)],
        "ears": [crop(center(65, 355), 52, 410, 317)],
        "shoulder": [crop(center(576, 715), 472, 480, 310)],
        "hands": sorted(hand),
        "heels": [crop(center(1694, 1822), 1583, 440, 289)],
        "all": [[0, 0, *SIZE]],
    }


def preview_image(image):
    data = io.BytesIO()
    image.save(data, format="WEBP", quality=92, alpha_quality=100, method=4)
    return "data:image/webp;base64," + base64.b64encode(data.getvalue()).decode("ascii")


def main():
    import hashlib

    from verify import verify

    cv2.setNumThreads(2)
    instep = Instep()
    source = json.loads(TAKE.with_suffix(".json").read_text())
    if file_sha(TAKE) != source["sha256"]:
        raise ValueError("The native RGB take differs from its recorded digest")
    with np.load(HERE / "support.npz") as fs:
        masks = np.unpackbits(fs["masks"], axis=2, count=SIZE[0]).astype(bool)
    if masks.shape != (COUNT, SIZE[1], SIZE[0]):
        raise ValueError("Unexpected tracked mask dimensions")
    joints = json.loads((ROOT / "audits/hand-source/cuff-landmarks.json").read_text())[
        "visible_joints_by_frame"
    ]
    CACHE.mkdir(parents=True, exist_ok=True)
    RELEASE.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CACHE) as folder:
        stage = Path(folder)
        encoder = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{SIZE[0] * 2}x{SIZE[1]}",
            "-framerate",
            f"{COUNT}/5",
            "-i",
            "pipe:0",
            "-an",
            "-vf",
            "scale=in_range=pc:out_range=pc:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,format=yuv420p",
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-preset",
            "fast",
            "-crf",
            "15",
            "-color_range",
            "pc",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "iec61966-2-1",
            "-movflags",
            "+faststart",
            str(stage / FILM),
        ]
        views = []
        with (
            zipfile.ZipFile(stage / PNGS, "w") as archive,
            subprocess.Popen(encoder, stdin=subprocess.PIPE) as film,
        ):
            for index, raw in enumerate(decoder(TAKE, SIZE)):
                if (
                    hashlib.sha256(raw.tobytes()).hexdigest()
                    != source["rgb24_sha256_by_frame"][index]
                ):
                    raise ValueError(f"Unexpected native pixels in frame {index}")
                if index == COUNT:
                    continue
                native = np.asarray(raw)
                rgba = Image.fromarray(
                    instep.restore(
                        plate(native, masks[index], joints[index]), native, index
                    )
                )
                entry = zipfile.ZipInfo(pngname(index), date_time=(1980, 1, 1, 0, 0, 0))
                entry.external_attr = 0o100644 << 16
                with archive.open(entry, "w") as p:
                    rgba.save(p, format="PNG", compress_level=6)
                film.stdin.write(pair(rgba).tobytes())
                views.append(
                    {
                        "raw": preview_image(raw),
                        "cut": preview_image(rgba),
                        "boxes": boxes(rgba, joints[index]),
                    }
                )
            film.stdin.close()
            if film.wait() or index != COUNT or len(views) != COUNT:
                raise RuntimeError("Incomplete orbit")
        template = (HERE / "viewer.html.in").read_text()
        (stage / VIEWER).write_text(
            template.replace(
                "__ORBIT_FRAMES__", json.dumps(views, separators=(",", ":"))
            )
        )
        result = verify(stage, masks)
        for name in (PNGS, FILM, VIEWER):
            (stage / name).replace(RELEASE / name)
        (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"Verified {COUNT} full-resolution cutouts and both film panels.")


if __name__ == "__main__":
    main()
