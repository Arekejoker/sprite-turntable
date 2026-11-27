"""Archive the native RGB take losslessly and make a synchronized five-second viewing copy."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1] / "release"
SIZE = (640, 1920)
FRAMES = 73
SHORT_SIZE = 1080
HAND_BOX = (8, 555, 632, 1231)
TILES = ((20, 52, 338, 1014), (388, 146, 672, 728))
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def decoder(path, size):
    with subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "3",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-threads",
            "2",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    ) as proc:
        length = size[0] * size[1] * 3
        while block := proc.stdout.read(length):
            if len(block) != length:
                raise RuntimeError(f"Incomplete decoded frame in {path.name}")
            yield Image.frombytes("RGB", size, block)
        if proc.wait():
            raise RuntimeError(f"Decoder failed for {path.name}")


def archive(directory, dest):
    temp = dest.with_name(dest.stem + ".partial.mkv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    hashes = []
    with subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "640x1920",
            "-framerate",
            "24",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264rgb",
            "-threads",
            "4",
            "-qp",
            "0",
            "-preset",
            "fast",
            "-map_metadata",
            "-1",
            str(temp),
        ],
        stdin=subprocess.PIPE,
    ) as encoder:
        for i in range(FRAMES):
            with Image.open(directory / f"{i:03}.png") as im:
                if im.size != SIZE:
                    raise ValueError(f"Unexpected source size in frame {i}")
                rgb = im.convert("RGB").tobytes()
            encoder.stdin.write(rgb)
            hashes.append(hashlib.sha256(rgb).hexdigest())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Could not archive H3 images")
    actual = [hashlib.sha256(im.tobytes()).hexdigest() for im in decoder(temp, SIZE)]
    if actual != hashes:
        raise ValueError("Decoded archive differs from the original 73 images")
    temp.replace(dest)
    dest.with_suffix(".json").write_text(
        json.dumps(
            {
                "image_size": SIZE,
                "sha256": file_sha(dest),
                "rgb24_sha256_by_frame": hashes,
                "last_frame_is_the_return_condition": True,
            },
            indent=2,
        )
        + "\n"
    )


def canvas():
    board = Image.new("RGB", (SHORT_SIZE, SHORT_SIZE), (238, 239, 236))
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype(FONT, 24)
    draw.text((20, 10), "全身", font=font, fill="#26312d")
    draw.text((388, 104), "同一帧的双臂与手", font=font, fill="#26312d")
    return board


def layout(frame, base):
    shot = base.copy()
    for image, (x, y, w, h) in zip([frame, frame.crop(HAND_BOX)], TILES):
        shot.paste(image.resize((w, h), Image.Resampling.LANCZOS), (x, y))
    return shot


def export(source, dest, whole=False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    base = None if whole else canvas()
    length = FRAMES - 1
    out_size = SIZE if whole else (SHORT_SIZE, SHORT_SIZE)
    temp = dest.with_name(dest.stem + ".partial.mp4")
    targets = []
    with subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{out_size[0]}x{out_size[1]}",
            "-framerate",
            "72/5",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-preset",
            "medium",
            "-crf",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp),
        ],
        stdin=subprocess.PIPE,
    ) as encoder:
        for i, image in enumerate(decoder(source, SIZE)):
            if i < length:
                shot = image if whole else layout(image, base)
                encoder.stdin.write(shot.tobytes())
                check = (
                    shot.crop((64, 877, 576, 1167))
                    if whole
                    else shot.resize((270, 270), Image.Resampling.BOX)
                )
                targets.append(np.int16(check))
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Could not encode the review clip")
    errors = []
    for i, shot in enumerate(decoder(temp, out_size)):
        check = (
            shot.crop((64, 877, 576, 1167))
            if whole
            else shot.resize((270, 270), Image.Resampling.BOX)
        )
        decoded = np.int16(check)
        errors.append(round(float(np.abs(targets[i] - decoded).mean()), 4))
    meta = json.loads(
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
                str(temp),
            ]
        )
    )["streams"][0]
    if (
        len(errors) != length
        or max(errors) > 2.5
        or abs(float(meta["duration"]) - 5) > 0.001
    ):
        raise ValueError(
            "The review clip differs from the 72 original views or the five-second duration"
        )
    temp.replace(dest)
    return {
        "video": meta,
        "source_sha256": file_sha(source),
        "film_sha256": file_sha(dest),
        "source_frame_range_inclusive": [0, 71],
        "examined_pixels": "Native RGB wrist band [64,877,576,1167)"
        if whole
        else "Whole 270x270 review board",
        "max_frame_mean_rgb_error": max(errors),
        "closeup_box_xyxy": None if whole else HAND_BOX,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    pack = commands.add_parser("archive")
    pack.add_argument("directory", type=Path)
    pack.add_argument("dest", type=Path)
    film = commands.add_parser("export")
    film.add_argument("source", type=Path)
    film.add_argument("dest", type=Path)
    film.add_argument("verification", type=Path)
    film.add_argument(
        "--whole", action="store_true", help="Keep the original 640 x 1920 framing."
    )
    args = parser.parse_args()
    if args.action == "archive":
        archive(args.directory, args.dest)
    else:
        result = export(args.source, args.dest, args.whole)
        args.verification.write_text(json.dumps(result, indent=2) + "\n")
