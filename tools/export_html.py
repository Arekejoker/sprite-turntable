"""Build the offline 兔女郎展示 viewer from the final 72 native PNG views."""
import argparse
import base64
import json
import struct
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "release/全身环绕_正面手臂精修_72帧.zip"
DEFAULT_OUTPUT = ROOT / "release/兔女郎展示.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template", type=Path, default=ROOT / "edge-matte/showcase.html")
    args = parser.parse_args()
    names = [f"orbit_{i:03}.png" for i in range(72)]
    frames = []
    expected_size = None
    with zipfile.ZipFile(args.frames) as archive:
        if sorted(archive.namelist()) != names:
            parser.error("Expected orbit_000.png through orbit_071.png at the root of the ZIP.")
        for name in names:
            data = archive.read(name)
            if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
                parser.error(f"{name} is not a valid PNG file.")
            width, height, depth, color_type = struct.unpack(">IIBB", data[16:26])
            if (depth, color_type) != (8, 6) or width <= 0 or height <= 0:
                parser.error(f"{name} must be an 8-bit RGBA PNG.")
            if expected_size is None:
                expected_size = (width, height)
            elif (width, height) != expected_size:
                parser.error(f"{name} size {width}×{height} differs from {expected_size[0]}×{expected_size[1]}.")
            frames.append("data:image/png;base64," + base64.b64encode(data).decode("ascii"))
    template = args.template.read_text(encoding="utf-8")
    if template.count("<script>") != 1:
        raise ValueError("Expected a single viewer script.")
    embedded = '<script type="application/json" id="orbit-frames">' + json.dumps(frames, separators=(",", ":")) + "</script>"
    page = template.replace("<script>", embedded + "\n<script>").replace(
        "<title>全身交互展示录制</title>", "<title>兔女郎展示</title>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staged = Path(tmp) / "viewer.html"
        staged.write_text(page, encoding="utf-8")
        staged.replace(args.output)
    print(f"{args.output} ({args.output.stat().st_size / 2**20:.1f} MiB, {expected_size[0]}×{expected_size[1]})")


if __name__ == "__main__":
    main()
