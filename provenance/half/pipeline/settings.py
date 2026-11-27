"""Shared paths; all generated outputs live in the run directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("SPRITE_WORK", ROOT / "work")).expanduser().resolve()
INPUTS = ROOT / "inputs"
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
SAM2_WEIGHTS = Path(os.environ.get("SAM2_WEIGHTS", ROOT / "models/sam2/sam2.1_hiera_tiny.pt"))
DEPTH_PRO_WEIGHTS = Path(os.environ.get("DEPTH_PRO_WEIGHTS", ROOT / "models/depth_pro/depth_pro.pt"))
FONT = os.environ.get("SPRITE_FONT", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
VIEWER_NAME = "兔女郎_3DSprite交互预览.html"


def input_path(name):
    """A regenerated input overrides the archived reference for this run."""
    generated = WORK / "inputs" / name
    return generated if generated.is_file() else INPUTS / name
