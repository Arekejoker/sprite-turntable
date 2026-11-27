"""Track the selected, uncut H3 take in both directions from its common front pose."""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "audits/hand-source"))
from cuff_input import LEFT, RIGHT, SOURCE, refine
from native import SIZE, decoder, file_sha

TAKE = ROOT / "audits/hand-source/inputs/cuff-take.mkv"
COUNT = 72
CACHE = ROOT / "work/edge-clean"
CHECKPOINT_SHA = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"


def front_mask():
    photo = np.array(Image.open(SOURCE / "01_front.png").convert("RGB"))
    alpha = np.array(Image.open(SOURCE / "01_front_rgba.png").getchannel("A"))
    alpha = refine(refine(alpha, photo, LEFT, 199), photo, RIGHT, 205)
    scale = 1.1137452711223204
    tx, ty = -202.3465321563682, 36.70517023959644
    transformed = Image.fromarray(alpha).transform(
        SIZE,
        Image.Transform.AFFINE,
        (1 / scale, 0, -tx / scale, 0, 1 / scale, -ty / scale),
        Image.Resampling.BICUBIC,
    )
    return np.asarray(transformed) >= 120


def main():
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    args = p.parse_args()
    if file_sha(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("This run expects the recorded SAM 2.1 Hiera Tiny checkpoint")
    CACHE.mkdir(parents=True, exist_ok=True)
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml", str(args.checkpoint), device="cuda"
    )
    with (
        tempfile.TemporaryDirectory(dir=CACHE) as td,
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        for i, image in enumerate(decoder(TAKE, SIZE)):
            image.save(Path(td) / f"{i:05}.jpg", quality=97, subsampling=0)
        if i != COUNT:
            raise ValueError("The recorded return frame is missing")
        state = predictor.init_state(video_path=td, offload_video_to_cpu=True)
        first = front_mask()
        fused = np.zeros((COUNT, SIZE[1], SIZE[0]), np.float16)
        for reverse in (False, True):
            predictor.reset_state(state)
            start = COUNT if reverse else 0
            predictor.add_new_mask(
                inference_state=state, frame_idx=start, obj_id=1, mask=first
            )
            for i, _, logits in predictor.propagate_in_video(
                state, start_frame_idx=start, reverse=reverse
            ):
                if i < COUNT:
                    weight = i / COUNT if reverse else 1 - i / COUNT
                    fused[i] += (
                        weight * logits[0, 0].float().clamp(-24, 24).cpu().numpy()
                    )
    packed = np.packbits(fused > 0, axis=2)
    out = HERE / "support.npz"
    temp = out.with_suffix(".partial.npz")
    np.savez_compressed(temp, masks=packed)
    temp.replace(out)
    print(f"Saved {COUNT} tracked silhouettes to {out.name}")


if __name__ == "__main__":
    main()
