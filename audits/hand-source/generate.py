"""Re-run the single-shot H3 experiment, or resume the recorded local ComfyUI job."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[1] / "work" / "hand-source" / "generation"
URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument(
        "--cuff-matte",
        action="store_true",
        help="Use the photo-checked gaps at the cuffs.",
    )
    variant.add_argument(
        "--instep-matte",
        action="store_true",
        help="Use the corrected cuff and instep reference for a new take.",
    )
    args = parser.parse_args()
    graph = json.loads((HERE / "hand-orbit.api.json").read_text())
    if args.instep_matte:
        from cuff_input import instep_condition

        ref = HERE / "inputs/fullbody-cuffs-instep-grey.png"
        instep_condition().save(ref)
        target_dir = CACHE.with_name("cuff-instep-generation")
        graph["17"]["inputs"]["filename_prefix"] = (
            "sprite-360-fullbody-hand-review/cuff-instep"
        )
    elif args.cuff_matte:
        ref = HERE / "inputs/fullbody-cuffs-grey.png"
        target_dir = CACHE.with_name("cuff-generation")
        graph["17"]["inputs"]["filename_prefix"] = (
            "sprite-360-fullbody-hand-review/cuff-source"
        )
    else:
        ref = HERE / "inputs/fullbody-grey.png"
        target_dir = CACHE
    target_dir.mkdir(parents=True, exist_ok=True)
    receipt = target_dir / "job.json"
    if args.resume:
        job_id = json.loads(receipt.read_text())["prompt_id"]
    else:
        with ref.open("rb") as stream:
            response = requests.post(
                URL + "/upload/image",
                files={"image": (ref.name, stream, "image/png")},
                data={"subfolder": "sprite-360-fullbody-hand-review"},
                timeout=90,
            )
        response.raise_for_status()
        uploaded = response.json()
        graph["4"]["inputs"]["image"] = "/".join(
            filter(None, [uploaded.get("subfolder"), uploaded["name"]])
        )
        run = requests.post(URL + "/prompt", json={"prompt": graph}, timeout=60)
        run.raise_for_status()
        job_id = run.json()["prompt_id"]
        receipt.write_text(
            json.dumps(
                {
                    "prompt_id": job_id,
                    "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
                    "submitted_graph": graph,
                },
                indent=2,
            )
            + "\n"
        )
    print(f"H3 job {job_id}", flush=True)
    while True:
        status = requests.get(f"{URL}/history/{job_id}", timeout=60)
        status.raise_for_status()
        job = status.json().get(job_id)
        if job:
            break
        time.sleep(3)
    if job["status"]["status_str"] != "success":
        raise RuntimeError(str(job["status"])[-1500:])
    images = job["outputs"]["17"]["images"]
    if len(images) != graph["10"]["inputs"]["length"]:
        raise ValueError("Incomplete native frame sequence")
    for i, item in enumerate(images):
        response = requests.get(URL + "/view", params=item, timeout=120)
        response.raise_for_status()
        target = target_dir / f"{i:03}.png"
        temp = target.with_suffix(".part")
        temp.write_bytes(response.content)
        temp.replace(target)
    print(f"Collected {len(images)} native images in {target_dir}", flush=True)


if __name__ == "__main__":
    main()
