"""Standalone OFFLINE inference for this experimental model.

Example: python infer.py --checkpoint .../best_train.pt --input source.png --output result.png
No downloads or image APIs. Not suitable for production; see experiment report.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from PIL import Image
from model import Reinker, compose
from prepare import features, read_alpha, save_alpha, sha


@torch.inference_mode()
def infer(model, x, device, core=512, halo=64):
    # Full-canvas distance fields and border mask: never recompute per tile.
    f, s = features(x)
    h, w = x.shape
    output = np.zeros_like(x)
    for top in range(0, h, core):
        for left in range(0, w, core):
            bottom, right = min(top + core, h), min(left + core, w)
            y0, y1 = max(0, top - halo), min(h, bottom + halo)
            x0, x1 = max(0, left - halo), min(w, right + halo)
            tensor = torch.from_numpy(f[None, :, y0:y1, x0:x1].copy()).to(device)
            support = torch.from_numpy(s[None, None, y0:y1, x0:x1].copy()).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                _, p = compose(tensor, model(tensor), support)
            p = p[0, 0].float().cpu().numpy()
            output[top:bottom, left:right] = p[top - y0 : bottom - y0, left - x0 : right - x0]
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()
    src, dest, cp = (
        Path(args.input).resolve(),
        Path(args.output).resolve(),
        Path(args.checkpoint).resolve(),
    )
    if src == dest or dest.exists() or dest.with_suffix(".json").exists():
        raise FileExistsError("Refusing to overwrite source or existing output")
    torch.set_num_threads(4)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ckpt = torch.load(cp, map_location="cpu", weights_only=True)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.60)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = Reinker(ckpt["config"]["width"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    x = read_alpha(src)
    started = time.perf_counter()
    y = infer(model, x, args.device)
    elapsed = time.perf_counter() - started
    assert x.shape == y.shape
    save_alpha(y, dest)
    meta = dict(
        input=str(src),
        input_sha256=sha(src),
        checkpoint=str(cp),
        checkpoint_sha256=sha(cp),
        checkpoint_mode=ckpt["config"].get("mode", "paired_native_pilot"),
        output=str(dest),
        output_sha256=sha(dest),
        shape=list(y.shape),
        elapsed_sec=elapsed,
        device=args.device,
        processing="offline local neural inference; grayscale alpha; no resize; no levels or sharpening",
        tiled_inference="512px core + 64px halo; global distance features; tile invariance not certified",
        warning="Research checkpoint: can erase fine lines and damage grey-tone areas. Not a finished style-transfer model.",
    )
    dest.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
