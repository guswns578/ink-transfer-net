"""Frozen full-panel probes and reusable input-only, two-stage tiled inference."""

import argparse, json, sys, time, shutil
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from PIL import Image
from prepare import read_alpha, save_alpha, features, sha
from fullres_model import FullResReinker
from detail_branch import DetailBranch, compose_detail
from infer_fullres import tiled as base_tiled
from eval_integrity import ROOT, render, noise

NAMES = ["042_007_c03.png", "042_007_c08.png", "042_008_c02.png", "042_009_c01.png"]
FIXED = ROOT / "outputs/OIT_full_panel_probes_20260911.json"


@torch.inference_mode()
def detail_tiled(model, f, s, core=512):
    h, w = s.shape
    result = np.empty_like(s)
    halo = 40
    for top in range(0, h, core):
        for left in range(0, w, core):
            bottom, right = min(h, top + core), min(w, left + core)
            a, z = max(0, top - halo), min(h, bottom + halo)
            b, q = max(0, left - halo), min(w, right + halo)
            ff = torch.from_numpy(f[None, :, a:z, b:q].copy()).cuda()
            ss = torch.from_numpy(s[None, None, a:z, b:q].copy()).cuda()
            p = compose_detail(ff, model(ff), ss)[1][0, 0].cpu().numpy()
            result[top:bottom, left:right] = p[top - a : bottom - a, left - b : right - b]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.freeze:
        if FIXED.exists():
            raise FileExistsError(FIXED)
        rows = []
        for name in NAMES:
            p = Path("data/panels_042") / name
            with Image.open(p) as im:
                size = list(im.size)
            rows.append(dict(path=str(p), sha256=sha(p), size=size))
        FIXED.write_text(
            json.dumps(
                dict(
                    note="Selected before current checkpoint inference; entire panels, no output-based selection. Not guaranteed never viewed in historical work.",
                    sources=rows,
                ),
                indent=2,
            ),
            "utf-8",
        )
        print(str(FIXED))
        return
    assert args.checkpoint and args.out
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.6)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DetailBranch(ck["config"]["width"]).cuda().eval()
    model.load_state_dict(ck["model"])
    basepath = Path(ck["config"]["base_checkpoint"])
    bc = torch.load(basepath, map_location="cpu", weights_only=True)
    assert sha(basepath) == ck["config"]["base_sha256"]
    base = FullResReinker(bc["config"]["width"]).cuda().eval()
    base.load_state_dict(bc["model"])
    rows = []
    for rec in json.loads(FIXED.read_text("utf-8"))["sources"]:
        src = Path(rec["path"])
        assert sha(src) == rec["sha256"]
        x = read_alpha(src)
        start = time.perf_counter()
        b = base_tiled(base, x)
        f, s = features(x)
        p = detail_tiled(model, np.concatenate([f, b[None]]), s)
        assert p.shape == x.shape
        shutil.copy2(src, out / f"{src.stem}_original.png")
        for k, a in [("input_grayscale", x), ("clean", b), ("output", p)]:
            save_alpha(a, out / f"{src.stem}_{k}.png")
        # Fixed middle window, not a cherry-picked best detail.
        t = max(0, (x.shape[0] - 384) // 2)
        l = max(0, (x.shape[1] - 384) // 2)
        sl = np.s_[t : t + 384, l : l + 384]
        render(
            [a[sl] for a in (x, b, p)],
            ["원본", "선 보존", "확장 학습"],
            out / f"00_{src.stem}.png",
            src.stem + " / 고정 중앙 영역 / 1:1",
        )
        rows.append(
            dict(
                source=rec,
                seconds=time.perf_counter() - start,
                clean=noise(x, b),
                output=noise(x, p),
            )
        )
    (out / "report.json").write_text(
        json.dumps(
            dict(
                checkpoint=str(args.checkpoint.resolve()),
                checkpoint_sha256=sha(args.checkpoint),
                results=rows,
                postprocess="none",
                manual_inference_masks=False,
                input_conversion="white-composited grayscale; original file copied intact, colored accessories become grayscale in model input/output",
            ),
            indent=2,
        ),
        "utf-8",
    )
    print("DONE " + str(out))


if __name__ == "__main__":
    main()
