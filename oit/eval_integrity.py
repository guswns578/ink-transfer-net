"""Controlled native-resolution comparison and phase/context diagnostics."""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label, gaussian_filter
from model import Reinker, compose
from fullres_model import FullResReinker
from prepare import features, read_alpha, save_alpha, sha
from evaluate import structure, target_score

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs" / "OIT_local_paired_pilot_20260907_v2"
OUT = ROOT / "outputs" / "OIT_line_integrity_20260908"


def render(arrays, labels, path, title):
    # 1:1 pixels only, no interpolation or noninteger enlargement.
    h = max(a.shape[0] for a in arrays)
    w = max(a.shape[1] for a in arrays)
    pad = 8
    sheet = Image.new("RGB", (len(arrays) * (w + pad), h + 90), "#e9edf0")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 16)
    d.text((8, 5), title, font=font, fill="#152132")
    for i, (a, l) in enumerate(zip(arrays, labels)):
        d.text((i * (w + pad) + 4, 40), l, font=font, fill="#152132")
        sheet.paste(
            Image.fromarray(np.uint8(np.rint((1 - np.clip(a, 0, 1)) * 255))).convert("RGB"),
            (i * (w + pad), 80),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def load(path, kind):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    m = FullResReinker() if kind in ("fullres", "integrity") else Reinker(24)
    m.load_state_dict(ck["model"])
    return m.cuda().eval()


@torch.inference_mode()
def predict(m, f, s):
    # FP32 for all compared models. No final blur/levels/sharpening.
    t = torch.from_numpy(f[None].copy()).cuda()
    sp = torch.from_numpy(s[None, None].copy()).cuda()
    _, p = compose(t, m(t), sp)
    return p[0, 0].cpu().numpy()


def noise(x, p):
    st = structure(x, p)
    ids, n = label(p > 0.5)
    counts = np.bincount(ids.ravel())[1:]
    st["tiny_components_1to4px"] = int(((counts > 0) & (counts <= 4)).sum())
    r = p - x
    st["residual_highfreq_rms"] = float(np.sqrt(np.mean((r - gaussian_filter(r, 1)) ** 2)))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-integrity", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.60)
    out = OUT / ("comparison_integrity" if args.include_integrity else "comparison_raw")
    out.mkdir(exist_ok=False)
    paths = {
        "old": OLD / "runs" / "paired_native_v1" / "best_train.pt",
        "fullres": OUT / "fullres_matched" / "best_train.pt",
    }
    if args.include_integrity:
        paths["integrity"] = OUT / "fullres_integrity" / "best_train.pt"
    models = {k: load(v, k) for k, v in paths.items()}
    manifest = json.loads((OLD / "manifest.json").read_text("utf-8"))
    probes = manifest["qualitative"]
    results = []
    for r in probes:
        x = read_alpha(r["source"])
        f, s = features(x)
        ps = {k: predict(m, f, s) for k, m in models.items()}
        save_alpha(x, out / f'{r["id"]}_0_original.png')
        for k, p in ps.items():
            save_alpha(p, out / f'{r["id"]}_{k}_raw.png')
        # Same fixed native-pixel window, including R09 region previously damaged.
        top = min(256, x.shape[0] - 256)
        left = min(256, x.shape[1] - 256)
        sl = np.s_[top : top + 256, left : left + 256]
        labels = {
            "old": "기존 축소·복원 모델",
            "fullres": "축소 없는 모델",
            "integrity": "축소 없음 + 선 보존 학습",
        }
        render(
            [x[sl]] + [ps[k][sl] for k in models],
            ["원본"] + [labels[k] for k in models],
            out / f'00_{r["id"]}_1to1.png',
            r["id"] + " / 미학습 페이지 / 원본 픽셀 1:1",
        )
        results.append(
            dict(
                id=r["id"],
                page=r["page"],
                shape=list(x.shape),
                scores={k: noise(x, p) for k, p in ps.items()},
            )
        )
    diagnostics = []
    for r in [probes[2], probes[4]]:
        x = read_alpha(r["source"])
        f, s = features(x)
        for k, m in models.items():
            p = predict(m, f, s)
            ff = np.pad(f, ((0, 0), (0, 0), (1, 0)), mode="edge")[:, :, :-1]
            ss = np.pad(s, ((0, 0), (1, 0)), mode="edge")[:, :-1]
            shifted = predict(m, ff, ss)
            a = p[96:-96, 96:-97]
            b = shifted[96:-96, 97:-96]
            patch = predict(m, f[:, 128:384, 128:384], s[128:384, 128:384])
            central = p[192:320, 192:320]
            diagnostics.append(
                dict(
                    id=r["id"],
                    model=k,
                    one_pixel_translation_mae=float(np.abs(a - b).mean()),
                    one_pixel_translation_max=float(np.abs(a - b).max()),
                    patch_vs_full_center_mae=float(np.abs(patch[64:-64, 64:-64] - central).mean()),
                )
            )
    paired = []
    for r in manifest["records"]:
        if r["id"] in manifest["quarantined"]:
            continue
        with np.load(OLD / r["data"]) as d:
            x, y, f, s = d["x"], d["y"], d["features"], d["support"]
        paired.append(
            dict(
                id=r["id"],
                scores={k: target_score(x, y, predict(m, f, s)) for k, m in models.items()},
            )
        )
    report = dict(
        checkpoints={k: dict(path=str(p), sha256=sha(p)) for k, p in paths.items()},
        results=results,
        diagnostics=diagnostics,
        paired=paired,
        precision="float32 for both",
        note="Same training budget/data/loss, different architectures; all predictions at native size. No claim that translation equivariance alone guarantees line quality.",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(
        json.dumps(
            dict(
                diagnostics=diagnostics,
                paired=paired,
                mean_far2={
                    k: float(
                        np.mean(
                            [r["scores"][k]["original_ink_farther_than_2px_pct"] for r in results]
                        )
                    )
                    for k in models
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
