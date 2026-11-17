"""Same inference size as memorization; all new probes are different source pages."""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from model import Reinker, compose
from prepare import read_alpha, features, save_alpha, sha
from evaluate import compare, structure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    args = ap.parse_args()
    root = Path(args.experiment).resolve()
    run = root / "runs" / "fixed_patch_memorization"
    out = run / "UNSEEN_DIAGNOSTIC"
    out.mkdir(exist_ok=False)
    cp = run / "memorized_only.pt"
    ckpt = torch.load(cp, map_location="cpu", weights_only=True)
    model = Reinker(24).cuda().eval()
    model.load_state_dict(ckpt["model"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    q = {r["id"]: r for r in manifest["qualitative"]}
    probes = [("R09", 256, 256), ("R11", 384, 512), ("R14", 256, 256)]
    results = []
    for name, left, top in probes:
        r = q[name]
        assert r["page"] not in ckpt["config"]["source_pages"]
        full = read_alpha(r["source"])
        f, s = features(full)
        left = min(left, full.shape[1] - 256)
        top = min(top, full.shape[0] - 256)
        sl = np.s_[top : top + 256, left : left + 256]
        x = full[sl]
        tensor = torch.from_numpy(f[None, :, sl[0], sl[1]].copy()).cuda()
        support = torch.from_numpy(s[None, None, sl[0], sl[1]].copy()).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, p = compose(tensor, model(tensor), support)
        p = p[0, 0].float().cpu().numpy()
        save_alpha(x, out / f"{name}_0_original.png")
        save_alpha(p, out / f"{name}_1_LOCAL_MEMORIZATION_MODEL_UNSEEN.png")
        compare(
            [x, p],
            ["원본", "암기 시험 모델 → 새 영역 출력"],
            "학습하지 않은 페이지 / " + name + " / 정답 이미지 없음",
            out / f"00_{name}_comparison.png",
            size=384,
        )
        results.append(
            dict(
                id=name,
                page=r["page"],
                crop=[left, top, left + 256, top + 256],
                source_sha256=sha(r["source"]),
                structure=structure(x, p),
            )
        )
    (out / "results.json").write_text(
        json.dumps(dict(checkpoint_sha256=sha(cp), results=results), indent=2), "utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
