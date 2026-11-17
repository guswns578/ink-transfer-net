"""Bounded local paired pilot; no downloads, production changes, or remote APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from model import Reinker, compose


def load_samples(root, ids, manifest):
    mapping = {r["id"]: r for r in manifest["records"]}
    samples = []
    for name in ids:
        with np.load(root / mapping[name]["data"]) as data:
            d = {k: data[k].copy() for k in data.files}
        union = (d["x"] > 0.1) | (d["y"] > 0.1)
        d["roi"] = (distance_transform_edt(~union) <= 3).astype("float32")
        d["weights"] = (1 + 4 * d["roi"]) * d["support"]
        # Loss excludes pixels unreachable by the input-only edit band.
        samples.append(d)
    return samples


def batch(samples, rng, patch, size, device, fixed=None):
    f, y, s, w = [], [], [], []
    for j in range(size):
        idx = int(rng.integers(len(samples)))
        d = samples[idx]
        h, wi = d["x"].shape
        for attempt in range(15):
            top, left = int(rng.integers(h - patch + 1)), int(rng.integers(wi - patch + 1))
            sl = np.s_[top : top + patch, left : left + patch]
            if d["roi"][sl].mean() > 0.08:
                break
        arrays = [
            d["features"][:, sl[0], sl[1]],
            d["y"][sl][None],
            d["support"][sl][None],
            d["weights"][sl][None],
        ]
        if rng.random() < 0.5:
            arrays = [a[..., ::-1] for a in arrays]
        for dst, a in zip((f, y, s, w), arrays):
            dst.append(np.ascontiguousarray(a))
    return [torch.from_numpy(np.stack(a)).to(device) for a in (f, y, s, w)]


@torch.inference_mode()
def evaluate(model, samples, device):
    model.eval()
    errors, bases, moved = [], [], []
    # Fixed native-pixel grid, no random crop selection during checkpoint scoring.
    for d in samples:
        for top in (64, 384, 704):
            for left in (64, 384, 704):
                sl = np.s_[top : top + 256, left : left + 256]
                f = torch.from_numpy(d["features"][:, sl[0], sl[1]][None].copy()).to(device)
                y = torch.from_numpy(d["y"][sl][None, None].copy()).to(device)
                s = torch.from_numpy(d["support"][sl][None, None].copy()).to(device)
                w = torch.from_numpy(d["weights"][sl][None, None].copy()).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, p = compose(f, model(f), s)
                errors.append(float(((p - y).abs() * w).sum() / w.sum().clamp_min(1)))
                bases.append(float(((f[:, :1] - y).abs() * w).sum() / w.sum().clamp_min(1)))
                moved.append(float((p - f[:, :1]).abs().mean()))
    model.train()
    return dict(
        weighted_mae=float(np.mean(errors)),
        identity_weighted_mae=float(np.mean(bases)),
        gain_percent=float((1 - np.mean(errors) / np.mean(bases)) * 100),
        moved=float(np.mean(moved)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--run", default="paired_native_v1")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--seed", type=int, default=907)
    ap.add_argument("--lr", type=float, default=0.0004)
    ap.add_argument("--evaluate-every", type=int, default=200)
    args = ap.parse_args()
    root = Path(args.experiment).resolve()
    run = root / "runs" / args.run
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    assert not set(manifest["train"]) & set(manifest["validation"])
    records = {r["id"]: r for r in manifest["records"]}
    assert not {records[k]["page"] for k in manifest["train"]} & {
        records[k]["page"] for k in manifest["validation"]
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.55)
    device = "cuda"
    train = load_samples(root, manifest["train"], manifest)
    val = load_samples(root, manifest["validation"], manifest)
    model = Reinker(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    cfg = vars(args) | dict(
        torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),
        train=manifest["train"],
        validation=manifest["validation"],
        parameter_count=sum(p.numel() for p in model.parameters()),
        initialization="scratch; old OIT checkpoint not used",
        manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        target_provenance="provisional GPT pseudo-pairs, NOT artist-certified",
        loss="support-masked weighted Charbonnier on raw composed alpha; weight=1+4*ink_union_3px",
        checkpoint_selection="training weighted MAE only; paired validation is reported, never selected",
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    )
    (run / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(cfg, ensure_ascii=False), flush=True)
    rng = np.random.default_rng(args.seed)
    history, best, started = [], float("inf"), time.perf_counter()
    baseline = dict(
        step=0, train=evaluate(model, train, device), validation=evaluate(model, val, device)
    )
    history.append(baseline)
    print(json.dumps(baseline), flush=True)
    for step in range(1, args.steps + 1):
        f, y, s, w = batch(train, rng, args.patch, args.batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = model(f)
            raw, p = compose(f, r, s)
            loss = (
                (torch.sqrt((raw - y).float().square() + 1e-6) - 0.001) * w
            ).sum() / w.sum().clamp_min(1)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if step == 1 and float(grad) == 0:
            raise RuntimeError("Dead gradient at initialization")
        optimizer.step()
        lr = args.lr * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        if step == 1 or step % 50 == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    dict(
                        step=step,
                        loss=float(loss.detach()),
                        grad=float(grad),
                        elapsed_sec=round(elapsed, 2),
                        avg_step_sec=round(elapsed / step, 3),
                        cuda_peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                    )
                ),
                flush=True,
            )
        if step % args.evaluate_every == 0 or step == args.steps:
            entry = dict(
                step=step,
                elapsed_sec=time.perf_counter() - started,
                train=evaluate(model, train, device),
                validation=evaluate(model, val, device),
            )
            history.append(entry)
            ckpt = dict(
                model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                config=cfg,
                step=step,
                metrics=entry,
            )
            torch.save(ckpt, run / "last.pt")
            if entry["train"]["weighted_mae"] < best:
                best = entry["train"]["weighted_mae"]
                torch.save(ckpt, run / "best_train.pt")
            (run / "history.json").write_text(json.dumps(history, indent=2), "utf-8")
            print(json.dumps(entry), flush=True)
    print("TRAINING_COMPLETE " + str(run / "best_train.pt"), flush=True)


if __name__ == "__main__":
    main()
