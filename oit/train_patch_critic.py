"""Matched continuation: regression only vs regression plus conditional critic."""

import argparse, json, math, sys, time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from detail_branch import DetailBranch, compose_detail
from patch_critic import PatchCritic
from curated_multi import curated_mask
from run_detail_branch import WINDOWS, infer
from eval_integrity import OLD, load, predict, render, noise
from prepare import features, read_alpha, save_alpha, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--adv", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--random-negatives", type=int, default=0)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    torch.manual_seed(913)
    torch.cuda.manual_seed_all(913)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.6)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = load(Path(ck["config"]["base_checkpoint"]), "integrity")
    model = DetailBranch(ck["config"]["width"]).cuda()
    model.load_state_dict(ck["model"])
    critic = PatchCritic().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    optd = torch.optim.Adam(critic.parameters(), lr=0.0001, betas=(0.5, 0.999))
    manifest = json.loads((OLD / "manifest.json").read_text("utf-8"))
    data = {}
    for rec in manifest["records"]:
        if rec["id"] not in manifest["train"]:
            continue
        with np.load(OLD / rec["data"]) as z:
            d = {k: z[k].copy() for k in z.files}
        b = predict(base, d["features"], d["support"])
        m = curated_mask(rec["id"], b.shape)
        d["clean"] = b
        d["mask"] = m
        d["y"] = m * d["y"] + (1 - m) * b
        d["input"] = np.concatenate([d["features"], b[None]])
        data[rec["id"]] = d

    def getcrop(index, dx=0, dy=0):
        name, l, t, label, pos = WINDOWS[index]
        d = data[name]
        h, w = d["x"].shape
        l = max(0, min(w - 256, l + dx))
        t = max(0, min(h - 256, t + dy))
        sl = np.s_[t : t + 256, l : l + 256]
        f = np.pad(d["input"], ((0, 0), (40, 40), (40, 40)), mode="edge")[
            :, t : t + 336, l : l + 336
        ]
        y = d["y"][sl]
        b = d["clean"][sl]
        s = d["support"][sl]
        changed = np.abs(y - b) > 0.12
        ink = distance_transform_edt(~((y > 0.1) | (b > 0.1))) <= 3
        weight = ((1 + 3 * ink + 12 * changed) * s).astype("float32")
        return [f, y[None], s[None], weight[None]], d["x"][sl], b, label

    pool = {
        (i, dx, dy): getcrop(i, dx, dy)[0]
        for i in range(5)
        for dx in (-16, 0, 16)
        for dy in (-16, 0, 16)
    }
    negatives = []
    negative_records = []
    nrng = np.random.default_rng(914)
    for attempt in range(20000):
        if len(negatives) >= args.random_negatives:
            break
        name = list(data)[int(nrng.integers(len(data)))]
        d = data[name]
        h, w = d["x"].shape
        l = int(nrng.integers(w - 256 + 1))
        t = int(nrng.integers(h - 256 + 1))
        sl = np.s_[t : t + 256, l : l + 256]
        if d["mask"][sl].max() > 0.01 or (d["x"][sl] > 0.1).mean() < 0.015:
            continue
        ff = np.pad(d["input"], ((0, 0), (40, 40), (40, 40)), mode="edge")[
            :, t : t + 336, l : l + 336
        ]
        yy = d["clean"][sl]
        ss = d["support"][sl]
        ink = distance_transform_edt(~(yy > 0.1)) <= 3
        ww = (3 * (1 + 3 * ink) * ss).astype("float32")
        negatives.append([ff, yy[None], ss[None], ww[None]])
        negative_records.append(dict(source=name, left=l, top=t, width=256, height=256))
    assert len(negatives) == args.random_negatives
    cfg = dict(
        ck["config"],
        parent_sha256=sha(args.checkpoint),
        parent=str(args.checkpoint.resolve()),
        continuation_steps=args.steps,
        adversarial_weight=args.adv,
        feature_matching_weight=0.01 if args.adv else 0,
        seed=913,
        random_negative_regions=negative_records,
        negative_weight=3 if negatives else 1,
        source="conditional patch critic concept: https://phillipi.github.io/pix2pix/ ; custom native-resolution generator, not pix2pix reproduction",
        code_sha256=sha(Path(__file__)),
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2), "utf-8")
    rng = np.random.default_rng(913)
    start = time.perf_counter()
    history = []
    for step in range(1, args.steps + 1):
        ids = [
            int(rng.integers(3)),
            int(rng.integers(3, 5)) if step % 2 == 0 else int(rng.integers(3)),
        ]
        arrays = []
        for j, i in enumerate(ids):
            if j == 1 and negatives:
                aa = negatives[int(rng.integers(len(negatives)))]
            else:
                aa = pool[i, int(rng.choice([-16, 0, 16])), int(rng.choice([-16, 0, 16]))]
            if rng.random() < 0.5:
                aa = [a[..., ::-1] for a in aa]
            arrays.append(aa)
        f, y, s, w = [
            torch.from_numpy(np.stack([np.ascontiguousarray(a[k]) for a in arrays])).cuda()
            for k in range(4)
        ]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw, p = compose_detail(f[..., 40:-40, 40:-40], model(f)[..., 40:-40, 40:-40], s)
            loss = (
                (torch.sqrt((raw - y).float().square() + 1e-6) - 0.001) * w
            ).sum() / w.sum().clamp_min(1)
            for dim in (2, 3):
                ww = w.narrow(dim, 0, w.shape[dim] - 1)
                loss = loss + 0.2 * (
                    (torch.diff(p, dim=dim) - torch.diff(y, dim=dim)).abs() * ww
                ).sum() / ww.sum().clamp_min(1)
        dloss = torch.tensor(0.0, device="cuda")
        if args.adv:
            cond = f[:, :1, 40:-40, 40:-40]
            critic.train()
            optd.zero_grad(set_to_none=True)
            real, _ = critic(cond, y)
            fake, _ = critic(cond, p.detach().float())
            dloss = 0.5 * ((real - 1).square().mean() + fake.square().mean())
            assert torch.isfinite(dloss)
            dloss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
            optd.step()
            critic.eval()
            for v in critic.parameters():
                v.requires_grad_(False)
            fake, ff = critic(cond, p.float())
            with torch.no_grad():
                _, rf = critic(cond, y)
            fm = sum((a - b).abs().mean() for a, b in zip(ff, rf)) / len(ff)
            loss = loss + args.adv * (fake - 1).square().mean() + 0.01 * fm
        assert torch.isfinite(loss)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        assert torch.isfinite(grad)
        opt.step()
        for v in critic.parameters():
            v.requires_grad_(True)
        for g in opt.param_groups:
            g["lr"] = 0.0001 * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        if step == 1 or step % 250 == 0:
            row = dict(
                step=step,
                loss=float(loss.detach()),
                dloss=float(dloss.detach()),
                seconds=time.perf_counter() - start,
            )
            print(json.dumps(row), flush=True)
            history.append(row)
    torch.save(
        dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, config=cfg),
        out / "candidate.pt",
    )
    if args.adv:
        torch.save(critic.cpu().state_dict(), out / "training_only_critic.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), "utf-8")
    model.eval()
    for i in range(5):
        aa, x, b, label = getcrop(i)
        ft = torch.from_numpy(aa[0][None].copy()).cuda()
        st = torch.from_numpy(aa[2][None].copy()).cuda()
        with torch.inference_mode():
            p = (
                compose_detail(ft[..., 40:-40, 40:-40], model(ft)[..., 40:-40, 40:-40], st)[1][0, 0]
                .cpu()
                .numpy()
            )
        for key, a in [("original", x), ("output", p), ("target", aa[1][0])]:
            save_alpha(a, out / "seen" / f"{label}_{key}.png")
        render(
            [x, b, p, aa[1][0]],
            ["원본", "선 보존", "추가 학습", "정리한 목표"],
            out / "seen" / f"00_{label}.png",
            "학습 영역 / 판별기 가중치 " + str(args.adv),
        )
    results = []
    for name in ("R09", "R11", "R14"):
        rec = next(r for r in manifest["qualitative"] if r["id"] == name)
        x = read_alpha(rec["source"])
        f, s = features(x)
        b = predict(base, f, s)
        p = infer(model, np.concatenate([f, b[None]]), s)
        for key, a in [("original", x), ("clean", b), ("output", p)]:
            save_alpha(a, out / "unseen" / f"{name}_{key}.png")
        t = min(256, x.shape[0] - 256)
        l = min(256, x.shape[1] - 256)
        sl = np.s_[t : t + 256, l : l + 256]
        render(
            [a[sl] for a in (x, b, p)],
            ["원본", "선 보존", "추가 학습"],
            out / "unseen" / f"00_{name}.png",
            name + " / 미학습 개발 평가",
        )
        results.append(dict(id=name, clean=noise(x, b), output=noise(x, p)))
    (out / "unseen.json").write_text(json.dumps(results, indent=2), "utf-8")
    print("DONE " + str(out), flush=True)


if __name__ == "__main__":
    main()
