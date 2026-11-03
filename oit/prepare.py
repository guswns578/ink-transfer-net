"""Read-only source audit; writes only new, task-owned pilot artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs' / 'OIT_direction_pilot_20260906'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_alpha(path):
    im = Image.open(path).convert('RGBA')
    white = Image.new('RGBA', im.size, (255, 255, 255, 255))
    white.alpha_composite(im)
    return 1 - np.asarray(white.convert('L'), dtype=np.float32) / 255


def save_alpha(a, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.rint((1 - np.clip(a, 0, 1)) * 255))).save(path)


def metrics(x, y):
    a, b = x > .5, y > .5
    da, db = distance_transform_edt(~a), distance_transform_edt(~b)
    roi = (distance_transform_edt(~(a | b)) <= 3)
    return dict(mae=float(np.abs(x-y).mean()), roi_mae=float(np.abs(x-y)[roi].mean()),
                source_to_target_p95=float(np.percentile(db[a], 95)),
                target_to_source_p95=float(np.percentile(da[b], 95)),
                source_ink_pct=float(a.mean()*100), target_ink_pct=float(b.mean()*100))


def old_bounds(x):
    from scipy.ndimage import maximum_filter
    ink = x > .5
    di = cv2.distanceTransform(ink.astype('uint8'), cv2.DIST_L2, 5)
    do = cv2.distanceTransform((~ink).astype('uint8'), cv2.DIST_L2, 5)
    rem = x * np.clip((maximum_filter(di, 9)-1.2)/.8, 0, 1) * np.clip((1.5-di)/.5, 0, 1)
    add = (1-x) * (do <= 2.5) * np.clip(1-(do-.5)/2, 0, 1)
    return np.clip(x-rem, 0, 1), np.clip(x+.8*add, 0, 1)


def features(x, band=16):
    ink = x > .1
    do = distance_transform_edt(~ink).astype('float32') if ink.any() else np.full_like(x, 1e4)
    di = distance_transform_edt(x > .5).astype('float32')
    f = np.stack([x, np.minimum(di/8, 1), np.minimum(do/16, 1)])
    # Conservative border holdout, not a semantic face/text detector.
    support = (do <= band).astype('float32')
    support[:8] = support[-8:] = 0
    support[:, :8] = support[:, -8:] = 0
    return f, support


def board(rows, dest, width=480):
    font_path = Path('C:/Windows/Fonts/malgun.ttf')
    font = ImageFont.truetype(str(font_path), 18)
    headers = ['원본', 'GPT 목표 후보 (원본 파일)', '위치 정렬 후보 (아직 학습 결과 아님)']
    sheet = Image.new('RGB', (width*3, len(rows)*(width+48)+48), '#e9edf0')
    d = ImageDraw.Draw(sheet)
    for col, label in enumerate(headers):
        d.text((col*width+8, 8), label, font=font, fill='#182333')
    for i, (label, xs) in enumerate(rows):
        top = 48+i*(width+48)
        for j, a in enumerate(xs):
            im = Image.fromarray(np.uint8(np.rint((1-np.clip(a, 0, 1))*255))).convert('RGB')
            im.thumbnail((width-8, width-8), Image.Resampling.LANCZOS)
            sheet.paste(im, (j*width+4, top))
            d.text((j*width+8, top+width), label, font=font, fill='#182333')
    sheet.save(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite experiment: {out}')
    out.mkdir(parents=True)
    candidates = json.loads((SOURCE/'candidate_manifest.json').read_text('utf-8'))
    regions = {r['id']: r for r in json.loads((SOURCE/'manifest.json').read_text('utf-8'))}
    records, boards = [], []
    for c in candidates:
        if not c['id'].endswith('_detail'):
            continue
        x, raw = read_alpha(c['source']), read_alpha(SOURCE/c['candidate_path'])
        assert x.shape == raw.shape
        warp = np.eye(2, 3, dtype='float32')
        # Smooth only ECC's measurement channels; training source stays untouched.
        try:
            ecc, warp = cv2.findTransformECC(gaussian_filter(x, 1.2), gaussian_filter(raw, 1.2),
                                           warp, cv2.MOTION_TRANSLATION,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6))
            valid = bool(np.max(np.abs(warp[:,2])) <= 12)
        except cv2.error:
            ecc, valid = None, False
        aligned = cv2.warpAffine(raw, warp, (x.shape[1], x.shape[0]),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0) if valid else raw.copy()
        before, after = metrics(x, raw), metrics(x, aligned)
        # Pure global placement correction only if mismatch demonstrably decreases.
        accept = valid and after['roi_mae'] < before['roi_mae']*.95
        y = aligned if accept else raw.copy()
        f, support = features(x)
        lo, hi = old_bounds(x)
        old_best = np.clip(y, lo, hi)
        new_best = np.where(support > 0, y, x)
        label = c['id']
        save_alpha(x, out/'data'/f'{label}_source.png')
        save_alpha(raw, out/'data'/f'{label}_gpt_raw.png')
        save_alpha(y, out/'data'/f'{label}_target.png')
        save_alpha(support, out/'data'/f'{label}_editable_mask.png')
        np.savez_compressed(out/'data'/f'{label}.npz', x=x, y=y, features=f, support=support)
        r = regions[c['source_region']]
        page = '_'.join(r['source_name'].split('_')[:2])
        records.append(dict(id=label, region=c['source_region'], page=page, panel=r['source_name'],
                            original_source=c['source'], source_sha256=sha(c['source']),
                            original_target=str(SOURCE/c['candidate_path']), target_sha256=sha(SOURCE/c['candidate_path']),
                            target_provenance='previous GPT image edit, provisional pseudo-target',
                            artist_certified=False, experimental_use_authorized_by_user=True,
                            translation_xy=warp[:,2].tolist(), ecc=ecc, translation_applied=accept,
                            before=before, after=metrics(x,y),
                            oracle_old_bound_mae=float(np.abs(old_best-y).mean()),
                            oracle_new_bound_mae=float(np.abs(new_best-y).mean()),
                            oracle_note='Target-known representability bound, NOT a learned prediction or safety certificate',
                            shape=list(x.shape), data=f'data/{label}.npz'))
        boards.append((label, [x, raw, y]))
    # Leave the entire 041_007 source page out. Other crops from any training page
    # are also excluded from qualitative generalization evaluation.
    # Cloth target remains visibly displaced after translation: quarantine it,
    # rather than teaching its contour relocation as style.
    train = [r['id'] for r in records if r['id'] in ('R01_detail', 'R03_detail')]
    val = [r['id'] for r in records if r['page'] == '041_007']
    pages = {r['page'] for r in records if r['id'] in train}
    visual = []
    for r in regions.values():
        page = '_'.join(r['source_name'].split('_')[:2])
        if page in pages or r['id'] == 'R15':
            continue
        visual.append(dict(id=r['id'], page=page, category=r['category'],
                           source=str(SOURCE/r['source_crop']),
                           old_lora=str(SOURCE/r['styled']['path']),
                           old_oit=str(SOURCE/r['oit_v0']['path']),
                           no_ground_truth=True))
    manifest = dict(name='OIT paired local pilot', python=platform.python_version(),
                    records=records, train=train, validation=val, qualitative=visual,
                    quarantined=['R06_detail'], quarantine_reason='Residual contour mismatch after global translation; excluded from training and independent evaluation.',
                    split_unit='source page before crop sampling',
                    validation_note='Previously viewed development data; not blind final test. One paired held-out page is insufficient for generalization claims.',
                    relaxed_edit_band_px=16, source_resize='none',
                    skeleton_input=False, protects='outer 8 pixels only; no guaranteed identity protection',
                    source_manifest_sha256=sha(SOURCE/'candidate_manifest.json'))
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), 'utf-8')
    board(boards, out/'00_data_alignment_review.png')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
