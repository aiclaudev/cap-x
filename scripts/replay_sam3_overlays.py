#!/usr/bin/env python
"""Offline re-render of SAM3 text-segmentation overlays from saved perception artifacts.

Regenerates the per-step mask-overlay images (with confidence-score chips) WITHOUT
re-running the simulation, LLM, or reflector. For every SAM3 text-segmentation step
recorded in a trial's ``execution_history_block_*.json`` we:

  1. read the exact input RGB that was fed to SAM3 (``block_{b}_step_{s}_img_0.jpg``),
  2. recover the text prompt from the step's log text,
  3. re-run SAM3 (deterministic) OR load a cached ``*_sam3.npz`` (``--from-npz``),
  4. re-draw the overlay via ``overlay_segmentation_masks(rgb, masks, scores=...)``
     and overwrite ``block_{b}_step_{s}_img_1.jpg``.

Step 3 also persists the kept masks+scores to ``block_{b}_step_{s}_sam3.npz`` so any
future overlay tweak can be re-rendered with ``--from-npz`` — no SAM3 GPU needed.

Requires a running SAM3 server (default http://127.0.0.1:8114) unless ``--from-npz``.

Usage:
    python scripts/replay_sam3_overlays.py \
        --perception-root outputs/agentv2/gemini-3.1-pro-preview/franka_robosuite_nut_assembly/perception \
        --trials 1-10
    python scripts/replay_sam3_overlays.py --perception-root <root> --trials 1-10 --from-npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from capx.utils.visualization_utils import overlay_segmentation_masks

SCORE_THRESH = 0.05   # match the live pipeline's viz filter (control_reduced.py)
SAVE_TOPK = 8         # overlay draws top-5; keep a little headroom in the cache


def _parse_trials(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _sam3_text_steps(hist_path: str):
    """Yield (block_index, step_index, prompt) for each SAM3 text-segmentation step."""
    hist = json.load(open(hist_path))
    bi = hist["code_block_index"]
    for s in hist.get("steps", []):
        tool = s.get("tool_name", "")
        text = s.get("text", "")
        if "SAM3 Text" in tool or "SAM3 text-prompt" in text:
            prompt = text.split("'")[1] if "'" in text else None
            if prompt:
                yield bi, s["step_index"], prompt


def replay_trial(tdir: str, segment_fn, from_npz: bool) -> int:
    n = 0
    for hist in sorted(glob.glob(os.path.join(tdir, "execution_history_block_*.json"))):
        for bi, si, prompt in _sam3_text_steps(hist):
            img0 = os.path.join(tdir, f"block_{bi}_step_{si}_img_0.jpg")
            img1 = os.path.join(tdir, f"block_{bi}_step_{si}_img_1.jpg")
            npz = os.path.join(tdir, f"block_{bi}_step_{si}_sam3.npz")
            if not os.path.exists(img0):
                print(f"    [skip] missing input rgb: {os.path.basename(img0)}")
                continue
            rgb = np.array(Image.open(img0).convert("RGB"))

            if from_npz:
                if not os.path.exists(npz):
                    print(f"    [skip] no cache: {os.path.basename(npz)}")
                    continue
                d = np.load(npz, allow_pickle=True)
                masks = [m for m in d["masks"]]
                scores = [float(x) for x in d["scores"]]
            else:
                results = segment_fn(rgb, text_prompt=prompt)
                kept = [r for r in results if r.get("score", 0) > SCORE_THRESH][:SAVE_TOPK]
                masks = [np.asarray(r["mask"], dtype=bool) for r in kept]
                scores = [float(r.get("score", 0.0)) for r in kept]
                masks_arr = (
                    np.stack(masks) if masks else np.zeros((0, *rgb.shape[:2]), dtype=bool)
                )
                np.savez_compressed(
                    npz, masks=masks_arr, scores=np.asarray(scores, dtype=np.float32),
                    prompt=prompt,
                )

            if masks:
                vis = overlay_segmentation_masks(rgb, masks, scores=scores)
                Image.fromarray(vis).save(img1, quality=95)
                n += 1
                print(f"    blk{bi} step{si:>3}  masks={len(masks)}  "
                      f"top={max(scores):.2f}  prompt={prompt!r}")
            else:
                print(f"    blk{bi} step{si:>3}  no masks > {SCORE_THRESH}  prompt={prompt!r}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception-root", required=True,
                    help="task perception dir containing trial_NN/ subdirs")
    ap.add_argument("--trials", required=True, help="e.g. '1-10' or '1,3,5'")
    ap.add_argument("--from-npz", action="store_true",
                    help="re-render from cached *_sam3.npz instead of calling SAM3")
    args = ap.parse_args()

    segment_fn = None
    if not args.from_npz:
        from capx.integrations.vision.sam3 import init_sam3
        segment_fn = init_sam3()

    total = 0
    for t in _parse_trials(args.trials):
        tdir = os.path.join(args.perception_root, f"trial_{t:02d}")
        if not os.path.isdir(tdir):
            print(f"[trial {t:02d}] no dir, skip")
            continue
        print(f"[trial {t:02d}] {tdir}")
        total += replay_trial(tdir, segment_fn, args.from_npz)
    print(f"\nDone. Re-rendered {total} SAM3 overlays.")


if __name__ == "__main__":
    main()
