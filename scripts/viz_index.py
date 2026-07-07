#!/usr/bin/env python3
"""Top-level dashboard (index.html) over all per-task report.html under a model output dir.

Scans <model_dir>/<task>/ for trial_* dirs, computes each task's success rate + avg reward
(deduping the per-turn snapshot dirs by trial number, same rule as viz_trial), and links each
task's report.html. Run after the per-task report.html files exist.

Usage:
    python scripts/viz_index.py outputs/Qwen_Qwen3-Coder-Next/
    python scripts/viz_index.py outputs/Qwen_Qwen3-Coder-Next/ --out /tmp/index.html
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_trial import CSS, esc, parse_trial_dirname  # reuse the design + parsing


def _completeness(d: Path):
    # Dedup key when a trial number has multiple rollout dirs (retries / accumulated re-runs):
    # prefer the MOST RECENT rollout (all_responses.json write time — not muddied by post-proc),
    # then video count, then reward. Matches the cleanup policy (keep latest attempt per trial).
    ar = d / "all_responses.json"
    try:
        rec = ar.stat().st_mtime if ar.exists() else d.stat().st_mtime
    except OSError:
        rec = 0.0
    try:
        rw = float(parse_trial_dirname(d.name).get("reward") or 0.0)
    except (TypeError, ValueError):
        rw = 0.0
    return (rec, len(list(d.glob("video_turn_*.mp4"))), rw)


def _friendly(task_dirname: str) -> str:
    """franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib -> cube_stack (M4)"""
    name = re.sub(r"^franka_robosuite_", "", task_dirname)
    name = re.sub(r"_multiturn_vdm_reduced_api_skill_lib$", "", name)
    return name


def task_stats(task_dir: Path) -> dict | None:
    """Dedup snapshot dirs by trial number, then aggregate success rate + avg reward."""
    trial_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("trial_")]
    if not trial_dirs:
        return None
    best: dict[str, Path] = {}
    for d in trial_dirs:
        m = re.match(r"(trial_\d+)", d.name)
        key = m.group(1) if m else d.name
        if key not in best or _completeness(d) > _completeness(best[key]):
            best[key] = d
    trials = list(best.values())
    rewards, completed = [], 0
    for d in trials:
        meta = parse_trial_dirname(d.name)
        try:
            rewards.append(float(meta.get("reward") or 0.0))
        except (TypeError, ValueError):
            rewards.append(0.0)
        if str(meta.get("task_completed")) == "1":
            completed += 1
    n = len(trials)
    return {
        "n": n,
        "completed": completed,
        "success": completed / n if n else 0.0,
        "avg_reward": sum(rewards) / n if n else 0.0,
        "has_report": (task_dir / "report.html").exists(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="model output dir, e.g. outputs/Qwen_Qwen3-Coder-Next")
    ap.add_argument("--out", default=None, help="output html (default: <model_dir>/index.html)")
    args = ap.parse_args()

    root = Path(args.model_dir)
    if not root.exists():
        raise SystemExit(f"model_dir not found: {root}")
    out_path = Path(args.out) if args.out else (root / "index.html")

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("trial_*")))

    rows, overall_succ, overall_n, done_tasks = [], 0.0, 0, 0
    for td in task_dirs:
        st = task_stats(td)
        if not st:
            continue
        done_tasks += 1
        overall_succ += st["success"]
        overall_n += st["n"]
        pct = st["success"] * 100
        bar = (f'<div class="bar"><div class="barfill" style="width:{pct:.0f}%"></div>'
               f'<span class="barlabel">{pct:.0f}%</span></div>')
        link = (f'<a href="{esc(td.name)}/report.html">report.html ▸</a>'
                if st["has_report"] else '<span class="muted">report 미생성</span>')
        rows.append(
            f'<tr><td><b>{esc(_friendly(td.name))}</b><div class="pin">{esc(td.name)}</div></td>'
            f'<td>{st["n"]}</td><td>{st["completed"]}/{st["n"]}</td><td>{bar}</td>'
            f'<td>{st["avg_reward"]:.4f}</td><td>{link}</td></tr>'
        )

    overall = (overall_succ / done_tasks * 100) if done_tasks else 0.0
    table = ('<table><tr><th>태스크</th><th>trials</th><th>완료</th><th>성공률</th>'
             '<th>평균 reward</th><th>리포트</th></tr>' + "\n".join(rows) + '</table>') if rows else \
        '<p class="muted">아직 완료된 태스크가 없습니다 (trial_* 폴더 없음).</p>'

    css = CSS + """
.bar{position:relative;background:#eef1f5;border-radius:6px;height:18px;min-width:120px;overflow:hidden}
.barfill{position:absolute;top:0;left:0;height:100%;background:#1a7f37;opacity:.85}
.barlabel{position:relative;font-size:11px;font-weight:700;color:#1f2328;padding-left:6px;line-height:18px}
.kpi{display:inline-block;background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px 16px;margin:4px 8px 4px 0}
.kpi b{font-size:20px}
"""
    htmldoc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CaP-Agent0 robosuite dashboard — {esc(root.name)}</title><style>{css}</style></head>
<body><div class="wrap">
<h1>CaP-Agent0 robosuite 대시보드</h1>
<p class="sub">{esc(str(root))}</p>
<div>
  <span class="kpi">태스크 <b>{done_tasks}</b></span>
  <span class="kpi">총 trial <b>{overall_n}</b></span>
  <span class="kpi">평균 성공률 <b>{overall:.0f}%</b></span>
</div>
<h2>태스크별 결과</h2>{table}
<p class="pin" style="margin-top:24px">각 리포트는 자체 포함 HTML(미디어 base64). 링크는 같은 폴더의 task별 report.html을 가리킵니다.<br>
생성: scripts/viz_index.py</p>
</div></body></html>"""

    out_path.write_text(htmldoc)
    print(f"wrote {out_path}  ({done_tasks} tasks, {overall_n} trials, avg success {overall:.0f}%)")


if __name__ == "__main__":
    main()
