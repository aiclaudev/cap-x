#!/usr/bin/env python3
"""CaP-X trial visualizer — turn an eval output_dir into one self-contained HTML report.

Shows, per trial: the model's turn-by-turn output (generated code + REGENERATE/FINISH
decision + reasoning), the rollout video(s), any perception debug images
(depth / segmentation / detection / grasp), and the prompts. Everything is base64-embedded
so the HTML opens directly with file:// (no server needed).

Usage:
    python scripts/viz_trial.py outputs/<Model>/<config_out>/            # -> writes report.html in that dir
    python scripts/viz_trial.py outputs/<...>/ --out /tmp/report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import re
from pathlib import Path

MAX_EMBED_BYTES = 12 * 1024 * 1024  # skip-embed media larger than this

# perception debug images the integrations save (cwd-relative; copied here if found)
PERCEPTION_IMG_NAMES = [
    "depth_image.jpg", "segmentation_image.jpg", "seg_crop_image.jpg",
    "owlvit_det.jpg",
]
PERCEPTION_GLOBS = ["*sam3*.png", "*sam3*.jpg", "*seg*.png", "*grasp*.png", "*grasp*.jpg", "*detection*.png"]

CSS = """
:root{--bg:#fff;--panel:#f7f8fa;--panel2:#fbfcfd;--ink:#1f2328;--muted:#5c6570;
--accent:#0a66c2;--green:#1a7f37;--orange:#9a6700;--pink:#bf3989;--red:#cf222e;--border:#d8dde3;--code:#f6f8fa;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;line-height:1.6;font-size:15px}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px;padding-bottom:7px;border-bottom:1px solid var(--border)}
h3{font-size:15px;margin:18px 0 7px;color:var(--accent)}
.sub{color:var(--muted);margin:0 0 14px;font-size:13px}
code{background:#eef1f4;padding:1px 6px;border-radius:5px;font-size:13px;font-family:"SF Mono",ui-monospace,Menlo,monospace}
pre{background:var(--code);border:1px solid #e2e6eb;border-radius:9px;padding:12px 14px;overflow:auto;font-size:12.5px;line-height:1.5;font-family:"SF Mono",ui-monospace,Menlo,monospace;margin:8px 0}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}
th,td{border:1px solid var(--border);padding:7px 10px;text-align:left}th{background:#eef1f5}
.trial{border:1px solid var(--border);border-radius:12px;padding:16px 18px;margin:16px 0;background:var(--panel)}
.badge{display:inline-block;font-size:11px;font-weight:700;padding:2px 9px;border-radius:999px;margin-right:6px;vertical-align:middle}
.b-initial{background:#dbeeff;color:#0a4ea3}.b-regenerate{background:#fff3d6;color:#9a6700}.b-finish{background:#dcf5e3;color:#1a7f37}
.b-ok{background:#dcf5e3;color:#1a7f37}.b-bad{background:#fde2e2;color:#cf222e}
.turn{border:1px solid var(--border);border-radius:9px;margin:9px 0;background:#fff;padding:10px 12px}
.turn-h{font-weight:600;font-size:13.5px;margin-bottom:4px}
.imgrid{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}
.imgrid figure{margin:0;max-width:300px}.imgrid img{max-width:300px;border:1px solid var(--border);border-radius:8px;display:block}
.imgrid figcaption{font-size:12px;color:var(--muted);margin-top:3px}
video{max-width:420px;border:1px solid var(--border);border-radius:8px;background:#000}
details{border:1px solid var(--border);border-radius:9px;margin:8px 0;background:var(--panel2)}
summary{cursor:pointer;padding:9px 13px;font-size:13.5px;font-weight:600;list-style:none}
summary::-webkit-details-marker{display:none}summary::before{content:"\\25b8 ";color:var(--accent)}
details[open] summary::before{content:"\\25be "}
.dbody{padding:2px 14px 12px}
.dbody pre{max-height:460px}
.turn pre{max-height:520px}
.pin{color:var(--muted);font-size:12.5px}.muted{color:var(--muted)}
.reasoning{color:#444;font-size:13px;background:#fffdf5;border-left:3px solid #e3c66b;padding:7px 11px;border-radius:0 7px 7px 0;margin:7px 0;white-space:pre-wrap;max-height:340px;overflow:auto}
.vdmimg{margin:8px 0}.vdmimg img{max-width:560px;width:100%;border:2px solid var(--orange);border-radius:8px;display:block;margin-top:4px}
.steprow{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--muted);padding:4px 2px;border-bottom:1px dashed #e8ebef}
.stepnum{display:inline-block;min-width:34px;font-weight:700;color:#8a93a0;font-variant-numeric:tabular-nums}
.b-vdm{background:#fff3d6;color:#9a6700}.b-perc{background:#dcf5e3;color:#1a7f37}.b-ctrl{background:#eef1f5;color:#5c6570}
.tlcard{border:1px solid var(--border);border-radius:9px;margin:9px 0;background:#fff;padding:10px 12px}
.tlcard.vdm{border-color:var(--orange);background:#fffdf5}
"""


def b64_data_uri(path: Path) -> str | None:
    if not path.exists() or path.stat().st_size > MAX_EMBED_BYTES:
        return None
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def parse_trial_dirname(name: str) -> dict:
    out = {"name": name, "reward": None, "sandbox_rc": None, "task_completed": None}
    for key, pat in [("sandbox_rc", r"sandboxrc_(-?\d+)"), ("reward", r"reward_([\d.]+)"),
                     ("task_completed", r"taskcompleted_(\d+)")]:
        m = re.search(pat, name)
        if m:
            out[key] = m.group(1)
    return out


def _flatten_prompt(obj) -> str:
    """Render a chat prompt (list of {role, content}) or plain string to readable text."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        out = []
        for m in obj:
            if not isinstance(m, dict):
                out.append(str(m)); continue
            role = m.get("role", "?")
            c = m.get("content", "")
            if isinstance(c, list):
                segs = []
                for part in c:
                    if isinstance(part, dict):
                        t = part.get("type")
                        if t == "text":
                            segs.append(part.get("text", ""))
                        elif t == "image_url":
                            segs.append("[image]")
                        else:
                            segs.append(f"[{t}]")
                    else:
                        segs.append(str(part))
                c = "\n".join(segs)
            out.append(f"────── {role} ──────\n{c}")
        return "\n\n".join(out)
    return str(obj)


def load_vdm_by_turn(perc_dir: Path | None) -> dict:
    """Parse execution_history VDM[turnN] steps → {turn_int: [(output_text, [input_img_paths])]}.

    trial.py tags each VDM call with the turn whose coder input it feeds:
    VDM[turn0] = initial scene description (→ turn 0 input); VDM[turnK] = diff (→ turn K input).
    """
    out: dict[int, list] = {}
    if not (perc_dir and perc_dir.exists()):
        return out
    for jf in sorted(perc_dir.glob("execution_history_block_*.json")):
        try:
            hist = json.load(open(jf))
        except Exception:
            continue
        blk = hist.get("code_block_index", 0)
        for s in hist.get("steps", []):
            m = re.match(r"VDM\[turn(\d+)\]", s.get("tool_name", "") or "")
            if not m:
                continue
            si = s.get("step_index")
            imgs = sorted(perc_dir.glob(f"block_{blk}_step_{si}_img_*.jpg"))
            out.setdefault(int(m.group(1)), []).append((s.get("text", "") or "", imgs))
    return out


def render_turns(responses: list, tdir: Path, vdm_by_turn: dict | None = None) -> str:
    """One self-contained card per turn, in inference order within the turn:
    🖼️ VDM (input image → 묘사) → 📥 Coder 입력 → 📤 Coder 출력 → 🎬 실행결과."""
    vdm_by_turn = vdm_by_turn or {}
    turn_videos = sorted(tdir.glob("video_turn_*.mp4"))
    parts = []
    exec_idx = 0  # index into turn_videos (only executed turns produce a video)
    for i, r in enumerate(responses):
        dec = r.get("decision", "?")
        code = "\n".join(r.get("code_blocks", []) or [])
        reasoning = r.get("reasoning", "") or ""
        bidx = r.get("block_idx") or [i]
        turn_key = bidx[0] if isinstance(bidx, list) and bidx else i
        bcls = {"initial": "b-initial", "regenerate": "b-regenerate", "finish": "b-finish"}.get(dec, "b-initial")
        parts.append(f'<div class="turn"><div class="turn-h"><span class="badge {bcls}">{esc(dec).upper()}</span> '
                     f'turn {i} <span class="pin">(코드 {len(code)} chars)</span></div>')

        # 🖼️ VDM that fed THIS turn's input — input prompt + input image(s) → output text
        for vtext, vimgs in vdm_by_turn.get(turn_key, []):
            in_prompt, _, out_text = vtext.partition("[VDM 출력]")
            in_prompt = in_prompt.replace("[VDM 입력 프롬프트]", "").strip()
            out_text = out_text.strip()
            card = ['<div class="tlcard vdm"><div class="turn-h"><span class="badge b-vdm">🖼️ VDM (이 턴 입력에 들어간 시각 묘사)</span></div>']
            # 📥 input prompt (text actually sent to the VDM)
            if in_prompt:
                card.append(f'<details><summary>📥 VDM 입력 프롬프트 (VLM에 보낸 텍스트 · {len(in_prompt)} chars) — 클릭</summary>'
                            f'<div class="dbody"><pre>{esc(in_prompt[:30000])}</pre></div></details>')
            # 📥 input image(s): diff sees 2 (previous → current), initial sees 1
            if vimgs:
                caps = (["이전 상태", "현재 상태"] if len(vimgs) >= 2 else ["입력 이미지"])
                card.append('<div><b>📥 VDM이 본 입력 이미지'
                            + (f' ({len(vimgs)}장: 이전→현재)' if len(vimgs) >= 2 else '') + '</b></div><div class="imgrid">')
                for k, p in enumerate(vimgs):
                    uri = b64_data_uri(p)
                    if uri:
                        cap = caps[k] if k < len(caps) else f"img {k}"
                        card.append(f'<figure><img src="{uri}"/><figcaption>{cap} · {esc(p.name)}</figcaption></figure>')
                card.append('</div>')
            # 📤 output text
            if out_text:
                codeish = ("```" in out_text) or ("get_observation()" in out_text) or ("import " in out_text)
                warn = ' <span class="pin">⚠️ 묘사 대신 코드 출력</span>' if codeish else ''
                card.append(f'<details open><summary>📤 VDM 출력 (이미지 보고 생성한 텍스트) · {len(out_text)} chars{warn}</summary>'
                            f'<div class="dbody"><pre>{esc(out_text[:12000])}</pre></div></details>')
            card.append('</div>')
            parts.append("".join(card))

        # 📥 Coder INPUT (prompt actually sent to the code model)
        inp = _flatten_prompt(r.get("initial_prompt"))
        src = "initial_prompt"
        if not inp:
            mt = r.get("multi_turn_prompt")
            inp = _flatten_prompt(mt) if mt else ""
            src = "multi_turn_prompt"
        if inp.strip():
            parts.append(f'<details><summary>📥 Coder 입력 프롬프트 ({src} · {len(inp)} chars) — 클릭</summary>'
                         f'<div class="dbody"><pre>{esc(inp[:40000])}</pre></div></details>')
        elif dec in ("regenerate", "finish"):
            parts.append('<p class="pin">📥 Coder 입력: 저장 안 됨 — <code>save_multiturn_prompts: true</code>로 재실행 필요</p>')

        # 📤 Coder OUTPUT (generated code [+ reasoning if any])
        out_bits = []
        if reasoning.strip():
            out_bits.append(f'<div class="reasoning"><b>reasoning:</b><br>{esc(reasoning[:4000])}</div>')
        if code.strip():
            out_bits.append(f'<pre>{esc(code)}</pre>')
        else:
            out_bits.append('<p class="pin">(빈 코드 — 모델이 코드 없이 응답)</p>')
        parts.append('<details open><summary>📤 Coder 출력 (생성 코드) — 클릭</summary>'
                     '<div class="dbody">' + "".join(out_bits) + '</div></details>')

        # 🎬 EXECUTION RESULT (before→after): the rollout video for this executed turn
        if dec in ("initial", "regenerate") and exec_idx < len(turn_videos):
            uri = b64_data_uri(turn_videos[exec_idx])
            if uri:
                parts.append(f'<div style="margin-top:8px"><b>🎬 실행결과 (전→후):</b> '
                             f'<span class="pin">{esc(turn_videos[exec_idx].name)}</span><br>'
                             f'<video src="{uri}" controls muted loop></video></div>')
            exec_idx += 1
        parts.append('</div>')
    return "\n".join(parts)


def render_timeline(perc_dir: Path | None = None) -> str:
    """Inference-order timeline from execution_history JSON (already chronological by step_index).

    Interleaves VDM steps, perception steps (SAM3/grasp), and control steps (IK/move/gripper).
    VDM steps render the model's **input image large + output text** (the model saw → produced).
    Repetitive control/no-image steps collapse to one-liners so VDM/perception stand out.
    """
    if not (perc_dir and perc_dir.exists()):
        return ""
    hist_files = sorted(perc_dir.glob("execution_history_block_*.json"))
    if not hist_files:
        return ""
    parts = ['<h3>추론 순서 타임라인 (inference order — VDM · 지각 · 제어 스텝이 실행된 순서)</h3>']
    for jf in hist_files:
        try:
            hist = json.load(open(jf))
        except Exception:
            continue
        blk = hist.get("code_block_index", 0)
        for s in hist.get("steps", []):
            tool = s.get("tool_name", "") or ""
            text = (s.get("text", "") or "").strip()
            si = s.get("step_index")
            imgfiles = sorted(perc_dir.glob(f"block_{blk}_step_{si}_img_*.jpg"))
            is_vdm = tool.startswith("VDM")
            # control / no-image step → compact one-liner
            if not is_vdm and not imgfiles:
                oneline = text.replace("\n", " ")[:130]
                parts.append(f'<div class="steprow"><span class="stepnum">#{esc(si)}</span>'
                             f'<span class="badge b-ctrl">{esc(tool)}</span><span>{esc(oneline)}</span></div>')
                continue
            badge = "b-vdm" if is_vdm else "b-perc"
            card = [f'<div class="tlcard{" vdm" if is_vdm else ""}"><div class="turn-h">'
                    f'<span class="stepnum">#{esc(si)}</span><span class="badge {badge}">{esc(tool)}</span></div>']
            # VDM: input image FIRST (large, bordered), then the text it produced
            if is_vdm and imgfiles:
                uri = b64_data_uri(imgfiles[0])
                if uri:
                    card.append(f'<div class="vdmimg"><b>📥 VDM이 본 입력 이미지</b> '
                                f'<span class="pin">({esc(imgfiles[0].name)})</span>'
                                f'<img src="{uri}"/></div>')
            if text:
                lbl = "📤 VDM 출력 (이 이미지를 보고 생성한 텍스트)" if is_vdm else "출력 텍스트"
                card.append(f'<details{" open" if is_vdm else ""}><summary>{lbl} · {len(text)} chars</summary>'
                            f'<div class="dbody"><pre>{esc(text[:12000])}</pre></div></details>')
            # perception images (SAM3 seg overlays etc.)
            if not is_vdm and imgfiles:
                card.append('<div class="imgrid">')
                for p in imgfiles:
                    uri = b64_data_uri(p)
                    if uri:
                        card.append(f'<figure><img src="{uri}"/><figcaption>{esc(p.name)}</figcaption></figure>')
                card.append('</div>')
            card.append('</div>')
            parts.append("".join(card))
    return "\n".join(parts)


def render_media(tdir: Path, perc_dir: Path | None = None) -> str:
    parts = []
    # fallback: bare debug images saved directly in the trial dir (e.g. depth_image.jpg)
    bare = []
    for nm in PERCEPTION_IMG_NAMES:
        p = tdir / nm
        if p.exists():
            bare.append(p)
    for g in PERCEPTION_GLOBS:
        bare += sorted(tdir.glob(g))
    if bare:
        parts.append('<h3>기타 perception 이미지</h3><div class="imgrid">')
        for p in bare:
            uri = b64_data_uri(p)
            if uri:
                parts.append(f'<figure><img src="{uri}"/><figcaption>{esc(p.name)}</figcaption></figure>')
        parts.append('</div>')
    # combined rollout video only (per-turn videos are now shown inline in each turn card)
    cv = tdir / "video_combined.mp4"
    if cv.exists():
        uri = b64_data_uri(cv)
        if uri:
            parts.append(f'<h3>전체 롤아웃 영상 (combined)</h3><video src="{uri}" controls muted loop></video>')
    return "\n".join(parts)


def render_prompts(tdir: Path) -> str:
    pr = tdir / "prompts_and_responses"
    files = []
    if pr.exists():
        files = sorted(pr.glob("*.txt"))
    files += [f for f in [tdir / "initial_prompt.txt"] if f.exists()]
    if not files:
        return ""
    body = []
    for f in files:
        try:
            txt = f.read_text(errors="replace")
        except Exception:
            continue
        body.append(f'<details><summary>{esc(f.name)} ({len(txt)} chars)</summary>'
                    f'<div class="dbody"><pre>{esc(txt[:20000])}</pre></div></details>')
    return '<h3>프롬프트</h3>' + "\n".join(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", help="eval output_dir (contains trial_* subdirs)")
    ap.add_argument("--out", default=None, help="output html path (default: <output_dir>/report.html)")
    args = ap.parse_args()

    root = Path(args.output_dir)
    if not root.exists():
        raise SystemExit(f"output_dir not found: {root}")
    out_path = Path(args.out) if args.out else (root / "report.html")

    trial_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("trial_")])

    # A single multiturn trial writes one dir per intermediate turn-snapshot (same trial number,
    # differing reward suffix). Collapse to the most complete dir per trial number so "1 run" shows
    # exactly 1 card: rank by (#turn videos, reward) — the final attempt has the videos + best reward.
    def _completeness(d: Path):
        try:
            rw = float(parse_trial_dirname(d.name).get("reward") or 0.0)
        except ValueError:
            rw = 0.0
        return (len(list(d.glob("video_turn_*.mp4"))), rw)

    best: dict[str, Path] = {}
    for d in trial_dirs:
        m = re.match(r"(trial_\d+)", d.name)
        key = m.group(1) if m else d.name
        if key not in best or _completeness(d) > _completeness(best[key]):
            best[key] = d
    trial_dirs = [best[k] for k in sorted(best)]

    # overview rows
    rows, sections = [], []
    for td in trial_dirs:
        meta = parse_trial_dirname(td.name)
        ar = td / "all_responses.json"
        responses = []
        if ar.exists():
            try:
                responses = json.load(open(ar))
            except Exception:
                responses = []
        decisions = [r.get("decision", "?") for r in responses]
        n_regen = decisions.count("regenerate")
        rc = meta["sandbox_rc"]
        rc_badge = '<span class="badge b-ok">OK</span>' if rc == "0" else '<span class="badge b-bad">rc=%s</span>' % esc(rc)
        rows.append(f'<tr><td><a href="#{esc(td.name)}">{esc(td.name)}</a></td><td>{rc_badge}</td>'
                    f'<td>{esc(meta["reward"])}</td><td>{esc(meta["task_completed"])}</td>'
                    f'<td>{len(responses)}</td><td>{n_regen}</td><td>{" → ".join(esc(d) for d in decisions)}</td></tr>')

        sec = [f'<div class="trial" id="{esc(td.name)}"><h2>{esc(td.name)}</h2>']
        sec.append(f'<p class="pin">reward <b>{esc(meta["reward"])}</b> · sandbox_rc {esc(rc)} · '
                   f'task_completed {esc(meta["task_completed"])} · turns {len(responses)} · regenerations {n_regen}</p>')
        _m = re.match(r"trial_(\d+)", td.name)
        perc_dir = (root / "perception" / f"trial_{_m.group(1)}") if _m else None
        vdm_by_turn = load_vdm_by_turn(perc_dir)
        sec.append('<h3>턴별 정리 (VDM 입력·출력 → Coder 입력·출력 → 실행결과)</h3>')
        sec.append(render_turns(responses, td, vdm_by_turn) if responses else '<p class="muted">all_responses.json 없음</p>')
        # raw fine-grained inference timeline as a collapsible appendix
        tl = render_timeline(perc_dir)
        if tl:
            sec.append('<details><summary>🔬 전체 추론 스텝 타임라인 (perception 포함, raw) — 클릭</summary>'
                       f'<div class="dbody">{tl}</div></details>')
        sec.append(render_media(td, perc_dir))
        sec.append(render_prompts(td))
        sec.append('</div>')
        sections.append("\n".join(sec))

    overview = ('<table><tr><th>trial</th><th>실행</th><th>reward</th><th>완료</th><th>턴</th>'
                '<th>재생성</th><th>결정 시퀀스</th></tr>' + "\n".join(rows) + '</table>') if rows else \
        '<p class="muted">trial_* 디렉토리를 찾지 못했습니다.</p>'

    htmldoc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CaP-X trial report — {esc(root.name)}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>CaP-X trial report</h1>
<p class="sub">{esc(str(root))} · trials: {len(trial_dirs)}</p>
<h2>개요</h2>{overview}
{''.join(sections)}
<p class="pin" style="margin-top:30px">생성: scripts/viz_trial.py — 미디어는 base64 임베드(파일 단독 열람 가능).</p>
</div></body></html>"""

    out_path.write_text(htmldoc)
    print(f"wrote {out_path}  ({out_path.stat().st_size//1024} KB, {len(trial_dirs)} trials)")


if __name__ == "__main__":
    main()
