#!/usr/bin/env python3
"""Auto-generate per-turn Korean summaries (intent + code diff) for a trial output dir.

For every turn of every (deduped) trial under an output_dir, calls a cheap letsur model to read
the turn's reasoning + code (and the previous turn's code) and writes a 1-2 sentence Korean
summary of (a) the agent's intent/plan and (b) what changed vs the previous turn. Results are
cached to `<trial>/turn_summaries.json` (idempotent — re-runs skip existing turns).

viz_trial.py renders these in each turn block. Run before viz_trial.py in report generation.

Usage:
    python scripts/summarize_turns.py outputs/<model>/<task>/   [--model gemini-3.5-flash] [--force]
"""
from __future__ import annotations
import argparse, json, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LETSUR_URL = "https://gw.letsur.ai/v1/chat/completions"
KEY_FILE = Path("/home/nas_main/dohyunlee/agentic-robotics-dev/.letsurkey")


def _flatten_code(r):
    return "\n".join(r.get("code_blocks", []) or [])


def _call(model, key, system, user, retries=3):
    payload = {"model": model, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": 600, "temperature": 0.3, "reasoning_effort": "low"}
    req = urllib.request.Request(LETSUR_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for a in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            if a == retries - 1:
                return f'{{"intent": "(요약 실패: {str(e)[:40]})", "diff": ""}}'
            time.sleep(3)


SYSTEM = ("당신은 로봇 조작 코드 에이전트의 행동을 분석하는 도우미입니다. 주어진 한 턴의 reasoning과 "
          "생성 코드를 읽고, 반드시 아래 JSON만 출력하세요(설명 금지): "
          '{"intent": "이 턴에서 에이전트의 계획/의도 1~2문장 한글 요약", '
          '"diff": "이전 코드 대비 무엇이 바뀌었는지 1~2문장 한글 요약 (이전 코드 없으면 \'최초 코드\')"}')


def _summarize_turn(model, key, idx, reasoning, code, prev_code):
    user = (f"[턴 {idx}]\n--- reasoning ---\n{(reasoning or '(없음)')[:4000]}\n"
            f"--- 생성 코드 ---\n{(code or '(코드 없음 — FINISH 등)')[:4000]}\n"
            f"--- 이전 턴 코드 ---\n{(prev_code or '(없음)')[:4000]}")
    out = _call(model, key, SYSTEM, user)
    m = re.search(r"\{.*\}", out, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:
        d = {}
    return {"intent": d.get("intent", "(요약 파싱 실패)"), "diff": d.get("diff", "")}


def _dedup_trials(root: Path):
    best = {}
    for d in root.iterdir():
        if not (d.is_dir() and d.name.startswith("trial_")):
            continue
        k = re.match(r"(trial_\d+)", d.name)
        key = k.group(1) if k else d.name
        try:
            n = len(json.load(open(d / "all_responses.json")))
        except Exception:
            n = -1
        sc = (len(list(d.glob("video_turn_*.mp4"))), n)
        if key not in best or sc > best[key][1]:
            best[key] = (d, sc)
    return [v[0] for v in best.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if not KEY_FILE.exists():
        print(f"!! no key at {KEY_FILE}"); sys.exit(1)
    key = KEY_FILE.read_text().strip()
    root = Path(args.output_dir)

    jobs = []  # (out_path, turns_data)
    for td in _dedup_trials(root):
        out = td / "turn_summaries.json"
        if out.exists() and not args.force:
            continue
        try:
            resp = json.load(open(td / "all_responses.json"))
        except Exception:
            continue
        jobs.append((out, resp))

    n_calls = 0
    def process(job):
        nonlocal n_calls
        out, resp = job
        sums = []
        prev = None
        for i, r in enumerate(resp):
            code = _flatten_code(r)
            sums.append(_summarize_turn(args.model, key, i, r.get("reasoning", ""), code, prev))
            prev = code or prev
            n_calls += 1
        out.write_text(json.dumps(sums, ensure_ascii=False, indent=1))
        return out

    print(f"summarizing {len(jobs)} trials under {root.name} (model={args.model})")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(process, jobs))
    print(f"done: {len(jobs)} trials, ~{n_calls} turn summaries written")


if __name__ == "__main__":
    main()
