"""Single-trial execution for CaP-X environments.

This module handles single trial execution including code generation,
multi-turn decisions, and visual feedback. It contains the core trial
loop extracted from launch.py, covering:

- Initial code generation and oracle code handling
- Code block execution with multi-turn regeneration
- Visual feedback capture and image/video differencing
- Trial artifact saving (code, logs, per-turn videos, combined video)
"""

from __future__ import annotations

import base64
import copy
import gc
import io
import json
import os
import time
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase

from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
    query_model_ensemble as _query_model_ensemble,
    query_single_model_ensemble as _query_single_model_ensemble,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _build_multi_turn_decision_prompt_legacy,
    _build_reflection_codegen_prompt,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
)
from capx.utils.video_utils import _encode_video_base64, _write_video

# Use TYPE_CHECKING to avoid circular imports for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs


MULTITURN_LIMIT = 10

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)


def _build_log_lines(
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    *,
    prefix: str = "",
    stderr_override: str | None = None,
) -> list[str]:
    """Build the standard log-line list used for both normal and timeout summaries."""
    stderr = stderr_override if stderr_override is not None else info_step.get("stderr", "")
    lines = ["-" * 100]
    if prefix:
        lines.append(prefix)
    lines.extend([
        "Generated program:",
        final_code if final_code else "(no program available)",
        "\n\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
        "-" * 100,
    ])
    return lines


# ---------------------------------------------------------------------------
# Trial video directory helper
# ---------------------------------------------------------------------------

def _trial_video_dir(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> str:
    """Return the trial output directory path used for video saving."""
    return os.path.join(
        config["output_dir"],
        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}"
        f"_taskcompleted_{int(info_step.get('task_completed', False))}",
    )


def _save_trial_video(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    num_code_blocks: int,
    *,
    suffix_extra: str = "",
) -> None:
    """Save recorded video frames from the environment, if available."""
    if not config["record_video"] or not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames or not config["output_dir"]:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)
    suffix = f"{reward:.3f}"
    if suffix_extra:
        suffix += f"_{suffix_extra}"

    if isinstance(frames, list):
        _write_video(frames, base_dir, suffix=suffix)
    elif isinstance(frames, dict):
        for key, frame in frames.items():
            _write_video(frame, base_dir, suffix=f"{suffix}_{key}")


def _save_turn_and_combined_videos(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    turn_frame_ranges: list[tuple[int, int]],
) -> None:
    """Save per-turn videos and a combined video of all turns.

    Gets all frames from the environment (clearing the buffer), then writes:
      - ``video_turn_00.mp4``, ``video_turn_01.mp4``, ... for each turn
      - ``video_combined.mp4`` for the full trial
      - If wrist camera is enabled: ``video_turn_00_wrist.mp4``, etc.
    """
    if not config["record_video"] or not config["output_dir"]:
        return
    if not hasattr(env, "get_video_frames"):
        return

    all_frames = env.get_video_frames(clear=True)
    if not all_frames:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)

    # all_frames may be a list (Robosuite) or a dict of lists (R1Pro multi-camera).
    # Normalise to a list for slicing; dict case is handled by _write_multi_video.
    if isinstance(all_frames, dict):
        # Multi-camera: write each camera stream as a combined video
        for key, frames in all_frames.items():
            if frames:
                _write_video(frames, base_dir, suffix=f"combined_{key}")
        return

    # Per-turn videos
    for i, (start, end) in enumerate(turn_frame_ranges):
        turn_frames = all_frames[start:end]
        if turn_frames:
            _write_video(turn_frames, base_dir, suffix=f"turn_{i:02d}")

    # Combined video
    _write_video(all_frames, base_dir, suffix="combined")

    # Wrist camera videos
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        wrist_frames = env.get_wrist_video_frames(clear=True)
        if wrist_frames:
            for i, (start, end) in enumerate(turn_frame_ranges):
                wrist_turn = wrist_frames[start:end]
                if wrist_turn:
                    _write_video(wrist_turn, base_dir, suffix=f"turn_{i:02d}_wrist")
            _write_video(wrist_frames, base_dir, suffix="combined_wrist")


# ---------------------------------------------------------------------------
# Visual feedback and image differencing
# ---------------------------------------------------------------------------

def _extract_task_goal(full_task_text: str) -> str:
    """Return a clean, code-free task goal for the VDM.

    The coder's user prompt bundles the goal together with the full API reference and an
    "ONLY write the executable Python code" instruction. Handing all of that to the VDM
    primes a code-capable VLM to emit code instead of a description. Keep only the ``Goal:``
    line; if there is none, fall back to the text before the ``APIs:`` section.
    """
    for line in full_task_text.splitlines():
        if line.strip().lower().startswith("goal:"):
            return line.strip()
    cut = full_task_text.find("APIs:")
    return (full_task_text[:cut] if cut != -1 else full_task_text).strip()


def _capture_initial_visual_feedback(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    config: dict[str, Any],
    args: LaunchArgs,
    visual_differencing_args: ModelQueryArgs,
) -> tuple[list, list[str], str]:
    """Capture the initial environment image and optionally describe it.

    Returns:
        (visual_feedback_imgs, visual_feedback_base64_history, task_description)
    """
    visual_feedback_imgs: list = []
    visual_feedback_base64_history: list[str] = []
    task_description = ""

    use_wrist = config.get("use_wrist_camera", False)

    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
        or config.get("use_video_differencing", False)
        or config.get("use_reflector", False)  # agentv1: Reflector needs task_description populated
    )
    if not (needs_visual and hasattr(env, "render")):
        return visual_feedback_imgs, visual_feedback_base64_history, task_description

    initial_base64, initial_img = _get_visual_feedback(env)
    visual_feedback_imgs.append(initial_img)
    visual_feedback_base64_history.append(initial_base64)
    _full_task_text = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])
    # The coder prompt carries the goal + full API reference + "ONLY write the executable
    # Python code"; feeding all of that to the VDM primes it to emit code. When vdm_goal_only
    # is set, hand the VDM just the task goal so it produces a scene description instead.
    task_description = (
        _extract_task_goal(_full_task_text)
        if config.get("vdm_goal_only", False) or config.get("use_reflector", False)
        else _full_task_text
    )

    # Also capture wrist camera image for multiview initial description
    initial_wrist_base64 = None
    if use_wrist and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            initial_wrist_base64 = (
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )
            visual_feedback_imgs.append(pil_wrist)

    # Append image to the prompt for VLM visual feedback
    if config["use_visual_feedback"]:
        obs["full_prompt"][-1]["content"][0]["text"] += (
            "\n\nIncluded below is an image of the initial state of the environment."
        )
        obs["full_prompt"][-1]["content"].append(
            {"type": "image_url", "image_url": {"url": initial_base64}}
        )
        if initial_wrist_base64 is not None:
            obs["full_prompt"][-1]["content"].append(
                {
                    "type": "text",
                    "text": "Included below is an image from the robot's wrist camera.",
                }
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_wrist_base64}}
            )

    # Initial-scene description: ask a VLM to describe the starting state so the FIRST code
    # generation isn't blind. This is generic image→text (no code/video yet), so agentv1 and
    # cap-agent0 use the SAME path/model here (visual_differencing_model). Only the multi-turn
    # Reflector loop uses the reflector model.
    if (
        config["use_img_differencing"]
        or config.get("use_video_differencing", False)
        or config.get("use_reflector", False)
    ):
        description = _describe_initial_scene(
            visual_differencing_args, task_description, initial_base64,
            wrist_image_base64=initial_wrist_base64,
            turn_tag="turn0",  # VDM I/O (prompt + input image → output text) logged inside for the viz tool
        )
        feedback = f"The initial state of the environment is described as follows:\n{description}"
        obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{feedback}"
        if args.debug:
            print(description)

    return visual_feedback_imgs, visual_feedback_base64_history, task_description


def _vdm_prompt_to_text(prompt: list[dict[str, Any]]) -> str:
    """Flatten a VDM chat prompt to readable text (images replaced by [이미지]) for the viz tool."""
    lines = []
    for m in prompt:
        c = m.get("content", "")
        if isinstance(c, list):
            segs = []
            for part in c:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        segs.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        segs.append("[이미지]")
                    else:
                        segs.append(f"[{part.get('type')}]")
                else:
                    segs.append(str(part))
            c = "\n".join(segs)
        lines.append(f"────── {m.get('role', '?')} ──────\n{c}")
    return "\n\n".join(lines)


def _log_vdm_io(tool: str, prompt: list[dict[str, Any]], output: str, images: list[str]) -> None:
    """Record one VDM call (input prompt + input image(s) → output text) for the viz tool."""
    try:
        from capx.utils.execution_logger import log_step as _log
        _log(tool, f"[VDM 입력 프롬프트]\n{_vdm_prompt_to_text(prompt)}\n\n[VDM 출력]\n{output}",
             images=[im for im in images if im])
    except Exception:
        pass


def _render_reflector_prompt_for_log(prompt: list[dict[str, Any]], frames: list[np.ndarray]) -> str:
    """Render the ACTUAL messages sent to the Reflector as readable text for the report.

    Text blocks are shown verbatim (exactly what the model receives); image_url (mp4 base64)
    blocks are replaced with a short marker since the base64 payload is huge and unreadable.
    """
    out: list[str] = []
    vid_idx = 0
    for msg in prompt:
        role = str(msg.get("role", "")).upper()
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append(f"=== {role} ===\n{content}")
            continue
        parts: list[str] = []
        for blk in content:
            if blk.get("type") == "text":
                parts.append(blk["text"])
            elif blk.get("type") == "image_url":
                if vid_idx == 0:
                    parts.append(
                        f"[영상 mp4(base64) 생략 — 메인 카메라 {len(frames)}프레임을 {REFLECTOR_ENCODE_FPS}fps로 "
                        f"인코딩해 전송 → gemini 실동작 ~{REFLECTOR_TARGET_FPS}fps 샘플. 아래 첫/마지막 프레임 참조]"
                    )
                else:
                    parts.append("[영상 mp4(base64) 생략 — wrist 카메라]")
                vid_idx += 1
        out.append(f"=== {role} ===\n" + "\n".join(parts))
    return "\n\n".join(out)


def _log_reflector_io(
    prompt: list[dict[str, Any]],
    frames: list[np.ndarray],
    output: str,
    turn_tag: str,
) -> None:
    """Record one Reflector call for the viz tool — logs the ACTUAL prompt sent to the model."""
    try:
        from capx.utils.execution_logger import log_step as _log
        input_text = _render_reflector_prompt_for_log(prompt, frames)
        imgs = ([frames[0], frames[-1]] if len(frames) > 1 else list(frames[:1])) if frames else []
        _log(f"Reflector[{turn_tag}] · 실제 전송 프롬프트 (입력 → 출력)",
             f"[Reflector 입력 — 모델에 실제 전송된 프롬프트]\n{input_text}\n\n[Reflector 출력]\n{output}",
             images=imgs)
    except Exception:
        pass


def _describe_initial_scene(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    image_base64: str,
    wrist_image_base64: str | None = None,
    turn_tag: str = "turn0",
) -> str:
    """Query a VLM to describe the initial environment state."""
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the initial state of the environment with the goal of the "
                "task in mind. You should try to provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera view:"},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    if wrist_image_base64 is not None:
        user_content.extend([
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image_url", "image_url": {"url": wrist_image_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the initial state of the "
                "environment with the goal of the task in mind. You should try to provide "
                "objective information and no assumptions. Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    content = _query_model(visual_differencing_args, prompt)["content"]
    _log_vdm_io(f"VDM[{turn_tag}] · 초기 장면 묘사 (input prompt + image → output text)",
                prompt, content, [image_base64, wrist_image_base64])
    return content


def _get_visual_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    visual_feedback_base64_history: list[str],
    wrist_base64_history: list[str] | None = None,
    turn_tag: str = "turn?",
) -> str | None:
    """Query a VLM to describe what changed between the two most recent frames.

    Args:
        wrist_base64_history: Optional history of wrist camera images.  When provided
            and has >=2 entries, the before/after wrist images are included in the prompt.
    """
    if len(visual_feedback_base64_history) < 2:
        return None

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the difference between the current state of the "
                "environment and the previous state of the environment with the "
                "goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code.."
            ),
        },
        {"type": "text", "text": "Previous state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
        {"type": "text", "text": "Current state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
    ]

    if wrist_base64_history and len(wrist_base64_history) >= 2:
        user_content.extend([
            {"type": "text", "text": "Previous state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-2]}},
            {"type": "text", "text": "Current state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-1]}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the difference between the "
                "current state of the environment and the previous state of the environment "
                "with the goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    content = _query_model(visual_differencing_args, prompt)["content"]
    # diff sees TWO images: previous state [-2] and current state [-1]
    _log_vdm_io(f"VDM[{turn_tag}] · 턴 차이 묘사 (input prompt + 2 images → output text)",
                prompt, content, [visual_feedback_base64_history[-2], visual_feedback_base64_history[-1]])
    return content


# ---------------------------------------------------------------------------
# Video differencing
# ---------------------------------------------------------------------------

def _get_video_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    turn_frames: list[np.ndarray],
    wrist_turn_frames: list[np.ndarray] | None = None,
) -> str | None:
    """Query a VLM with a video of the turn execution to describe what happened.

    Args:
        visual_differencing_args: Model query args for the VDM model.
        task_description: The task goal.
        turn_frames: RGB frames from the main camera for this turn.
        wrist_turn_frames: RGB frames from the wrist camera for this turn (optional).

    Returns:
        Text description of the execution, or None if no frames.
    """
    if not turn_frames:
        return None

    video_base64 = _encode_video_base64(turn_frames)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "The following video shows the robot executing code in the "
                "environment from the main camera view. Describe what happened "
                "during execution, including what actions the robot took, how "
                "the objects in the scene changed, and whether the task appears "
                "to have been completed. Provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera video:"},
        {"type": "image_url", "image_url": {"url": video_base64}},
    ]

    if wrist_turn_frames:
        wrist_video_base64 = _encode_video_base64(wrist_turn_frames)
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the same execution from the "
                    "robot's wrist-mounted camera (eye-in-hand view), providing "
                    "a close-up perspective of the gripper and objects being "
                    "manipulated."
                ),
            },
            {"type": "text", "text": "Wrist camera video:"},
            {"type": "image_url", "image_url": {"url": wrist_video_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that analyzes robot execution "
                "videos. You describe what happened during the robot's code "
                "execution, what actions were taken, how the environment "
                "changed, and whether the task appears to have been completed. "
                "Provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# agentv1 Reflector
# ---------------------------------------------------------------------------

REFLECTOR_SYSTEM_PROMPT = (
    "You are a Reflector agent for a robot-manipulation coding agent. You are given the task "
    "goal, the Python code that was just executed, its console output, and a video of the robot "
    "executing that code. Your job is to REFLECT on what happened: what the code was trying to do, "
    "what actually happened in the video, what went wrong or is incomplete, and concrete, specific "
    "guidance on what the code-generation agent should change next. "
    "The environment is NOT reset between turns — the executed code's effects persist, and the next "
    "code will continue from the CURRENT state (the state shown at the end of the video). Frame your "
    "guidance as what to do NEXT from this current state; do not tell it to redo steps that already succeeded. "
    "You do NOT write code — describe fixes in words, never Python. "
    "End your response with a verdict line, exactly one of:\n"
    "VERDICT: FINISH   (the task is fully and correctly completed)\n"
    "VERDICT: CONTINUE (more work is needed — your reflection above tells the next agent what to fix)"
)

# The Reflector sends the turn's execution as a VIDEO (mp4 base64 via image_url). letsur/gemini
# processes it (verified functionally: gemini reads objects + motion), tokenizing each sampled
# frame at LOW resolution (~65 tok/frame, measured; vs ~1089 for a full still image).
#
# HOW WE HIT ~5 fps: letsur ignores `video_metadata.fps`, and gemini re-samples the mp4 at ~1
# frame per second of the clip's DURATION (measured: mp4 of D seconds → ~D frames seen). So to make
# gemini sample the REAL motion at 5 fps, we stretch the timeline by encoding at capture/target fps:
#   encode_fps = SIM_CAPTURE_FPS / REFLECTOR_TARGET_FPS  (= 20/5 = 4)
# → an N-frame turn (N/20 s real) becomes an N/4 s mp4 → gemini sees ~N/4 frames = 5 per real second.
REFLECTOR_TARGET_FPS = 5      # frames per REAL second we want gemini to sample
SIM_CAPTURE_FPS = 20         # robosuite/libero control_freq (frames rendered per real second)
REFLECTOR_ENCODE_FPS = max(1, round(SIM_CAPTURE_FPS / REFLECTOR_TARGET_FPS))  # = 4


def _parse_reflection(content: str) -> tuple[str, str]:
    """Parse the Reflector's response into (verdict, reflection_text).

    verdict is "finish" or "continue" (defaults to "continue" if no explicit verdict is found,
    so the loop keeps going rather than stopping prematurely). The verdict line is stripped from
    the returned reflection text.
    """
    if not content:
        return "continue", ""
    verdict = "continue"
    lines = content.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("VERDICT:"):
            payload = stripped.split(":", 1)[1].strip().upper()
            verdict = "finish" if payload.startswith("FINISH") else "continue"
            continue  # drop the verdict line from the reflection text
        kept.append(line)
    return verdict, "\n".join(kept).strip()


def _get_reflection(
    reflector_args: ModelQueryArgs,
    task_description: str,
    executed_code: str,
    turn_frames: list[np.ndarray],
    stdout: str,
    stderr: str,
    wrist_turn_frames: list[np.ndarray] | None = None,
    turn_tag: str = "",
) -> dict[str, str] | None:
    """agentv1 Reflector: review executed code + execution video and reflect on what to fix.

    The Reflector does not generate code. It returns a reflection (guidance for the next
    code-generation turn) and a FINISH/CONTINUE verdict for whether the trial should stop.

    Args:
        reflector_args: Model query args for the Reflector model (must be a VLM — takes video).
        task_description: The task goal.
        executed_code: The code executed so far this trial (for context).
        turn_frames: RGB frames from the main camera for the turn just executed.
        stdout: Console stdout from the executed code.
        stderr: Console stderr from the executed code.
        wrist_turn_frames: Optional wrist-camera frames for the same turn.

    Returns:
        {"verdict": "finish"|"continue", "reflection": str, "raw": str}. When no video frames
        were captured (code crashed / no motion before the sim advanced), still reflects on the
        code + stdout/stderr alone rather than returning None, so pure-code crashes get retried.
    """
    # A turn produces NO video frames when the code crashes (or does no motion) BEFORE the
    # simulator advances — e.g. perception/IK returns None and raises during planning. Do NOT
    # skip the Reflector in that case (skipping silently ends the trial with no corrective turn);
    # reflect on the code + stdout/stderr traceback alone so pure-code crashes still get retried.
    has_video = bool(turn_frames)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Task goal:\n{task_description}"},
        {"type": "text", "text": "The Python code just executed in the environment was:"},
        {"type": "text", "text": f"```python\n{executed_code}\n```"},
        {"type": "text", "text": f"Console stdout:\n{stdout or '(empty)'}"},
        {"type": "text", "text": f"Console stderr:\n{stderr or '(empty)'}"},
    ]

    if has_video:
        video_base64 = _encode_video_base64(turn_frames, fps=REFLECTOR_ENCODE_FPS)  # → gemini ~5fps of real motion
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the robot executing that code from the main camera view. "
                    "Reflect on what happened and what to fix next."
                ),
            },
            {"type": "image_url", "image_url": {"url": video_base64}},
        ])
    else:
        user_content.append({
            "type": "text",
            "text": (
                "No execution video is available for this turn: the code raised an error (or performed "
                "no robot motion) before the simulator advanced, so nothing was rendered. Reflect on the "
                "code and the console output/traceback above and give concrete, specific guidance on what "
                "to fix in the next code (e.g. which API call returned None, what to check before using it)."
            ),
        })

    if has_video and wrist_turn_frames:
        wrist_video_base64 = _encode_video_base64(wrist_turn_frames, fps=REFLECTOR_ENCODE_FPS)
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the same execution from the robot's wrist-mounted "
                    "(eye-in-hand) camera, a close-up of the gripper and manipulated objects."
                ),
            },
            {"type": "image_url", "image_url": {"url": wrist_video_base64}},
        ])

    prompt = [
        {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    raw = _query_model(reflector_args, prompt)["content"]
    verdict, reflection = _parse_reflection(raw)
    _log_reflector_io(prompt, turn_frames, raw or "", turn_tag)
    return {"verdict": verdict, "reflection": reflection, "raw": raw or ""}


# ---------------------------------------------------------------------------
# Initial code generation
# ---------------------------------------------------------------------------

def _query_initial_code(
    args: LaunchArgs,
    config: dict[str, Any],
    obs: dict[str, Any],
) -> tuple[str, str | None, dict | None]:
    """Query the model for the initial code generation.

    Returns:
        (raw_code, reasoning, ensemble_data)
    """
    # Save the initial prompt
    with open(os.path.join(config["output_dir"], "initial_prompt.txt"), "w") as f:
        f.write(str(obs["full_prompt"]))

    ensemble_data = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTIMODEL ENSEMBLE QUERY")
            out = _query_model_ensemble(args, obs["full_prompt"], is_multiturn=False)
        else:
            print("RUNNING SINGLE MODEL ENSEMBLE QUERY")
            out = _query_single_model_ensemble(args, obs["full_prompt"], args.model, is_multiturn=False)
        ensemble_data = {
            "ensemble_candidates_txt": out["ensemble_candidates_txt"],
            "ensemble_synthesis_txt": out["ensemble_synthesis_txt"],
        }
    else:
        out = _query_model(args, obs["full_prompt"])

    return out["content"], out["reasoning"], ensemble_data


# ---------------------------------------------------------------------------
# Multi-turn decision handling
# ---------------------------------------------------------------------------

def _handle_multi_turn_step(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    args: LaunchArgs,
    config: dict[str, Any],
    visual_differencing_args: ModelQueryArgs,
    multi_turn_prompt: str,
    code_blocks: list[str],
    code_block_idx: int,
    info_step: dict[str, Any],
    task_description: str,
    visual_feedback_imgs: list,
    visual_feedback_base64_history: list[str],
    stderr_history: list[str],
    turn_frames: list[np.ndarray] | None = None,
    wrist_turn_frames: list[np.ndarray] | None = None,
    wrist_base64_history: list[str] | None = None,
    reflector_args: ModelQueryArgs | None = None,
) -> tuple[str, str | None, str | None, dict | None, list | None]:
    """Execute one multi-turn decision step.

    Captures visual feedback, builds the decision prompt, queries the model,
    and returns the parsed decision.

    Args:
        turn_frames: Frames from the main camera for this turn (for video differencing).
        wrist_turn_frames: Frames from the wrist camera for this turn (for video differencing).
        wrist_base64_history: History of wrist camera base64 images for image-based
            differencing with multiview.

    Returns:
        (decision, new_code, reasoning, multiturn_ensemble_entry)
        where decision is "regenerate", "finish", or "continue".
    """
    use_wrist = config.get("use_wrist_camera", False)

    executed_code = "\n".join(code_blocks[:code_block_idx])
    complete_multi_turn_prompt = multi_turn_prompt.format(
        executed_code=executed_code,
        console_stdout=info_step["stdout"],
        console_stderr=info_step["stderr"],
    )

    if info_step["stderr"] != "":
        stderr_history.append(info_step["stderr"])

    # --- agentv1 Reflector path ---
    # Reflect on the executed code + execution video, decide FINISH/CONTINUE, and on CONTINUE
    # generate corrected code from the reflection. This replaces the VDM decision path.
    if config.get("use_reflector"):
        reflection_out = _get_reflection(
            reflector_args if reflector_args is not None else visual_differencing_args,
            task_description,
            executed_code,
            turn_frames or [],
            info_step["stdout"],
            info_step["stderr"],
            wrist_turn_frames,
            turn_tag=f"turn{code_block_idx}",
        )
        if reflection_out is None:
            # Reflector could not run at all (should be rare — it now reflects on code+stderr even
            # with no video). Proceed without regenerating rather than ending the trial.
            return "continue", None, None, None, None
        reflection = reflection_out["reflection"]
        if reflection_out["verdict"] == "finish":
            print("Reflector chose to finish")
            return "finish", None, reflection, None, None
        # CONTINUE: hand the reflection to the code-generation agent (pure code gen, no decision).
        codegen_prompt = _build_reflection_codegen_prompt(obs, complete_multi_turn_prompt, reflection)
        content = _query_model(args, codegen_prompt)
        # The CodeGen agent only writes code; its "reasoning" (if any) is its own think trace.
        # Do NOT prepend the reflection here — it's already shown in the Coder input + Reflector
        # cards, so prepending it made the Coder-output card falsely look like the coder reflected.
        codegen_reasoning = content.get("reasoning") or "(CodeGen은 순수 코드 생성 — 반성은 입력/Reflector 카드 참조)"
        return "regenerate", content["content"], codegen_reasoning, None, codegen_prompt

    # Capture visual feedback if applicable
    visual_feedback_base64 = None
    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
    )
    if needs_visual and hasattr(env, "render"):
        vf_base64, vf_img = _get_visual_feedback(env)
        visual_feedback_imgs.append(vf_img)
        visual_feedback_base64_history.append(vf_base64)

        # Also capture wrist camera snapshot for image-based multiview
        if use_wrist and hasattr(env, "render_wrist") and wrist_base64_history is not None:
            wrist_result = _get_visual_feedback(env, use_wrist_camera=True)
            if wrist_result[0] is not None and isinstance(wrist_result[0], list) and len(wrist_result[0]) > 1:
                wrist_base64_history.append(wrist_result[0][1])  # index 1 = wrist image

    # Determine differencing feedback
    differencing_feedback = None
    is_video_feedback = False

    if config.get("use_video_differencing") and turn_frames:
        # Video-based differencing: pass video of this turn to VDM
        differencing_feedback = _get_video_differencing_feedback(
            visual_differencing_args, task_description, turn_frames, wrist_turn_frames,
        )
        is_video_feedback = True
    elif config["use_img_differencing"] and len(visual_feedback_base64_history) >= 2:
        # Image-based differencing: pass before/after images to VDM
        # (VDM I/O — prompt + 2 images → text — is logged inside the function for the viz tool)
        differencing_feedback = _get_visual_differencing_feedback(
            visual_differencing_args, task_description, visual_feedback_base64_history,
            wrist_base64_history=wrist_base64_history,
            turn_tag=f"turn{code_block_idx}",
        )

    # Only pass visual feedback to prompt if visual_feedback is enabled
    if not config["use_visual_feedback"]:
        visual_feedback_base64 = None
    elif needs_visual and hasattr(env, "render"):
        visual_feedback_base64 = visual_feedback_base64_history[-1] if visual_feedback_base64_history else None

    # Build decision prompt
    if args.use_legacy_multi_turn_decision_prompt:
        print("Using legacy multi-turn decision prompt")
        decision_prompt = _build_multi_turn_decision_prompt_legacy(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )
    else:
        decision_prompt = _build_multi_turn_decision_prompt(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )

    # Query model
    multiturn_ensemble_entry = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTITURN MULTIMODEL ENSEMBLE QUERY")
            content = _query_model_ensemble(args, decision_prompt, is_multiturn=True)
        else:
            print("RUNNING MULTITURN SINGLE MODEL ENSEMBLE QUERY")
            content = _query_single_model_ensemble(args, decision_prompt, args.model, is_multiturn=True)
        multiturn_ensemble_entry = {
            "ensemble_candidates_txt": content.get("ensemble_candidates_txt", ""),
            "ensemble_synthesis_txt": content.get("ensemble_synthesis_txt", ""),
        }
    else:
        content = _query_model(args, decision_prompt)

    reasoning = content["reasoning"]
    decision, new_code = _parse_multi_turn_decision(content["content"])

    return decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt


# ---------------------------------------------------------------------------
# Core single-trial execution
# ---------------------------------------------------------------------------

def _run_single_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Execute a single trial end-to-end.

    Steps:
        1. Reset the environment.
        2. Capture initial visual feedback (if configured).
        3. Query the model for initial code generation.
        4. Execute code blocks one-by-one, with optional multi-turn regeneration.
        5. Save artifacts (code, logs, per-turn videos, combined video) and return a TrialSummary.
    """
    trial_start_time = time.time()

    # Reset perception step logger so this trial's images don't accumulate from prior trials,
    # and enable webui-style logging on the APIs so perception steps (SAM3/grasp images) get
    # recorded even in headless mode (otherwise _log_step is a no-op and nothing is captured).
    try:
        from capx.utils.execution_logger import clear_all_histories
        clear_all_histories()
        for _api in getattr(env, "_apis", {}).values():
            if hasattr(_api, "enable_webui"):
                _api.enable_webui(True)
    except Exception:
        pass

    use_video_diff = config.get("use_video_differencing", False)
    use_wrist = config.get("use_wrist_camera", False)
    use_reflector = config.get("use_reflector", False)  # agentv1: needs per-turn video frames

    # --- 1. Reset environment ---
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    # Reset the SIGALRM timer AFTER env.reset() so the timeout only covers
    # actual task execution, not scene loading / cuRobo JIT compilation.
    import signal
    remaining = signal.alarm(0)  # cancel current alarm
    if remaining > 0:
        signal.alarm(1000)  # restart fresh 1000s from now
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    _patch_libero_goal(env, obs)

    if config["record_video"] and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)
    elif (use_video_diff or use_reflector) and hasattr(env, "enable_video_capture"):
        # Video differencing and the agentv1 Reflector need frame recording even without record_video
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

    # --- Shared trial state ---
    code_blocks: list[str] = []
    code_block_metadata: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    stderr_history: list[str] = []
    num_regenerations = 0
    num_finishes = 0
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    sandbox_rc_override = None
    ensemble_data = None
    multiturn_ensemble_data: list[dict[str, Any]] = []

    # Per-turn frame tracking (for video differencing and per-turn video saving)
    turn_frame_ranges: list[tuple[int, int]] = []

    # Wrist camera base64 history for image-based multiview differencing
    wrist_base64_history: list[str] | None = [] if use_wrist else None

    visual_differencing_args = ModelQueryArgs(
        model=args.visual_differencing_model,
        server_url=args.visual_differencing_model_server_url,
        api_key=args.visual_differencing_model_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )

    if config["use_img_differencing"] or use_video_diff or use_reflector:
        assert visual_differencing_args.model in VLM_MODELS, (
            "Image/video differencing model (also used for agentv1's initial-scene "
            "description) must be in the list of VLM models"
        )

    # agentv1: build the Reflector's query args. Reuses the visual-differencing endpoint
    # (server_url / api_key) but the model defaults to visual_differencing_model unless
    # reflector_model is set. Must be a VLM — it consumes the per-turn execution video.
    reflector_args: ModelQueryArgs | None = None
    if use_reflector:
        reflector_args = ModelQueryArgs(
            model=config.get("reflector_model") or args.visual_differencing_model,
            server_url=args.visual_differencing_model_server_url,
            api_key=args.visual_differencing_model_api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            debug=args.debug,
        )
        assert reflector_args.model in VLM_MODELS, (
            "agentv1 Reflector model must be a VLM (it consumes execution video)"
        )

    # --- 2. Capture initial visual feedback ---
    visual_feedback_imgs, visual_feedback_base64_history, task_description = (
        _capture_initial_visual_feedback(env, obs, config, args, visual_differencing_args)
    )

    # Seed wrist base64 history with initial wrist image
    if use_wrist and wrist_base64_history is not None and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            wrist_base64_history.append(
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )

    # --- 3. Initial code generation ---
    if config["use_oracle_code"]:
        raw_code = env.oracle_code
        with open(os.path.join(config["output_dir"], "oracle_code.py"), "w") as f:
            f.write(raw_code)
        reasoning = None
        ensemble_data = None
    else:
        raw_code, reasoning, ensemble_data = _query_initial_code(args, config, obs)

    # Initialize partial artifacts for timeout recovery
    if partial_artifacts is not None:
        partial_artifacts.update({
            "raw_code": raw_code,
            "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata,
            "all_responses": all_responses,
            "visual_feedback_imgs": visual_feedback_imgs,
            "info_step": info_step,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_regenerations": num_regenerations,
            "num_finishes": num_finishes,
            "num_code_blocks": 0,
            "ensemble_data": ensemble_data,
            "multiturn_ensemble_data": multiturn_ensemble_data,
        })

    # Parse initial code into blocks
    initial_blocks = _extract_code(raw_code)
    code_blocks.extend(initial_blocks)
    code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))
    all_responses.append({
        "block_idx": [0],
        "code_blocks": initial_blocks,
        "decision": "initial",
        "initial_prompt": copy.deepcopy(obs["full_prompt"]),
        "reasoning": reasoning if reasoning is not None else "",
    })

    with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
        json.dump(all_responses, f)

    if args.debug:
        with open(os.path.join(config["output_dir"], "code_init.txt"), "w") as f:
            f.write("\n".join(initial_blocks))

    # --- 4. Execute code blocks (with optional multi-turn) ---
    info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    code_block_idx = 0

    # Track whether we're recording frames (for video diff or record_video)
    recording_frames = (
        (config["record_video"] or use_video_diff or use_reflector)
        and hasattr(env, "get_video_frame_count")
    )

    while code_block_idx < len(code_blocks) and code_block_idx <= MULTITURN_LIMIT:
        code = code_blocks[code_block_idx]
        code_block_idx += 1

        # Record frame index before step
        frame_start = env.get_video_frame_count() if recording_frames else 0

        obs_next, reward, terminated, truncated, info_step = env.step(code)

        # Record frame index after step
        frame_end = env.get_video_frame_count() if recording_frames else 0
        turn_frame_ranges.append((frame_start, frame_end))

        if partial_artifacts is not None:
            partial_artifacts.update({
                "info_step": info_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            })

        obs = obs_next

        # Multi-turn decision
        if multi_turn_prompt:
            if "terminated episode" in info_step["stderr"]:
                truncated = True
                break

            # Get turn frames for video differencing
            turn_frames = None
            wrist_turn_frames = None
            if (use_video_diff or use_reflector) and recording_frames:
                turn_frames = env.get_video_frames_range(frame_start, frame_end)
                if use_wrist and hasattr(env, "get_wrist_video_frames_range"):
                    wrist_turn_frames = env.get_wrist_video_frames_range(
                        frame_start, frame_end,
                    )

            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, code_block_idx, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                stderr_history,
                turn_frames=turn_frames,
                wrist_turn_frames=wrist_turn_frames,
                wrist_base64_history=wrist_base64_history,
                reflector_args=reflector_args,
            )

            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)

            if decision == "regenerate":
                print("Model chose to regenerate code")
                new_blocks = _extract_code(new_code)
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": new_blocks,
                    "decision": "regenerate",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                })
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(
                    [{"generation": num_regenerations + 1, "regenerated": True,
                      "regenerated_at_idx": code_block_idx}]
                    * len(new_blocks)
                )
                num_regenerations += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_regenerations"] = num_regenerations

            elif decision == "finish":
                all_responses.append({
                    "decision": "finish",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                })
                print("Model chose to finish")
                num_finishes += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_finishes"] = num_finishes
                break

        print(f"Code block {code_block_idx} done")
        print(f"Number of code blocks: {len(code_blocks)}")

        # Save intermediate artifacts (code, logs) per code block
        final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
        _save_trial_artifacts(
            config, trial, info_step["sandbox_rc"], reward,
            info_step.get("task_completed", False), final_code, raw_code,
            all_responses, ["-" * 100, "Generated program:", final_code],
            visual_feedback_imgs,
        )

        # Only save intermediate video if NOT doing per-turn saving
        # (per-turn saving is deferred to after the loop to avoid clearing the buffer)
        if not recording_frames:
            _save_trial_video(
                env, config, trial, info_step, reward, len(code_blocks),
                suffix_extra=str(len(code_blocks)),
            )

    print("Code blocks done")

    # --- 5. Build final summary ---
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
    num_code_blocks = len(code_blocks)

    if partial_artifacts is not None:
        partial_artifacts["final_code"] = final_code
        partial_artifacts["num_code_blocks"] = num_code_blocks

    # Override sandbox_rc for terminated-episode stderr
    if "executing action in terminated episode" in info_step["stderr"]:
        sandbox_rc_override = 0
    if sandbox_rc_override is not None:
        info_step["sandbox_rc"] = sandbox_rc_override

    stderr = "\n\n".join(stderr_history) if stderr_history else info_step["stderr"]
    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        stderr_override=stderr,
    )

    code_path = _save_trial_artifacts(
        config, trial, info_step["sandbox_rc"], reward,
        info_step.get("task_completed", False), final_code, raw_code,
        all_responses, log_lines, visual_feedback_imgs,
        ensemble_data=ensemble_data,
        multiturn_ensemble_data=multiturn_ensemble_data,
    )

    # Save per-turn and combined videos
    if recording_frames and turn_frame_ranges:
        _save_turn_and_combined_videos(
            env, config, trial, info_step, reward, turn_frame_ranges,
        )
    else:
        _save_trial_video(env, config, trial, info_step, reward, num_code_blocks)

    success = info_step["sandbox_rc"] == 0

    # --- Evolving skill library integration (opt-in) ---
    if config.get("evolve_skill_library", False) and info_step.get("task_completed", False):
        try:
            from capx.skills import SkillLibrary

            skill_lib_path = config.get("skill_library_path", None)
            skill_lib = SkillLibrary(path=skill_lib_path)
            task_name = config.get("task_name", f"trial_{trial}")
            new_skills = skill_lib.extract_from_code(final_code, task_name=task_name)
            skill_lib.save()
            if new_skills:
                print(f"[SkillLibrary] Extracted {len(new_skills)} new skill(s): {new_skills}")
        except Exception as exc:
            print(f"[SkillLibrary] Skill extraction failed: {exc}")

    print(f"Trial {trial} took {time.time() - trial_start_time:.2f} seconds")

    # Persist perception step images (SAM3 / segmentation / grasp) for this trial so the viz tool can show them
    try:
        from capx.utils.execution_logger import get_all_histories, get_current_history
        _perc_dir = os.path.join(config["output_dir"], "perception", f"trial_{trial:02d}")
        _hists = list(get_all_histories())
        _cur = get_current_history()           # steps live here until finalized
        if _cur is not None and _cur not in _hists:
            _hists.append(_cur)
        for _h in _hists:
            _h.save_to_directory(_perc_dir)
    except Exception as _e:
        print(f"[perception-save] failed: {_e}")

    gc.collect()

    return TrialSummary(
        trial=trial,
        success=success,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", None),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )


def _patch_libero_goal(env: CodeExecutionEnvBase, obs: dict[str, Any]) -> None:
    """Inject the LIBERO task language into the prompt template if applicable."""
    if not hasattr(env.low_level_env, "handle"):
        return
    handle = env.low_level_env.handle
    if (
        hasattr(handle, "task_language")
        and "libero_environment_goal" in obs["full_prompt"][-1]["content"][0]["text"]
    ):
        goal = getattr(handle, "task_language")
        obs["full_prompt"][-1]["content"][0]["text"] = (
            obs["full_prompt"][-1]["content"][0]["text"].format(
                libero_environment_goal=goal
            )
        )
