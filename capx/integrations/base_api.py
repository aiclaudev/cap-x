import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.base import BaseEnv


class ApiBase(ABC):
    """Base class for tool APIs.

    Guidelines:
    - Expose public functions via `functions()`.
    - Each function should have a Google-style docstring with Args/Returns.
    - `combined_doc()` returns a standardized aggregate doc for prompts.
    - If your API needs access to the environment, implement `set_env(env)`.

    Example:
    class GraspPlanApi(ApiBase):
        def functions(self) -> dict[str, Callable[..., Any]]:
            # optionally, we can expose only one of the functions if needed
            return {"grasp_plan": self.grasp_plan, "grasp_plan_sim": self.grasp_plan_sim}

        def grasp_plan(self, depth: np.ndarray, intrinsics: np.ndarray) -> list[dict]:
            \"""
            Plan parallel-jaw grasps from a depth map.

            Args:
            depth: (H, W) float32 meters.
            intrinsics: (3, 3) float32 pinhole intrinsics.

            Returns:
            List of dicts: {pose: (4,4) float32, width: float, score: float}.
            \"""
            ...

        def grasp_plan_sim(self, depth: np.ndarray, intrinsics: np.ndarray) -> list[dict]:
            ...
    """

    def __init__(self, env: BaseEnv) -> None:
        self._env = env
        self._webui_enabled: bool = False

    def enable_webui(self, enabled: bool = True) -> None:
        """Enable or disable web UI execution logging for this API instance."""
        self._webui_enabled = enabled

    def _log_step(
        self,
        tool_name: str,
        text: str,
        images: list[np.ndarray | Image.Image | str]
        | np.ndarray
        | Image.Image
        | str
        | None = None,
        highlight: bool = False,
    ) -> None:
        """Log an execution step if web UI mode is enabled.

        This is a thin wrapper around :func:`capx.utils.execution_logger.log_step`
        that no-ops when the web UI flag is off, keeping control API code free of
        conditional checks.
        """
        if not self._webui_enabled:
            return
        from capx.utils.execution_logger import log_step

        log_step(tool_name=tool_name, text=text, images=images, highlight=highlight)

    def _log_step_update(
        self,
        text: str | None = None,
        images: list[np.ndarray | Image.Image | str]
        | np.ndarray
        | Image.Image
        | str
        | None = None,
    ) -> None:
        """Update the last logged step if web UI mode is enabled."""
        if not self._webui_enabled:
            return
        from capx.utils.execution_logger import log_step_update

        log_step_update(text=text, images=images)

    @abstractmethod
    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return mapping of public function name -> callable."""

    def combined_doc(self) -> str:
        """Aggregate function docs in a simple, consistent format.

        Format per function:
            name(signature)
              Summary: first line of function doc
              Doc: full function docstring (Google style recommended)
        """
        # we need to discuss this design further down the line
        lines: list[str] = []
        # lines: list[str] = [f"API: {self.__class__.__name__}", ""]
        for name, fn in self.functions().items():
            try:
                sig = str(inspect.signature(fn))
            except Exception:
                sig = "(…)"
            doc = inspect.getdoc(fn) or ""
            # first = doc.splitlines()[0] if doc else ""
            lines.append(f"{name}{sig}")
            # if first:
            #     lines.append(f"  Summary: {first}")
            if doc:
                lines.append("  Doc:")
                lines.extend(f"    {ln}" for ln in doc.splitlines())
            lines.append("")
        doc = "\n".join(lines).strip()
        # agentv2 multi-view: when the env exposes >1 render camera, append a dynamic block
        # listing the actual view names so the model queries them (the static get_observation
        # docstring only documents robot0_robotview). Single-view configs (cap-agent0/agentv1)
        # have <=1 view here, so this is skipped and their prompt is unchanged.
        views = getattr(getattr(self, "_env", None), "render_camera_names", None)
        if views and len(views) > 1:
            doc += "\n\n" + self._multiview_camera_doc(list(views))
        return doc

    def _multiview_camera_doc(self, views: list[str]) -> str:
        """Dynamic 'available camera views' block for multi-view (agentv2) envs."""
        scene = views[0]
        wrist = next((v for v in views if "eye_in_hand" in v), views[-1])
        return (
            "AVAILABLE CAMERA VIEWS (IMPORTANT — READ THIS):\n"
            f"  get_observation() returns a SEPARATE entry for EACH of these camera views: {', '.join(views)}.\n"
            '  Each entry has ["images"]["rgb"] (H,W,3), ["images"]["depth"] (H,W), ["intrinsics"] (3,3), ["pose_mat"] (4,4).\n'
            "  These are DIFFERENT viewpoints of the same scene. Use whichever view(s) best see your target —\n"
            "  do NOT restrict yourself to robot0_robotview. A wrist view (..._eye_in_hand) gives a close-up of\n"
            "  the gripper/object; a scene view gives global layout. Example:\n"
            "    obs = get_observation()\n"
            f'    rgb_scene = obs["{scene}"]["images"]["rgb"]\n'
            f'    rgb_wrist = obs["{wrist}"]["images"]["rgb"]\n'
            f'    masks = segment_sam3_text_prompt(rgb_wrist, "<your target>")   # run perception on the best view\n'
            f'    depth = obs["{wrist}"]["images"]["depth"]; K = obs["{wrist}"]["intrinsics"]; pose = obs["{wrist}"]["pose_mat"]'
        )


_API_FACTORIES: dict[str, Callable[[], ApiBase]] = {}


def register_api(name: str, factory: Callable[[], ApiBase]) -> None:
    _API_FACTORIES[name] = factory


@lru_cache(maxsize=256)
def get_api(name: str) -> Callable[[BaseEnv], ApiBase]:
    if name not in _API_FACTORIES:
        raise KeyError(f"API '{name}' not registered")
    return _API_FACTORIES[name]


def list_apis() -> list[str]:
    return list(_API_FACTORIES.keys())
