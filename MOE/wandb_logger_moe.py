"""WandB logger with a couple of MoE-friendly helpers.
- Gracefully degrades to no-op if wandb isn't available or disabled.
- Supports logging scalars and histograms (if you choose to add them later).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


class WandBLogger:
    def __init__(
        self,
        enabled: Optional[bool] = None,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        mode: Optional[str] = None,  # 'offline' or 'online'
    ) -> None:
        if enabled is None:
            disabled_env = os.getenv("WANDB_DISABLED", "false").lower() in ("1", "true", "yes")
            enabled = not disabled_env

        self.enabled = bool(enabled)
        self._wandb = None
        self._active = False

        if not self.enabled:
            return

        try:
            import wandb  # type: ignore
            kwargs = {
                "project": project or os.getenv("WANDB_PROJECT", "glm-mini-prmoe"),
                "name": run_name or os.getenv("WANDB_RUN_NAME"),
                "entity": entity or os.getenv("WANDB_ENTITY"),
                "config": config or {},
            }
            _mode = mode or os.getenv("WANDB_MODE")
            if _mode is not None:
                kwargs["mode"] = _mode
            self._wandb = wandb
            self._wandb.init(**kwargs)
            self._active = True
        except Exception as e:
            print(f"[wandb] disabled: {e}")
            self.enabled = False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._active:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass

    def log_histogram(self, name: str, values: Any, step: Optional[int] = None) -> None:
        if self._active:
            try:
                hist = self._wandb.Histogram(values)
                self._wandb.log({name: hist}, step=step)
            except Exception:
                pass

    def set_summary(self, **kwargs: Any) -> None:
        if self._active:
            try:
                for k, v in kwargs.items():
                    self._wandb.run.summary[k] = v
            except Exception:
                pass

    def watch(self, model, log: str = "gradients", log_freq: int = 0) -> None:
        if self._active:
            try:
                self._wandb.watch(model, log=log, log_freq=log_freq)
            except Exception:
                pass

    def finish(self) -> None:
        if self._active:
            try:
                self._wandb.finish()
            except Exception:
                pass