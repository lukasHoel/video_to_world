"""
Shared logging helpers.

Many scripts in this repo are intended to be run as standalone modules. To keep log output
consistent (and avoid copy/pasting boilerplate handler setup everywhere), use `get_logger`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional


class _TqdmCompatibleStreamHandler(logging.StreamHandler):
    """StreamHandler that plays nicely with tqdm progress bars.

    If tqdm is available, uses tqdm.write(...) so output does not corrupt bars.
    Falls back to standard StreamHandler behavior otherwise.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            try:
                from tqdm.auto import tqdm  # type: ignore

                tqdm.write(msg)
                return
            except Exception:
                pass
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: Optional[str] = None, *, level: int = logging.INFO) -> logging.Logger:
    """
    Get a module logger with a simple stream handler (idempotent).

    - Avoids adding duplicate handlers on repeated imports / re-runs.
    - Uses a compact format: `[LEVEL] message`
    """
    logger = logging.getLogger(name if name is not None else __name__)

    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream_handler:
        handler = _TqdmCompatibleStreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


def try_create_tensorboard_writer(log_dir: str):
    """Best-effort TensorBoard writer creation (never hard-fails)."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning("TensorBoard not available (%s). Disabling tensorboard logging.", e)
        return None
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def tb_log_hparams(tb_writer, hparams: dict[str, Any], step: int = 0) -> None:
    """Log hyperparameters to TensorBoard (JSON blob + simple table)."""
    logger = get_logger(__name__)

    try:
        tb_writer.add_text(
            "hparams/json",
            "```json\n" + json.dumps(hparams, indent=2, sort_keys=True, default=str) + "\n```",
            step,
        )
    except Exception as e:
        logger.warning("Failed to write hparams/json to TensorBoard: %s", e)

    try:
        lines = ["| key | value |", "| --- | --- |"]
        for k in sorted(hparams.keys()):
            v = hparams[k]
            lines.append(f"| `{k}` | `{v}` |")
        tb_writer.add_text("hparams/table", "\n".join(lines), step)
    except Exception as e:
        logger.warning("Failed to write hparams/table to TensorBoard: %s", e)
