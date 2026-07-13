"""
ov-cli server: 全局状态、常量、日志。
"""

import json
import os
import threading
import time
from typing import Any


# ── 全局状态 ────────────────────────────────────────────────

_model_lock = threading.Lock()
_generate_lock = threading.Lock()  # 串行化 generate 调用
_model_state: dict[str, Any] = {}  # pipe, model_type, is_vlm, model_path


# ── 生成锁 ──────────────────────────────────────────────────

# 串行化 generate 调用，避免多线程并发访问 pipe


# ── 日志 ────────────────────────────────────────────────────

def _log(msg: str, end="\n"):
    import sys
    sys.stdout.write(msg + end)
    sys.stdout.flush()


def _log_request(method: str, path: str, detail: str = ""):
    ts = time.strftime("%H:%M:%S")
    _log(f"  [{ts}] {method} {path}  {detail}")


def _log_generation(model_name: str, prompt_tokens: int, generated_tokens: int, elapsed: float):
    speed = generated_tokens / elapsed if elapsed > 0 else 0
    _log(f"  📊 {model_name} | "
         f"prompt {prompt_tokens} tok | "
         f"generated {generated_tokens} tok | "
         f"{elapsed:.1f}s | "
         f"{speed:.1f} tok/s")