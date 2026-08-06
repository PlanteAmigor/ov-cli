"""ov-cli: OpenVINO LLM 命令行工具"""

import os

_LANG = "zh" if any(x in os.environ.get("LANG", "") for x in ("zh_CN", "zh-", "zh_")) else "en"

def TR(zh, en):
    return zh if _LANG == "zh" else en


def has_gpu():
    """是否有可用 GPU 设备（匹配 GPU / GPU.0 / GPU.1 等）。"""
    import openvino as ov
    return any(d.startswith("GPU") for d in ov.Core().available_devices)
