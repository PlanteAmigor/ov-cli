"""ov-cli: OpenVINO LLM 命令行工具"""

import os

_LANG = "zh" if any(x in os.environ.get("LANG", "") for x in ("zh_CN", "zh-", "zh_")) else "en"

def TR(zh, en):
    return zh if _LANG == "zh" else en


def has_gpu():
    """是否有可用 GPU 设备（匹配 GPU / GPU.0 / GPU.1 等）。"""
    import openvino as ov
    return any(d.startswith("GPU") for d in ov.Core().available_devices)


import functools as _functools

# OpenVINO GPU 插件只支持 Intel GPU；其他厂商（NVIDIA/AMD 等）会被枚举但
# 复杂模型 kernel 编译必然失败，通过设备名黑名单排除。
_NON_INTEL_GPU_MARKERS = (
    # NVIDIA
    "nvidia", "geforce", "rtx", "gtx", "quadro",
    # AMD
    "radeon", "amdgpu", "amd ", "advanced micro devices", "firepro", "instinct",
    # 其他 / 软件渲染
    "llvmpipe", "mesa", "apple", "qualcomm", "mali", "adreno",
)


def _probe_gpu(core, device):
    """判断 GPU 是否为可用的 Intel GPU（排除 NVIDIA/AMD 等非 Intel 设备）。"""
    try:
        name = core.get_property(device, "FULL_DEVICE_NAME") or ""
        low = name.lower()
        # 黑名单命中 → 非 Intel GPU，排除
        if any(k in low for k in _NON_INTEL_GPU_MARKERS):
            return False
        return True
    except Exception:
        return False


@_functools.lru_cache(maxsize=8)
def pick_device(device=""):
    """选择推理设备：显式指定优先；否则选择可用的 Intel GPU（排除 NVIDIA 等），失败回退 CPU。"""
    if device:
        return device
    import openvino as ov
    core = ov.Core()
    for d in sorted(core.available_devices):
        if d.startswith("GPU") and _probe_gpu(core, d):
            return d  # GPU.0 优先
    return "CPU"
