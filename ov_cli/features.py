"""
ov-cli features: 按需安装的功能模块管理。

支持的功能:
  chat      — 聊天终端 (+PyMuPDF)
  image     — 文生图
  asr       — 语音识别 (+qwen-asr, soundfile, scipy)
  tts       — 语音合成 (+qwen-tts, soundfile)
  ui        — Gradio Web 界面 (+gradio)
  mcp       — MCP 协议服务器
  server    — API 服务器 (+fastapi, uvicorn)
  convert   — 模型转换 (+torch, optimum-intel...)
"""

import os

_FEATURES_FILE = ".ov-cli-features"


def get_installed(venv_path: str) -> set[str]:
    """读取已安装的功能列表。"""
    path = os.path.join(venv_path, _FEATURES_FILE)
    if not os.path.isfile(path):
        # 旧版本未记录 → 假设全装
        return {"chat", "image", "asr", "tts", "ui", "mcp", "server", "convert"}
    with open(path) as f:
        return {s.strip() for s in f.read().strip().split(",") if s.strip()}


def save(venv_path: str, features: set[str]):
    """写入已安装的功能列表。"""
    path = os.path.join(venv_path, _FEATURES_FILE)
    with open(path, "w") as f:
        f.write(",".join(sorted(features)) + "\n")


def has(venv_path: str, feature: str) -> bool:
    """检查某功能是否已安装。"""
    return feature in get_installed(venv_path)


# ── 功能 → pip 包映射 ──

_FEATURE_PACKAGES = {
    "chat":     [],
    "image":    [],
    "asr":      ["soundfile", "scipy"],
    "tts":      ["soundfile"],
    "ui":       ["gradio"],
    "mcp":      [],
    "server":   [],
    "convert":  ["torch", "torchvision"],
}

_FEATURE_EXTRA_PIPS = {
    "asr":  ["qwen-asr"],
    "tts":  ["qwen-tts"],
    "convert": ["optimum-intel@git+https://github.com/huggingface/optimum-intel.git",
                 "transformers", "nncf>=3.0", "safetensors", "sentencepiece",
                 "accelerate"],
}


def get_packages(features: set[str]) -> list[str]:
    """根据功能列表获取需要 pip 安装的包。"""
    pkgs = set()
    # 基础依赖（始终安装）
    pkgs.update([
        "openvino>=2026.2", "openvino-tokenizers", "openvino-genai",
        "pillow", "numpy", "jinja2", "huggingface-hub",
        "wcwidth", "PyMuPDF",  # chat 用
        "soundfile", "scipy",  # asr/tts 用
        "fastapi>=0.100", "uvicorn[standard]>=0.20",  # server 用
        "gradio",  # ui 用
    ])
    for f in features:
        pkgs.update(_FEATURE_PACKAGES.get(f, []))
    return sorted(pkgs)


def get_extra_pips(features: set[str]) -> list[str]:
    """获取额外 pip 包（qwen-tts/asr 等外部包）。"""
    pkgs = []
    for f in features:
        pkgs.extend(_FEATURE_EXTRA_PIPS.get(f, []))
    return pkgs
