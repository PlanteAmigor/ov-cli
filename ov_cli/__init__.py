"""ov-cli: OpenVINO LLM 命令行工具"""

import os

_LANG = "zh" if any(x in os.environ.get("LANG", "") for x in ("zh_CN", "zh-", "zh_")) else "en"

def TR(zh, en):
    return zh if _LANG == "zh" else en
