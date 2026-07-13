"""
ov-cli server: Prompt 构建、图片提取。
"""

import base64
import io

import numpy as np
import openvino as ov
from PIL import Image

from .config import _log


def build_prompt(messages: list, tokenizer) -> str:
    """将 OpenAI 消息列表转为 GenAI prompt。

    纯文本生成，不做工具注入/解析。
    """
    from ov_cli.chat import _build_prompt as _bp

    processed = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        entry = {"role": role, "content": content if content is not None else ""}
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in m:
                entry[key] = m[key]
        if isinstance(content, list):
            text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
            entry["content"] = " ".join(text_parts)
        processed.append(entry)

    return _bp(processed, tokenizer, enable_thinking=True)


def extract_images(messages: list) -> list:
    """从消息中提取所有图片，返回 list[ov.Tensor]。"""
    result = []
    try:
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:image"):
                            b64 = url.split(",", 1)[1]
                            raw = base64.b64decode(b64)
                            img = Image.open(io.BytesIO(raw)).convert("RGB")
                            w, h = img.size
                            cur_pixels = w * h
                            max_pixels = 384 * 384
                            if cur_pixels > max_pixels:
                                ratio = (max_pixels / cur_pixels) ** 0.5
                                w, h = int(w * ratio), int(h * ratio)
                            w = max(32, (w // 32) * 32)
                            h = max(32, (h // 32) * 32)
                            img = img.resize((w, h))
                            arr = np.array(img).astype(np.uint8)[None]
                            result.append(ov.Tensor(arr))
    except Exception as e:
        _log(f"  ⚠ 图片解析失败: {e}")
    return result
