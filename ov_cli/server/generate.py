"""
ov-cli server: 非流式和流式生成逻辑。

纯文本生成，不做工具调用注入/解析。
"""

import json
import time
from typing import AsyncGenerator

import openvino_genai as ov_genai

from .config import (
    _generate_lock,
    _log,
    _log_generation,
    _log_request,
)
from .prompt import build_prompt
from .model import load_model


def non_stream_generate(
    pipe, messages, gen_cfg, is_vlm, has_image, images,
):
    """非流式生成。"""
    tokenizer = pipe.get_tokenizer()
    prompt = build_prompt(messages, tokenizer)
    try:
        enc = tokenizer.encode(prompt)
        prompt_tokens = enc.input_ids.shape[-1] if hasattr(enc, 'input_ids') else 0
    except Exception:
        prompt_tokens = 0

    t0 = time.time()
    try:
        with _generate_lock:
            kwargs = {"generation_config": gen_cfg}
            if is_vlm and has_image and images:
                img_tag = "<|vision_start|><|image_pad|><|vision_end|>\n"
                prompt = img_tag * len(images) + prompt
                kwargs["images"] = images
            result = pipe.generate(prompt, **kwargs)
    except RuntimeError as e:
        return {"error": f"生成失败: {str(e)[:200]}"}, None
    except Exception as e:
        return {"error": f"内部错误: {str(e)[:200]}"}, None

    elapsed = time.time() - t0
    full_text = str(result) if result else ""
    _log(f"  🔍 OUTPUT: {len(full_text)} chars")
    _log_generation("", prompt_tokens, 0, elapsed)

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 0,
        "total_tokens": prompt_tokens,
    }
    result_choice = {
        "index": 0,
        "message": {"role": "assistant", "content": full_text},
        "finish_reason": "stop",
        "usage": usage,
    }
    return result_choice, {"role": "assistant", "content": full_text}


async def stream_chat(
    request_id: str, model_path: str, device: str,
    messages: list, gen_cfg: ov_genai.GenerationConfig,
    has_image: bool, images: list,
) -> AsyncGenerator[str, None]:
    """异步生成 SSE 事件流。

    直接用主线程调 pipe.generate()，拿到完整结果后切块发送。
    不做工具调用解析（纯观察派，由 agent 框架自行处理）。
    """
    state = load_model(model_path, device)
    is_vlm = state["is_vlm"]
    pipe = state["pipe"]
    tokenizer = pipe.get_tokenizer()
    prompt = build_prompt(messages, tokenizer)
    try:
        enc = tokenizer.encode(prompt)
        prompt_tokens = enc.input_ids.shape[-1] if hasattr(enc, 'input_ids') else 0
    except Exception:
        prompt_tokens = 0

    _log_request("POST", "/v1/chat/completions",
                 f"stream | prompt {prompt_tokens} tok{' 📷' if has_image else ''}")

    task_id = request_id

    yield f"data: {json.dumps({'id': f'chatcmpl-{task_id}', 'object': 'chat.completion.chunk', 'choices': [{'delta': {'role': 'assistant'}, 'index': 0}]})}\n\n"

    gen_start = time.time()
    full_text = ""

    try:
        kwargs = {"generation_config": gen_cfg}
        final_prompt = prompt
        if is_vlm and has_image and images:
            img_tag = "<|vision_start|><|image_pad|><|vision_end|>\n"
            final_prompt = img_tag * len(images) + prompt
            kwargs["images"] = images
        with _generate_lock:
            result = pipe.generate(final_prompt, **kwargs)
        full_text = str(result) if result else ""
    except RuntimeError as e:
        yield f"data: {json.dumps({'error': {'message': str(e)[:200]}})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        yield f"data: {json.dumps({'error': {'message': str(e)[:200]}})}\n\n"
        yield "data: [DONE]\n\n"
        return

    elapsed = time.time() - gen_start
    _log(f"  🔍 tokens={len(full_text)}, time={elapsed:.1f}s")

    for i in range(0, len(full_text), 4):
        chunk = full_text[i:i+4]
        yield f"data: {json.dumps({'id': f'chatcmpl-{task_id}', 'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': chunk}, 'index': 0}]})}\n\n"

    yield f"data: {json.dumps({'id': f'chatcmpl-{task_id}', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': 0, 'total_tokens': prompt_tokens}})}\n\n"
    yield "data: [DONE]\n\n"
