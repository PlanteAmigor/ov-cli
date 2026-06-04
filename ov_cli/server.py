"""
ov-cli server: OpenAI 兼容 API 服务。

用法:
  ./ov-cli server --model ./model-ov --port 8080

API 端点 (OpenAI 兼容):
  GET  /v1/models              模型信息 + capabilities
  POST /v1/chat/completions    聊天补全 (SSE 流式)
  POST /v1/chat/completions/control  停止生成
  GET  /health                 健康检查
"""

import asyncio
import base64
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from pydantic import BaseModel

# FastAPI / uvicorn 按需安装
try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
except ImportError:
    raise ImportError(
        "需要安装 fastapi + uvicorn:\n"
        "  pip install 'fastapi>=0.100' 'uvicorn[standard]>=0.20'"
    )

import openvino as ov
import openvino_genai as ov_genai

# ── 全局状态 ────────────────────────────────────────────────

_model_lock = threading.Lock()
_model_state: dict[str, Any] = {}  # pipe, model_type, is_vlm, model_path
_running_tasks: dict[str, threading.Thread] = {}  # task_id -> thread
_running_task_lock = threading.Lock()


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


# ── 模型加载 ────────────────────────────────────────────────

def _load_model(model_path: str, device: str = "CPU") -> dict:
    """加载模型（线程安全，全局单例）。"""
    global _model_state
    with _model_lock:
        if _model_state and _model_state.get("model_path") == model_path:
            return _model_state

        is_vlm = os.path.isfile(os.path.join(model_path, "openvino_vision_embeddings_model.xml"))
        is_optimum = (
            os.path.isfile(os.path.join(model_path, "openvino_text_embeddings_per_layer_model.xml"))
            or (is_vlm and not os.path.isfile(os.path.join(model_path, "openvino_vision_embeddings_merger_model.xml")))
        )

        if is_optimum:
            _log(f"  📦 加载 Optimum: {os.path.basename(model_path)} ({device})", end=" ")
            t0 = time.time()
            import os as _os
            _os.environ["TRUST_REMOTE_CODE"] = "1"
            from optimum.intel import OVModelForVisualCausalLM
            from transformers import AutoConfig, AutoProcessor
            # 先加载 config（trust_remote_code 必须显式传）
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = OVModelForVisualCausalLM.from_pretrained(model_path, device=device, config=cfg, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            _log(f"✓ {time.time()-t0:.1f}s")
            pipe = None
            model_type = None
            cfg_path = os.path.join(model_path, "config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                model_type = cfg.get("model_type")
            model_name = cfg.get("_name_or_path", os.path.basename(model_path)) if os.path.isfile(cfg_path) else os.path.basename(model_path)
            _log(f"  📋 模型: {model_name} | 类型: {model_type or 'Optimum VLM'}")
            _model_state = {
                "pipe": None, "model": model, "processor": processor,
                "is_optimum": True, "is_vlm": is_vlm,
                "model_path": model_path, "model_name": model_name, "device": device,
            }
            return _model_state

        tag = "VLM" if is_vlm else "LLM"
        _log(f"  📦 加载 {tag}: {os.path.basename(model_path)} ({device})", end=" ")
        t0 = time.time()
        pipe = ov_genai.VLMPipeline(model_path, device) if is_vlm else ov_genai.LLMPipeline(model_path, device)
        _log(f"✓ {time.time()-t0:.1f}s")

        model_type = None
        cfg_path = os.path.join(model_path, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type")

        model_name = cfg.get("_name_or_path", os.path.basename(model_path)) if os.path.isfile(cfg_path) else os.path.basename(model_path)

        # Tokenizer 统计
        try:
            tok = pipe.get_tokenizer()
            test = tok.encode("Hello world")
            vocab = test.input_ids.shape[-1] if hasattr(test, 'input_ids') else 0
        except Exception:
            vocab = 0

        _log(f"  📋 模型: {model_name} | 类型: {model_type or tag} | 词表: {vocab}")

        _model_state = {
            "pipe": pipe,
            "model_type": model_type,
            "is_vlm": is_vlm,
            "model_path": model_path,
            "model_name": model_name,
            "device": device,
            "vocab_size": vocab,
        }
        return _model_state


def _build_prompt(messages: list, tokenizer) -> str:
    """将 OpenAI 消息列表转为 GenAI prompt。"""
    from ov_cli.chat import _build_prompt as _bp
    conv = []
    has_system = False
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            conv.insert(0, {"role": role, "content": content})
            has_system = True
        else:
            if isinstance(content, list):
                # VLM 消息: content 是 [{type: "text", text: ...}, {type: "image_url", ...}]
                text_parts = [c.get("text") or c.get("content", "") for c in content if c.get("type") == "text"]
                conv.append({"role": role, "content": " ".join(text_parts)})
            else:
                conv.append({"role": role, "content": content})
    return _bp(conv, tokenizer, enable_thinking=True)


def _extract_images_pil(messages):
    """从消息中提取所有图片，返回 list[PIL.Image]。"""
    from PIL import Image as _PIL
    import io as _io, base64 as _b64
    result = []
    try:
        for m in messages:
            for p in (m.get("content", []) if isinstance(m.get("content"), list) else []):
                if isinstance(p, dict) and p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    if url.startswith("data:image"):
                        raw = _b64.b64decode(url.split(",", 1)[1])
                        img = _PIL.open(_io.BytesIO(raw)).convert("RGB")
                        w, h = img.size
                        cur = w * h
                        max_px = 384 * 384
                        if cur > max_px:
                            ratio = (max_px / cur) ** 0.5
                            w, h = int(w * ratio), int(h * ratio)
                        w = max(32, (w // 32) * 32)
                        h = max(32, (h // 32) * 32)
                        result.append(img.resize((w, h)))
    except Exception:
        pass
    return result


def _extract_images(messages: list) -> list:
    """从消息中提取所有图片，返回 list[ov.Tensor]。
    与 chat.py._load_image 逻辑一致。
    """
    from PIL import Image
    import numpy as np
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


# ── Pydantic 请求模型 ──────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str | list

class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    stream: bool = True
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    presence_penalty: Optional[float] = None

class ControlRequest(BaseModel):
    task_id: str


# ── SSE 流式聊天 ────────────────────────────────────────────

async def _stream_chat(request_id: str, model_path: str, device: str,
                       messages: list, gen_cfg: ov_genai.GenerationConfig,
                       has_image: bool, images: list) -> AsyncGenerator[str, None]:
    """异步生成 SSE 事件流。"""
    state = _load_model(model_path, device)
    is_vlm = state["is_vlm"]

    if state.get("is_optimum"):
        # ── Optimum 格式流式 ──
        model = state["model"]
        processor = state["processor"]
        prompt = processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, chat_template_kwargs={"enable_thinking": True})
        from transformers import TextIteratorStreamer
        from threading import Thread

        # 图片 → processor（不是 model.generate）
        pil_images = _extract_images_pil(messages)

        # 为每张图插入占位标记
        img_tag = "<|vision_start|><|image_pad|><|vision_end|>\n"
        prompt_lines = messages[-1].get("content", "") if messages else ""
        if isinstance(prompt_lines, list):
            text = " ".join(p.get("text", "") for p in prompt_lines if isinstance(p, dict) and p.get("type") == "text")
        else:
            text = prompt_lines if isinstance(prompt_lines, str) else ""
        prompt = processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, chat_template_kwargs={"enable_thinking": True})

        if pil_images:
            prompt = img_tag * len(pil_images) + prompt
            inputs = processor(text=[prompt], images=pil_images, return_tensors="pt")
        else:
            inputs = processor(text=[prompt], return_tensors="pt")

        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs, max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=gen_cfg.temperature >= 0.01,
            temperature=gen_cfg.temperature if gen_cfg.temperature >= 0.01 else None,
            top_p=gen_cfg.top_p, top_k=gen_cfg.top_k,
            streamer=streamer,
        )

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        yield f"data: {json.dumps({'id': f'chatcmpl-{request_id}', 'choices': [{'delta': {'role': 'assistant'}, 'index': 0}]})}\n\n"
        for t in streamer:
            yield f"data: {json.dumps({'id': f'chatcmpl-{request_id}', 'choices': [{'delta': {'content': t}, 'index': 0}]})}\n\n"
        thread.join()
        yield "data: [DONE]\n\n"
        return

    # ── GenAI 格式流式 ──
    pipe = state["pipe"]
    tokenizer = pipe.get_tokenizer()

    prompt = _build_prompt(messages, tokenizer)

    # Token 计数
    try:
        enc = tokenizer.encode(prompt)
        prompt_tokens = enc.input_ids.shape[-1] if hasattr(enc, 'input_ids') else 0
    except Exception:
        prompt_tokens = 0

    _log_request("POST", "/v1/chat/completions",
                 f"stream | prompt {prompt_tokens} tok{' 📷' if has_image else ''}")

    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    stop_flag = [False]
    task_id = request_id
    token_count = [0]
    gen_start = time.time()

    def streamer_cb(t: str) -> bool:
        if stop_flag[0]:
            return True
        token_count[0] += 1
        queue.put_nowait(t)
        return False

    def run_generate():
        try:
            kwargs = {"generation_config": gen_cfg, "streamer": streamer_cb}
            if is_vlm and has_image and images:
                img_tag = "<|vision_start|><|image_pad|><|vision_end|>\n"
                prompt = img_tag * len(images) + prompt
                kwargs["images"] = images
            pipe.generate(prompt, **kwargs)
        except RuntimeError as e:
            err = str(e)
            if "reshape" in err.lower():
                queue.put_nowait("[ERROR: 该模型不支持图像输入]")
            else:
                queue.put_nowait(f"[ERROR: {err[:200]}]")
        except Exception as e:
            queue.put_nowait(f"[ERROR: {e}]")
        finally:
            queue.put_nowait(None)

    thread = threading.Thread(target=run_generate, daemon=True)
    with _running_task_lock:
        _running_tasks[task_id] = thread
    thread.start()

    try:
        # 发送角色标识
        yield f"data: {json.dumps({'id': f'chatcmpl-{task_id}', 'choices': [{'delta': {'role': 'assistant'}, 'index': 0}]})}\n\n"

        while True:
            t = await queue.get()
            if t is None:
                break
            if t.startswith("[ERROR:"):
                _log_request("POST", "/v1/chat/completions", f"✗ ERROR: {t[8:-1]}")
                yield f"data: {json.dumps({'error': {'message': t[8:-1]}})}\n\n"
                break
            if t:
                yield f"data: {json.dumps({'id': f'chatcmpl-{task_id}', 'choices': [{'delta': {'content': t}, 'index': 0}]})}\n\n"
    finally:
        elapsed = time.time() - gen_start
        _log_generation(state.get("model_name", "?"), prompt_tokens, token_count[0], elapsed)
        # 发送结束 + usage 统计
        yield f"data: {json.dumps({'id': f'chatcmpl-{task_id}', 'object': 'chat.completion.chunk', 'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': token_count[0], 'total_tokens': prompt_tokens + token_count[0]}})}\n\n"
        with _running_task_lock:
            _running_tasks.pop(task_id, None)

    yield "data: [DONE]\n\n"


# ── FastAPI 应用 ────────────────────────────────────────────

def create_app(model_path: str, device: str = "", host: str = "0.0.0.0", port: int = 8080) -> FastAPI:
    if not device:
        device = "GPU" if "GPU" in ov.Core().available_devices else "CPU"
    """创建 FastAPI 应用实例。"""
    # 预加载模型
    state = _load_model(model_path, device)

    app = FastAPI(title="ov-cli", version="0.1.0")

    # ── 服务器属性（llama.cpp UI 兼容） ──
    @app.get("/props")
    async def get_props():
        _log_request("GET", "/props")
        is_vlm = state.get("is_vlm", False)
        try:
            gen_cfg = state["pipe"].get_generation_config()
            n_ctx = gen_cfg.max_new_tokens if hasattr(gen_cfg, 'max_new_tokens') else 2048
        except Exception:
            n_ctx = 2048
        return {
            "default_generation_settings": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 1024,
            },
            "total_slots": 1,
            "model_alias": state.get("model_name", ""),
            "model_path": state.get("model_path", ""),
            "modalities": {
                "vision": is_vlm,
                "audio": False,
            },
            "build_info": f"OpenVINO {ov.__version__} | optimum-intel | ov-cli",
        }

    # ── 槽位信息（UI 展示生成统计） ──
    @app.get("/slots")
    async def get_slots():
        return [{
            "id": 0,
            "state": "idle",
            "n_ctx": 2048,
        }]

    # ── 健康检查 ──
    @app.get("/health")
    async def health():
        _log_request("GET", "/health")
        return {"status": "ok"}

    # ── 服务器属性 ──
    @app.get("/properties")
    async def properties():
        _log_request("GET", "/properties")
        return {
            "model_path": state.get("model_path", ""),
            "device": state.get("device", "CPU"),
            "model_type": state.get("model_type", ""),
            "is_vlm": state.get("is_vlm", False),
            "vocab_size": state.get("vocab_size", 0),
        }

    # ── 模型列表 ──
    @app.get("/v1/models")
    async def list_models():
        is_vlm = state.get("is_vlm", False)
        model_name = state.get("model_name", "default")
        _log_request("GET", "/v1/models", f"{model_name} {'📷' if is_vlm else ''}")
        return {
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "capabilities": {
                    "vision": is_vlm,
                    "chat": True,
                    "tools": False,
                    "mcp": False,
                    "skills": False,
                    "agent": False,
                    "rag": False,
                    "search": False,
                    "voice": False,
                    "speech": False,
                    "knowledge": False,
                },
            }],
        }

    # ── 聊天补全 ──
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        state = _load_model(model_path, device)
        pipe = state.get("pipe")

        if state.get("is_optimum"):
            tokenizer = state["processor"].tokenizer
        else:
            tokenizer = pipe.get_tokenizer()

        gen_cfg = ov_genai.GenerationConfig()
        gen_cfg.max_new_tokens = req.max_tokens
        gen_cfg.temperature = req.temperature
        gen_cfg.top_p = req.top_p
        gen_cfg.top_k = req.top_k
        gen_cfg.do_sample = req.temperature >= 0.01
        if req.presence_penalty is not None:
            gen_cfg.presence_penalty = req.presence_penalty

        has_image = any(
            isinstance(m.content, list) and any(p.get("type") == "image_url" for p in m.content)
            for m in req.messages
        )
        messages_dict = [m.model_dump() for m in req.messages]
        images = _extract_images(messages_dict) if has_image else []

        request_id = uuid.uuid4().hex[:12]

        if req.stream:
            return StreamingResponse(
                _stream_chat(request_id, model_path, device,
                            [m.model_dump() for m in req.messages],
                            gen_cfg, has_image, images),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # 非流式：收集所有 token
            _log_request("POST", "/v1/chat/completions",
                         f"non-stream{' 📷' if has_image else ''}")

            if state.get("is_optimum"):
                # ── Optimum 格式 ──
                model = state["model"]
                processor = state["processor"]
                prompt = processor.apply_chat_template(
                    [m.model_dump() for m in req.messages],
                    tokenize=False, add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": True}
                )
                from transformers import TextIteratorStreamer
                from threading import Thread

                # 图片 → processor
                pil_images = _extract_images_pil([m.model_dump() for m in req.messages]) if has_image else []

                img_tag = "<|vision_start|><|image_pad|><|vision_end|>\n"
                if pil_images:
                    prompt = img_tag * len(pil_images) + prompt
                    inputs = processor(text=[prompt], images=pil_images, return_tensors="pt")
                else:
                    inputs = processor(text=[prompt], return_tensors="pt")

                streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
                gen_kwargs = dict(
                    **inputs, max_new_tokens=req.max_tokens,
                    do_sample=req.temperature >= 0.01,
                    temperature=req.temperature if req.temperature >= 0.01 else None,
                    top_p=req.top_p, top_k=req.top_k,
                    streamer=streamer,
                )
                t0 = time.time()
                collected = []
                thread = Thread(target=model.generate, kwargs=gen_kwargs)
                thread.start()
                for t in streamer:
                    collected.append(t)
                thread.join()
                elapsed = time.time() - t0
                content = "".join(collected)
                return {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                }

            # ── GenAI 格式 ──
            collected = []
            stop_flag = [False]
            t0 = time.time()

            def cb(t):
                if stop_flag[0]:
                    return True
                collected.append(t)
                return False

            prompt = _build_prompt([m.model_dump() for m in req.messages], tokenizer)

            try:
                enc = tokenizer.encode(prompt)
                prompt_tokens = enc.input_ids.shape[-1] if hasattr(enc, 'input_ids') else 0
            except Exception:
                prompt_tokens = 0

            kwargs = {"generation_config": gen_cfg, "streamer": cb}
            if state["is_vlm"] and has_image and images:
                img_tag = "<|vision_start|><|image_pad|><|vision_end|>\n"
                prompt = img_tag * len(images) + prompt
                kwargs["images"] = images
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, lambda: pipe.generate(prompt, **kwargs))
            except RuntimeError as e:
                err = str(e)
                if "reshape" in err.lower():
                    return JSONResponse(
                        status_code=400,
                        content={"error": {"message": "该模型不支持图像输入 (This model does not support image input)", "type": "vlm_error"}}
                    )
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": f"生成失败: {err[:200]}", "type": "generation_error"}}
                )
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": f"内部错误: {str(e)[:200]}", "type": "internal_error"}}
                )

            elapsed = time.time() - t0
            content = "".join(collected)
            token_count = len(collected)
            _log_generation(state.get("model_name", "?"), prompt_tokens, token_count, elapsed)

            return {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                "usage": {"completion_tokens": token_count},
            }

    # ── Token 计数 ──
    @app.post("/token")
    async def count_token(req: Request):
        body = await req.json()
        content = body.get("content", "")
        try:
            tok = state.get("pipe").get_tokenizer()
            enc = tok.encode(content)
            count = enc.input_ids.shape[-1] if hasattr(enc, 'input_ids') else 0
        except Exception:
            count = 0
        return {"tokens": [count]}

    # ── WebSocket（用于 tokenizer 等实时通信） ──
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                data = await ws.receive_text()
                # 简单回显
                await ws.send_text(json.dumps({"echo": data}))
        except WebSocketDisconnect:
            pass

    # ── 停止生成 ──
    @app.post("/v1/chat/completions/control")
    async def stop_generation(req: ControlRequest):
        with _running_task_lock:
            if req.task_id in _running_tasks:
                pass
        return {"status": "stopped"}

    # ── 根路径 / 前端（未部署 UI 时返回提示） ──
    @app.get("/{path:path}")
    async def serve_root(path: str):
        if path == "" or path == "/":
            return HTMLResponse("""<!doctype html><html><head><meta charset="utf-8"><title>ov-cli</title><style>body{font-family:system-ui;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#1a1a2e;color:#e0e0e0}.card{text-align:center;max-width:480px;padding:40px}h1{font-size:24px;margin-bottom:8px}p{color:#888;line-height:1.6}code{background:#0f3460;padding:2px 8px;border-radius:4px;font-size:14px}</style></head><body><div class="card"><h1>ov-cli</h1><p>OpenVINO LLM 服务运行中</p><p>API: <code>/v1/chat/completions</code><br>模型: <code>SEE /v1/models</code><br>健康检查: <code>/health</code></p></div></body></html>""", status_code=200)
        # API 路径
        if path.startswith(("v1/", "docs", "openapi.json", "health", "properties")):
            from fastapi.responses import JSONResponse
            return JSONResponse({"error": "not found"}, status_code=404)
        if path in ("tools", "mcp-servers"):
            return []
        if path == "token":
            return {}
        return JSONResponse({"error": "not found"}, status_code=404)

    return app


# ── 入口 ────────────────────────────────────────────────────

def run_server(model_path: str, device: str = "",
               host: str = "0.0.0.0", port: int = 8080):
    """启动 ov-cli server。"""
    import uvicorn

    model_name = os.path.basename(model_path.rstrip("/"))
    _log(f"  ╔══════════════════════════════════════════════╗")
    _log(f"  ║        ov-cli API Server                     ║")
    _log(f"  ╠══════════════════════════════════════════════╣")
    _log(f"  ║  模型: {model_name:<36s}║")
    _log(f"  ║  设备: {device:<36s}║")
    _log(f"  ╠══════════════════════════════════════════════╣")
    _log(f"  ║  Web UI:  http://localhost:{port:<5d}               ║")
    _log(f"  ║  API:     http://localhost:{port:<5d}/v1          ║")
    _log(f"  ║  Docs:    http://localhost:{port:<5d}/docs         ║")
    _log(f"  ╚══════════════════════════════════════════════╝")
    _log(f"")

    app = create_app(model_path, device, host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
