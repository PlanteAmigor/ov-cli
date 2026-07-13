"""
ov-cli server: FastAPI 应用 + HTTP 路由。

纯文本生成，不做工具调用注入/解析。
"""

import json
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import openvino as ov
import openvino_genai as ov_genai

from .config import _log, _log_request
from .model import load_model
from .prompt import build_prompt, extract_images
from .schemas import ChatCompletionRequest, ChatMessage
from .generate import non_stream_generate, stream_chat


def create_app(model_path: str, device: str = "", host: str = "0.0.0.0", port: int = 8080) -> FastAPI:
    if not device:
        devices = ov.Core().available_devices
        device = next((d for d in devices if "GPU" in d), "CPU")

    state = load_model(model_path, device)
    app = FastAPI(title="ov-cli", version="0.1.0")

    @app.get("/props")
    async def get_props():
        _log_request("GET", "/props")
        is_vlm = state.get("is_vlm", False)
        n_ctx_train = state.get("n_ctx_train", 4096)
        try:
            gen_cfg = state["pipe"].get_generation_config()
            raw = gen_cfg.max_new_tokens if hasattr(gen_cfg, 'max_new_tokens') else 16384
            max_new_tokens = min(raw, 131072) if raw > 1000000 else raw
        except Exception:
            max_new_tokens = 16384
        return {
            "default_generation_settings": {
                "temperature": 0.7, "top_p": 0.9, "top_k": 40, "max_tokens": max_new_tokens,
            },
            "total_slots": 1,
            "n_ctx_train": n_ctx_train,
            "model_alias": state.get("model_name", ""),
            "model_path": state.get("model_path", ""),
            "modalities": {"vision": is_vlm, "audio": False},
            "build_info": f"OpenVINO {ov.__version__} | optimum-intel | ov-cli",
        }

    @app.get("/slots")
    async def get_slots():
        n_ctx_train = state.get("n_ctx_train", 4096)
        return [{"id": 0, "state": "idle", "n_ctx": n_ctx_train}]

    @app.get("/health")
    async def health():
        _log_request("GET", "/health")
        return {"status": "ok"}

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

    @app.get("/v1/models")
    async def list_models():
        is_vlm = state.get("is_vlm", False)
        model_name = state.get("model_name", "default")
        n_ctx_train = state.get("n_ctx_train", 4096)
        _log_request("GET", "/v1/models", f"{model_name} {'📷' if is_vlm else ''}")
        return {
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ov-cli",
                "capabilities": {
                    "vision": is_vlm, "chat": True, "tools": False,
                },
                "meta": {
                    "n_ctx_train": n_ctx_train,
                    "n_ctx": n_ctx_train,
                },
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        state = load_model(model_path, device)
        pipe = state.get("pipe")

        raw_messages = [m.model_dump() for m in req.messages]
        _log(f"  🔍 REQUEST: temp={req.temperature}, max_tokens={req.max_tokens}, messages={len(raw_messages)}")

        if os.environ.get("OV_CLI_DEBUG_REQUEST"):
            dump_path = os.path.join(os.path.dirname(model_path.rstrip("/")), "..", ".last_request.json")
            try:
                with open(dump_path, "w") as f:
                    json.dump(req.model_dump() if hasattr(req, 'model_dump') else {
                        "messages": raw_messages, "temperature": req.temperature,
                        "max_tokens": req.max_tokens, "stream": req.stream,
                        "model": req.model,
                    }, f, ensure_ascii=False, indent=2, default=str)
                _log(f"  💾 完整请求已保存: {os.path.abspath(dump_path)}")
            except Exception as e:
                _log(f"  ⚠ 保存请求失败: {e}")

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
        images = extract_images(messages_dict) if has_image else []
        request_id = uuid.uuid4().hex[:12]

        if req.stream:
            return StreamingResponse(
                stream_chat(request_id, model_path, device,
                           messages_dict, gen_cfg, has_image, images),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            _log_request("POST", "/v1/chat/completions",
                         f"non-stream{' 📷' if has_image else ''}")
            result, assistant_dict = non_stream_generate(
                pipe, messages_dict, gen_cfg,
                state["is_vlm"], has_image, images,
            )
            if "error" in result:
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": result["error"]}},
                )
            return {
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "choices": [result],
                "usage": result.get("usage", {}),
            }

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

    @app.get("/{path:path}")
    async def serve_root(path: str):
        if path == "" or path == "/":
            return HTMLResponse("""<!doctype html><html><head><meta charset="utf-8"><title>ov-cli</title>"""
                                """<style>body{font-family:system-ui;display:flex;justify-content:center;"""
                                """align-items:center;height:100vh;margin:0;background:#1a1a2e;color:#e0e0e0}"""
                                """.card{text-align:center;max-width:480px;padding:40px}h1{font-size:24px;"""
                                """margin-bottom:8px}p{color:#888;line-height:1.6}code{background:#0f3460;"""
                                """padding:2px 8px;border-radius:4px;font-size:14px}</style></head><body>"""
                                """<div class="card"><h1>ov-cli</h1><p>OpenVINO LLM 服务运行中</p>"""
                                """<p>API: <code>/v1/chat/completions</code><br>模型: <code>SEE /v1/models</code><br>"""
                                """健康检查: <code>/health</code></p></div></body></html>""", status_code=200)
        if path.startswith(("v1/", "docs", "openapi.json", "health", "properties")):
            return JSONResponse({"error": "not found"}, status_code=404)
        if path in ("tools", "mcp-servers"):
            return []
        if path == "token":
            return {}
        return JSONResponse({"error": "not found"}, status_code=404)

    return app


def run_server(model_path: str, device: str = "",
               host: str = "0.0.0.0", port: int = 8080):
    """启动 ov-cli server。"""
    import uvicorn

    if not device:
        devices = ov.Core().available_devices
        device = next((d for d in devices if "GPU" in d), "CPU")

    model_name = os.path.basename(model_path.rstrip("/"))
    _log(f"  ov-cli API Server")
    _log(f"  • 模型: {model_name}")
    _log(f"  • 设备: {device}")
    _log(f"  • UI:  http://localhost:{port}")
    _log(f"  • API: http://localhost:{port}/v1")
    _log(f"")

    app = create_app(model_path, device, host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
