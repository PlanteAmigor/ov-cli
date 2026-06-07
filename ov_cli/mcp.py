"""
ov-cli mcp: MCP (Model Context Protocol) Server.

通过 stdin/stdout JSON-RPC 暴露 LLM/TTS/ASR/Image 工具，
供 AI 编码助手（VS Code Copilot, Cursor, Claude Desktop 等）调用。

启动方式:
  ov-cli mcp --model ./Qwen3/8B-ov
"""

import os, sys, json, threading, queue, time
from pathlib import Path
import openvino_genai as ov_genai


# ── 工具注册 ────────────────────────────────────────────────

TOOLS = [
    {
        "name": "chat",
        "description": "Send a prompt to the local LLM and get a text response",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "User prompt / question"},
                "system": {"type": "string", "description": "System prompt"},
                "max_tokens": {"type": "integer", "default": 1024},
                "temperature": {"type": "number", "default": 0.7},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "chat_stream",
        "description": "Send a prompt to the local LLM and receive streaming text chunks (for real-time output)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "User prompt / question"},
                "system": {"type": "string", "description": "System prompt"},
                "max_tokens": {"type": "integer", "default": 1024},
                "temperature": {"type": "number", "default": 0.7},
            },
            "required": ["prompt"],
        },
    },
]


# ── JSON-RPC 编解码 ─────────────────────────────────────────

def _read_msg() -> dict | None:
    """从 stdin 读取一条 JSON-RPC 消息（以换行为分隔）。"""
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _send_msg(msg: dict):
    """发送一条 JSON-RPC 消息到 stdout。"""
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _error(req, code: int, message: str):
    _send_msg({"id": req.get("id"), "error": {"code": code, "message": message}})


def _result(req, data):
    _send_msg({"id": req.get("id"), "result": data})


# ── 模型加载（复用 chat.py） ────────────────────────────────

_model_ctx = None
_model_lock = threading.Lock()


def _ensure_model(model_path: str):
    """线程安全地加载模型（全局单例）。"""
    global _model_ctx
    with _model_lock:
        if _model_ctx is not None:
            return _model_ctx
        from .chat import load_model
        print(f"  📦 加载模型: {os.path.basename(model_path)}", file=sys.stderr)
        _model_ctx = load_model(model_path)
        return _model_ctx


# ── 工具处理 ────────────────────────────────────────────────

def _handle_chat(args: dict, stream: bool = False) -> str | list[str]:
    """执行 LLM 推理。"""
    ctx = _ensure_model(args["_model_path"])
    pipe = ctx["pipe"]
    is_vlm = ctx.get("is_vlm", False)
    prompt = args.get("prompt", "")
    system = args.get("system", "You are a helpful AI assistant.")
    max_tokens = args.get("max_tokens", 1024)
    temperature = args.get("temperature", 0.7)

    # 构建 prompt
    conv = []
    if system:
        conv.append({"role": "system", "content": system})
    conv.append({"role": "user", "content": prompt})
    from .chat import _build_prompt as build_prompt
    full_prompt = build_prompt(conv, pipe.get_tokenizer(), enable_thinking=True)

    # 生成配置
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_tokens
    cfg.temperature = temperature
    cfg.do_sample = temperature >= 0.01

    if stream:
        chunks = []
        q = queue.Queue()

        class Streamer:
            def __call__(self, word):
                q.put(word)
                return False

        def run():
            try:
                if is_vlm:
                    pipe.generate(full_prompt, generation_config=cfg, streamer=Streamer(), images=[])
                else:
                    pipe.generate(full_prompt, cfg, Streamer())
            except Exception:
                pass
            finally:
                q.put(None)

        t = threading.Thread(target=run, daemon=True)
        t.start()

        while True:
            word = q.get()
            if word is None:
                break
            chunks.append(word)

        return chunks
    else:
        result = pipe.generate(full_prompt, cfg)
        return str(result)


# ── 请求处理 ────────────────────────────────────────────────

def _process_request(req: dict, model_path: str):
    method = req.get("method", "")
    req_id = req.get("id")

    # 初始化
    if method == "initialize":
        _result(req, {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "ov-cli", "version": "0.0.0-dev"},
        })
        return

    # Ping
    if method == "ping":
        _result(req, {})
        return

    # 列出工具
    if method == "tools/list":
        _result(req, {"tools": TOOLS})
        return

    # 调用工具
    if method == "tools/call":
        name = req["params"]["name"]
        arguments = req["params"].get("arguments", {})
        arguments["_model_path"] = model_path

        try:
            if name == "chat":
                text = _handle_chat(arguments, stream=False)
                _result(req, {"content": [{"type": "text", "text": text}]})
            elif name == "chat_stream":
                chunks = _handle_chat(arguments, stream=True)
                full = "".join(chunks)
                _result(req, {"content": [{"type": "text", "text": full}]})
            else:
                _error(req, -32601, f"Unknown tool: {name}")
        except Exception as e:
            _error(req, -32603, str(e))
        return

    # 未知方法
    _error(req, -32601, f"Unknown method: {method}")


# ── 入口 ────────────────────────────────────────────────────

def run_mcp(model_path: str):
    """运行 MCP Server（阻塞，直到 stdin 关闭）。"""
    if not os.path.isdir(model_path):
        print(f"  ❌ 模型目录不存在: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"  🧩 MCP Server 已启动 (stdin/stdout)", file=sys.stderr)
    print(f"  模型: {os.path.basename(model_path)}", file=sys.stderr)
    print(f"  工具: chat, chat_stream", file=sys.stderr)

    while True:
        req = _read_msg()
        if req is None:
            break
        try:
            _process_request(req, model_path)
        except Exception as e:
            _error(req, -32603, str(e))

    print(f"  👋 MCP Server 已关闭", file=sys.stderr)
