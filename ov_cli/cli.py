"""
ov-cli: OpenVINO LLM 命令行工具
"""

import os, sys, argparse
import ov_cli
from ov_cli import TR
from ov_cli.setup import cmd_setup

# 工作区根目录
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 功能检查装饰器 ──────────────────────────────────────────

def _require(feature, hint=None):
    """装饰器：检查指定 feature 是否已安装，否则退出。"""
    def deco(fn):
        def wrapper(*a, **kw):
            from .features import has
            if not has(feature):
                hint_text = hint or f"./ov-cli setup --with {feature}"
                print(f"  ⚠ {TR(f'{feature} 模块未安装，请运行:', f'{feature} not installed, run:')} {hint_text}")
                sys.exit(1)
            return fn(*a, **kw)
        return wrapper
    return deco


def _check_wsl2_gpu():
    """WSL2 下检查 Intel GPU runtime。"""
    try:
        with open("/proc/version") as f:
            if "microsoft" not in f.read().lower():
                return
    except Exception:
        return
    try:
        import openvino as ov
        if "GPU" not in ov.Core().available_devices:
            print(f"  {TR('⚠ WSL2 检测到 Intel GPU 但缺少 runtime，请安装:', '⚠ WSL2: Intel GPU detected but runtime missing, install:')}")
            print(f"    sudo apt install intel-level-zero-gpu libze1")
    except Exception:
        pass





@_require("chat", hint="./ov-cli setup --with chat")
def cmd_benchmark(args):
    """ov-cli benchmark"""
    from .benchmark import run_benchmark
    ov_path = os.path.abspath(args.model)
    run_benchmark(ov_path, args.reasoning == "on", device=args.device)


@_require("server", hint="./ov-cli setup --with server")
def cmd_server(args):
    """ov-cli server: 启动 API 服务"""
    from .server import run_server
    model_path = os.path.abspath(args.model)
    if not os.path.isdir(model_path):
        print(f"  ⚠ {TR('模型路径不存在', 'Model path not found')}: {model_path}")
        sys.exit(1)
    run_server(model_path, args.device, args.host, args.port)


@_require("image", hint="./ov-cli setup --with image")
def cmd_image(args):
    """ov-cli image: 文生图"""
    from .image import load_model, run_once, run_generate, run_pipe
    ov_path = os.path.abspath(args.model)
    if not os.path.isdir(ov_path):
        print(f"{TR('错误: 找不到模型目录', 'Error: model directory not found')}: {ov_path}")
        sys.exit(1)
    ctx = load_model(ov_path, device=args.device)
    if args.mode == "once":
        if not args.prompt:
            print(f"  ⚠ {TR('once 模式需要 --prompt 参数', 'once mode requires --prompt')}")
            sys.exit(1)
        run_once(ctx, prompt=args.prompt, output=args.output,
                 width=args.width, height=args.height,
                 steps=args.steps, guidance=args.guidance,
                 seed=args.seed, json_output=args.json)
    elif args.mode == "pipe":
        run_pipe(ctx, width=args.width, height=args.height,
                 steps=args.steps, guidance=args.guidance)
    else:
        run_generate(ctx, width=args.width, height=args.height,
                     steps=args.steps, guidance=args.guidance)


@_require("tts", hint="./ov-cli setup --with tts")
def cmd_tts(args):
    """ov-cli tts: 语音合成"""
    from .tts import load_model, run_once, run_pipe, detect_model_type
    ov_path = os.path.abspath(args.model)
    if not os.path.isdir(ov_path):
        print(f"{TR('错误: 找不到模型目录', 'Error: model directory not found')}: {ov_path}")
        sys.exit(1)
    mtype = detect_model_type(ov_path)
    if mtype is None:
        print(f"  ⚠ {TR('不是有效的 TTS 模型', 'Not a valid TTS model')}")
        sys.exit(1)
    ctx = load_model(ov_path, device=args.device)
    if mtype == "custom_voice":
        print(f"  {TR('类型: CustomVoice', 'Type: CustomVoice')}", file=sys.stderr)
        speakers = ctx["model"].get_supported_speakers()
        print(f"  {TR('预设声音:', 'Preset voices:')} {', '.join(speakers)}", file=sys.stderr)
    else:
        print(f"  {TR('类型: Base (声音克隆)', 'Type: Base (Voice Clone)')}", file=sys.stderr)
    print(file=sys.stderr)
    if args.mode == "pipe":
        run_pipe(ctx, speaker=args.speaker, language=args.lang,
                 instruct=args.instruct, ref_audio=args.ref_audio,
                 warmup=not args.no_warmup)
        return
    if not args.prompt:
        print(f"  ⚠ {TR('需要 --prompt 参数', 'requires --prompt')}")
        sys.exit(1)
    run_once(ctx, text=args.prompt, output=args.output,
             speaker=args.speaker, language=args.lang,
             instruct=args.instruct, ref_audio=args.ref_audio,
             warmup=not args.no_warmup, json_output=args.json)

@_require("ui", hint="./ov-cli setup --with ui")
def cmd_ui(args):
    """ov-cli ui: 网页界面"""
    from .ui import launch_ui
    launch_ui(model_path=args.model, device=args.device, port=args.port, share=args.share, reasoning=args.reasoning == "on")

@_require("chat", hint="./ov-cli setup --with chat")
def cmd_chat(args):
    """ov-cli chat"""
    from .chat import load_model, run_chat, run_translate
    if args.reasoning == "off" and args.mode != "translate":
        print(f"  {TR('💡 提示', '💡 Hint')}: "
              f"{TR('若当前是简易模式，--reasoning off 仅过滤 <think> 块显示...',
                   'In simple mode, --reasoning off only filters <think> blocks...')}")
    mode = args.mode
    if mode == "once" and not args.prompt and not args.file:
        print(f"  ⚠ {TR('once 模式需要 --prompt 和/或 --file', 'once mode requires --prompt and/or --file')}")
        sys.exit(1)
    ov_path = os.path.abspath(args.model)
    if not os.path.isdir(ov_path):
        print(f"{TR('错误: 找不到模型目录', 'Error: model directory not found')}: {ov_path}")
        sys.exit(1)
    if not os.path.isfile(os.path.join(ov_path, "openvino_model.xml")) and \
       not os.path.isfile(os.path.join(ov_path, "openvino_config.json")):
        print(f"{TR('错误: 找不到模型文件', 'Error: model file not found')}: {ov_path}")
        sys.exit(1)
    ctx = load_model(ov_path, device=args.device)
    if mode == "translate":
        run_translate(ctx, max_tokens=args.max_tokens)
    elif mode == "once":
        from .chat import run_once
        prompt = args.prompt.replace("\\n", "\n") if args.prompt else ""
        run_once(ctx, prompt=prompt, files=args.file or [],
                 output=args.output, temperature=args.temp, top_p=args.top_p,
                 top_k=args.top_k, max_tokens=args.max_tokens,
                 reasoning=args.reasoning == "on", json_output=args.json)
    elif mode == "pipe":
        from .chat import run_pipe
        run_pipe(ctx, reasoning=args.reasoning == "on", max_tokens=args.max_tokens, temperature=args.temp)
    else:
        run_chat(ctx, system=args.system, temperature=args.temp, top_p=args.top_p,
                 top_k=args.top_k, max_tokens=args.max_tokens, image_path=args.image,
                 reasoning=args.reasoning == "on")


@_require("asr", hint="./ov-cli setup --with asr")
def cmd_asr(args):
    """ov-cli asr: 语音转文字"""
    from .asr import load_model, run_once, run_whisper, run_pipe
    ov_path = os.path.abspath(args.model)
    if not os.path.isdir(ov_path):
        print(f"{TR('错误: 找不到模型目录', 'Error: model directory not found')}: {ov_path}")
        sys.exit(1)
    ctx = load_model(ov_path, device=args.device)
    if args.mode == "once":
        if not args.file:
            print(f"  ⚠ {TR('once 模式需要 --file 参数', 'once mode requires --file')}")
            sys.exit(1)
        run_once(ctx, file_path=args.file, lang=args.lang, output=args.output, json_output=args.json)
    elif args.mode == "pipe":
        run_pipe(ctx, lang=args.lang)
    else:
        run_whisper(ctx, lang=args.lang)


@_require("mcp", hint="./ov-cli setup --with mcp")
def cmd_mcp(args):
    """ov-cli mcp: MCP 协议服务器"""
    from .mcp import run_mcp
    ov_path = os.path.abspath(args.model)
    run_mcp(ov_path)


# ── 帮助文本 ──

_HELP_DESC_ZH = "ov-cli — 基于 OpenVINO 的 LLM 本地推理工具箱\n轻量、离线、CPU/GPU 皆可运行。"
_HELP_DESC_EN = "ov-cli — OpenVINO-powered LLM local inference toolkit."

_HELP_EPILOG_ZH = (
    "📖 使用示例:\n\n"
    "  ./ov-cli setup\n"
    "  ./ov-cli chat --model ./gemma-4-E2B-it-ov-int4\n"
    "  ./ov-cli asr --model ./whisper/ov-large\n"
    "  ./ov-cli image --model ./FLUX/ov-int4\n"
    "  ./ov-cli tts --model ./0.6B-CV-ov --prompt 你好 --speaker Vivian\n"
    "  ./ov-cli ui --model ./model-ov\n"
    "  ./ov-cli server --model ./model-ov --port 8080\n"
    "  ./ov-cli setup --fix\n"
)
_HELP_EPILOG_EN = (
    "📖 Examples:\n\n"
    "  ./ov-cli setup\n"
    "  ./ov-cli chat --model ./gemma-4-E2B-it-ov-int4\n"
    "  ./ov-cli asr --model ./whisper/ov-large\n"
    "  ./ov-cli image --model ./FLUX/ov-int4\n"
    "  ./ov-cli tts --model ./0.6B-CV-ov --prompt hello --speaker vivian\n"
    "  ./ov-cli ui --model ./model-ov\n"
    "  ./ov-cli server --model ./model-ov --port 8080\n"
    "  ./ov-cli mcp --model ./model-ov\n"
    "  ./ov-cli setup --fix\n"
)


# ── 入口 ──

def main():
    # W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--lang" and i + 1 < len(sys.argv):
            ov_cli._LANG = sys.argv[i + 1]
            break
        if a.startswith("--lang="):
            ov_cli._LANG = a.split("=", 1)[1]
            break

    zh = ov_cli._LANG == "zh"
    desc = _HELP_DESC_ZH if zh else _HELP_DESC_EN
    epilog = _HELP_EPILOG_ZH if zh else _HELP_EPILOG_EN
    parser = argparse.ArgumentParser(prog="ov-cli", description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=epilog)
    parser.add_argument("--lang", choices=["zh", "en"])
    sub = parser.add_subparsers(dest="cmd", required=True, title=TR("子命令", "commands"))

    # setup
    p = sub.add_parser("setup", help=TR("创建环境", "Setup"))
    p.add_argument("--optimum-dir", help=TR("optimum-intel 源码目录", "optimum-intel source"))
    p.add_argument("--with", dest="with_features", default="all",
        help=TR("按需安装 (chat,image,asr,tts,ui,mcp,server)", "Features (chat,image,asr,tts,ui,mcp,server)"))
    p.add_argument("--remove", dest="remove_features", default="",
        help=TR("移除模块 (chat,image,asr,tts,ui,mcp,server)", "Remove features (chat,image,asr,tts,ui,mcp,server)"))
    p.add_argument("--fix", action="store_true", help=TR("修复模式", "Fix mode"))

    # chat
    p = sub.add_parser("chat", help=TR("聊天/翻译", "Chat"))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--mode", choices=["chat","translate","once","pipe"], default="chat")
    p.add_argument("--prompt"), p.add_argument("--file", action="append", default=None)
    p.add_argument("--output"), p.add_argument("--system", default="You are a helpful AI assistant.")
    p.add_argument("--json", action="store_true", help=TR("JSON 格式输出", "JSON output"))
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    p.add_argument("--top-k", type=int, default=40, dest="top_k")
    p.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    p.add_argument("--image", "-i"), p.add_argument("--reasoning", choices=["on","off"], default="on")
    p.add_argument("--device", default="", help=TR("推理设备 (CPU/GPU/GPU.N/NPU)", "Device (CPU/GPU/GPU.N/NPU)") + TR("，留空自动选择", ", leave empty for auto)"))

    # benchmark
    p = sub.add_parser("benchmark", help=TR("基准测试", "Benchmark"))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--reasoning", choices=["on","off"], default="on")
    p.add_argument("--device", default="", help=TR("推理设备 (CPU/GPU/GPU.N/NPU)", "Device (CPU/GPU/GPU.N/NPU)") + TR("，留空自动选择", ", leave empty for auto)"))

    # server
    p = sub.add_parser("server", help=TR("API服务", "Server"))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--device", default="", help=TR("推理设备 (CPU/GPU/GPU.N/NPU)", "Device (CPU/GPU/GPU.N/NPU)") + TR("，留空自动选择", ", leave empty for auto)"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)

    # image
    p = sub.add_parser("image", help=TR("文生图", "Image"),
        description=TR(
            "使用 OpenVINO GenAI Text2ImagePipeline 生成图片。\n\n"
            "示例:\n"
            "  ov-cli image --model ./FLUX-ov --prompt 'a cat' --width 1024 --height 768",
            "Image generation via Text2ImagePipeline."))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--mode", choices=["interactive","once","pipe"], default="interactive")
    p.add_argument("--prompt"), p.add_argument("--output", "-o")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None, help=TR("随机种子", "Random seed"))
    p.add_argument("--json", action="store_true", help=TR("JSON 格式输出", "JSON output"))

    # tts
    p = sub.add_parser("tts", help=TR("语音合成", "TTS"),
        description=TR(
            "使用 OpenVINO Qwen3-TTS 生成语音。\n\n"
            "CustomVoice 示例 (预设声音):\n"
            "  ov-cli tts --model ./0.6B-CV-ov --prompt 你好 --speaker Vivian --output voice.wav\n\n"
            "Base 声音克隆示例 (需参考音频):\n"
            "  ov-cli tts --model ./0.6B-ov --prompt 你好 --ref-audio ref.mp3 --output voice.wav",
            "Text-to-speech via Qwen3-TTS."))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--prompt"), p.add_argument("--output", "-o")
    p.add_argument("--mode", choices=["once","pipe"], default="once",
        help=TR("once=单次输出 pipe=管道模式", "once=single pipe=pipeline"))
    p.add_argument("--speaker", help=TR("预设声音 (CustomVoice)", "Speaker (CustomVoice)"))
    p.add_argument("--lang", help=TR("语言 (auto/chinese/english...)", "Language"))
    p.add_argument("--instruct", help=TR("语气指令", "Voice instruction"))
    p.add_argument("--ref-audio", help=TR("参考音频路径 (Base 模型)", "Reference audio (Base model)"))
    p.add_argument("--device", default=None, help=TR("推理设备 (auto/CPU/GPU)", "Device"))
    p.add_argument("--no-warmup", action="store_true", help=TR("跳过预热", "Skip warmup"))
    p.add_argument("--json", action="store_true", help=TR("JSON 格式输出", "JSON output"))

    # asr
    p = sub.add_parser("asr", help=TR("语音转文字", "ASR"),
        description=TR(
            "语音转文字，自动识别 Whisper / Qwen3-ASR。\n\n"
            "  interactive  交互式终端 (默认)\n"
            "  once         单次转录 --file speech.mp3\n"
            "  pipe         管道模式: echo audio.wav | ov-cli asr --mode pipe\n\n"
            "Whisper 示例:\n"
            "  ov-cli asr --model ./whisper/ov-large --mode once --file speech.mp3\n\n"
            "Qwen3-ASR 示例:\n"
            "  ov-cli asr --model ./Qwen3-ASR-0.6B-ov --mode once --file speech.mp3",
            "Speech-to-text. Auto-detects Whisper / Qwen3-ASR."))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--mode", choices=["interactive","once","pipe"], default="interactive")
    p.add_argument("--file"), p.add_argument("--output", "-o")
    p.add_argument("--lang")
    p.add_argument("--json", action="store_true", help=TR("JSON 格式输出", "JSON output"))

    # ui
    p = sub.add_parser("ui", help=TR("网页界面", "Web UI"),
        description=TR(
            "启动 Gradio 网页界面。自动检测模型类型。\n\n"
            "示例:\n"
            "  ov-cli ui --model ./Qwen3-ov\n"
            "  ov-cli ui --model ./0.6B-CV-ov --port 7860\n"
            "  ov-cli ui --model ./FLUX-ov --share",
            "Launch Gradio web UI. Auto-detects model type."))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--device", default=None, help=TR("推理设备", "Device"))
    p.add_argument("--port", type=int, default=7860, help=TR("端口", "Port"))
    p.add_argument("--share", action="store_true", help=TR("生成公链", "Public link"))
    p.add_argument("--reasoning", choices=["on","off"], default="on", help=TR("思考模式", "Reasoning"))

    # mcp
    p = sub.add_parser("mcp", help=TR("MCP 协议服务器", "MCP Server"),
        description=TR(
            "启动 MCP (Model Context Protocol) 服务器。\n"
            "通过 stdin/stdout JSON-RPC 暴露 LLM 工具。\n\n"
            "示例:\n"
            "  ov-cli mcp --model ./Qwen3-ov\n"
            "  ov-cli mcp --model ./deepseek/7B-ov",
            "MCP (Model Context Protocol) server.\n"
            "Exposes LLM tools via stdin/stdout JSON-RPC."))
    p.add_argument("--model", "-m", required=True)

    args = parser.parse_args()
    if args.lang:
        ov_cli._LANG = args.lang
    if args.cmd not in ("setup",):
        _check_wsl2_gpu()

    dispatch = {
        "setup": lambda a: cmd_setup(a, _WORKSPACE), "chat": cmd_chat,
        "benchmark": cmd_benchmark, "server": cmd_server,
        "image": cmd_image, "tts": cmd_tts, "asr": cmd_asr, "ui": cmd_ui, "mcp": cmd_mcp,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
