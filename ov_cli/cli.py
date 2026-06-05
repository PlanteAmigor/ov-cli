"""
ov-cli: OpenVINO LLM 命令行工具
"""

import os, sys, argparse
import ov_cli
from ov_cli import TR
from ov_cli.setup import cmd_setup, _activate_path


def _get_version():
    """从 pyproject.toml 读取版本号。"""
    wp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pp = os.path.join(wp, "pyproject.toml")
    if os.path.isfile(pp):
        with open(pp) as f:
            for line in f:
                if line.startswith("version"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.0.0"


def _version_stamp_path(venv_path):
    return os.path.join(venv_path, ".ov-cli-version")


def _check_version_warning(venv_path):
    """检查版本变化，如有则打印提示。"""
    sp = _version_stamp_path(venv_path)
    if not os.path.isfile(sp):
        return
    with open(sp) as f:
        installed = f.read().strip()
    current = _get_version()
    if installed != current:
        print(f"  \u26a0 {TR('检测到版本变化 ({i} \u2192 {c})，建议运行:', 'Version changed ({i} \u2192 {c}), run:').format(i=installed, c=current)}")
        print(f"     ./ov-cli setup --fix")


def _check_wsl2_gpu():
    """WSL2 下检查 Intel GPU runtime。"""
    import subprocess
    try:
        subprocess.run(["grep", "-qi", "microsoft", "/proc/version"], capture_output=True, check=True)
    except Exception:
        return
    try:
        import openvino as ov
        if "GPU" not in ov.Core().available_devices:
            print(f"  {TR('⚠ WSL2 检测到 Intel GPU 但缺少 runtime，请安装:', '⚠ WSL2: Intel GPU detected but runtime missing, install:')}")
            print(f"    sudo apt install intel-level-zero-gpu libze1")
    except Exception:
        pass


def cmd_convert(args):
    """ov-cli convert"""
    from .convert import convert_model
    model_path = os.path.abspath(args.model)
    if not os.path.isdir(model_path):
        print(f"{TR('错误: 找不到模型目录', 'Error: model directory not found')}: {model_path}")
        sys.exit(1)
    output_path = args.output or model_path.rstrip("/") + "-ov"
    output_path = os.path.abspath(output_path)
    convert_model(model_path, output_path, args.format,
                  ratio=args.ratio, group_size=args.group_size)


def cmd_venv(args):
    """ov-cli venv: 进入 setup 创建的虚拟环境"""
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_path = args.venv or os.path.join(workspace, ".venv")
    activate = _activate_path(venv_path)
    if not os.path.isfile(activate):
        print(f"{TR('错误: 找不到虚拟环境', 'Error: venv not found')}: {activate}")
        print(f"  {TR('请先运行', 'Run first')}: ./ov-cli setup")
        sys.exit(1)
    print(f"  source {activate}")


def cmd_benchmark(args):
    """ov-cli benchmark"""
    from .benchmark import run_benchmark
    ov_path = os.path.abspath(args.model)
    run_benchmark(ov_path, args.reasoning == "on")


def cmd_server(args):
    """ov-cli server: 启动 API 服务"""
    from .server import run_server
    model_path = os.path.abspath(args.model)
    if not os.path.isdir(model_path):
        print(f"  ⚠ {TR('模型路径不存在', 'Model path not found')}: {model_path}")
        sys.exit(1)
    run_server(model_path, args.device, args.host, args.port)


def cmd_generate(args):
    """ov-cli generate: 文生图"""
    from .generate import load_model, run_once, run_generate
    ov_path = os.path.abspath(args.model)
    if not os.path.isdir(ov_path):
        print(f"{TR('错误: 找不到模型目录', 'Error: model directory not found')}: {ov_path}")
        sys.exit(1)
    ctx = load_model(ov_path)
    if args.mode == "once":
        if not args.prompt:
            print(f"  ⚠ {TR('once 模式需要 --prompt 参数', 'once mode requires --prompt')}")
            sys.exit(1)
        run_once(ctx, prompt=args.prompt, output=args.output,
                 width=args.width, height=args.height,
                 steps=args.steps, guidance=args.guidance)
    else:
        run_generate(ctx, width=args.width, height=args.height,
                     steps=args.steps, guidance=args.guidance)


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
    ctx = load_model(ov_path)
    if mode == "translate":
        run_translate(ctx, max_tokens=args.max_tokens)
    elif mode == "once":
        from .chat import run_once
        prompt = args.prompt.replace("\\n", "\n") if args.prompt else ""
        run_once(ctx, prompt=prompt, files=args.file or [],
                 output=args.output, temperature=args.temp, top_p=args.top_p,
                 top_k=args.top_k, max_tokens=args.max_tokens,
                 reasoning=args.reasoning == "on")
    else:
        run_chat(ctx, system=args.system, temperature=args.temp, top_p=args.top_p,
                 top_k=args.top_k, max_tokens=args.max_tokens, image_path=args.image,
                 reasoning=args.reasoning == "on")


def cmd_whisper(args):
    """ov-cli whisper: 语音转文字"""
    from .whisper import load_model, run_once, run_whisper
    ov_path = os.path.abspath(args.model)
    if not os.path.isdir(ov_path):
        print(f"{TR('错误: 找不到模型目录', 'Error: model directory not found')}: {ov_path}")
        sys.exit(1)
    ctx = load_model(ov_path)
    if args.mode == "once":
        if not args.file:
            print(f"  ⚠ {TR('once 模式需要 --file 参数', 'once mode requires --file')}")
            sys.exit(1)
        run_once(ctx, file_path=args.file, lang=args.lang, output=args.output)
    else:
        run_whisper(ctx, lang=args.lang)


# ── 帮助文本 ──

def _build_help():
    zh = ov_cli._LANG == "zh"
    if zh:
        desc = "ov-cli — 基于 OpenVINO 的 LLM 本地推理工具箱\n轻量、离线、CPU/GPU 皆可运行。"
        epilog = (
            "📖 使用示例:\n\n"
            "  ./ov-cli setup\n  ./ov-cli convert --model ./Qwen3.5 --format int8\n"
            "  ./ov-cli chat --model ./gemma-4-E2B-it-ov-int4\n"
            "  ./ov-cli whisper --model ./whisper/ov-large\n"
            "  ./ov-cli generate --model ./FLUX/ov-int4\n"
            "  ./ov-cli server --model ./model-ov --port 8080\n"
            "  ./ov-cli setup --fix\n"
        )
    else:
        desc = "ov-cli — OpenVINO-powered LLM local inference toolkit."
        epilog = (
            "📖 Examples:\n\n"
            "  ./ov-cli setup\n  ./ov-cli convert --model ./Qwen3.5 --format int8\n"
            "  ./ov-cli chat --model ./gemma-4-E2B-it-ov-int4\n"
            "  ./ov-cli whisper --model ./whisper/ov-large\n"
            "  ./ov-cli generate --model ./FLUX/ov-int4\n"
            "  ./ov-cli server --model ./model-ov --port 8080\n"
            "  ./ov-cli setup --fix\n"
        )
    return desc, epilog


# ── 入口 ──

def main():
    W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--lang" and i + 1 < len(sys.argv):
            ov_cli._LANG = sys.argv[i + 1]
            break
        if a.startswith("--lang="):
            ov_cli._LANG = a.split("=", 1)[1]
            break

    desc, epilog = _build_help()
    parser = argparse.ArgumentParser(prog="ov-cli", description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=epilog)
    parser.add_argument("--lang", choices=["zh", "en"])
    sub = parser.add_subparsers(dest="cmd", required=True, title=TR("子命令", "commands"))

    # setup
    p = sub.add_parser("setup", help=TR("创建环境", "Setup"))
    p.add_argument("--venv", help=TR("venv 路径", "venv path"))
    p.add_argument("--optimum-dir", help=TR("optimum-intel 源码目录", "optimum-intel source"))
    p.add_argument("--fix", action="store_true", help=TR("修复模式", "Fix mode"))

    # convert
    p = sub.add_parser("convert", help=TR("转换模型", "Convert"))
    p.add_argument("--model", "-m", required=True, help=TR("模型目录", "model dir"))
    p.add_argument("--output", "-o", help=TR("输出目录", "output dir"))
    p.add_argument("--format", choices=["fp32","fp16","int8","int4","mxfp4","nf4","cb4"], default="fp32")
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--group-size", type=int, default=128, dest="group_size")

    # chat
    p = sub.add_parser("chat", help=TR("聊天/翻译", "Chat"))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--mode", choices=["chat","translate","once"], default="chat")
    p.add_argument("--prompt"), p.add_argument("--file", action="append", default=None)
    p.add_argument("--output"), p.add_argument("--system", default="You are a helpful AI assistant.")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    p.add_argument("--top-k", type=int, default=40, dest="top_k")
    p.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    p.add_argument("--image", "-i"), p.add_argument("--reasoning", choices=["on","off"], default="on")

    # benchmark
    p = sub.add_parser("benchmark", help=TR("基准测试", "Benchmark"))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--reasoning", choices=["on","off"], default="on")

    # server
    p = sub.add_parser("server", help=TR("API服务", "Server"))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--device", choices=["CPU","GPU"], default="")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)

    # generate
    p = sub.add_parser("generate", help=TR("文生图", "Generate"))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--mode", choices=["interactive","once"], default="interactive")
    p.add_argument("--prompt"), p.add_argument("--output", "-o")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guidance", type=float, default=0.0)

    # whisper
    p = sub.add_parser("whisper", help=TR("语音转文字", "Whisper"))
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--mode", choices=["interactive","once"], default="interactive")
    p.add_argument("--file"), p.add_argument("--output", "-o")
    p.add_argument("--lang")

    # venv
    p = sub.add_parser("venv", help=TR("进入环境", "Venv"))
    p.add_argument("--venv")

    args = parser.parse_args()
    if args.lang:
        ov_cli._LANG = args.lang
    if args.cmd != "setup":
        _venv = getattr(args, "venv", None) or os.path.join(W, ".venv")
        _check_version_warning(_venv)
    if args.cmd not in ("setup", "venv"):
        _check_wsl2_gpu()

    dispatch = {
        "setup": lambda a: cmd_setup(a, W), "convert": cmd_convert, "chat": cmd_chat,
        "benchmark": cmd_benchmark, "venv": cmd_venv, "server": cmd_server,
        "generate": cmd_generate, "whisper": cmd_whisper,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
