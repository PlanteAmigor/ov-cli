"""
ov-cli: OpenVINO LLM 命令行工具

用法:
  ov-cli chat --model path/to/ov-dir          通用聊天
  ov-cli chat --model path --mode translate   翻译模式
  ov-cli convert --model path --format int8   模型转换
  ov-cli setup [--venv path]                  创建环境
"""

import os, sys, argparse, json
import ov_cli
from ov_cli import TR


def _ensure_vscode_settings(venv_path):
    """创建 VS Code 工作区设置，使终端自动激活虚拟环境"""
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vscode_dir = os.path.join(workspace, ".vscode")
    settings_path = os.path.join(vscode_dir, "settings.json")
    os.makedirs(vscode_dir, exist_ok=True)

    settings = {}
    if os.path.isfile(settings_path):
        with open(settings_path) as f:
            try:
                settings = json.load(f)
            except json.JSONDecodeError:
                settings = {}

    # 计算相对于工作区的 venv 路径
    rel_venv = os.path.relpath(venv_path, workspace)
    py_path = os.path.join("${workspaceFolder}", rel_venv, "bin", "python")

    changed = False
    if settings.get("python.defaultInterpreterPath") != py_path:
        settings["python.defaultInterpreterPath"] = py_path
        changed = True
    if not settings.get("python.terminal.activateEnvironment"):
        settings["python.terminal.activateEnvironment"] = True
        changed = True
    if not settings.get("python.terminal.activateEnvInCurrentTerminal"):
        settings["python.terminal.activateEnvInCurrentTerminal"] = True
        changed = True
    if "files.exclude" not in settings:
        settings["files.exclude"] = {"**/__pycache__": True, "**/*.egg-info": True, "**/.venv": True}
        changed = True
    if "search.exclude" not in settings:
        settings["search.exclude"] = {"**/.venv": True}
        changed = True

    if changed:
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
            f.write("\n")
        print(f"  ✓ VS Code 设置已更新: .vscode/settings.json")


def _apply_gemma4_patch():
    """自动修复 optimum-intel 中 Gemma-4 共享 KV 层的属性引用错误。

    model_patcher.py 的 gemma4_text_attention_forward 引用了不存在的
    self.kv_shared_layer_index，应使用 self.layer_type（与 transformers 5.9.0
    的 Gemma4TextAttention 实际 API 一致）。
    """
    import re
    patcher_path = None
    # 查找已安装的 model_patcher.py
    try:
        import optimum.exporters.openvino.model_patcher as mp
        patcher_path = mp.__file__
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass

    if not patcher_path or not os.path.isfile(patcher_path):
        return  # optimum-intel 未安装，跳过

    with open(patcher_path) as f:
        content = f.read()

    old = "self.kv_shared_layer_index"
    new = "self.layer_type"
    if old in content:
        content = content.replace(old, new)
        with open(patcher_path, "w") as f:
            f.write(content)
        print(f"  ✓ {TR('Gemma-4 补丁已应用', 'Gemma-4 patch applied')}: {os.path.basename(patcher_path)}")
    else:
        # 检查是否已经是修复后的版本
        if "past_key_values.shared_layers[self.layer_type]" in content:
            print(f"  ✓ {TR('Gemma-4 补丁已存在', 'Gemma-4 patch already applied')}")
        else:
            print(f"  - {TR('Gemma-4 补丁不需要或已不适用', 'Gemma-4 patch not needed or N/A')}")


def cmd_setup(args):
    """ov-cli setup: 创建虚拟环境并安装依赖"""
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_path = args.venv or os.path.join(workspace, ".venv")
    print(f"  {TR('创建虚拟环境', 'Creating venv')}: {venv_path}")
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "venv", venv_path, "--clear"])
    pip = os.path.join(venv_path, "bin", "pip")
    print(f"  {TR('安装依赖...', 'Installing dependencies...')}")
    pkgs = [
        "openvino>=2026.2",
        "openvino-tokenizers",
        "openvino-genai",
        "nncf>=3.0",
        "torch",
        "torchvision",
        "tokenizers",
        "jinja2",
        "pillow",
        "numpy",
        "huggingface-hub",
        "safetensors",
        "sentencepiece",
    ]
    cmd = [pip, "install", "-v"] + pkgs
    subprocess.check_call(cmd)

    # 安装最新版 optimum-intel（必须从 GitHub 源码装，PyPI 版对新架构支持不完善）
    _optimum_src = args.optimum_dir
    if _optimum_src:
        _optimum_src = os.path.abspath(_optimum_src)
    if not _optimum_src or not os.path.isdir(_optimum_src):
        _optimum_src = os.path.join(workspace, "optimum-intel-main")
    if os.path.isdir(_optimum_src):
        print(f"  {TR('安装 optimum-intel (本地源码)...', 'Installing optimum-intel (local)...')}: {_optimum_src}")
        subprocess.check_call([pip, "install", _optimum_src])
    else:
        print(f"  {TR('安装 optimum-intel (GitHub)...', 'Installing optimum-intel (GitHub)...')}")
        subprocess.check_call([pip, "install", "optimum-intel@git+https://github.com/huggingface/optimum-intel.git"])

    # 强制安装最新版 transformers（--no-deps 避免 optimum-intel 的 <5.1 约束降级）
    print(f"  {TR('安装 transformers (no-deps)...', 'Installing transformers (no-deps)...')}")
    subprocess.check_call([pip, "install", "--no-deps", "--force-reinstall", "transformers>=5.9"])

    # 应用 Gemma-4 补丁（修改 model_patcher.py 中不存在的属性引用）
    _apply_gemma4_patch()

    # 自动配置 VS Code 工作区设置
    _ensure_vscode_settings(venv_path)

    print()
    print(f"  {TR('✅ 完成!', '✅ Done!')}")
    print(f"  {TR('💡 激活虚拟环境:', '💡 Activate venv:')}")
    print(f"     source {venv_path}/bin/activate")
    print(f"  {TR('💡 或在 VS Code 中重新打开终端即可自动激活', '💡 Or just reopen terminal in VS Code for auto-activation')}")


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
    activate = os.path.join(venv_path, "bin", "activate")
    if not os.path.isfile(activate):
        print(f"{TR('错误: 找不到虚拟环境', 'Error: venv not found')}: {activate}")
        print(f"  {TR('请先运行', 'Run first')}: ./ov-cli setup")
        sys.exit(1)
    print(f"  source {activate}")


def cmd_benchmark(args):
    """ov-cli benchmark"""
    from .benchmark import run_benchmark
    ov_path = os.path.abspath(args.model)
    run_benchmark(ov_path)


def cmd_chat(args):
    """ov-cli chat"""
    from .chat import load_model, run_chat, run_translate

    ov_path = os.path.abspath(args.model)
    if not os.path.isdir(ov_path):
        print(f"{TR('错误: 找不到模型目录', 'Error: model directory not found')}: {ov_path}")
        sys.exit(1)
    # 支持两种格式: 传统 openvino_model.xml 或 GenAI openvino_config.json
    _has_legacy = os.path.isfile(os.path.join(ov_path, "openvino_model.xml"))
    _has_genai = os.path.isfile(os.path.join(ov_path, "openvino_config.json"))
    if not _has_legacy and not _has_genai:
        print(f"{TR('错误: 找不到 OpenVINO 模型文件', 'Error: OpenVINO model not found')}: {ov_path}")
        print(f"  {TR('需要 openvino_model.xml 或 openvino_config.json', 'Need openvino_model.xml or openvino_config.json')}")
        sys.exit(1)

    ctx = load_model(ov_path)

    mode = args.mode
    if mode == "auto":
        # 根据 model_type 自动判断
        mt = ctx.get("model_type", "")
        if mt == "hunyuan_dense":
            mode = "translate"
        else:
            mode = "chat"

    if mode == "translate":
        run_translate(ctx, max_tokens=args.max_tokens)
    else:
        run_chat(ctx, system=args.system,
                 temperature=args.temp, top_p=args.top_p,
                 top_k=args.top_k, max_tokens=args.max_tokens,
                 image_path=args.image)


def _build_help():
    """构建语言感知的帮助文本"""
    zh = ov_cli._LANG == "zh"
    if zh:
        desc = (
            "ov-cli — 基于 OpenVINO 的 LLM 本地推理工具箱\n"
            "\n"
            "轻量、离线、CPU/GPU 皆可运行。支持模型转换、量化、聊天、翻译。\n"
            "基于 Optimum Intel + OpenVINO GenAI，对标 llama.cpp 设计。"
            "\n"
            "工作流:\n"
            "  1. ./ov-cli setup             创建环境\n"
            "  2. ./ov-cli convert            转换模型 → OpenVINO IR\n"
          "  3. ./ov-cli chat / translate   推理\n"
          "  4. ./ov-cli benchmark          基准测试"
        )
        epilog = (
            "📖 使用示例:\n"
            "\n"
            "  # 首次使用: 创建环境\n"
            "  ./ov-cli setup\n"
            "  eval \"$(./ov-cli venv)\"      # 进入虚拟环境\n"
            "\n"
            "  # 转换 HuggingFace 模型为 OpenVINO IR\n"
            "  ./ov-cli convert --model ./Qwen3.5 --format int8\n"
            "  ./ov-cli convert --model ./Qwen3.5 --format int4 -o ./Qwen3.5-ov-int4\n"
            "  ./ov-cli convert --model ./Qwen3.5 --format fp16    # 半精度\n"
            "\n"
            "  # 聊天模式 (通用对话)\n"
            "  ./ov-cli chat --model ./gemma-4-E2B-it-ov-int4\n"
            "  ./ov-cli chat --model ./model-ov --temp 0.9 --max-tokens 2048\n"
            "\n"
            "  # 翻译模式 (Hy-MT2 等翻译模型)\n"
            "  ./ov-cli chat --model ./Hy-MT2-1.8B-ov --mode translate\n"
            "\n"
            "  # 指定语言\n"
            "  ./ov-cli --lang en chat --model ./model-ov\n"
            "\n"
            "💡 更多信息: 每个子命令后加 -h 查看更多选项, 如 ./ov-cli chat -h\n"
        )
    else:
        desc = (
            "ov-cli — OpenVINO-powered LLM local inference toolkit\n"
            "\n"
            "Lightweight, offline, runs on CPU & GPU. Convert, quantize, chat & translate.\n"
            "Powered by Optimum Intel + OpenVINO GenAI, inspired by llama.cpp."
            "\n"
            "Workflow:\n"
            "  1. ./ov-cli setup             create environment\n"
            "  2. ./ov-cli convert            convert model → OpenVINO IR\n"
            "  3. ./ov-cli chat / translate   inference\n"
            "  4. ./ov-cli benchmark          benchmark"
        )
        epilog = (
            "📖 Examples:\n"
            "\n"
            "  # First time: setup environment\n"
            "  ./ov-cli setup\n"
            "  eval \"$(./ov-cli venv)\"       # activate venv\n"
            "\n"
            "  # Convert HuggingFace model to OpenVINO IR\n"
            "  ./ov-cli convert --model ./Qwen3.5 --format int8\n"
            "  ./ov-cli convert --model ./Qwen3.5 --format int4 -o ./Qwen3.5-ov-int4\n"
            "  ./ov-cli convert --model ./Qwen3.5 --format fp16     # half precision\n"
            "\n"
            "  # Chat mode (general conversation)\n"
            "  ./ov-cli chat --model ./gemma-4-E2B-it-ov-int4\n"
            "  ./ov-cli chat --model ./model-ov --temp 0.9 --max-tokens 2048\n"
            "\n"
            "  # Translate mode (Hy-MT2 and similar)\n"
            "  ./ov-cli chat --model ./Hy-MT2-1.8B-ov --mode translate\n"
            "\n"
            "  # Set UI language\n"
            "  ./ov-cli --lang en chat --model ./model-ov\n"
            "\n"
            "💡 Tip: use ./ov-cli <command> -h for detailed options\n"
        )
    return desc, epilog


def main():
    # 预先解析 --lang，使帮助文本能正确切换语言
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--lang" and i + 1 < len(sys.argv):
            ov_cli._LANG = sys.argv[i + 1]
            break
        if a.startswith("--lang="):
            ov_cli._LANG = a.split("=", 1)[1]
            break

    desc, epilog = _build_help()

    parser = argparse.ArgumentParser(
        prog="ov-cli",
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("--lang", choices=["zh", "en"],
                        help=TR("界面语言 (zh/en)", "UI language (zh/en)"))
    sub = parser.add_subparsers(dest="cmd", required=True,
                                title=TR("子命令", "commands"))

    # ── setup ──
    p_setup = sub.add_parser(
        "setup",
        help=TR("创建虚拟环境并安装依赖", "Create venv & install dependencies"),
        description=TR(
            "一键创建 Python 虚拟环境并安装所有运行时依赖：\n"
            "  • openvino / openvino-tokenizers   推理引擎\n"
            "  • openvino-genai                  GenAI 管道\n"
            "  • nncf                           模型量化\n"
            "  • transformers 5.2 / tokenizers   分词与模板\n"
            "  • optimum-intel                  官方导出工具\n"
            "  • torch                          模型加载 (转换用)\n"
            "  • fastapi / uvicorn              HTTP 服务（预留）",
            "One-command venv setup with all runtime dependencies:\n"
            "  • openvino / openvino-tokenizers  inference engine\n"
            "  • openvino-genai                  GenAI pipeline\n"
            "  • nncf                            model quantization\n"
            "  • transformers 5.2 / tokenizers   tokenization & templates\n"
            "  • optimum-intel                   official export tool\n"
            "  • torch                           model loading (convert)\n"
            "  • fastapi / uvicorn               HTTP server (reserved)",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_setup.add_argument("--venv",
                         help=TR("虚拟环境路径 (默认 ./.venv)", "venv path (default: ./.venv)"))
    p_setup.add_argument("--optimum-dir",
                         help=TR("optimum-intel 源码目录 (默认自动检测)",
                                 "optimum-intel source dir (default: auto-detect)"))

    # ── convert ──
    p_conv = sub.add_parser(
        "convert",
        help=TR("转换 HuggingFace 模型 → OpenVINO IR", "Convert HF model → OpenVINO IR"),
        description=TR(
            "转换 HuggingFace 模型 → OpenVINO IR (FP16/INT8/INT4 量化可选)\n"
            "\n"
            "使用 Optimum Intel 官方工具，自动推断 task 类型。\n"
            "\n"
            "流程:\n"
            "  1. optimum-cli export openvino (自动下载 + 转换 + 量化)\n"
            "  2. 保存 .xml/.bin + tokenizer + 配置文件\n"
            "\n"
            "量化格式:\n"
            "  fp32  浮点 (无损, 体积最大)\n"
            "  fp16  半精度 (体积减半, 几乎无损)\n"
            "  int8  8-bit (体积~25%, 几乎无损)\n"
            "  int4  4-bit (体积~12.5%, 有精度损失)\n"
            "\n"
            "高级参数:\n"
            "  --ratio RATIO     INT4 混合精度比例 (0-1, 默认1.0)\n"
            "  --group-size GS   量化分组大小 (默认128, 越大精度越高)",
            "Convert HuggingFace model → OpenVINO IR (FP16/INT8/INT4 optional)\n"
            "\n"
            "Uses Optimum Intel official tool, auto-infers task type.\n"
            "\n"
            "Pipeline:\n"
            "  1. optimum-cli export openvino (download + convert + quantize)\n"
            "  2. Save .xml/.bin + tokenizer + configs\n"
            "\n"
            "Quantization formats:\n"
            "  fp32  full precision (lossless, largest)\n"
            "  fp16  half precision (size ~half, near-lossless)\n"
            "  int8  8-bit (size ~25%, near-lossless)\n"
            "  int4  4-bit (size ~12.5%, some quality loss)\n"
            "\n"
            "Advanced:\n"
            "  --ratio RATIO     INT4 mixed precision ratio (0-1, default 1.0)\n"
            "  --group-size GS   quantization group size (default 128)",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_conv.add_argument("--model", "-m", required=True,
                        help=TR("HuggingFace 模型目录路径", "path to HuggingFace model dir"))
    p_conv.add_argument("--output", "-o",
                        help=TR("输出目录 (默认: {model}-ov)", "output dir (default: {model}-ov)"))
    p_conv.add_argument("--format", choices=["fp32", "fp16", "int8", "int4"], default="fp32",
                        help=TR("量化格式 (默认: fp32)", "quantization format (default: fp32)"))
    p_conv.add_argument("--ratio", type=float, default=1.0,
                        help=TR("INT4 混合精度比例 0-1 (默认 1.0)", "INT4 mixed precision ratio 0-1 (default 1.0)"))
    p_conv.add_argument("--group-size", type=int, default=128, dest="group_size",
                        help=TR("量化分组大小 (默认 128)", "quantization group size (default 128)"))

    # ── chat ──
    p_chat = sub.add_parser(
        "chat",
        help=TR("交互式聊天 / 翻译终端", "Interactive chat / translate terminal"),
        description=TR(
            "加载 OpenVINO 模型并启动交互终端。支持聊天和翻译两种模式。\n"
            "\n"
            "聊天模式 (VLM 支持图片):\n"
            "  • --image PATH  启动时加载图片\n"
            "  • //img PATH    对话中加载/切换图片\n"
            "  • 流式输出 + tok/s 性能统计\n"
            "\n"
            "翻译模式 (仅 Hy-MT2 等翻译模型):\n"
            "  • 自动检测语言方向 (中↔英)\n"
            "  • //en 文本 → 强制译英, //zh 文本 → 强制译中\n"
            "\n"
            "采样参数:\n"
            "  temperature  随机性 (0=贪婪, >0 越高越随机)\n"
            "  top-p        nucleus sampling 累积概率阈值\n"
            "  top-k        仅从概率最高的 k 个 token 中采样",
            "Load an OpenVINO model and start an interactive terminal.\n"
            "\n"
            "Chat mode (VLM supports images):\n"
            "  • --image PATH  load image at startup\n"
            "  • //img PATH    load/switch image during chat\n"
            "  • Streaming output + tok/s stats\n"
            "\n"
            "Translate mode (Hy-MT2 and similar):\n"
            "  • Auto-detect language direction (Chinese↔English)\n"
            "  • //en text → force English, //zh text → force Chinese\n"
            "\n"
            "Sampling parameters:\n"
            "  temperature  randomness (0=greedy, higher=more random)\n"
            "  top-p        nucleus sampling cumulative prob threshold\n"
            "  top-k        only sample from top-k highest prob tokens",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_chat.add_argument("--model", "-m", required=True,
                        help=TR("OpenVINO 模型目录 (须包含 openvino_model.xml)",
                                "OpenVINO model dir (must contain openvino_model.xml)"))
    p_chat.add_argument("--mode", choices=["chat", "translate", "auto"], default="auto",
                        help=TR("运行模式 (默认: auto 自动检测)", "mode (default: auto-detect)"))
    p_chat.add_argument("--system", default="You are a helpful AI assistant.",
                        help=TR("系统提示词 (仅 chat 模式)", "system prompt (chat mode only)"))
    p_chat.add_argument("--temp", type=float, default=0.7,
                        help=TR("采样温度, 0-2 (默认: 0.7)", "temperature, 0-2 (default: 0.7)"))
    p_chat.add_argument("--top-p", type=float, default=0.9, dest="top_p",
                        help=TR("nucleus 采样阈值, 0-1 (默认: 0.9)", "nucleus sampling threshold (default: 0.9)"))
    p_chat.add_argument("--top-k", type=int, default=40, dest="top_k",
                        help=TR("top-k 采样数 (默认: 40, 0=禁用)", "top-k sampling count (default: 40, 0=off)"))
    p_chat.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens",
                        help=TR("每次回复最大 token 数 (默认: 1024)",
                                "max tokens per reply (default: 1024)"))
    p_chat.add_argument("--image", "-i",
                        help=TR("初始图片路径 (VLM 模型)", "initial image path (VLM model)"))

    # ── benchmark ──
    p_bench = sub.add_parser(
        "benchmark",
        help=TR("运行性能基准测试", "Run performance benchmark"),
        description=TR(
            "测试模型的推理性能：首 token 延迟、生成速度、内存占用。\n"
            "自动在 32 和 1024 token 两种输入大小下测试。",
            "Benchmark model inference performance: first token latency,\n"
            "generation speed, and memory usage. Tests with input sizes\n"
            "of 32 and 1024 tokens.",
        ),
    )
    p_bench.add_argument("--model", "-m", required=True,
                         help=TR("OpenVINO 模型目录", "OpenVINO model dir"))

    # ── venv ──
    p_venv = sub.add_parser(
        "venv",
        help=TR("进入虚拟环境", "Enter the virtual environment"),
        description=TR(
            "打印 source 命令以进入 setup 创建的虚拟环境。\n"
            "用法: eval \"$(./ov-cli venv)\"",
            "Print the source command to activate the venv created by setup.\n"
            "Usage: eval \"$(./ov-cli venv)\"",
        ),
    )
    p_venv.add_argument("--venv",
                        help=TR("虚拟环境路径 (默认 ./.venv)", "venv path (default: ./.venv)"))

    args = parser.parse_args()

    # 语言覆盖
    if args.lang:
        ov_cli._LANG = args.lang

    # 分发
    if args.cmd == "setup":
        cmd_setup(args)
    elif args.cmd == "convert":
        cmd_convert(args)
    elif args.cmd == "chat":
        cmd_chat(args)
    elif args.cmd == "benchmark":
        cmd_benchmark(args)
    elif args.cmd == "venv":
        cmd_venv(args)


if __name__ == "__main__":
    main()
