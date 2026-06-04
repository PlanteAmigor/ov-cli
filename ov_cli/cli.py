"""
ov-cli: OpenVINO LLM 命令行工具

用法:
  ov-cli chat --model path/to/ov-dir          通用聊天
  ov-cli chat --model path --mode translate   翻译模式
  ov-cli convert --model path --format int8   模型转换
  ov-cli setup [--venv path]                  创建环境
"""

import os, sys, argparse, json, shutil, tempfile
import ov_cli
from ov_cli import TR


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


def _write_version_stamp(venv_path):
    ver = _get_version()
    sp = _version_stamp_path(venv_path)
    with open(sp, "w") as f:
        f.write(ver + "\n")


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
    """WSL2 下检查 Intel GPU runtime 是否可用，不可用时给提示。"""
    import subprocess
    try:
        subprocess.run(["grep", "-qi", "microsoft", "/proc/version"],
                       capture_output=True, check=True)
    except Exception:
        return  # 不是 WSL2
    try:
        import openvino as ov
        if "GPU" not in ov.Core().available_devices:
            print(f"  {TR('⚠ WSL2 检测到 Intel GPU 但缺少 runtime，请安装:', '⚠ WSL2: Intel GPU detected but runtime missing, install:')}")
            print(f"    sudo apt install intel-level-zero-gpu libze1")
    except Exception:
        pass


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
    self.kv_shared_layer_index。需要在运行时根据 layer_type 动态计算
    对应的非共享层索引（transformers 5.x 的 Gemma4TextAttention 实际 API）。
    """
    patcher_path = None
    try:
        import optimum.exporters.openvino.model_patcher as mp  # type: ignore[reportMissingImports]
        patcher_path = mp.__file__
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass

    if not patcher_path or not os.path.isfile(patcher_path):
        return

    with open(patcher_path) as f:
        content = f.read()

    old_line = "    if self.is_kv_shared_layer and past_key_values is not None:"
    new_lines = """    if self.is_kv_shared_layer and past_key_values is not None:
        # kv_shared_layer_index 需要在运行时计算
        if not hasattr(self, "kv_shared_layer_index"):
            first_shared = self.config.num_hidden_layers - getattr(self.config, "num_kv_shared_layers", 0)
            prev_types = self.config.layer_types[:first_shared]
            self.kv_shared_layer_index = len(prev_types) - 1 - prev_types[::-1].index(self.layer_type)
        key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]"""

    old = old_line + "\n        key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]"
    new = new_lines
    if old in content:
        content = content.replace(old, new)
        with open(patcher_path, "w") as f:
            f.write(content)
        print(f"  ✓ {TR('Gemma-4 补丁已应用', 'Gemma-4 patch applied')}: {os.path.basename(patcher_path)}")
    else:
        # 检查是否已经是修复后的版本
        if "hasattr(self, \"kv_shared_layer_index\")" in content:
            print(f"  ✓ {TR('Gemma-4 补丁已存在', 'Gemma-4 patch already applied')}")
        else:
            print(f"  - {TR('Gemma-4 补丁不需要或已不适用', 'Gemma-4 patch not needed or N/A')}")


def _apply_qwen35_patch():
    """自动修复 optimum-intel 中 Qwen3.5 的 DynamicCache 导入错误。

    transformers 5.9 将 Qwen3_5DynamicCache 合并到了通用的 DynamicCache
    （transformers.cache_utils），optimum-intel 仍从旧路径导入。
    """
    patcher_path = None
    try:
        import optimum.exporters.openvino.model_patcher as mp
        patcher_path = mp.__file__
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass

    if not patcher_path or not os.path.isfile(patcher_path):
        return

    with open(patcher_path) as f:
        content = f.read()

    old = "from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache"
    new = "from transformers.cache_utils import DynamicCache as Qwen3_5DynamicCache"
    patched = False
    if old in content:
        content = content.replace(old, new)
        patched = True

    # 还要补上 layer_types 属性（DynamicCache 没有这个成员）
    old2 = "super().__init__(config=config)\n\n                self.conv_states"
    new2 = "super().__init__(config=config)\n                self.layer_types = config.layer_types\n\n                self.conv_states"
    if old2 in content:
        content = content.replace(old2, new2)
        patched = True

    if patched:
        with open(patcher_path, "w") as f:
            f.write(content)
        print(f"  ✓ {TR('Qwen3.5 补丁已应用', 'Qwen3.5 patch applied')}: {os.path.basename(patcher_path)}")
    else:
        if new in content:
            print(f"  ✓ {TR('Qwen3.5 补丁已存在', 'Qwen3.5 patch already applied')}")
        else:
            print(f"  - {TR('Qwen3.5 补丁不需要或已不适用', 'Qwen3.5 patch not needed or N/A')}")


def _is_windows():
    return sys.platform == "win32"


def _activate_path(venv_path):
    """返回虚拟环境的 activate 脚本路径。"""
    if _is_windows():
        return os.path.join(venv_path, "Scripts", "activate")
    return os.path.join(venv_path, "bin", "activate")


def _pip_path(venv_path):
    """返回虚拟环境的 pip 路径。"""
    if _is_windows():
        return os.path.join(venv_path, "Scripts", "pip.exe")
    return os.path.join(venv_path, "bin", "pip")


def _build_genai_from_source(venv_path, genai_src):
    """从源码编译 openvino-genai 并安装到虚拟环境。

    这是临时方案，直到 upstream 合入 reasoning_budget 支持。
    """
    import subprocess, shutil
    import sysconfig

    build_dir = os.path.join(genai_src, "build")
    if os.path.isdir(build_dir):
        shutil.rmtree(build_dir)

    # 找到 venv 中的 OpenVINO cmake 配置
    site_packages = os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    openvino_dir = os.path.join(site_packages, "openvino", "cmake")
    if not os.path.isdir(openvino_dir):
        print(f"  ⚠ {TR('找不到 OpenVINO cmake 配置', 'OpenVINO cmake not found')}")
        print(f"    {TR('请确保已安装 openvino', 'Make sure openvino is installed')}")
        return False

    env = os.environ.copy()
    env["OpenVINO_DIR"] = openvino_dir

    print(f"  {TR('配置 GenAI 源码...', 'Configuring GenAI source...')}")
    try:
        subprocess.check_call([
            "cmake", "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TOKENIZERS=OFF",
            "-DENABLE_TOOLS=OFF",
            "-DENABLE_TESTS=OFF",
            "-DENABLE_SAMPLES=OFF",
            "-DENABLE_GGUF=OFF",
            "-DENABLE_XGRAMMAR=OFF",
            "-S", genai_src, "-B", build_dir,
        ], env=env)
    except Exception as e:
        print(f"  ⚠ {TR('cmake 配置失败', 'cmake configuration failed')}: {e}")
        return False

    print(f"  {TR('编译 GenAI...', 'Building GenAI...')}")
    try:
        subprocess.check_call(["cmake", "--build", build_dir, "--config", "Release", "-j"], env=env)
    except Exception as e:
        print(f"  ⚠ {TR('编译失败', 'Build failed')}: {e}")
        return False

    # 安装到 venv 的 site-packages
    genai_out = os.path.join(build_dir, "openvino_genai")
    target = os.path.join(site_packages, "openvino_genai")
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    libsrc = os.path.join(genai_out, "libopenvino_genai.so")
    pysrc = os.path.join(genai_out, f"py_openvino_genai{ext}")
    if os.path.isfile(pysrc) and os.path.isfile(libsrc):
        # Fix RUNPATH to $ORIGIN so .so files find each other regardless of install path
        import subprocess as _sp
        try:
            _sp.check_call(["patchelf", "--set-rpath", f"$ORIGIN:{target}", pysrc])
        except Exception:
            pass  # patchelf not available, fallback to symlink
        shutil.copy2(libsrc, os.path.join(target, "libopenvino_genai.so"))
        shutil.copy2(libsrc, os.path.join(target, "libopenvino_genai.so.2620"))
        shutil.copy2(pysrc, target)
        print(f"  ✓ {TR('已安装', 'Installed')}: libopenvino_genai.so + py_openvino_genai{ext}")

    print(f"  ✓ {TR('GenAI 编译安装完成', 'GenAI build & install complete')}")
    return True


def _prompt_mode(has_genai_src):
    """交互选择安装模式：1=简易 2=完整（编译 GenAI）"""
    if not has_genai_src:
        return 1
    while True:
        try:
            r = input(f"  {TR('选择安装模式', 'Select mode')}:\n"
                      f"    1. {TR('简易模式 - 仅 pip 安装', 'Simple - pip only')}\n"
                      f"    2. {TR('完整模式 - 编译 GenAI 源码启用 thinking budget', 'Full - build GenAI from source')}\n"
                      f"  {TR('请输入 [1/2]', 'Enter [1/2]')} (1): ")
            if r.strip() == "":
                return 1
            m = int(r.strip())
            if m in (1, 2):
                return m
        except (ValueError, EOFError):
            return 1


def cmd_setup(args):
    """ov-cli setup: 创建虚拟环境并安装依赖"""
    import shutil, subprocess, sys as _sys
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── 修复模式 ──
    if args.fix:
        venv_path = args.venv or os.path.join(workspace, ".venv")
        if not os.path.isdir(venv_path):
            print(f"  {TR('错误: 未找到虚拟环境', 'Error: venv not found')}: {venv_path}")
            print(f"  {TR('请先运行', 'Run first')}: ./ov-cli setup")
            sys.exit(1)
        pip = _pip_path(venv_path)
        print(f"  {TR('修复模式: 升级依赖 + 重打补丁', 'Fix mode: upgrade deps + repatch')}")
        try:
            subprocess.check_call([pip, "install", "--upgrade", workspace])
            _optimum_src = args.optimum_dir or os.path.join(workspace, "optimum-intel-main")
            if os.path.isdir(_optimum_src):
                subprocess.check_call([pip, "install", "--upgrade", _optimum_src])
            else:
                subprocess.check_call([pip, "install", "--upgrade",
                                       "optimum-intel@git+https://github.com/huggingface/optimum-intel.git"])
            subprocess.check_call([pip, "install", "--upgrade", "--no-deps", "transformers>=5.9"])
            _apply_gemma4_patch()
            _apply_qwen35_patch()
            _write_version_stamp(venv_path)
            print(f"  {TR('✅ 修复完成', '✅ Fix done')}")
        except KeyboardInterrupt:
            print()
            print(f"  {TR('修复已取消', 'Fix cancelled')}")
            sys.exit(1)
        return

    # 检查目录写入权限
    if not os.access(workspace, os.W_OK):
        _user = os.environ.get("USER", "")
        print(f"  {TR('错误: 当前目录没有写入权限', 'Error: no write permission')}")
        print(f"  {TR('请执行以下命令后重试:', 'Run the following command and retry:')}")
        print(f"    sudo chown -R {_user}:{_user} {workspace}")
        sys.exit(1)

    genai_src = os.path.join(workspace, "openvino.genai-2026.2.0.0-optimization")
    try:
        mode = _prompt_mode(os.path.isdir(genai_src))
    except KeyboardInterrupt:
        print()
        print(f"  {TR('安装已取消', 'Setup cancelled')}")
        sys.exit(1)

    if mode == 2:
        print()
        print("=" * 54)
        print(f"  {TR('完整模式将执行以下操作', 'Full mode will:')}")
        print(f"  {TR('1. 检查编译环境 (cmake, gcc)', '1. Check build env (cmake, gcc)')}")
        print(f"  {TR('2. 创建虚拟环境并 pip 安装依赖', '2. Create venv & pip install deps')}")
        print(f"  {TR('3. 编译 GenAI 源码 → 启用 thinking budget', '3. Build GenAI from source')}")
        print(f"  {TR('4. 安装编译产物到虚拟环境', '4. Install to venv')}")
        print()
        print(f"  {TR('前置条件', 'Prerequisites')}:")
        print(f"  • cmake, gcc/g++, make, patchelf")
        print(f"  • {TR('首次编译需联网下载依赖 (~100MB)', 'First build downloads deps (~100MB)')}")
        print(f"  • {TR('编译耗时约 2-5 分钟', 'Build takes ~2-5 minutes')}")

    # 设置专属临时目录，避免残留
    import tempfile
    _build_tmp = tempfile.mkdtemp(prefix="ov-cli-setup-")
    _old_tmpdir = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = _build_tmp
    print("=" * 54)
    try:
        r = input(f"  {TR('是否继续?', 'Continue?')} [y/N]: ")
        if r.strip().lower() != "y":
            mode = 1
    except (EOFError, KeyboardInterrupt):
        print()
        print(f"  {TR('安装已取消', 'Setup cancelled')}")
        sys.exit(1)

    # 检查系统依赖
    for _pkg, _hint in [("venv", "python3-venv"), ("pip", "python3-pip")]:
        _ok = subprocess.run(
            [sys.executable, "-c", f"import {_pkg}"],
            capture_output=True, text=True
        ).returncode == 0
        if not _ok:
            print(f"  {TR('错误: 缺少 {_hint}', 'Error: missing {_hint}').format(_hint=_hint)}")
            print(f"  {TR('请执行:', 'Run:')} sudo apt install {_hint}")
            sys.exit(1)

    try:
        venv_path = args.venv or os.path.join(workspace, ".venv")
        print(f"  {TR('创建虚拟环境', 'Creating venv')}: {venv_path}")
        subprocess.check_call([_sys.executable, "-m", "venv", venv_path, "--clear"])
        pip = _pip_path(venv_path)
        print(f"  {TR('安装依赖...', 'Installing dependencies...')}")
        # 先装 CPU-only torch（避免拉 CUDA 全家桶 ~3.4G）
        subprocess.check_call([pip, "install", "-v",
                               "torch", "torchvision",
                               "--index-url", "https://download.pytorch.org/whl/cpu"])

        pkgs = [
            "openvino>=2026.2",
            "openvino-tokenizers",
            "openvino-genai",
            "nncf>=3.0",
            "pillow",
            "numpy",
            "jinja2",
            "huggingface-hub",
            "safetensors",
            "sentencepiece",
            "tokenizers",
            "fastapi>=0.100",
            "uvicorn[standard]>=0.20",
            "accelerate",
            "wcwidth",
            "PyMuPDF",
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
    except KeyboardInterrupt:
        print()
        print(f"  {TR('安装已取消', 'Setup cancelled')}")
        sys.exit(1)

    # 应用 Gemma-4 补丁（修改 model_patcher.py 中不存在的属性引用）
    _apply_gemma4_patch()

    # 应用 Qwen3.5 补丁（transformers 5.9 中 Qwen3_5DynamicCache → DynamicCache）
    _apply_qwen35_patch()

    # 自动配置 VS Code 工作区设置
    _ensure_vscode_settings(venv_path)

    # 编译 GenAI 源码（模式2）
    if mode == 2:
        deps_ok = True
        for dep, hint in [("cmake", "sudo apt install cmake"),
                          ("gcc", "sudo apt install gcc"),
                          ("g++", "sudo apt install g++"),
                          ("make", "sudo apt install make"),
                          ("patchelf", "sudo apt install patchelf")]:
            if not shutil.which(dep):
                print(f"  ❌ {TR('未找到 {dep}，请先安装 ({hint})', '{dep} not found, install: {hint}').format(dep=dep, hint=hint)}")
                deps_ok = False
        if not deps_ok:
            sys.exit(1)
        _build_genai_from_source(venv_path, genai_src)

    # 写入版本戳
    _write_version_stamp(venv_path)

    # 清理临时目录
    shutil.rmtree(_build_tmp, ignore_errors=True)
    if _old_tmpdir:
        os.environ["TMPDIR"] = _old_tmpdir
    else:
        os.environ.pop("TMPDIR", None)

    print()
    print(f"  {TR('✅ 完成!', '✅ Done!')}")
    print(f"  {TR('💡 激活虚拟环境:', '💡 Activate venv:')}")
    print(f"     source {_activate_path(venv_path)}")
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
    """ov-cli server: 启动 API 服务 + Web UI"""
    from .server import run_server
    model_path = os.path.abspath(args.model)
    if not os.path.isdir(model_path):
        print(f"  ⚠ 模型路径不存在: {model_path}")
        sys.exit(1)
    run_server(model_path, args.device, args.host, args.port)


def cmd_chat(args):
    """ov-cli chat"""
    from .chat import load_model, run_chat, run_translate

    mode = args.mode
    if mode == "auto":
        # auto 模式下先加载模型再判断（需要 model_type）
        pass
    elif mode == "once":
        if not args.prompt and not args.file:
            print(f"  ⚠ {TR('once 模式需要 --prompt 和/或 --file 参数',
                           'once mode requires --prompt and/or --file')}")
            sys.exit(1)

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
    elif mode == "once":
        from .chat import run_once
        prompt = args.prompt.replace("\\n", "\n") if args.prompt else ""
        run_once(ctx, prompt=prompt, files=args.file or [],
                 output=args.output,
                 temperature=args.temp, top_p=args.top_p,
                 top_k=args.top_k, max_tokens=args.max_tokens,
                 reasoning=args.reasoning == "on")
    else:
        run_chat(ctx, system=args.system,
                 temperature=args.temp, top_p=args.top_p,
                 top_k=args.top_k, max_tokens=args.max_tokens,
                 image_path=args.image,
                 reasoning=args.reasoning == "on")


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
            "  ./ov-cli convert --model ./Qwen3.5 --format nf4     # NF4 (QLoRA)\n"
            "\n"
            "  # 聊天模式 (通用对话)\n"
            "  ./ov-cli chat --model ./gemma-4-E2B-it-ov-int4\n"
            "  ./ov-cli chat --model ./model-ov --temp 0.9 --max-tokens 2048\n"
            "\n"
            "  # 翻译模式 (Hy-MT2 等翻译模型)\n"
            "  ./ov-cli chat --model ./Hy-MT2-1.8B-ov --mode translate\n"
            "\n"
            "  # API 服务\n"
            "  ./ov-cli server --model ./model-ov --port 8080\n"
            "  # → 浏览器打开 http://localhost:8080\n"
            "\n"
            "  # 修复环境（更新代码后使用）\n"
            "  ./ov-cli setup --fix\n"
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
            "  ./ov-cli convert --model ./Qwen3.5 --format nf4      # NF4 (QLoRA)\n"
            "\n"
            "  # Chat mode (general conversation)\n"
            "  ./ov-cli chat --model ./gemma-4-E2B-it-ov-int4\n"
            "  ./ov-cli chat --model ./model-ov --temp 0.9 --max-tokens 2048\n"
            "\n"
            "  # Translate mode (Hy-MT2 and similar)\n"
            "  ./ov-cli chat --model ./Hy-MT2-1.8B-ov --mode translate\n"
            "\n"
            "  # API server\n"
            "  ./ov-cli server --model ./model-ov --port 8080\n"
            "  # → Open http://localhost:8080 in browser\n"
            "\n"
            "  # Fix environment (after git pull)\n"
            "  ./ov-cli setup --fix\n"
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
    p_setup.add_argument("--fix", action="store_true",
                         help=TR("修复模式（不重建 venv，仅升级依赖+重打补丁）",
                                 "Fix mode (no venv recreate, only upgrade + repatch)"))

    # ── convert ──
    p_conv = sub.add_parser(
        "convert",
        help=TR("转换 HuggingFace 模型 → OpenVINO IR", "Convert HF model → OpenVINO IR"),
        description=TR(
            "转换 HuggingFace 模型 → OpenVINO IR (支持多种量化格式)\n"
            "\n"
            "使用 Optimum Intel 官方工具，自动推断 task 类型。\n"
            "\n"
            "流程:\n"
            "  1. optimum-cli export openvino (自动下载 + 转换 + 量化)\n"
            "  2. 保存 .xml/.bin + tokenizer + 配置文件\n"
            "\n"
            "量化格式:\n"
            "  fp32   浮点 (无损, 体积最大)\n"
            "  fp16   半精度 (体积减半, 几乎无损)\n"
            "  int8   8-bit (体积~25%, 几乎无损)\n"
            "  int4   4-bit (体积~12.5%, 有精度损失)\n"
            "  mxfp4  MXFP4 (微缩放格式, OCP 标准)\n"
            "  nf4    NF4 (QLoRA 格式, 高质量 4-bit)\n"
            "  cb4    CodeBook (16 固定 fp8 码本)\n"

            "\n"
            "高级参数:\n"
            "  --ratio RATIO     INT4 混合精度比例 (0-1, 默认1.0)\n"
            "  --group-size GS   量化分组大小 (默认128, 越大精度越高)",
            "Convert HuggingFace model → OpenVINO IR (multiple formats)\n"
            "\n"
            "Uses Optimum Intel official tool, auto-infers task type.\n"
            "\n"
            "Pipeline:\n"
            "  1. optimum-cli export openvino (download + convert + quantize)\n"
            "  2. Save .xml/.bin + tokenizer + configs\n"
            "\n"
            "Quantization formats:\n"
            "  fp32   full precision (lossless, largest)\n"
            "  fp16   half precision (size ~half, near-lossless)\n"
            "  int8   8-bit (size ~25%, near-lossless)\n"
            "  int4   4-bit (size ~12.5%, some quality loss)\n"
            "  mxfp4  MXFP4 (micro-scaling, OCP standard)\n"
            "  nf4    NF4 (QLoRA format, high-quality 4-bit)\n"
            "  cb4    CodeBook (16 fixed fp8 codebook)\n"
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
    p_conv.add_argument("--format", choices=["fp32", "fp16", "int8", "int4",
                                               "mxfp4", "nf4", "cb4"], default="fp32",
                        help=TR("量化格式: fp32/fp16/int8/int4/mxfp4/nf4/cb4 (默认: fp32)",
                                "format: fp32/fp16/int8/int4/mxfp4/nf4/cb4 (default: fp32)"))
    p_conv.add_argument("--ratio", type=float, default=1.0,
                        help=TR("INT4 混合精度比例 0-1 (默认 1.0)", "INT4 mixed precision ratio 0-1 (default 1.0)"))
    p_conv.add_argument("--group-size", type=int, default=128, dest="group_size",
                        help=TR("量化分组大小 (默认 128)", "quantization group size (default 128)"))

    # ── chat ──
    p_chat = sub.add_parser(
        "chat",
        help=TR("交互式聊天 / 翻译终端", "Interactive chat / translate terminal"),
        description=TR(
            "加载 OpenVINO 模型并启动交互终端。支持聊天、翻译和单次输出三种模式。\n"
            "\n"
            "聊天模式 (VLM 支持图片和 PDF):\n"
            "  • //img PATH  加载图片（支持多文件）\n"
            "  • //pdf PATH  加载 PDF（自动转图片，最多 24 页）\n"
            "  • //txt PATH  加载文本文件（支持多文件）\n"
            "  • /file       查看已加载文件\n"
            "  • /clear [ids] 清空上下文或指定文件 ID\n"
            "  • /temp N     调节温度\n"
            "  • /system T   设置系统提示词\n"
            "  • 流式输出 + visual token 统计\n"
            "\n"
            "翻译模式 (仅 Hy-MT2 等翻译模型):\n"
            "  • 自动检测语言方向 (中↔英)\n"
            "  • //en 文本 → 强制译英, //zh 文本 → 强制译中\n"
            "\n"
            "单次输出模式 (once):\n"
            "  • --prompt TEXT  文字输入（支持 \\n 换行）\n"
            "  • --file PATH    上传文件（可多次，自动检测类型）\n"
            "  • --output PATH  保存结果为 .md 文件\n"
            "  • 输出完自动退出\n"
            "\n"
            "采样参数:\n"
            "  temperature  随机性 (0=贪婪, >0 越高越随机)\n"
            "  top-p        nucleus sampling 累积概率阈值\n"
            "  top-k        仅从概率最高的 k 个 token 中采样",
            "Load an OpenVINO model and start an interactive terminal.\n"
            "\n"
            "Chat mode (VLM supports images & PDFs):\n"
            "  • //img PATH  load image(s)\n"
            "  • //pdf PATH  load PDF (auto-convert to images, max 24 pages)\n"
            "  • //txt PATH  load text file(s)\n"
            "  • /file       list loaded files\n"
            "  • /clear [ids] clear context or specific files by ID\n"
            "  • /temp N     set temperature\n"
            "  • /system T   set system prompt\n"
            "  • Streaming + visual token stats\n"
            "\n"
            "Translate mode (Hy-MT2 and similar):\n"
            "  • Auto-detect language direction (Chinese↔English)\n"
            "  • //en text → force English, //zh text → force Chinese\n"
            "\n"
            "Once mode (single output):\n"
            "  • --prompt TEXT  input text (supports \\n)\n"
            "  • --file PATH    upload file(s), auto-detect type\n"
            "  • --output PATH  save result as .md file\n"
            "  • Exits after output\n"
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
    p_chat.add_argument("--mode", choices=["chat", "translate", "once", "auto"], default="auto",
                        help=TR("运行模式: chat=交互, translate=翻译, once=单次输出 (默认: auto)",
                                "mode: chat=interactive, translate=translation, once=single output (default: auto)"))
    p_chat.add_argument("--prompt",
                        help=TR("输入文字 (支持 \\n 换行，仅 once 模式)", "input text (supports \\n, once mode only)"))
    p_chat.add_argument("--file", action="append", default=None,
                        help=TR("上传文件 (可多次使用，仅 once 模式)", "file path (can be used multiple times, once mode only)"))
    p_chat.add_argument("--output",
                        help=TR("输出目录或文件路径 (自动生成 .md，仅 once 模式)",
                                "output dir or file path (auto .md, once mode only)"))
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
    p_chat.add_argument("--reasoning", choices=["on", "off"], default="on",
                        help=TR("思考模式 (默认: on，仅支持思考的模型有效)", "reasoning mode (default: on)"))

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
    p_bench.add_argument("--reasoning", choices=["on", "off"], default="on",
                         help=TR("思考模式 (默认: on)", "reasoning mode (default: on)"))

    # ── server ──
    p_serve = sub.add_parser(
        "server",
        help=TR("启动 API 服务", "Start API server"),
        description=TR(
            "启动 OpenAI 兼容 API 服务。\n"
            "支持多图输入（VLM）和流式/非流式输出。\n"
            "\n"
            "端点:\n"
            "  POST /v1/chat/completions    聊天 (SSE 流式 + 非流式)\n"
            "  GET  /v1/models               模型信息 + 能力\n"
            "  GET  /props                   服务器属性\n"
            "  GET  /health                  健康检查\n"
            "  GET  /                        服务状态\n"
            "\n"
            "多图推理:\n"
            "  content 中传入多个 image_url 即可，自动添加占位标记\n"
            "\n"
            "示例:\n"
            "  ./ov-cli server --model ./model-ov --port 8080\n"
            "  curl http://localhost:8080/v1/models",
            "Start OpenAI-compatible API server.\n"
            "Supports multi-image input (VLM) and stream/non-stream output.\n"
            "\n"
            "Endpoints:\n"
            "  POST /v1/chat/completions    chat (SSE streaming + non-stream)\n"
            "  GET  /v1/models               model info + capabilities\n"
            "  GET  /props                   server properties\n"
            "  GET  /health                  health check\n"
            "  GET  /                        server status\n"
            "\n"
            "Multi-image:\n"
            "  Pass multiple image_url entries in content, image tags auto-added\n"
            "\n"
            "Examples:\n"
            "  ./ov-cli server --model ./model-ov --port 8080\n"
            "  curl http://localhost:8080/v1/models",
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_serve.add_argument("--model", "-m", required=True,
                         help=TR("OpenVINO 模型目录路径", "path to OpenVINO model dir"))
    p_serve.add_argument("--device", default="", choices=["CPU", "GPU"],
                         help=TR("推理设备 (默认: 自动检测)", "inference device (default: auto)"))
    p_serve.add_argument("--host", default="0.0.0.0",
                         help=TR("监听地址 (默认: 0.0.0.0)", "listen address (default: 0.0.0.0)"))
    p_serve.add_argument("--port", type=int, default=8080,
                         help=TR("监听端口 (默认: 8080)", "listen port (default: 8080)"))

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

    # 版本检测（非 setup 命令）
    if args.cmd != "setup":
        _venv = getattr(args, "venv", None) or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv")
        _check_version_warning(_venv)

    # WSL2 GPU 检测（非 setup/venv 命令）
    if args.cmd not in ("setup", "venv"):
        _check_wsl2_gpu()

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
    elif args.cmd == "server":
        cmd_server(args)


if __name__ == "__main__":
    main()
