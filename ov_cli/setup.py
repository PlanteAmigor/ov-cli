"""
ov-cli setup: 虚拟环境创建与依赖安装。

支持按需安装:
  ./ov-cli setup --with chat,image
  ./ov-cli setup --with all
  ./ov-cli setup --fix
"""

import os, sys, json, shutil, subprocess, tempfile
import ov_cli
from ov_cli import TR
from ov_cli.features import get_packages, get_extra_pips, get_exclusive_packages, get_installed, save as _save_features


_ALL_FEATURES = {"chat", "image", "asr", "tts", "ui", "mcp", "server", "convert"}

_FEATURE_HINTS = {
    "chat":    "聊天终端（PyMuPDF ~15MB）",
    "image":   "文生图（无额外依赖）",
    "asr":     "语音识别（soundfile + qwen-asr ~50MB）",
    "tts":     "语音合成（soundfile + qwen-tts ~50MB）",
    "ui":      "Web 界面（gradio ~30MB）",
    "mcp":     "MCP 协议服务器（无额外依赖）",
    "server":  "API 服务（fastapi + uvicorn ~15MB）",
    "convert": "模型转换（torch ~3GB, optimum-intel, 需 5-10 分钟）",
}

_CONVERT_WARN = "⚠ convert 模块需要安装 torch + optimum-intel（约 3GB，耗时 5-10 分钟）"


def _is_windows():
    return sys.platform == "win32"


def _features_path(venv_path):
    return os.path.join(venv_path, ".ov-cli-features")


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


def _ensure_vscode_settings(venv_path, workspace):
    """创建 VS Code 工作区设置，使终端自动激活虚拟环境"""
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


_APT_DEPS = {
    "sox": {
        "pkg": "sox",
        "hint": "音频处理（TTS/ASR 需要）",
        "features": ["asr", "tts"],
    },
    "libsndfile1": {
        "pkg": "libsndfile1",
        "hint": "音频 I/O（soundfile 需要）",
        "features": ["asr", "tts"],
    },
}


def _check_apt_deps(features):
    """检测系统级 apt 依赖是否安装，缺失则给出安装提示。"""
    missing = []
    for cmd, info in _APT_DEPS.items():
        if not any(f in features for f in info["features"]):
            continue
        if shutil.which(cmd) if cmd != "libsndfile1" else _check_ld_lib(cmd):
            continue
        missing.append((info["pkg"], info["hint"]))

    if not missing:
        return

    print(f"  ⚠ {TR('检测到系统依赖缺失', 'Missing system dependencies')}:")
    for pkg, hint in missing:
        print(f"    • {pkg} — {hint}")
    print(f"  {TR('请执行以下命令安装:', 'Run the following to install:')}")
    print(f"    sudo apt install {' '.join(pkg for pkg, _ in missing)}")


def _check_ld_lib(lib):
    """检查共享库是否可用（ldconfig / ld 查找）。"""
    try:
        r = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=10)
        return lib in r.stdout
    except Exception:
        try:
            r = subprocess.run(["ld", f"-l{lib}"], capture_output=True, text=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return True  # 无法检测时放行


def _apply_gemma4_patch():
    """自动修复 optimum-intel 中 Gemma-4 共享 KV 层的属性引用错误。"""
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
        if "hasattr(self, \"kv_shared_layer_index\")" in content:
            print(f"  ✓ {TR('Gemma-4 补丁已存在', 'Gemma-4 patch already applied')}")
        else:
            print(f"  - {TR('Gemma-4 补丁不需要或已不适用', 'Gemma-4 patch not needed or N/A')}")


def _build_genai_from_source(venv_path, genai_src):
    """从源码编译 openvino-genai 并安装到虚拟环境。"""
    import sysconfig

    build_dir = os.path.join(genai_src, "build")
    if os.path.isdir(build_dir):
        shutil.rmtree(build_dir)

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

    genai_out = os.path.join(build_dir, "openvino_genai")
    target = os.path.join(site_packages, "openvino_genai")
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    libsrc = os.path.join(genai_out, "libopenvino_genai.so")
    pysrc = os.path.join(genai_out, f"py_openvino_genai{ext}")
    if os.path.isfile(pysrc) and os.path.isfile(libsrc):
        try:
            subprocess.check_call(["patchelf", "--set-rpath", f"$ORIGIN:{target}", pysrc])
        except Exception:
            pass
        shutil.copy2(libsrc, os.path.join(target, "libopenvino_genai.so"))
        shutil.copy2(libsrc, os.path.join(target, "libopenvino_genai.so.2620"))
        shutil.copy2(pysrc, target)
        print(f"  ✓ {TR('已安装', 'Installed')}: libopenvino_genai.so + py_openvino_genai{ext}")

    print(f"  ✓ {TR('GenAI 编译安装完成', 'GenAI build & install complete')}")
    return True


def _prompt_mode(has_genai_src):
    """交互选择安装模式：1=简易 2=完整（编译 GenAI）"""
    if not has_genai_src:
        print(f"  - {TR('GenAI 源码目录不存在，使用简易模式', 'No GenAI source, using simple mode')}")
        return 1
    while True:
        try:
            r = input(f"  {TR('选择安装模式', 'Select mode')}:\n"
                      f"    1. {TR('简易模式 - 仅 pip 安装', 'Simple - pip only')}\n"
                      f"    2. {TR('完整模式 - 安装修改版 openvino-genai 启用 thinking budget', 'Full - install patched openvino-genai for thinking budget')}\n"
                      f"  {TR('请输入 [1/2]', 'Enter [1/2]')} (1): ")
            if r.strip() == "":
                return 1
            m = int(r.strip())
            if m in (1, 2):
                return m
        except (ValueError, EOFError):
            return 1


def _install_features(pip, features: set[str], workspace, fix_mode=False):
    """安装指定功能需要的 pip 包。"""
    pkgs = get_packages(features)
    if pkgs:
        print(f"  {TR('安装基础依赖...', 'Installing base deps...')}")
        subprocess.check_call([pip, "install", "-v"] + pkgs)

    # 修复模式下：升级依赖 + 重打补丁
    if fix_mode:
        subprocess.check_call([pip, "install", "--upgrade", "huggingface-hub", "transformers"])
        # 检查是否已安装 optimum-intel，是则重装为固定版本
        r = subprocess.run([sys.executable, "-c", "import optimum.intel"], capture_output=True)
        if r.returncode == 0:
            print(f"  {TR('重装 optimum-intel==1.27.0...', 'Reinstalling optimum-intel==1.27.0...')}")
            subprocess.check_call([pip, "install", "--force-reinstall", "optimum-intel==1.27.0"])
        return

    # Install optimum-intel + transformers if convert is requested
    if "convert" in features:
        print(f"  {TR('安装转换依赖...', 'Installing convert deps...')}")
        print(f"  {TR('安装 optimum-intel==1.27.0 (pip)...', 'Installing optimum-intel==1.27.0 (pip)...')}")
        subprocess.check_call([pip, "install", "optimum-intel==1.27.0"])
        print(f"  {TR('安装 transformers (no-deps)...', 'Installing transformers (no-deps)...')}")
        subprocess.check_call([pip, "install", "--no-deps", "--force-reinstall", "transformers"])

    # 额外 pip 包（qwen-tts/asr 等）
    extra = get_extra_pips(features)
    for pkg in extra:
        print(f"  ⚡ {TR('安装 {}...', 'Installing {}...').format(pkg)}")
        subprocess.check_call([pip, "install", "--quiet", pkg], timeout=180)

    # qwen 包可能拉入 CUDA torch，强制换回 CPU 版
    if "asr" in features or "tts" in features:
        print(f"  ⚡ {TR('修复 torch 为 CPU 版...', 'Fixing torch to CPU version...')}")
        subprocess.check_call([pip, "install", "--force-reinstall", "--no-deps",
                               "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])

    # torch 先安装（CPU 版）
    if "convert" in features:
        subprocess.check_call([pip, "install", "torch", "torchvision",
                               "--index-url", "https://download.pytorch.org/whl/cpu"])


def _remove_features(pip, venv_path, removed: set[str]):
    """卸载指定功能独有（不被其他已装功能需要）的 pip 包。"""
    installed = get_installed(venv_path)
    remaining = installed - removed

    if not removed:
        print(f"  {TR('没有指定要移除的模块', 'No features to remove')}")
        return

    invalid = removed - _ALL_FEATURES
    if invalid:
        print(f"  ❌ {TR('不支持的功能', 'Unsupported features')}: {', '.join(sorted(invalid))}")
        print(f"     {TR('支持', 'Supported')}: {', '.join(sorted(_ALL_FEATURES))}")
        return

    not_installed = removed - installed
    if not_installed:
        print(f"  - {TR('以下模块未安装，跳过', 'Already not installed')}: {', '.join(sorted(not_installed))}")

    to_remove = removed - not_installed
    if not to_remove:
        print(f"  {TR('没有需要移除的模块', 'Nothing to remove')}")
        return

    exclusive = get_exclusive_packages(to_remove, remaining)
    if not exclusive:
        print(f"  {TR('所有包均为其他模块共享，无需卸载', 'All packages are shared, nothing to uninstall')}")
    else:
        print(f"  {TR('将卸载以下独有包', 'Will uninstall exclusive packages')}:")
        for f, pkgs in sorted(exclusive.items()):
            print(f"    • {f}: {', '.join(pkgs)}")
        print()
        try:
            r = input(f"  {TR('确认卸载？(y/N)', 'Confirm uninstall? (y/N)')}: ").strip().lower()
            if r != "y":
                print(f"  {TR('已取消', 'Cancelled')}")
                return
        except (EOFError, KeyboardInterrupt):
            print()
            print(f"  {TR('已取消', 'Cancelled')}")
            return
        all_pkgs = sorted(set(p for pkgs in exclusive.values() for p in pkgs))
        try:
            subprocess.check_call([pip, "uninstall", "-y"] + all_pkgs)
            print(f"  ✓ {TR('卸载完成', 'Uninstall done')}")
        except subprocess.CalledProcessError:
            print(f"  ⚠ {TR('部分包卸载失败（可能已被手动移除）', 'Some packages may already be removed')}")

    _save_features(venv_path, remaining)
    print(f"  ✓ {TR('已更新安装记录', 'Features list updated')}: {', '.join(sorted(remaining))}")
    print(f"  {TR('基础依赖（openvino 等）始终保持不变', 'Base deps (openvino etc.) are kept')}")
    print(f"  {TR('如需彻底清理，可重建环境', 'For full cleanup, recreate with ./ov-cli setup')}")


def cmd_setup(args, workspace):
    """ov-cli setup: 创建虚拟环境并安装依赖"""

    # ── 解析 --with ──
    if args.with_features:
        raw = args.with_features.strip()
        features = _ALL_FEATURES if raw == "all" else {s.strip() for s in raw.split(",") if s.strip()}
    else:
        features = _ALL_FEATURES  # 默认全装

    invalid = features - _ALL_FEATURES
    if invalid:
        print(f"  ❌ {TR('不支持的功能', 'Unsupported features')}: {', '.join(sorted(invalid))}")
        print(f"     {TR('支持', 'Supported')}: {', '.join(sorted(_ALL_FEATURES))}")
        sys.exit(1)

    # ── 修复模式 ──
    if args.fix:
        venv_path = args.venv or os.path.join(workspace, ".venv")
        if not os.path.isdir(venv_path):
            print(f"  {TR('错误: 未找到虚拟环境', 'Error: venv not found')}: {venv_path}")
            print(f"  {TR('请先运行', 'Run first')}: ./ov-cli setup")
            sys.exit(1)
        pip = _pip_path(venv_path)

        # 读取已装功能
        installed = set()
        fp = _features_path(venv_path)
        if os.path.isfile(fp):
            with open(fp) as f:
                installed = {s.strip() for s in f.read().strip().split(",") if s.strip()}
        if not installed:
            installed = _ALL_FEATURES

        _mode_file = os.path.join(venv_path, ".ov-cli-mode")
        _prev_mode = None
        if os.path.isfile(_mode_file):
            with open(_mode_file) as f:
                _prev_mode = f.read().strip()
        _genai_src = os.path.join(workspace, "openvino.genai-2026.2.0.0-optimization")

        # 简易→完整升级路径
        if _prev_mode != "2":
            _thinking_whl = os.path.join(workspace, "openvino-genai-thinking", "dist",
                                         "openvino_genai_thinking-2026.2.0.0-cp313-cp313-manylinux_2_41_x86_64.whl")
            if os.path.isfile(_thinking_whl):
                print(f"  {TR('检测到修改版 openvino-genai whl，可升级到完整模式', 'Patched openvino-genai whl found, can upgrade to full mode')}")
                r = input(f"  {TR('升级到完整模式？(y/N)', 'Upgrade to full mode? (y/N)')}: ").strip().lower()
                if r == "y":
                    subprocess.check_call([pip, "install", _thinking_whl])
                    with open(_mode_file, "w") as f:
                        f.write("2")
                    print(f"  {TR('✅ 已升级到完整模式', '✅ Upgraded to full mode')}")
                    return

        print(f"  {TR('修复模式: 升级依赖 + 重打补丁', 'Fix mode: upgrade deps + repatch')}")
        # 只修复已装的功能
        _install_features(pip, installed, workspace, fix_mode=True)
        if "convert" in installed:
            _apply_gemma4_patch()
        print(f"  {TR('✅ 修复完成', '✅ Fix done')}")
        return

    # ── 移除模式 ──
    if args.remove_features:
        raw = args.remove_features.strip()
        to_remove = {s.strip() for s in raw.split(",") if s.strip()}
        venv_path = args.venv or os.path.join(workspace, ".venv")
        if not os.path.isdir(venv_path):
            print(f"  {TR('错误: 未找到虚拟环境', 'Error: venv not found')}: {venv_path}")
            print(f"  {TR('请先运行', 'Run first')}: ./ov-cli setup")
            sys.exit(1)
        pip = _pip_path(venv_path)
        _remove_features(pip, venv_path, to_remove)
        return

    # ── 打印安装概要 ──
    print(f"  {TR('即将安装以下模块', 'Will install:')}")
    for f in sorted(features):
        hint = _FEATURE_HINTS.get(f, f)
        print(f"    • {f} — {hint}")

    if "convert" in features:
        print(f"  {_CONVERT_WARN}")
        try:
            r = input(f"  {TR('是否继续?', 'Continue?')} [Y/n]: ")
            if r.strip().lower() == "n":
                sys.exit(0)
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

    # 检查目录写入权限
    if not os.access(workspace, os.W_OK):
        _user = os.environ.get("USER", "")
        print(f"  {TR('错误: 当前目录没有写入权限', 'Error: no write permission')}")
        print(f"  {TR('请执行以下命令后重试:', 'Run the following command and retry:')}")
        print(f"    sudo chown -R {_user}:{_user} {workspace}")
        sys.exit(1)

    genai_src = os.path.join(workspace, "openvino.genai-2026.2.0.0-optimization")

    # ── 只有装了 chat 才问 mode ──
    mode = 1
    if "chat" in features:
        print()
        print(f"  {TR('chat 模块需要选择安装模式', 'chat module needs mode selection')}")
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
        print(f"  {TR('1. 创建虚拟环境并 pip 安装依赖', '1. Create venv & pip install deps')}")
        print(f"  {TR('2. 安装修改版 openvino-genai（预编译 whl，数秒完成）', '2. Install patched openvino-genai (prebuilt whl, seconds)')}")
        print()
        print(f"  {TR('前置条件', 'Prerequisites')}:")
        print(f"  • {TR('Python 3.10+', 'Python 3.10+')}")
        print(f"  • {TR('Intel GPU / CPU', 'Intel GPU / CPU')}")

    # ── venv 就绪检查 + 系统依赖 ──
    _check_apt_deps(features)
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
        subprocess.check_call([sys.executable, "-m", "venv", venv_path, "--clear"])
        pip = _pip_path(venv_path)

        # 安装所选功能
        _install_features(pip, features, workspace)

    except KeyboardInterrupt:
        print()
        print(f"  {TR('安装已取消', 'Setup cancelled')}")
        sys.exit(1)

    # 补丁（只有装了 convert 才需要）
    if "convert" in features:
        _apply_gemma4_patch()

    _ensure_vscode_settings(venv_path, workspace)

    # 编译 GenAI（仅 mode 2 且包含 chat）
    if mode == 2 and "chat" in features:
        _thinking_whl = os.path.join(workspace, "openvino-genai-thinking", "dist", "openvino_genai_thinking-2026.2.0.0-cp313-cp313-manylinux_2_41_x86_64.whl")
        if os.path.isfile(_thinking_whl):
            print(f"  ⚡ {TR('安装修改版 openvino-genai（含 ThinkingBudgetTransform）...', 'Installing patched openvino-genai (with ThinkingBudgetTransform)...')}")
            subprocess.check_call([pip, "install", _thinking_whl])
        else:
            print(f"  ⚠ {TR('未找到预编译 whl，回退到源码编译...', 'Prebuilt whl not found, falling back to source build...')}")
            for dep, hint in [("cmake", "sudo apt install cmake"),
                              ("gcc", "sudo apt install gcc"),
                              ("g++", "sudo apt install g++"),
                              ("make", "sudo apt install make"),
                              ("patchelf", "sudo apt install patchelf")]:
                if not shutil.which(dep):
                    print(f"  ❌ {TR('未找到 {dep}，请先安装 ({hint})', '{dep} not found, install: {hint}').format(dep=dep, hint=hint)}")
                    sys.exit(1)
            _build_genai_from_source(venv_path, genai_src)

    # ── 记录安装信息 ──
    _save_features(venv_path, features)

    _mode_file = os.path.join(venv_path, ".ov-cli-mode")
    with open(_mode_file, "w") as f:
        f.write(str(mode))

    print()
    print(f"  {TR('✅ 完成!', '✅ Done!')}")
    print(f"  {TR('💡 激活虚拟环境:', '💡 Activate venv:')}")
    print(f"     source {_activate_path(venv_path)}")
    print(f"  {TR('💡 或在 VS Code 中重新打开终端即可自动激活', '💡 Or just reopen terminal in VS Code for auto-activation')}")


def _write_version_stamp(venv_path, workspace):
    """（已废弃）保留桩函数避免引用错误。"""
    pass