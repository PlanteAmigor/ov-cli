"""
ov-cli setup: 虚拟环境创建与依赖安装。
"""

import os, sys, json, shutil, subprocess, tempfile
import ov_cli
from ov_cli import TR


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


def cmd_setup(args, workspace):
    """ov-cli setup: 创建虚拟环境并安装依赖"""
    # ── 修复模式 ──
    if args.fix:
        venv_path = args.venv or os.path.join(workspace, ".venv")
        if not os.path.isdir(venv_path):
            print(f"  {TR('错误: 未找到虚拟环境', 'Error: venv not found')}: {venv_path}")
            print(f"  {TR('请先运行', 'Run first')}: ./ov-cli setup")
            sys.exit(1)
        pip = _pip_path(venv_path)
        _mode_file = os.path.join(venv_path, ".ov-cli-mode")
        _prev_mode = None
        if os.path.isfile(_mode_file):
            with open(_mode_file) as f:
                _prev_mode = f.read().strip()
        _genai_src = os.path.join(workspace, "openvino.genai-2026.2.0.0-optimization")
        # 简易→完整升级路径
        if _prev_mode != "2" and os.path.isdir(_genai_src):
            print(f"  {TR('检测到 GenAI 源码目录，可升级到完整模式', 'GenAI source found, can upgrade to full mode')}")
            r = input(f"  {TR('升级到完整模式？(y/N)', 'Upgrade to full mode? (y/N)')}: ").strip().lower()
            if r == "y":
                _build_genai_from_source(venv_path, _genai_src)
                with open(_mode_file, "w") as f:
                    f.write("2")
                print(f"  {TR('✅ 已升级到完整模式', '✅ Upgraded to full mode')}")
                return
        print(f"  {TR('修复模式: 升级依赖 + 重打补丁', 'Fix mode: upgrade deps + repatch')}")
        try:
            # 升级 ov-cli（排除 openvino-genai 避免覆盖编译版）
            _install_cmd = [pip, "install", "--upgrade", workspace]
            if _prev_mode == "2":
                _install_cmd += ["--no-deps"]
                subprocess.check_call(_install_cmd)
                subprocess.check_call([pip, "install", "--upgrade", "--no-deps",
                                       "optimum-intel@git+https://github.com/huggingface/optimum-intel.git"])
                subprocess.check_call([pip, "install", "--upgrade", "--no-deps", "transformers"])
            else:
                subprocess.check_call(_install_cmd)
            _apply_gemma4_patch()
            _write_version_stamp(venv_path, workspace)
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
        print(f"  {TR('安装依赖...', 'Installing dependencies...')}")
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
            "soundfile",
            "scipy",
            "gradio",
        ]
        subprocess.check_call([pip, "install", "-v"] + pkgs)

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

        print(f"  {TR('安装 transformers (no-deps)...', 'Installing transformers (no-deps)...')}")
        subprocess.check_call([pip, "install", "--no-deps", "--force-reinstall", "transformers"])
    except KeyboardInterrupt:
        print()
        print(f"  {TR('安装已取消', 'Setup cancelled')}")
        sys.exit(1)

    _apply_gemma4_patch()
    _ensure_vscode_settings(venv_path, workspace)

    if mode == 2:
        for dep, hint in [("cmake", "sudo apt install cmake"),
                          ("gcc", "sudo apt install gcc"),
                          ("g++", "sudo apt install g++"),
                          ("make", "sudo apt install make"),
                          ("patchelf", "sudo apt install patchelf")]:
            if not shutil.which(dep):
                print(f"  ❌ {TR('未找到 {dep}，请先安装 ({hint})', '{dep} not found, install: {hint}').format(dep=dep, hint=hint)}")
                sys.exit(1)
        _build_genai_from_source(venv_path, genai_src)

    # 记录安装模式（供 --fix 使用）
    _mode_file = os.path.join(venv_path, ".ov-cli-mode")
    with open(_mode_file, "w") as f:
        f.write(str(mode))

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


def _write_version_stamp(venv_path, workspace):
    """（已废弃）保留桩函数避免引用错误。"""
    pass

