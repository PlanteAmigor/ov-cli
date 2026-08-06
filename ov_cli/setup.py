"""
ov-cli setup: 虚拟环境创建与依赖安装。

支持按需安装:
  ./ov-cli setup --with chat,image
  ./ov-cli setup --with all
  ./ov-cli setup --fix
  ./ov-cli setup --remove asr
"""

import os, sys, json, shutil, subprocess, tempfile
import ov_cli
from ov_cli import TR
from ov_cli.features import get_packages, get_extra_pips, get_exclusive_packages, get_installed, save as _save_features

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_ALL_FEATURES = {"chat", "image", "asr", "tts", "yolo"}

_FEATURE_HINTS = {
    "chat":    "聊天终端（PyMuPDF ~15MB）",
    "image":   "文生图（无额外依赖）",
    "asr":     "语音识别（soundfile + qwen-asr ~50MB）",
    "tts":     "语音合成（soundfile + qwen-tts ~50MB）",
    "yolo":    "目标检测（ultralytics ~30MB）",
}


def _activate_path(venv_path):
    """返回虚拟环境的 activate 脚本路径。"""
    return os.path.join(venv_path, "bin", "activate")


def _pip_path(venv_path):
    """返回虚拟环境的 pip 路径。"""
    return os.path.join(venv_path, "bin", "pip")



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


def _install_features(pip, features: set[str], workspace, fix_mode=False):
    """安装指定功能需要的 pip 包。"""
    pkgs = get_packages(features)
    if pkgs:
        print(f"  {TR('安装基础依赖...', 'Installing base deps...')}")
        subprocess.check_call([pip, "install", "-v"] + pkgs)

    # 修复模式下：仅升级 huggingface-hub + transformers
    if fix_mode:
        subprocess.check_call([pip, "install", "--upgrade", "huggingface-hub", "transformers"])
        return

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




def _remove_features(pip, venv_path, removed: set[str]):
    """卸载指定功能独有（不被其他已装功能需要）的 pip 包。"""
    installed = get_installed()
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

    _save_features(remaining)
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
        venv_path = os.path.join(workspace, ".venv")
        if not os.path.isdir(venv_path):
            print(f"  {TR('错误: 未找到虚拟环境', 'Error: venv not found')}: {venv_path}")
            print(f"  {TR('请先运行', 'Run first')}: ./ov-cli setup")
            sys.exit(1)
        pip = _pip_path(venv_path)

        installed = get_installed()

        print(f"  {TR('修复模式: 升级依赖', 'Fix mode: upgrade deps')}")
        # 只修复已装的功能
        _install_features(pip, installed, workspace, fix_mode=True)
        print(f"  {TR('✅ 修复完成', '✅ Fix done')}")
        return

    # ── 移除模式 ──
    if args.remove_features:
        raw = args.remove_features.strip()
        to_remove = {s.strip() for s in raw.split(",") if s.strip()}
        venv_path = os.path.join(workspace, ".venv")
        if not os.path.isdir(venv_path):
            print(f"  {TR('错误: 未找到虚拟环境', 'Error: venv not found')}: {venv_path}")
            print(f"  {TR('请先运行', 'Run first')}: ./ov-cli setup")
            sys.exit(1)
        pip = _pip_path(venv_path)
        _remove_features(pip, venv_path, to_remove)
        return

    # ── 交互确认 ──
    print(f"  {TR('即将安装以下模块', 'Will install:')}")
    selected = set()
    for f in sorted(features):
        hint = _FEATURE_HINTS.get(f, f)
        print(f"    • {f} — {hint}")

        try:
            r = input(f"  {TR('是否安装?', 'Install?')} [Y/n]: ")
            if r.strip().lower() == "n":
                continue  # 跳过该模块，不退出
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        selected.add(f)

    features = selected
    if not features:
        print(f"  {TR('没有选择任何模块，退出', 'No features selected, exiting')}")
        sys.exit(0)

    # 检查目录写入权限
    if not os.access(workspace, os.W_OK):
        _user = os.environ.get("USER", "")
        print(f"  {TR('错误: 当前目录没有写入权限', 'Error: no write permission')}")
        print(f"  {TR('请执行以下命令后重试:', 'Run the following command and retry:')}")
        print(f"    sudo chown -R {_user}:{_user} {workspace}")
        sys.exit(1)

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
        venv_path = os.path.join(workspace, ".venv")
        print(f"  {TR('创建虚拟环境', 'Creating venv')}: {venv_path}")
        subprocess.check_call([sys.executable, "-m", "venv", venv_path, "--clear"])
        pip = _pip_path(venv_path)

        # 安装所选功能
        _install_features(pip, features, workspace)

    except KeyboardInterrupt:
        print()
        print(f"  {TR('安装已取消', 'Setup cancelled')}")
        sys.exit(1)

    # ── 记录安装信息 ──
    _save_features(features)

    print()
    print(f"  {TR('✅ 完成!', '✅ Done!')}")