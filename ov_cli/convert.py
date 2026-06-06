"""
ov-cli convert: 模型转换。

使用 Optimum Intel 官方工具将 HuggingFace 模型导出为 OpenVINO IR 格式。
支持 INT8/INT4/NF4/MXFP4/CB4 等多种量化，自动推断 task 类型。
也支持自定义 pipeline 转换（如 Qwen3-TTS）。
"""

import os, sys, time, json, subprocess
from pathlib import Path
from ov_cli import TR

# 模型类型 → 转换所需 transformers 版本（None = 使用当前版本）
# 推理统一用 5.9，转换完成后自动恢复
_MODEL_TF_VERSION = {
    "gemma4":     ">=5.9",        # Gemma-4 需要高版本 transformers + 补丁
    "qwen3_5":    "==5.2",        # Qwen3.5 必须 5.2（否则 DynamicCache 兼容问题）
    "qwen3_6":    "==5.2",        # Qwen3.6 MoE 同 Qwen3.5
    "qwen3_tts":  None,           # Qwen3-TTS 用自定义 helper，不依赖 transformers 版本
    "qwen3_asr":  None,           # Qwen3-ASR 同
}


def _detect_model_type(model_path):
    """从 config.json 读取 model_type。"""
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        # 优先检查 architectures（Qwen3-TTS / Qwen3-ASR 等自定义模型）
        archs = cfg.get("architectures", [])
        if any("Qwen3TTS" in a for a in archs):
            return "qwen3_tts"
        if any("Qwen3ASR" in a for a in archs):
            return "qwen3_asr"
        # 多模态模型 model_type 在 text_config 里
        mt = cfg.get("model_type", "")
        if mt in ("gemma4", "qwen3_5", "qwen3_6"):
            return mt
        tc = cfg.get("text_config", {})
        return tc.get("model_type", mt)
    return ""


def _ensure_transformers(model_type):
    """转换前确保 transformers 版本符合要求，返回是否需要恢复。"""
    needed = _MODEL_TF_VERSION.get(model_type)
    if needed is None:
        return False  # 不需要切换

    # 用 subprocess 查当前版本，避免 import 缓存问题
    try:
        cur_ver = subprocess.check_output(
            [sys.executable, "-c", "import transformers; print(transformers.__version__)"],
            timeout=10, text=True,
        ).strip()
    except Exception:
        return False

    need_switch = False
    if needed.startswith("=="):
        need_switch = cur_ver != needed[2:]
    elif needed.startswith(">="):
        needed_ver = tuple(int(x) for x in needed[2:].split("."))
        cur_ver_t = tuple(int(x) for x in cur_ver.split("."))
        need_switch = cur_ver_t < needed_ver

    if not need_switch:
        return False

    print(f"  ⚡ {model_type} 需要 transformers {needed}（当前 {cur_ver}），临时切换...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
             f"transformers{needed}"],
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f"  \u274c {TR('pip 下载超时 (120s)，请检查网络或手动执行:', 'pip timed out (120s), check network or run:')}")
        print(f"    pip install transformers{needed}")
        sys.exit(1)
    new_ver = subprocess.check_output(
        [sys.executable, "-c", "import transformers; print(transformers.__version__)"],
        timeout=10, text=True,
    ).strip()
    print(f"  ✓ 已切换至 transformers {new_ver}")
    return True


def _get_pkg_version(pkg):
    """获取已安装包的版本，不存在返回 None。（用 pip list，避免 import 副作用）"""
    return _pip_get_version(pkg)


def _ensure_qwen_tts():
    """转换 Qwen3-TTS 前安装 qwen-tts，并修复 torchaudio CPU 兼容性。
    返回 (old_transformers, old_hf_hub) 用于恢复。"""
    old_tf = _get_pkg_version("transformers")
    old_hf = _get_pkg_version("huggingface_hub")
    print(f"  ⚡ 安装 qwen-tts（转换 Qwen3-TTS 所需）...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "qwen-tts"],
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        print(f"  \u274c pip 下载超时 (180s)，请检查网络或手动执行: pip install qwen-tts")
        sys.exit(1)
    # 修复 torchaudio：qwen-tts 可能带了 CUDA 版，强制换 CPU 版
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
             "torchaudio", "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pass  # torchaudio 非必需，超时无所谓
    new_tf = _get_pkg_version("transformers")
    if new_tf != old_tf:
        print(f"  ⚡ transformers 已从 {old_tf} 变为 {new_tf}，转换后恢复")
    return old_tf, old_hf


def _restore_qwen_transformers(old_tf, old_hf):
    """Qwen3-TTS 转换后恢复 transformers 和 huggingface_hub 到之前版本。"""
    # 先恢复 transformers（当前 huggingface_hub 还兼容当前 transformers）
    if old_tf:
        cur_tf = _pip_get_version("transformers")
        if cur_tf != old_tf:
            print(f"  ⚡ 恢复 transformers {old_tf}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
                     f"transformers=={old_tf}"],
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                print(f"  \u274c pip 下载超时，请手动执行: pip install transformers=={old_tf}")
                sys.exit(1)
    # 再恢复 huggingface_hub（此时 transformers 已恢复，版本兼容没问题）
    if old_hf:
        cur_hf = _pip_get_version("huggingface_hub")
        if cur_hf != old_hf:
            print(f"  ⚡ 恢复 huggingface_hub {old_hf}...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
                     f"huggingface_hub=={old_hf}"],
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                print(f"  \u274c pip 下载超时")
    # 用 pip list 查版本，避免 import 冲突
    tf_ver = _pip_get_version("transformers")
    hf_ver = _pip_get_version("huggingface_hub")
    print(f"  ✓ 已恢复: transformers={tf_ver}, huggingface_hub={hf_ver}")


def _pip_get_version(pkg):
    """用 pip list 查已安装包版本，不触发 import。"""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=columns"],
            timeout=10, text=True,
        )
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].lower().replace("-", "_") == pkg.lower().replace("-", "_"):
                return parts[1]
        return None
    except Exception:
        return None


def _restore_transformers():
    """转换完成后恢复 transformers 到最新版。"""
    try:
        cur_ver = subprocess.check_output(
            [sys.executable, "-c", "import transformers; print(transformers.__version__)"],
            timeout=10, text=True,
        ).strip()
    except Exception:
        return
    # 当前版本已 >= 5.9 则跳过（不需要降级过）
    parts = cur_ver.split(".")
    if len(parts) >= 2 and int(parts[0]) >= 6:
        return
    if len(parts) >= 2 and int(parts[0]) == 5 and int(parts[1]) >= 9:
        return
    print(f"  ⚡ 恢复 transformers 到最新版...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
             "transformers"],
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f"  \u274c {TR('pip 下载超时 (120s)，请检查网络或手动执行:', 'pip timed out (120s), check network or run:')}")
        print(f"    pip install transformers")
        sys.exit(1)
    new_ver = subprocess.check_output(
        [sys.executable, "-c", "import transformers; print(transformers.__version__)"],
        timeout=10, text=True,
    ).strip()
    print(f"  ✓ 已恢复至 transformers {new_ver}")


def _infer_task(model_path):
    """根据 config.json 推断 optimum 的 task 参数。"""
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        print(f"  {TR('错误: 找不到 config.json', 'Error: config.json not found')}")
        print(f"  {TR('请确认模型目录存在且包含 config.json:', 'Make sure the model directory has config.json:')}")
        print(f"    {model_path}")
        print(f"  {TR('提示:', 'Hint:')} {TR('如果该路径确实指向一个模型目录，请检查参数 --model', 'If the path is correct, check the --model argument')}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)
    mt = cfg.get("model_type", "")

    # 多模态模型（VLM）
    if mt in ("gemma4", "qwen3_vl", "qwen3_5", "llama4", "qwen3_6"):
        return "image-text-to-text"
    # 翻译模型（Hy-MT2 等）
    if mt in ("hunyuan_v1_dense",):
        return "text-generation-with-past"
    # 默认文本生成（带 KV cache）
    return "text-generation-with-past"


def _convert_qwen3_tts(model_path, output_path, weight_format):
    """用自定义 helper 转换 Qwen3-TTS 模型。"""
    import sys as _sys
    # 找 helper：优先 dlc/，再找 notebook 目录
    helper_dirs = [
        Path(__file__).parent.parent / "dlc",
        Path(__file__).parent.parent / "model/openvino_notebooks-latest/notebooks/qwen3-tts",
    ]
    helper = None
    for d in helper_dirs:
        p = d / "qwen_tts_helper.py"
        if p.exists():
            helper = p
            break
    if helper is None:
        print(f"  \u274c 找不到 qwen_tts_helper.py，请确认 dlc/ 或 notebooks 目录存在")
        sys.exit(1)

    _sys.path.insert(0, str(helper.parent))
    # 安装/确保 qwen-tts 依赖
    saved_tf, saved_hf = _ensure_qwen_tts()

    # 量化配置（Qwen3-TTS 只支持 fp32/fp16/int8/int4）
    _QWEN_TTS_QUANT = {
        "fp32": None,
        "fp16": None,
        "int8": '{"mode": "int8"}',
        "int4": '{"mode": "int4_sym", "ratio": 0.8, "group_size": 128}',
    }
    if weight_format not in _QWEN_TTS_QUANT:
        supported = ", ".join(_QWEN_TTS_QUANT.keys())
        print(f"  ⚠ Qwen3-TTS 不支持 {weight_format} 量化，支持: {supported}")
        print(f"  将使用 fp32（不量化）继续...")
        quant_config = "None"
    else:
        quant_config = _QWEN_TTS_QUANT[weight_format]

    # 执行转换
    helper_mod = helper.stem  # "qwen_tts_helper"
    print(f"  使用 helper: {helper}")
    print(f"  模型: {model_path}")
    print(f"  量化: {weight_format}")
    print(f"  输出: {output_path}")
    print()
    t0 = time.time()
    try:
        subprocess.check_call([
            sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{helper.parent}')
from {helper_mod} import convert_qwen3_tts_model
convert_qwen3_tts_model(
    model_id=r'{model_path}',
    output_dir=r'{output_path}',
    quantization_config={quant_config},
    use_local_dir=False,
)
"""], timeout=3600)
        print(f"\n  ✓ ({time.time()-t0:.1f}s)")
        # 统计
        total_mb = 0
        for f in Path(output_path).rglob("*.bin"):
            total_mb += f.stat().st_size
        print(f"  保存到: {output_path}  ({total_mb / 1024 / 1024:.0f} MB)")
    except subprocess.CalledProcessError:
        print(f"\n  ✗ 导出失败")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"\n  ✗ 导出超时 (60分钟)")
        sys.exit(1)
    finally:
        # 恢复 transformers 和 huggingface_hub
        _restore_qwen_transformers(saved_tf, saved_hf)


def _ensure_qwen_asr():
    """转换 Qwen3-ASR 前安装 qwen-asr，并修复 torchaudio CPU 兼容性。
    返回 (old_transformers, old_hf_hub)。"""
    old_tf = _get_pkg_version("transformers")
    old_hf = _get_pkg_version("huggingface_hub")
    print(f"  ⚡ 安装 qwen-asr（转换 Qwen3-ASR 所需）...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "qwen-asr"],
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        print(f"  \u274c pip 下载超时 (180s)，请检查网络或手动执行: pip install qwen-asr")
        sys.exit(1)
    # 修复 torchaudio
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
             "torchaudio", "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pass
    new_tf = _get_pkg_version("transformers")
    if new_tf != old_tf:
        print(f"  ⚡ transformers 已从 {old_tf} 变为 {new_tf}，转换后恢复")
    return old_tf, old_hf


def _convert_qwen3_asr(model_path, output_path, weight_format):
    """用自定义 helper 转换 Qwen3-ASR 模型。"""
    import sys as _sys
    helper_dirs = [
        Path(__file__).parent.parent / "dlc",
        Path(__file__).parent.parent / "model/openvino_notebooks-latest/notebooks/qwen3-asr",
    ]
    helper = None
    for d in helper_dirs:
        p = d / "qwen_3_asr_helper.py"
        if p.exists():
            helper = p
            break
    if helper is None:
        print(f"  \u274c 找不到 qwen_3_asr_helper.py，请确认 dlc/ 或 notebooks 目录存在")
        sys.exit(1)

    _sys.path.insert(0, str(helper.parent))
    # 安装/确保 qwen-asr 依赖
    saved_tf, saved_hf = _ensure_qwen_asr()

    # 量化配置（Qwen3-ASR 只支持 fp32/fp16/int8/int4）
    _QWEN_ASR_QUANT = {
        "fp32": None,
        "fp16": None,
        "int8": '{"mode": "int8"}',
        "int4": '{"mode": "int4_sym", "ratio": 0.8, "group_size": 128}',
    }
    if weight_format not in _QWEN_ASR_QUANT:
        supported = ", ".join(_QWEN_ASR_QUANT.keys())
        print(f"  ⚠ Qwen3-ASR 不支持 {weight_format} 量化，支持: {supported}")
        print(f"  将使用 fp32（不量化）继续...")
        quant_config = "None"
    else:
        quant_config = _QWEN_ASR_QUANT[weight_format]

    helper_mod = helper.stem
    print(f"  使用 helper: {helper}")
    print(f"  模型: {model_path}")
    print(f"  量化: {weight_format}")
    print(f"  输出: {output_path}")
    print()
    t0 = time.time()
    try:
        subprocess.check_call([
            sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{helper.parent}')
from {helper_mod} import convert_qwen3_asr_model
convert_qwen3_asr_model(
    model_id=r'{model_path}',
    output_dir=r'{output_path}',
    quantization_config={quant_config},
    use_local_dir=False,
)
"""], timeout=3600)
        print(f"\n  ✓ ({time.time()-t0:.1f}s)")
        total_mb = 0
        for f in Path(output_path).rglob("*.bin"):
            total_mb += f.stat().st_size
        print(f"  保存到: {output_path}  ({total_mb / 1024 / 1024:.0f} MB)")
    except subprocess.CalledProcessError:
        print(f"\n  ✗ 导出失败")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"\n  ✗ 导出超时 (60分钟)")
        sys.exit(1)
    finally:
        _restore_qwen_transformers(saved_tf, saved_hf)


def convert_model(model_path, output_path, weight_format,
                  ratio=1.0, group_size=128):
    """用 optimum-cli 导出模型。"""
    # 检测模型类型并确保 transformers 版本
    model_type = _detect_model_type(model_path)

    # Qwen3-TTS 走自定义转换路径
    if model_type == "qwen3_tts":
        return _convert_qwen3_tts(model_path, output_path, weight_format)
    if model_type == "qwen3_asr":
        return _convert_qwen3_asr(model_path, output_path, weight_format)

    needs_restore = _ensure_transformers(model_type)

    task = _infer_task(model_path)
    print(f"  任务类型: {task}")
    print(f"  量化格式: {weight_format}")
    if weight_format == "int4":
        print(f"  混合精度: ratio={ratio}, group_size={group_size}")
    print()

    wf_map = {"fp32": "fp32", "fp16": "fp16", "int8": "int8", "int4": "int4",
              "mxfp4": "mxfp4", "nf4": "nf4", "cb4": "cb4"}
    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli", "export", "openvino",
        "--model", model_path,
        "--task", task,
        "--weight-format", wf_map.get(weight_format, "fp32"),
        "--trust-remote-code",
        output_path,
    ]

    # INT4 混合精度参数
    if weight_format == "int4":
        cmd += ["--ratio", str(ratio), "--group-size", str(group_size)]

    print(f"  {' '.join(cmd)}")
    print()
    t0 = time.time()
    try:
        subprocess.check_call(cmd, timeout=1800)
        print(f"\n  ✓ ({time.time()-t0:.1f}s)")
        # 统计文件大小
        for xml_name in ["openvino_model.xml", "openvino_language_model.xml",
                         "openvino_text_embeddings_model.xml"]:
            xml_path = os.path.join(output_path, xml_name)
            if os.path.isfile(xml_path):
                bin_path = xml_path.replace(".xml", ".bin")
                bin_size = os.path.getsize(bin_path) / (1024 * 1024) if os.path.isfile(bin_path) else 0
                print(f"  保存到: {output_path}  ({bin_size:.0f} MB)")
                break
    except subprocess.CalledProcessError:
        print(f"\n  ✗ 导出失败")
        print(f"  提示: 该模型可能不被 optimum 支持，请检查模型类型或更新 optimum-intel")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"\n  ✗ 导出超时 (30分钟)")
        sys.exit(1)
    finally:
        # 恢复 transformers 版本（如果需要）
        if needs_restore:
            _restore_transformers()

    print("  完成!")
