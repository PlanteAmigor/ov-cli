"""
ov-cli convert: 模型转换。

使用 Optimum Intel 官方工具将 HuggingFace 模型导出为 OpenVINO IR 格式。
支持 INT8/INT4/NF4/MXFP4/CB4 等多种量化，自动推断 task 类型。
"""

import os, sys, time, json, subprocess
from ov_cli import TR

# 模型类型 → 转换所需 transformers 版本（None = 使用当前版本）
# 推理统一用 5.9，转换完成后自动恢复
_MODEL_TF_VERSION = {
    "gemma4":   ">=5.9",        # Gemma-4 需要高版本 transformers + 补丁
    "qwen3_5":  "==5.2",        # Qwen3.5 必须 5.2（否则 DynamicCache 兼容问题）
    "qwen3_6":  "==5.2",        # Qwen3.6 MoE 同 Qwen3.5
}


def _detect_model_type(model_path):
    """从 config.json 读取 model_type。"""
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
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
        need_switch = cur_ver_t < needed_ver_t

    if not need_switch:
        return False

    print(f"  ⚡ {model_type} 需要 transformers {needed}（当前 {cur_ver}），临时切换...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
         f"transformers{needed}"],
        timeout=120,
    )
    new_ver = subprocess.check_output(
        [sys.executable, "-c", "import transformers; print(transformers.__version__)"],
        timeout=10, text=True,
    ).strip()
    print(f"  ✓ 已切换至 transformers {new_ver}")
    return True


def _restore_transformers():
    """转换完成后恢复 transformers 到 5.9。"""
    try:
        cur_ver = subprocess.check_output(
            [sys.executable, "-c", "import transformers; print(transformers.__version__)"],
            timeout=10, text=True,
        ).strip()
    except Exception:
        return
    if cur_ver.startswith("5.9") or cur_ver.startswith("5.") and int(cur_ver.split(".")[1]) >= 9:
        return
    print(f"  ⚡ 恢复 transformers 5.9...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
         "transformers>=5.9"],
        timeout=120,
    )
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


def convert_model(model_path, output_path, weight_format,
                  ratio=1.0, group_size=128):
    """用 optimum-cli 导出模型。"""
    # 检查模型目录
    if not os.path.isdir(model_path):
        print(f"  {TR('错误: 模型目录不存在', 'Error: model directory not found')}")
        print(f"    {model_path}")
        print(f"  {TR('请检查 --model 参数', 'Please check the --model argument')}")
        sys.exit(1)

    # 检测模型类型并确保 transformers 版本
    model_type = _detect_model_type(model_path)
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
