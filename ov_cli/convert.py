"""
ov-cli convert: 模型转换。

使用 Optimum Intel 官方工具将 HuggingFace 模型导出为 OpenVINO IR 格式。
支持 INT8/INT4 量化，自动推断 task 类型（text-generation / image-text-to-text）。
"""

import os, sys, time, json, subprocess


def _infer_task(model_path):
    """根据 config.json 推断 optimum 的 task 参数。"""
    with open(os.path.join(model_path, "config.json")) as f:
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
    task = _infer_task(model_path)
    print(f"  任务类型: {task}")
    print(f"  量化格式: {weight_format}")
    if weight_format == "int4":
        print(f"  混合精度: ratio={ratio}, group_size={group_size}")
    print()

    wf_map = {"fp32": "fp32", "fp16": "fp16", "int8": "int8", "int4": "int4"}
    cmd = [
        "optimum-cli", "export", "openvino",
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

    print("  完成!")
