"""
ov-cli yolo: YOLO 目标检测工具。

基于 Ultralytics YOLO + OpenVINO 后端。
支持 .pt 自动导出和 OpenVINO IR 直接推理。
"""

import os, sys, json, shutil
from pathlib import Path
from ultralytics import YOLO
from ov_cli import TR


# ── 路径解析 ──────────────────────────────────────────────

def _is_pt(path: str) -> bool:
    return path.endswith(".pt")


def _is_ir(path: str) -> bool:
    """判断路径是否为有效的 OpenVINO IR 目录（含 .xml 文件）。"""
    if os.path.isdir(path):
        return any(f.endswith(".xml") for f in os.listdir(path))
    if path.endswith(".xml"):
        return os.path.isfile(path)
    return False


def _get_ir_path(pt_path: str) -> str:
    """根据 .pt 路径计算默认的 OpenVINO IR 导出目录。"""
    d = os.path.dirname(pt_path)
    stem = os.path.splitext(os.path.basename(pt_path))[0]
    return os.path.join(d, f"{stem}_openvino_model")


def _ir_is_valid(ir_dir: str) -> bool:
    """检查 IR 目录是否包含必要的文件。"""
    if not os.path.isdir(ir_dir):
        return False
    has_xml = any(f.endswith(".xml") for f in os.listdir(ir_dir))
    has_bin = any(f.endswith(".bin") for f in os.listdir(ir_dir))
    return has_xml and has_bin


def resolve_model(model_arg: str) -> str:
    """解析 --model 参数，返回可传给 YOLO() 的路径。

    - .pt 文件：检查旁边 {stem}_openvino_model/，有则返回 IR，无则导出
    - IR 目录：直接返回
    - 其他：报错
    """
    model_path = os.path.abspath(model_arg)

    if _is_ir(model_path):
        return model_path

    if _is_pt(model_path):
        ir_dir = _get_ir_path(model_path)
        if _ir_is_valid(ir_dir):
            print(f"  {TR('✓ 加载已导出的 IR:', '✓ Loading cached IR:')} {ir_dir}")
            return ir_dir
        # 导出 .pt → IR
        print(f"  {TR('⚡ 导出 OpenVINO IR...', '⚡ Exporting to OpenVINO IR...')}")
        model = YOLO(model_path)
        result = model.export(format="openvino", imgsz=640, nms=True, dynamic=False)
        print(f"  {TR('  ✓ 导出完成:', '  ✓ Export done:')} {result}")
        if _ir_is_valid(ir_dir):
            return ir_dir
        # 如果 ultralytics 导出到其他地方（如缓存目录），尝试定位
        export_path = result.strip()
        if os.path.isdir(export_path) and _ir_is_valid(export_path):
            return export_path
        print(f"  ⚠ {TR('导出后找不到 IR 目录', 'Export completed but IR dir not found')}: {ir_dir}")
        sys.exit(1)

    print(f"  ⚠ {TR('不支持的模型格式，请使用 .pt 或 OpenVINO IR 目录', 'Unsupported model format, use .pt or OpenVINO IR directory')}")
    print(f"     {TR('示例:', 'Example:')} --model yolo11n.pt {TR('或', 'or')} --model yolo11n_openvino_model/")
    sys.exit(1)


# ── 设备名映射 ────────────────────────────────────────────

def _map_device(device: str) -> str:
    """将 ov-cli 设备名映射到 ultralytics 格式。

    CPU     → intel:cpu
    GPU     → intel:gpu
    GPU.N   → intel:gpu.N
    NPU     → intel:npu
    其他    → 原样传递
    """
    if not device:
        return ""
    d = device.upper().strip()
    if d == "CPU":
        return "intel:cpu"
    if d.startswith("GPU"):
        # GPU.0 → intel:gpu.0,  GPU → intel:gpu
        return "intel:" + d.lower().replace("gpu", "gpu")
    if d == "NPU":
        return "intel:npu"
    # 原样传递
    return device


# ── 推理 ──────────────────────────────────────────────────

def _read_task(ir_dir: str) -> str | None:
    """从 metadata.yaml 读取 task 类型。"""
    meta = os.path.join(ir_dir, "metadata.yaml")
    if not os.path.isfile(meta):
        return None
    try:
        with open(meta) as f:
            for line in f:
                line = line.strip()
                if line.startswith("task:"):
                    return line.split(":", 1)[1].strip().strip("\"'")
    except Exception:
        pass
    return None


def _parse_classes(classes_str: str, names: dict[int, str] | None = None) -> list[int]:
    """解析 --classes 参数为 class ID 列表。

    支持格式:
      "0,2,5"    → [0, 2, 5]  (数字 ID)
      "person,car" → 通过 names 映射为 ID（需要加载模型后调用）
    """
    if not classes_str:
        return []
    parts = [p.strip() for p in classes_str.split(",") if p.strip()]
    # 按数字解析
    ids = []
    unknown_names = []
    for p in parts:
        if p.isdigit():
            ids.append(int(p))
        else:
            unknown_names.append(p)

    # 如果有 name → ID 映射且有无数字的部分，尝试映射
    if names and unknown_names:
        name_to_id = {v.lower(): k for k, v in names.items()}
        for name in unknown_names:
            nl = name.lower()
            if nl in name_to_id:
                ids.append(name_to_id[nl])
            else:
                print(f"  ⚠ {TR(f'忽略未知类别:', f'Ignoring unknown class:')} {name}", file=sys.stderr)

    return sorted(set(ids))


_SUPPORTED_IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
})


def _find_images(path: str) -> list[str]:
    """递归查找目录下所有支持的图片文件。"""
    imgs = []
    if os.path.isfile(path):
        return [path]
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in _SUPPORTED_IMAGE_EXTS:
                imgs.append(os.path.join(root, f))
    return imgs


def _print_detections(detections: list[dict], file=sys.stdout):
    """终端输出检测结果。"""
    for d in detections:
        bbox = d["bbox"]
        print(f"    {d['label']:20s}  {d['confidence']:.3f}  "
              f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]", file=file)


def _build_detections(result, names: dict) -> list[dict]:
    """从 ultralytics 结果构建结构化检测列表。"""
    boxes = result.boxes
    if boxes is None:
        return []
    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        conf_val = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        label = names.get(cls_id, str(cls_id))
        detections.append({
            "class_id": cls_id,
            "label": label,
            "confidence": round(conf_val, 4),
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        })
    return detections


def run_once(model_path: str, image_path: str, *,
             device: str = "",
             output: str | None = None,
             json_output: bool = False,
             imgsz: int = 640,
             conf: float = 0.25,
             iou: float = 0.45,
             classes_str: str | None = None):
    """单次推理：加载模型 → 检测 → 输出。

    Args:
        classes_str: 只检测指定类别，如 "person,car" 或 "0,5"，None 表示全部。
    """
    model_arg = resolve_model(model_path)
    ult_device = _map_device(device) if device else ""

    # 加载模型（从 metadata.yaml 读取 task，避免 ultralytics 的 WARNING）
    task = _read_task(model_arg) if os.path.isdir(model_arg) else None
    print(f"  {TR('加载模型...', 'Loading model...')}", file=sys.stderr)
    model = YOLO(model_arg, task=task) if task else YOLO(model_arg)

    # 解析类别过滤（需要模型加载后的 names）
    classes = _parse_classes(classes_str, model.names) if classes_str else None

    # 推理
    kwargs = dict(device=ult_device, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    if classes is not None:
        kwargs["classes"] = classes
    print(f"  {TR('推理中...', 'Running inference...')}", file=sys.stderr)
    results = model(image_path, **kwargs)

    result = results[0]
    names = result.names
    detections = _build_detections(result, names)

    print(f"\n  {TR('检测结果:', 'Detections:')} {len(detections)}", file=sys.stderr)

    # 保存画框图
    if output and not json_output:
        print(f"  {TR('保存结果:', 'Saving result:')} {output}", file=sys.stderr)
        annotated = result.plot()
        from PIL import Image
        Image.fromarray(annotated[..., ::-1]).save(output)

    # JSON 输出
    if json_output:
        print(json.dumps(detections, ensure_ascii=False))
        return

    # 终端输出
    _print_detections(detections)


def run_batch(model_path: str, input_dir: str, *,
              device: str = "",
              output_dir: str | None = None,
              json_output: bool = False,
              imgsz: int = 640,
              conf: float = 0.25,
              iou: float = 0.45,
              classes_str: str | None = None):
    """批量处理目录下所有图片。"""
    images = _find_images(input_dir)
    if not images:
        print(f"  ⚠ {TR('目录中未找到图片', 'No images found in')}: {input_dir}", file=sys.stderr)
        return

    print(f"  {TR('找到', 'Found')} {len(images)} {TR('张图片', 'images')}", file=sys.stderr)

    model_arg = resolve_model(model_path)
    ult_device = _map_device(device) if device else ""

    # 加载模型
    task = _read_task(model_arg) if os.path.isdir(model_arg) else None
    print(f"  {TR('加载模型...', 'Loading model...')}", file=sys.stderr)
    model = YOLO(model_arg, task=task) if task else YOLO(model_arg)

    # 解析类别过滤
    classes = _parse_classes(classes_str, model.names) if classes_str else None

    # 推理参数（逐张处理，因为 OV IR 是静态 batch=1）
    pred_kwargs = dict(device=ult_device, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    if classes is not None:
        pred_kwargs["classes"] = classes

    print(f"  {TR('批量推理中...', 'Batch inference...')}", file=sys.stderr)

    all_detections = {}
    for i, img_path in enumerate(images):
        if json_output:
            print(f"  [{i+1}/{len(images)}] {os.path.basename(img_path)}", file=sys.stderr)
        result = model(img_path, **pred_kwargs)[0]
        names = result.names
        detections = _build_detections(result, names)
        all_detections[img_path] = detections

        # 保存画框图
        rel = os.path.relpath(img_path, input_dir)
        if output_dir and not json_output:
            out_path = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            annotated = result.plot()
            from PIL import Image
            Image.fromarray(annotated[..., ::-1]).save(out_path)

    # 汇总输出
    total = sum(len(d) for d in all_detections.values())
    print(f"\n  {TR('处理完成:', 'Done:')} {len(images)} {TR('张, 共', 'images,')} {total} {TR('个检测', 'detections')}", file=sys.stderr)

    if json_output:
        # 输出 { 文件名: [检测列表], ... }
        result_json = {
            os.path.relpath(k, input_dir): v
            for k, v in all_detections.items()
            if v
        }
        print(json.dumps(result_json, ensure_ascii=False))
        return

    # 终端输出每张图的结果
    for img_path, detections in all_detections.items():
        rel = os.path.relpath(img_path, input_dir)
        if detections:
            print(f"\n  {rel}: {len(detections)} {TR('个检测', 'detections')}")
            _print_detections(detections)
        else:
            print(f"\n  {rel}: \u2014")
