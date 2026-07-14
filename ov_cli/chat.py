"""
ov-cli chat: LLM 聊天/翻译终端。

使用 OpenVINO GenAI LLMPipeline/VLMPipeline。
"""

import os, sys, time, json, re, signal, threading
import readline  # 激活 input() 的 ←→↑↓ + 历史记录
import openvino as ov
import openvino_genai as ov_genai


def _make_streamer(reply_parts, stop_flag, on_first_token=None, thinking_filter=False):
    """创建 streamer callback。

    on_first_token: 首个 token 到达时的回调（用于停止进度指示器）。
    thinking_filter: 是否过滤 <think> 标签及思考内容。
    """
    _first = [True]
    # 启用 filter 时，初始假设在 think 块内（模型可能先输出思考才输出 </think>）
    in_think = [thinking_filter]

    def cb(t):
        if stop_flag[0]:
            return True
        if _first[0] and on_first_token:
            _first[0] = False
            on_first_token()

        if thinking_filter:
            if in_think[0]:
                # 在 think 块内：只找 </think>
                if '</think>' in t:
                    idx = t.index('</think>')
                    after = t[idx + 8:]
                    if after:
                        reply_parts.append(after)
                        sys.stdout.write(after)
                    in_think[0] = False
                # 否则丢弃（思考内容）
            else:
                if '<think>' in t:
                    # 进入 think 块，丢弃前面的内容
                    idx = t.index('<think>')
                    after = t[idx + 7:]
                    if '</think>' in after:
                        idx2 = after.index('</think>')
                        rest = after[idx2 + 8:]
                        if rest:
                            reply_parts.append(rest)
                            sys.stdout.write(rest)
                        # 同一块内打开了又关闭，不改变状态
                    else:
                        in_think[0] = True
                else:
                    reply_parts.append(t)
                    sys.stdout.write(t)
            sys.stdout.flush()
        else:
            reply_parts.append(t)
            sys.stdout.write(t)
            sys.stdout.flush()
        return False

    return cb


def _is_genai_format(model_path):
    """检测模型目录是否为 OpenVINO GenAI 格式。"""
    return (os.path.isfile(os.path.join(model_path, "openvino_config.json")) or
            os.path.isfile(os.path.join(model_path, "openvino_model.xml")))



def _is_multimodal(model_path):
    """检测模型是否包含视觉组件。"""
    return os.path.isfile(os.path.join(model_path, "openvino_vision_embeddings_model.xml"))


def load_model(ov_path, device=""):
    """加载 OpenVINO 模型。自动检测 GenAI/传统格式。"""
    if not device:
        devices = ov.Core().available_devices
        device = next((d for d in devices if "GPU" in d), "CPU")

    if _is_genai_format(ov_path):
        is_vlm = _is_multimodal(ov_path)
        tag = "VLM" if is_vlm else "LLM"
        print(f"  加载 {tag}Pipeline ({device})...", end=" ", flush=True, file=sys.stderr)
        t0 = time.time()
        if is_vlm:
            pipe = ov_genai.VLMPipeline(ov_path, device)
        else:
            pipe = ov_genai.LLMPipeline(ov_path, device)
        print(f"✓ ({time.time()-t0:.1f}s)", file=sys.stderr)

        # 从 config.json 读 model_type
        model_type = None
        cfg_path = os.path.join(ov_path, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type")

        return {
            "pipe": pipe,
            "device": device,
            "model_type": model_type,
            "genai": True,
            "is_vlm": is_vlm,
        }




def _make_genai_config(temperature=0.7, top_p=0.9, top_k=40, max_tokens=0, presence_penalty=None):
    """创建 GenAI GenerationConfig。
    
    max_tokens=0 表示不限制，由模型自行决定何时输出 EOS 结束。
    """
    cfg = ov_genai.GenerationConfig()
    if max_tokens > 0:
        cfg.max_new_tokens = max_tokens
    cfg.temperature = temperature
    cfg.top_p = top_p
    cfg.top_k = top_k
    cfg.do_sample = temperature >= 0.01
    if presence_penalty is not None:
        cfg.presence_penalty = presence_penalty
    return cfg


# ── 管道模式 ────────────────────────────────────────────

def run_pipe(ctx, max_tokens=0, temperature=0.7):
    """管道模式：从 stdin 读提示词，向 stdout 写 JSON 结果。"""
    import json as _json
    pipe = ctx.get("pipe")
    is_vlm = ctx.get("is_vlm", False)
    from . import TR as _TR

    print(f"  🧪 {_TR('管道模式已启动 (stdin/stdout)', 'Pipe mode started (stdin/stdout)')}", file=sys.stderr)
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            prompt = line.strip()
            if not prompt:
                continue

            conv = [{"role": "user", "content": prompt}]
            full = _build_prompt(conv, pipe.get_tokenizer(), enable_thinking=True)

            cfg = ov_genai.GenerationConfig(max_new_tokens=max_tokens, temperature=temperature)
            cfg.do_sample = temperature >= 0.01

            t0 = time.time()
            try:
                if is_vlm:
                    result = pipe.generate(full, generation_config=cfg, images=[])
                else:
                    result = pipe.generate(full, cfg)
            except Exception as e:
                print(_json.dumps({"error": str(e)[:200]}, ensure_ascii=False), flush=True)
                continue

            elapsed = time.time() - t0
            resp = str(result).strip()
            print(_json.dumps({"text": resp, "time": round(elapsed, 1)}, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        pass


# ── 行编辑器（简单版本） ──────────────────────────────


def has_chinese(text):
    return any('\u4e00' <= c <= '\u9fff' for c in text[:30])


def readline():
    try:
        return input(">>> ")
    except EOFError:
        return ""


def _count_tokens(ctx, text):
    if not text:
        return 0
    try:
        r = ctx["pipe"].get_tokenizer().encode(text)
        return r.input_ids.shape[-1]
    except Exception:
        return 0


def run_once(ctx, prompt="", files=None, output=None,
             temperature=0.7, top_p=0.9, top_k=40, max_tokens=0,
             json_output=False):
    """单次输出模式：读取文件 + 文字，一次生成，输出后退出。"""
    import numpy as np
    import json as _json
    from . import TR

    is_vlm = ctx.get("is_vlm", False)
    pipe = ctx.get("pipe")
    processor = ctx.get("processor")

    # 收集所有输入
    all_pages = []
    text_parts = []

    if files:
        for fpath in files:
            fpath = os.path.abspath(fpath)
            if not os.path.isfile(fpath):
                print(f"  ⚠ {TR('找不到文件', 'File not found')}: {fpath}")
                continue
            ext = os.path.splitext(fpath)[1].lower()
            if ext == ".pdf" and is_vlm:
                print(f"  ⚠ {TR('PDF 暂时禁用—见 #36386', 'PDF temporarily disabled—see #36386')} (https://github.com/openvinotoolkit/openvino/issues/36386)")
            elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp") and is_vlm:
                from PIL import Image
                all_pages.append(_load_image(fpath))
                print(f"  ✓ {TR('已加载图片', 'Image loaded')}: {fpath}")
            elif ext in _TEXT_EXTENSIONS:
                text_parts.append(f"[{os.path.basename(fpath)}]\n```\n{_load_text_file(fpath)}\n```")
                print(f"  ✓ {TR('已加载文件', 'File loaded')}: {fpath}")
            else:
                print(f"  ⚠ {TR('不支持的文件类型', 'Unsupported file type')}: {fpath}")

    # 合并 prompt
    user_text = prompt
    if text_parts:
        prefix = "\n\n".join(text_parts)
        user_text = prefix + "\n\n" + user_text if user_text else prefix

    messages = [{"role": "user", "content": user_text}]

    # GenAI 路径
    img_tag = "<|vision_start|><|image_pad|><|vision_end|>\n"
    if all_pages:
        user_text = img_tag * len(all_pages) + user_text
        messages[0]["content"] = user_text

    tokenizer = pipe.get_tokenizer()
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            extra_context={"enable_thinking": True})
    except Exception:
        prompt_text = f"<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n"

    gen_cfg = _make_genai_config(temperature, top_p, top_k, max_tokens)

    # VLM 预编码进度
    n_vis = len(all_pages)
    progress_stop = threading.Event()
    if is_vlm and all_pages:
        _t_prefill = time.time()
        def _on_first():
            progress_stop.set()
            pt = time.time() - _t_prefill
            print(f"\r  ✓ {TR('视觉编码 + prefill 完成', 'Vision + prefill done')} ({pt:.1f}s, ~{n_vis})  ")
            print(f"  {TR('回复', 'Reply')}:", end=" ", flush=True)
        on_first = _on_first
        def _prog():
            while not progress_stop.is_set():
                el = time.time() - _t_prefill
                print(f"\r  ⏳ {TR('正在处理', 'Processing')} {n_vis} {TR('张图', 'images')}... ({el:.0f}s)", end="", flush=True)
                progress_stop.wait(1.0)
        threading.Thread(target=_prog, daemon=True).start()
    else:
        on_first = None

    # 构建 tensors
    image_tensors = [ov.Tensor(np.array(img)[None]) for img in all_pages] if is_vlm and all_pages else None

    reply_parts = []
    stop_flag = [False]
    streamer_cb = _make_streamer(reply_parts, stop_flag, on_first, thinking_filter=False)

    kwargs = {"generation_config": gen_cfg, "streamer": streamer_cb}
    if image_tensors is not None:
        kwargs["images"] = image_tensors

    t0 = time.time()
    try:
        pipe.generate(prompt_text, **kwargs)
    except RuntimeError as e:
        print(f"\n  ⚠ {TR('生成失败', 'Generation failed')}: {str(e)[:200]}")
        sys.exit(1)
    finally:
        if not progress_stop.is_set():
            progress_stop.set()

    reply_text = "".join(reply_parts)

    # 输出统计
    elapsed = time.time() - t0
    char_count = len(reply_text.replace(" ", ""))
    print(f"\n  [{elapsed:.1f}s | {char_count} chars | {char_count/elapsed:.1f} ch/s]", file=sys.stderr)

    # 保存到文件
    if output:
        out_path = output
        if os.path.isdir(out_path) or out_path.endswith(os.sep):
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_path, f"{ts}.md")
        meta_parts = [f"mode: once | {time.strftime('%Y-%m-%d %H:%M:%S')}"]
        if prompt:
            meta_parts.append(f"prompt: {prompt}")
        if files:
            meta_parts.append(f"files: {', '.join(files)}")
        meta = f"<!-- ov-cli | {' | '.join(meta_parts)} -->\n\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(meta + reply_text)
        print(f"  💾 {TR('已保存', 'Saved')}: {out_path}", file=sys.stderr)

    # stdout: 纯结果 / JSON
    if json_output:
        print(_json.dumps({"text": reply_text, "tokens": 0, "time": round(elapsed, 1)}, ensure_ascii=False))
    else:
        print(reply_text)


def run_chat(ctx, system="You are a helpful AI assistant.",
             temperature=0.7, top_p=0.9, top_k=40, max_tokens=0,
             image_path=None):
    """通用聊天模式"""
    from . import TR

    print()
    print("        ██████╗ ██╗   ██╗     ██████╗██╗     ██╗")
    print("       ██╔═══██╗██║   ██║    ██╔════╝██║     ██║")
    print("       ██║   ██║██║   ██║    ██║     ██║     ██║")
    print("       ██║   ██║╚██╗ ██╔╝    ██║     ██║     ██║")
    print("       ╚██████╔╝ ╚████╔╝     ╚██████╗███████╗██║")
    print("        ╚═════╝   ╚═══╝       ╚═════╝╚══════╝╚═╝")
    print("=" * 50)
    print("  ov-cli " + TR("聊天终端", "Chat Terminal"))
    print(f"  {TR('设备', 'Device')}: {ctx['device']} | OpenVINO")
    print("=" * 50)
    if ctx.get("is_vlm"):
        print("  //img PATH  " + TR("加载图片", "load image"))
    print("  //txt PATH  " + TR("加载文本文件", "load text file"))
    print("  /temp N     " + TR("温度 (0-2)", "temperature"))
    print("  /system T   " + TR("系统提示词", "system prompt"))
    print("  /help       " + TR("帮助", "help"))
    print("  /exit       " + TR("退出", "quit"))
    print("=" * 50)
    print()

    _run_chat_genai(ctx, system, temperature, top_p, top_k, max_tokens, image_path)


def _build_prompt(messages, tokenizer=None, enable_thinking=True, tools=None):
    """将消息列表转为纯文本 prompt。

    优先使用模型的 chat template（通过 tokenizer.apply_chat_template），
    回退到手动构建。
    """
    if tokenizer is not None:
        try:
            kwargs = {
                "add_generation_prompt": True,
                "extra_context": {"enable_thinking": enable_thinking},
            }
            if tools is not None:
                kwargs["tools"] = tools
            prompt = tokenizer.apply_chat_template(messages, **kwargs)
            # 去掉空的 <think>\n\n</think>\n\n（仅当 thinking 关闭时的空壳）
            prompt = prompt.replace("<think>\n\n</think>\n\n", "")
            prompt = prompt.replace("<think>\n</think>\n\n", "")
            return prompt
        except Exception:
            pass
    # 回退：手动构建 ChatML（通用兜底）
    prompt = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            prompt += f"<|im_start|>system\n{content}\n<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}\n<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def _load_image(path, max_pixels=1024*1024):
    """加载图片为 PIL Image，缩放到像素预算内。"""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        ratio = (max_pixels / (w * h)) ** 0.5
        w, h = int(w * ratio), int(h * ratio)
    w = max(32, (w // 32) * 32)
    h = max(32, (h // 32) * 32)
    return img.resize((w, h))


# ── 文件加载辅助 ──────────────────────────────────────────

_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".json", ".py", ".c", ".cpp", ".h", ".hpp",
    ".yaml", ".yml", ".toml", ".xml", ".csv", ".sh", ".env",
    ".conf", ".cfg", ".ini", ".log", ".rst", ".tex", ".sql",
    ".js", ".ts", ".tsx", ".jsx", ".vue", ".css", ".scss",
    ".go", ".rs", ".java", ".kt", ".swift", ".rb", ".php",
    ".pl", ".lua", ".r", ".m", ".mm",
})


def _is_text_file(path):
    """判断文件是否可直接读为文本。"""
    ext = os.path.splitext(path)[1].lower()
    return ext in _TEXT_EXTENSIONS


def _load_text_file(path):
    """读取文本文件内容。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _get_gpu_driver():
    """检测当前 GPU 使用的内核驱动。

    Returns:
        "i915" | "xe" | None
    """
    import glob
    try:
        for card in glob.glob("/sys/class/drm/card*"):
            dev = os.path.join(card, "device", "driver", "module")
            if os.path.isdir(os.path.join(card, "device")):
                link = os.path.join(card, "device", "driver")
                if os.path.islink(link):
                    driver = os.path.basename(os.readlink(link))
                    if driver in ("i915", "xe"):
                        return driver
    except Exception:
        pass
    return None


def _pdf_to_images(path):
    """把 PDF 每页转成 PIL Image，返回列表。

    i915 驱动下 PDF 超 20 页会触发 GPU fence timeout，返回 None 并提示切换到 Xe。
    Xe 驱动或 CPU 下无限制，使用 300 DPI 高清晰度渲染。
    """
    from . import TR
    try:
        import fitz
    except ImportError:
        print("  ⚠ 需要安装 PyMuPDF: pip install PyMuPDF")
        return None

    import numpy as np
    from PIL import Image

    # 检测 GPU 驱动
    driver = _get_gpu_driver()
    if driver == "i915":
        print(f"  \u26a0 {TR('\u5f53\u524d\u4f7f\u7528 i915 \u9a71\u52a8\uff0cPDF \u591a\u9875\u7f16\u7801\u4f1a\u89e6\u53d1 GPU fence timeout', 'i915 driver detected: multi-page PDF encoding triggers GPU fence timeout')}")
        print(f"    {TR('\u9700\u8981\u5728\u5f15\u5bfc\u65f6\u5207\u6362\u5230 Xe \u9a71\u52a8\uff08\u5982\u8bbe\u7f6e initramfs \u6216\u5185\u6838\u53c2\u6570\uff09', 'Switch to Xe driver at boot time (initramfs or kernel parameter)')}")
        return None

    # 屏蔽 MuPDF 的 C 层 + Python 层 stderr 警告
    old_stderr_fd = os.dup(2)
    old_sys_stderr = sys.stderr
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    sys.stderr = None
    try:
        doc = fitz.open(path)
    except Exception:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        sys.stderr = old_sys_stderr
        raise
    total = len(doc)
    # 统一 300 DPI 高清渲染，448px 截断
    max_pixels = 1024 * 1024
    dpi = 300
    px = 1024
    tok_per_page = max(1, max_pixels // (32 * 32))
    total_tokens = tok_per_page * total
    images = []
    print(f"\r  \U0001f4c4 {os.path.basename(path)}: {total} \u9875 ({px}px, ~{tok_per_page} tok/\u9875, ~{total_tokens} tok \u5408\u8ba1)", end="", flush=True)
    try:
        for i in range(total):
            pix = doc[i].get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            w, h = img.size
            cur_pixels = w * h
            if cur_pixels > max_pixels:
                ratio = (max_pixels / cur_pixels) ** 0.5
                w, h = int(w * ratio), int(h * ratio)
            w = max(32, (w // 32) * 32)
            h = max(32, (h // 32) * 32)
            img = img.resize((w, h))
            images.append(img)
            # \u5355\u884c\u52a8\u6001\u66f4\u65b0
            print(f"\r  \U0001f4c4 {os.path.basename(path)}: \u6b63\u5728\u8f6c\u6362  {i+1}/{total} \u9875", end="", flush=True)
    finally:
        doc.close()
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        sys.stderr = old_sys_stderr
    print()
    return images


def _images_to_ov_tensor(images):
    """多张图片垂直拼接为一张后转为 openvino.Tensor。"""
    import numpy as np
    import openvino as ov
    # 找出最大宽度，所有图缩放到同宽
    max_w = max(img.width for img in images)
    resized = []
    for img in images:
        if img.width != max_w:
            ratio = max_w / img.width
            new_h = int(img.height * ratio)
            img = img.resize((max_w, new_h))
        resized.append(img)
    total_h = sum(img.height for img in resized)
    canvas = np.zeros((total_h, max_w, 3), dtype=np.uint8)
    y = 0
    for img in resized:
        arr = np.array(img)
        h = arr.shape[0]
        canvas[y:y+h] = arr
        y += h
    return ov.Tensor(canvas[None])


# ── GenAI 聊天模式 ──────────────────────────────────────

def _run_chat_genai(ctx, system, temperature, top_p, top_k, max_tokens, image_path=None):
    """GenAI 格式聊天模式。"""
    pipe = ctx["pipe"]
    is_vlm = ctx.get("is_vlm", False)
    from . import TR

    # 文件管理: {id, path, type, pages:[PIL]}
    loaded_files = []
    _next_id = 1

    # 预加载初始图片
    if image_path and is_vlm:
        if os.path.isfile(image_path):
            loaded_files.append({"id": _next_id, "path": image_path, "type": "image", "pages": [_load_image(image_path)]})
            print(f"  {TR('已加载图片', 'Image loaded')}: {image_path}")
            _next_id += 1
        else:
            print(f"  ⚠ {TR('找不到图片', 'Image not found')}: {image_path}")

    conv = []
    while True:
        try:
            text = readline()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue

        if text in ("/exit", "exit"):
            break
        if text == "/help":
            if is_vlm:
                print("  //img PATH   " + TR("加载图片", "load image"))
            print("  //txt PATH   " + TR("加载文本文件", "load text file"))
            print("  /temp N      " + TR("温度 (0-2)", "temperature"))
            print("  /system T    " + TR("系统提示词", "system prompt"))
            print("  /help        " + TR("帮助", "help"))
            print("  /exit        " + TR("退出", "quit"))
            print()
            continue
        if text.startswith("/temp "):
            try:
                temperature = max(0, min(2, float(text[6:])))
                print(f"  temperature = {temperature}")
            except:
                print("  ⚠ /temp 0.7")
            print()
            continue
        if text.startswith("/system "):
            system = text[8:]
            print(f"  {TR('系统提示词已更新', 'System prompt updated')}")
            print()
            continue
        if text.startswith("//img ") and is_vlm:
            import shlex
            paths = shlex.split(text[6:])
            if not paths:
                print(f"  ⚠ {TR('用法', 'Usage')}: //img PATH1 [PATH2 ...]")
                print()
                continue
            for img_path in paths:
                if os.path.isfile(img_path):
                    loaded_files.append({"id": _next_id, "path": img_path, "type": "image", "pages": [_load_image(img_path)]})
                    print(f"  #{_next_id} {TR('已加载图片', 'Image loaded')}: {img_path}")
                    _next_id += 1
                else:
                    print(f"  ⚠ {TR('找不到图片', 'Image not found')}: {img_path}")
            print()
            continue

        if text.startswith("//pdf ") and is_vlm:
            print(f"  ⚠ {TR('PDF 暂时禁用—见 #36386', 'PDF temporarily disabled—see #36386')} (https://github.com/openvinotoolkit/openvino/issues/36386)")
            print()
            continue

        if text.startswith("//txt "):
            import shlex
            paths = shlex.split(text[6:])
            if not paths:
                print(f"  ⚠ {TR('用法', 'Usage')}: //txt PATH1 [PATH2 ...]")
                print()
                continue
            for txt_path in paths:
                if os.path.isfile(txt_path):
                    file_content = _load_text_file(txt_path)
                    loaded_files.append({"id": _next_id, "path": txt_path, "type": "text", "content": file_content})
                    print(f"  #{_next_id} {TR('已加载文件', 'File loaded')}: {txt_path}")
                    _next_id += 1
                else:
                    print(f"  ⚠ {TR('找不到文件', 'File not found')}: {txt_path}")
            print()
            continue

        # 构建消息：合并已加载文件与当前输入
        all_pages = []
        txt_prefix = ""
        for f in loaded_files:
            if f["type"] == "text":
                fname = os.path.basename(f["path"])
                txt_prefix += f"[文件 {fname}]\n```\n{f['content']}\n```\n\n"
            else:
                all_pages.extend(f["pages"])
        # 为每张图片插入模型对应的图片占位符
        if all_pages:
            img_tag = "<|vision_start|><|image_pad|><|vision_end|>\n"
            text = img_tag * len(all_pages) + text
        if txt_prefix:
            text = txt_prefix + text
        loaded_files.clear()
        conv.append({"role": "user", "content": text})
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(conv)

        # 合并所有文件的页面为多张独立图片
        import numpy as np
        image_tensors = [ov.Tensor(np.array(img)[None]) for img in all_pages] if all_pages and is_vlm else None

        gen_cfg = _make_genai_config(temperature, top_p, top_k, max_tokens)

        # VLM prefill 进度指示器
        reply_parts = []
        stop_flag = [False]
        n_vlm_pages = len(all_pages)
        progress_stop = threading.Event()
        on_first_token = None
        if is_vlm and image_tensors and n_vlm_pages > 0:
            n_vis_tokens = n_vlm_pages * (max(img.width * img.height for img in all_pages) // (32 * 32))
            _t_prefill_start = time.time()
            def _on_first():
                progress_stop.set()
                pt = time.time() - _t_prefill_start
                print(f"\r  \u2713 {TR('视觉编码 + prefill 完成', 'Vision + prefill done')} ({pt:.1f}s, ~{n_vis_tokens} tok)  ")
                print(f"  {TR('回复', 'Reply')}:", end=" ", flush=True)
            on_first_token = _on_first
            def _show_progress():
                while not progress_stop.is_set():
                    elapsed = time.time() - _t_prefill_start
                    print(f"\r  \u23f3 {TR('正在处理', 'Processing')} {n_vlm_pages} {TR('页', 'pages')}... ({elapsed:.0f}s)", end="", flush=True)
                    progress_stop.wait(1.0)
            threading.Thread(target=_show_progress, daemon=True).start()

        streamer_callback = _make_streamer(reply_parts, stop_flag, on_first_token, thinking_filter=False)

        t0 = time.time()
        try:
            prompt = _build_prompt(messages, pipe.get_tokenizer(), True)
            kwargs = {"generation_config": gen_cfg, "streamer": streamer_callback}
            if is_vlm and image_tensors is not None:
                kwargs["images"] = image_tensors
            pipe.generate(prompt, **kwargs)
        except RuntimeError as e:
            err = str(e)
            if "reshape" in err:
                print(f"\n  ⚠ {TR('该模型不支持图像输入', 'This model does not support image input')}")
            else:
                print(f"\n  ⚠ {TR('生成失败', 'Generation failed')}: {err[:200]}")
        reply_text = "".join(reply_parts)

        elapsed = time.time() - t0
        conv.append({"role": "assistant", "content": reply_text})
        char_count = len(reply_text.replace(" ", ""))
        tok_count = _count_tokens(ctx, reply_text)
        print()
        print(f"  [{elapsed:.1f}s | {char_count} chars | {char_count/elapsed:.1f} ch/s | {tok_count/elapsed:.1f} tok/s]")
        print()



