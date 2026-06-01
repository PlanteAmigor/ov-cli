"""
ov-cli chat: LLM 聊天/翻译终端。

支持两种官方格式:
  GenAI 格式 (optimum-cli 导出): 使用 openvino_genai LLMPipeline/VLMPipeline
  Optimum 格式 (optimum-intel OVModelForVisualCausalLM): Gemma-4 等
"""

import os, sys, time, json, signal
import openvino as ov
import openvino_genai as ov_genai


# ── Gemma-4 补丁 ────────────────────────────────────────────
# model_patcher.py 中 gemma4_text_attention_forward 引用了不存在的
# self.kv_shared_layer_index，已直接修改库文件将其替换为 self.layer_type。
# 如果重装 optimum-intel 后需要重新应用此补丁。


def _is_genai_format(model_path):
    """检测模型目录是否为 OpenVINO GenAI 格式。"""
    return os.path.isfile(os.path.join(model_path, "openvino_config.json"))


def _is_optimum_format(model_path):
    """检测模型是否为 Optimum Intel 导出的 VLM 格式
    （需要 OVModelForVisualCausalLM + AutoProcessor）。
    特征：有 openvino_config.json 且包含逐层输入模型文件。"""
    if not _is_genai_format(model_path):
        return False
    # 有逐层输入模型 => 必须用 Optimum 格式加载
    if os.path.isfile(os.path.join(model_path, "openvino_text_embeddings_per_layer_model.xml")):
        return True
    # 有 vision 但不含 merger → Gemma-4 等 VLM
    if _is_multimodal(model_path):
        if not os.path.isfile(os.path.join(model_path, "openvino_vision_embeddings_merger_model.xml")):
            return True
    return False


def _is_multimodal(model_path):
    """检测模型是否包含视觉组件。"""
    return os.path.isfile(os.path.join(model_path, "openvino_vision_embeddings_model.xml"))


def load_model(ov_path):
    """加载 OpenVINO 模型。自动检测 GenAI/传统格式。"""
    device = "GPU" if "GPU" in ov.Core().available_devices else "CPU"

    if _is_genai_format(ov_path):
        if _is_optimum_format(ov_path):
            # === Optimum 格式（OVModelForVisualCausalLM + AutoProcessor） ===
            return _load_optimum(ov_path, device)

        # === GenAI 格式（optimum-cli 导出，openvino-genai 推理） ===
        is_vlm = _is_multimodal(ov_path)
        tag = "VLM" if is_vlm else "LLM"
        print(f"  加载 {tag}Pipeline ({device})...", end=" ", flush=True)
        t0 = time.time()
        if is_vlm:
            pipe = ov_genai.VLMPipeline(ov_path, device)
        else:
            pipe = ov_genai.LLMPipeline(ov_path, device)
        print(f"✓ ({time.time()-t0:.1f}s)")

        # 从 config.json 读 model_type
        model_type = None
        cfg_path = os.path.join(ov_path, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type")

        # 翻译 prompt 模板
        t_zh = "将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
        t_en = "Translate the following text into {target}. Note that you should only output the translated result without any additional explanation:\n\n{text}"

        return {
            "pipe": pipe,
            "device": device,
            "model_type": model_type,
            "genai": True,
            "is_vlm": is_vlm,
            "t_zh": t_zh,
            "t_en": t_en,
        }




def _make_genai_config(temperature=0.7, top_p=0.9, top_k=40, max_tokens=1024):
    """创建 GenAI GenerationConfig。"""
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_tokens
    cfg.temperature = temperature
    cfg.top_p = top_p
    cfg.top_k = top_k
    cfg.do_sample = temperature >= 0.01
    return cfg


# ── 行编辑器 ────────────────────────────────────────────
# 支持方向键、Home/End、退格、Delete、历史记录（仿 llama.cpp）


def has_chinese(text):
    return any('\u4e00' <= c <= '\u9fff' for c in text[:30])


def read_multiline(prompt=">>> "):
    import sys as _sys
    if _sys.platform == "win32":
        try:
            return input(prompt)
        except EOFError:
            return ""
    try:
        line = input(prompt)
    except EOFError:
        return ""
    if not line:
        return ""
    import select
    lines = [line.rstrip("\n")]
    try:
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                break
            more = sys.stdin.readline()
            if not more:
                break
            lines.append(more.rstrip("\n"))
    except EOFError:
        pass
    return "\n".join(lines)


def _count_tokens(ctx, text):
    if not text:
        return 0
    try:
        if ctx.get("optimum"):
            return len(ctx["processor"].tokenizer.encode(text))
        else:
            r = ctx["pipe"].get_tokenizer().encode(text)
            return r.input_ids.shape[-1]
    except Exception:
        return 0


_hist = []
_hist_idx = -1
_in_readline = False



def _char_width(ch):
    cp = ord(ch)
    if cp < 0x80:
        return 1
    if 0x4E00 <= cp <= 0x9FFF or 0x3000 <= cp <= 0x303F or 0xFF00 <= cp <= 0xFFEF:
        return 2
    return 1


def _total_width(chars, start=0, end=None):
    if end is None:
        end = len(chars)
    return sum(_char_width(c) for c in chars[start:end])


def _move_cursor(delta):
    if delta < 0:
        import sys as _sys
        _sys.stdout.write("\b" * (-delta))
    elif delta > 0:
        import sys as _sys
        _sys.stdout.write("\033[C" * delta)


def _clear_line(_total_w):
    import sys as _sys
    _sys.stdout.write("\r\033[K")


def _draw_prompt():
    import sys as _sys
    _sys.stdout.write(">>> ")


def readline():
    global _hist, _hist_idx, _in_readline
    import os as _os
    if _in_readline or _os.name == "nt":
        try:
            return input(">>> ")
        except EOFError:
            return ""
    _in_readline = True

    import sys as _sys, tty, termios
    fd = _sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setraw(fd)

    buf, widths = [], []
    char_pos = 0

    try:
        _draw_prompt()
        _sys.stdout.flush()
        while True:
            raw = _os.read(fd, 1)
            if not raw:
                break
            b = raw[0]
            if b == 3:
                raise KeyboardInterrupt
            if b == 4:
                break
            if b in (13, 10):
                _sys.stdout.write("\r\n")
                _sys.stdout.flush()
                break
            if b == 9:
                continue

            if b & 0xE0 == 0xC0:
                raw += _os.read(fd, 1)
            elif b & 0xF0 == 0xE0:
                raw += _os.read(fd, 2)
            elif b & 0xF8 == 0xF0:
                raw += _os.read(fd, 3)
            ch = raw.decode("utf-8", errors="replace")

            if b == 27:
                nxt = _os.read(fd, 1)
                if nxt == b"[":
                    params = b""
                    while True:
                        c = _os.read(fd, 1)
                        if c in [b"A", b"B", b"C", b"D", b"H", b"F", b"~"] or (c.isalpha() and c.isupper()):
                            code = c
                            if code == b"3":
                                _os.read(fd, 1)
                            break
                        params += c
                    if code == b"D" and char_pos > 0:
                        char_pos -= 1
                        _move_cursor(-_char_width(buf[char_pos]))
                    elif code == b"C" and char_pos < len(buf):
                        _move_cursor(_char_width(buf[char_pos]))
                        char_pos += 1
                    elif code == b"H":
                        _move_cursor(-_total_width(buf[:char_pos]))
                        char_pos = 0
                    elif code == b"F":
                        _move_cursor(_total_width(buf[char_pos:]))
                        char_pos = len(buf)
                    elif code == b"A":
                        char_pos = _hist_up(buf, widths, char_pos)
                    elif code == b"B":
                        char_pos = _hist_down(buf, widths, char_pos)
                    elif code == b"~" and params == b"3":
                        if char_pos < len(buf):
                            dw = _char_width(buf[char_pos])
                            del buf[char_pos]
                            del widths[char_pos]
                            tail = "".join(buf[char_pos:])
                            _sys.stdout.write(tail + " ")
                            _move_cursor(-_total_width(buf[char_pos:]) - dw - 1)
                        _sys.stdout.flush()
                _sys.stdout.flush()
                continue

            if b in (127, 8):
                if char_pos > 0:
                    dw = _char_width(buf[char_pos - 1])
                    del buf[char_pos - 1]
                    del widths[char_pos - 1]
                    char_pos -= 1
                    _move_cursor(-dw)
                    tail = "".join(buf[char_pos:])
                    _sys.stdout.write(tail + " ")
                    _move_cursor(-_total_width(buf[char_pos:]) - 1)
                _sys.stdout.flush()
                continue

            w = _char_width(ch)
            if char_pos == len(buf):
                _sys.stdout.write(ch)
            else:
                tail = "".join(buf[char_pos:])
                _sys.stdout.write(ch + tail)
                _move_cursor(-_total_width(buf[char_pos:]))
            buf.insert(char_pos, ch)
            widths.insert(char_pos, w)
            char_pos += 1
            _sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old)
        _in_readline = False

    line = "".join(buf)
    if line:
        _hist.append(line)
    _hist_idx = -1
    return line


def _hist_up(buf, widths, char_pos):
    global _hist_idx, _hist_buf
    import sys as _sys
    if not _hist:
        return char_pos
    if _hist_idx == -1:
        _hist_buf = "".join(buf)
    if _hist_idx + 1 < len(_hist):
        _hist_idx += 1
        s = _hist[-(_hist_idx + 1)]
        _sys.stdout.write("\r")
        _clear_line(_total_width(buf))
        _draw_prompt()
        for c in s:
            _sys.stdout.write(c)
        buf.clear()
        buf.extend(s)
        widths.clear()
        widths.extend(_char_width(c) for c in s)
        return len(s)
    return char_pos


def _hist_down(buf, widths, char_pos):
    global _hist_idx, _hist_buf
    import sys as _sys
    if _hist_idx <= 0:
        if _hist_idx == 0:
            _sys.stdout.write("\r")
            _clear_line(_total_width(buf))
            _draw_prompt()
            for c in _hist_buf:
                _sys.stdout.write(c)
            buf.clear()
            buf.extend(_hist_buf)
            widths.clear()
            widths.extend(_char_width(c) for c in _hist_buf)
            _hist_idx = -1
            return len(_hist_buf)
        return char_pos
    _hist_idx -= 1
    s = _hist[-(_hist_idx + 1)]
    _move_cursor(-_total_width(buf[:char_pos]) - 4)
    _clear_line(_total_width(buf))
    _draw_prompt()
    for c in s:
        _sys.stdout.write(c)
    buf.clear()
    buf.extend(s)
    widths.clear()
    widths.extend(_char_width(c) for c in s)
    return len(s)


_hist = []
_hist_idx = -1
_in_readline = False
_hist_buf = ""
def run_chat(ctx, system="You are a helpful AI assistant.",
             temperature=0.7, top_p=0.9, top_k=40, max_tokens=1024,
             image_path=None):
    """通用聊天模式"""
    from . import TR

    print()
    print("=" * 50)
    print("  ov-cli " + TR("聊天终端", "Chat Terminal"))
    print(f"  {TR('设备', 'Device')}: {ctx['device']} | OpenVINO")
    print("=" * 50)
    if ctx.get("is_vlm"):
        print("  //img PATH " + TR("加载图片", "load image"))
    print("  /temp 0.7   " + TR("设置温度", "set temperature"))
    print("  /system ... " + TR("设置系统提示词", "set system prompt"))
    print("  /clear      " + TR("清空上下文", "clear context"))
    print("  /help       " + TR("帮助", "help"))
    print("  /exit       " + TR("退出", "quit"))
    print("=" * 50)
    print()

    if ctx.get("optimum"):
        _run_chat_optimum(ctx, system, temperature, top_p, top_k, max_tokens, image_path)
    else:
        _run_chat_genai(ctx, system, temperature, top_p, top_k, max_tokens, image_path)


def _build_prompt(messages):
    """将消息列表转为纯文本 prompt（用于 VLMPipeline 等不支持 ChatHistory 的管道）。"""
    prompt = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    prompt += "<|assistant|>\n"
    return prompt


def _load_image(path):
    """加载图片为 openvino.Tensor。

    Qwen3.5 视觉编码器要求 H×W 能被 512 整除 (patch_size² × merge_size² = 16² × 2²)。
    自动缩放使宽高为 32 的倍数，长边不超过 2048。
    """
    from PIL import Image
    import numpy as np, openvino as ov
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # 长边不超过 2048
    max_sz = 2048
    if w > max_sz or h > max_sz:
        ratio = max_sz / max(w, h)
        w, h = int(w * ratio), int(h * ratio)

    # 确保宽高为 32 的倍数 (patch_size=16, spatial_merge=2, 乘积=32)
    w = (w // 32) * 32
    h = (h // 32) * 32
    if w < 32: w = 32
    if h < 32: h = 32

    img = img.resize((w, h))
    # 官方方式: [None] 加 batch 维 → [1, H, W, 3]
    arr = np.array(img).astype(np.uint8)[None]
    return ov.Tensor(arr)


def _load_optimum(ov_path, device):
    """加载 Optimum 格式模型（OVModelForVisualCausalLM + AutoProcessor）。"""
    from . import TR

    from optimum.intel import OVModelForVisualCausalLM
    from transformers import AutoProcessor

    print(f"  加载 OVModelForVisualCausalLM ({device})...", end=" ", flush=True)
    t0 = time.time()
    model = OVModelForVisualCausalLM.from_pretrained(ov_path, device=device)
    processor = AutoProcessor.from_pretrained(ov_path, trust_remote_code=True)
    print(f"✓ ({time.time()-t0:.1f}s)")

    # model_type
    model_type = None
    cfg_path = os.path.join(ov_path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type")

    return {
        "model": model,
        "processor": processor,
        "device": device,
        "model_type": model_type,
        "optimum": True,
        "is_vlm": _is_multimodal(ov_path),
    }


def _count_tokens(ctx, text):
    """统计文本的 token 数。"""
    if not text:
        return 0
    try:
        if ctx.get("optimum"):
            return len(ctx["processor"].tokenizer.encode(text))
        else:
            r = ctx["pipe"].get_tokenizer().encode(text)
            return r.input_ids.shape[-1]
    except Exception:
        return 0


def _run_chat_optimum(ctx, system, temperature, top_p, top_k, max_tokens, image_path=None):
    """Optimum 格式聊天模式（OVModelForVisualCausalLM + AutoProcessor）。"""
    model = ctx["model"]
    processor = ctx["processor"]
    is_vlm = ctx.get("is_vlm", False)
    from . import TR

    # 预加载初始图片
    current_image = None
    if image_path and is_vlm:
        if os.path.isfile(image_path):
            from PIL import Image
            current_image = Image.open(image_path).convert("RGB")
            print(f"  {TR('已加载图片', 'Image loaded')}: {image_path}")
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
        if text in ("/clear", "clear"):
            conv.clear()
            current_image = None
            print(f"  ✓ {TR('上下文已清空', 'Context cleared')}")
            print()
            continue
        if text == "/help":
            if is_vlm:
                print("  //img PATH " + TR("加载图片", "load image"))
            print("  /temp N      " + TR("温度 (0-2)", "temperature"))
            print("  /system T    " + TR("系统提示词", "system prompt"))
            print("  /clear       " + TR("清空上下文", "clear context"))
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
            img_path = text[6:].strip()
            if os.path.isfile(img_path):
                from PIL import Image
                current_image = Image.open(img_path).convert("RGB")
                print(f"  ✓ {TR('已加载图片', 'Image loaded')}: {img_path}")
            else:
                print(f"  ⚠ {TR('找不到图片', 'Image not found')}: {img_path}")
            print()
            continue

        # 带图片 → 消息体引用图片，chat template 才能插入 [IMAGE] token
        if current_image is not None and is_vlm:
            conv.append({"role": "user", "content": [{"type": "image", "image": current_image}, {"type": "text", "text": text}]})
        else:
            conv.append({"role": "user", "content": text})

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(conv)

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if current_image is not None and is_vlm:
            inputs = processor(text=[prompt], images=[current_image], return_tensors="pt")
        else:
            inputs = processor(text=[prompt], return_tensors="pt")

        t0 = time.time()
        print(f"  {TR('回复', 'Reply')}:", end=" ", flush=True)

        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature >= 0.01,
            temperature=temperature if temperature >= 0.01 else None,
            top_p=top_p,
            top_k=top_k,
            streamer=streamer,
        )

        stop_flag = [False]
        old_handler = signal.signal(signal.SIGINT, lambda s, f: stop_flag.__setitem__(0, True))
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        reply_parts = []
        try:
            for t in streamer:
                if stop_flag[0]:
                    break
                sys.stdout.write(t)
                sys.stdout.flush()
                reply_parts.append(t)
        finally:
            signal.signal(signal.SIGINT, old_handler)

        if stop_flag[0]:
            print(f"\n  ⚠ {TR('已中断', 'Interrupted')}")
            print()
            thread.join(timeout=5)
            continue

        thread.join()
        reply_text = "".join(reply_parts)
        elapsed = time.time() - t0
        conv.append({"role": "assistant", "content": reply_text})
        # 图片用完即清，下次需要重新 //img
        if current_image is not None:
            current_image = None
        char_count = len(reply_text.replace(" ", ""))
        tok_count = _count_tokens(ctx, reply_text)
        print()
        print(f"  [{elapsed:.1f}s | {char_count} chars | {char_count/elapsed:.1f} ch/s | {tok_count/elapsed:.1f} tok/s]")
        print()


def _run_chat_genai(ctx, system, temperature, top_p, top_k, max_tokens, image_path=None):
    """GenAI 格式聊天模式。"""
    pipe = ctx["pipe"]
    is_vlm = ctx.get("is_vlm", False)
    from . import TR

    # 预加载初始图片
    current_image = None
    if image_path and is_vlm:
        if os.path.isfile(image_path):
            current_image = _load_image(image_path)
            print(f"  {TR('已加载图片', 'Image loaded')}: {image_path}")
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
        if text in ("/clear", "clear"):
            conv.clear()
            current_image = None
            print(f"  ✓ {TR('上下文已清空', 'Context cleared')}")
            print()
            continue
        if text == "/help":
            if is_vlm:
                print("  //img PATH " + TR("加载图片", "load image"))
            print("  /temp N      " + TR("温度 (0-2)", "temperature"))
            print("  /system T    " + TR("系统提示词", "system prompt"))
            print("  /clear       " + TR("清空上下文", "clear context"))
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
            img_path = text[6:].strip()
            if os.path.isfile(img_path):
                current_image = _load_image(img_path)
                print(f"  ✓ {TR('已加载图片', 'Image loaded')}: {img_path}")
            else:
                print(f"  ⚠ {TR('找不到图片', 'Image not found')}: {img_path}")
            print()
            continue

        conv.append({"role": "user", "content": text})
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(conv)

        gen_cfg = _make_genai_config(temperature, top_p, top_k, max_tokens)
        t0 = time.time()
        print(f"  {TR('回复', 'Reply')}:", end=" ", flush=True)

        reply_parts = []
        stop_flag = [False]

        def streamer_callback(t):
            if stop_flag[0]:
                return True
            reply_parts.append(t)
            sys.stdout.write(t)
            sys.stdout.flush()
            return False

        old_handler = signal.signal(signal.SIGINT, lambda s, f: stop_flag.__setitem__(0, True))
        try:
            if is_vlm:
                kwargs = {"generation_config": gen_cfg, "streamer": streamer_callback}
                if current_image is not None:
                    kwargs["image"] = current_image
                pipe.generate(_build_prompt(messages), **kwargs)
            else:
                pipe.generate(ov_genai.ChatHistory(messages), gen_cfg, streamer_callback)
        finally:
            signal.signal(signal.SIGINT, old_handler)
        reply_text = "".join(reply_parts)

        elapsed = time.time() - t0
        conv.append({"role": "assistant", "content": reply_text})
        # 图片用完即清
        if current_image is not None:
            current_image = None
        char_count = len(reply_text.replace(" ", ""))
        tok_count = _count_tokens(ctx, reply_text)
        print()
        print(f"  [{elapsed:.1f}s | {char_count} chars | {char_count/elapsed:.1f} ch/s | {tok_count/elapsed:.1f} tok/s]")
        print()


def _run_translate_genai(ctx, max_tokens):
    """GenAI 格式翻译模式。"""
    pipe = ctx["pipe"]
    t_zh = ctx["t_zh"]
    t_en = ctx["t_en"]
    from ov_cli import TR

    print()
    print("=" * 50)
    print("  ov-cli " + TR("翻译终端", "Translation Terminal"))
    print(f"  {TR('设备', 'Device')}: {ctx['device']} | OpenVINO")
    print("=" * 50)
    print("  " + TR("直接输入文本 → 自动检测翻译方向", "Type text → auto detect language"))
    print("  //en " + TR("文本 → 强制译英", "text → force English"))
    print("  //zh " + TR("文本 → 强制译中", "text → force Chinese"))
    print("  /help " + TR("→ 帮助", "→ help"))
    print("  /exit " + TR("→ 退出", "→ quit"))
    print("=" * 50)
    print()

    while True:
        try:
            text = readline()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/exit", "exit", TR("退出", "exit")):
            break
        if text in ("/help", "help", TR("帮助", "help")):
            print("  //en " + TR("文本 → 强制译英", "text → force English"))
            print("  //zh " + TR("文本 → 强制译中", "text → force Chinese"))
            print("  /exit " + TR("退出", "quit"))
            print()
            continue

        force_target = None
        if text.startswith("//en "):
            force_target = TR("英语", "English")
            text = text[5:]
        elif text.startswith("//zh "):
            force_target = TR("中文", "Chinese")
            text = text[5:]
        elif text.startswith("//"):
            print("  ⚠ " + TR("未知指令，可用 //en 或 //zh", "Unknown command, use //en or //zh"))
            continue

        if force_target:
            target = force_target
        elif has_chinese(text):
            target = TR("英语", "English")
        else:
            target = TR("中文", "Chinese")

        prompt = t_zh.format(target=target, text=text) if has_chinese(text) else t_en.format(target=target, text=text)
        gen_cfg = _make_genai_config(temperature=0, max_tokens=max_tokens)

        print(f"  → {target}", flush=True)
        t0 = time.time()
        sys.stdout.write("  ")
        sys.stdout.flush()
        reply_parts = []

        stop_flag = [False]
        def streamer_callback(t):
            if stop_flag[0]:
                return True
            reply_parts.append(t)
            sys.stdout.write(t)
            sys.stdout.flush()
            return False

        old_handler = signal.signal(signal.SIGINT, lambda s, f: stop_flag.__setitem__(0, True))
        try:
            if ctx.get("is_vlm"):
                pipe.generate(prompt, generation_config=gen_cfg, streamer=streamer_callback)
            else:
                pipe.generate(prompt, gen_cfg, streamer_callback)
        finally:
            signal.signal(signal.SIGINT, old_handler)
        elapsed = time.time() - t0
        reply_text = "".join(reply_parts)
        char_count = len(reply_text.replace(" ", ""))
        tok_count = _count_tokens(ctx, reply_text)
        print()
        print(f"  [{elapsed:.1f}s | {char_count} chars | {char_count/elapsed:.1f} ch/s | {tok_count/elapsed:.1f} tok/s]")
        print()


def run_translate(ctx, max_tokens=512):
    """翻译模式（用于 Hy-MT2 等翻译模型）。"""
    from ov_cli import TR

    if ctx.get("optimum"):
        # Optimum 格式翻译（退化为聊天模式）
        print(f"  ⚠ {TR('翻译模式在 Optimum 格式下不可用，进入聊天模式', 'Translate mode not available, using chat mode')}")
        _run_chat_optimum(ctx, None, 0, 0.9, 40, max_tokens, None)
        return
    _run_translate_genai(ctx, max_tokens)
