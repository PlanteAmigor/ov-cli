"""
ov-cli chat: LLM 聊天/翻译终端。

支持两种官方格式:
  GenAI 格式 (optimum-cli 导出): 使用 openvino_genai LLMPipeline/VLMPipeline
  Optimum 格式 (optimum-intel OVModelForVisualCausalLM): Gemma-4 等
"""

import os, sys, time, json, signal
import openvino as ov
import openvino_genai as ov_genai
from wcwidth import wcwidth, wcswidth


# 翻译语言映射：代码 → (中文名, 英文名)
TRANSLATE_LANGS = {
    "zh":   ("中文",     "Chinese"),
    "en":   ("英语",     "English"),
    "ja":   ("日语",     "Japanese"),
    "ko":   ("韩语",     "Korean"),
    "fr":   ("法语",     "French"),
    "de":   ("德语",     "German"),
    "es":   ("西班牙语", "Spanish"),
    "pt":   ("葡萄牙语", "Portuguese"),
    "ru":   ("俄语",     "Russian"),
    "ar":   ("阿拉伯语", "Arabic"),
    "it":   ("意大利语", "Italian"),
    "tr":   ("土耳其语", "Turkish"),
    "th":   ("泰语",     "Thai"),
    "vi":   ("越南语",   "Vietnamese"),
    "ms":   ("马来语",   "Malay"),
    "id":   ("印尼语",   "Indonesian"),
    "tl":   ("菲律宾语", "Filipino"),
    "hi":   ("印地语",   "Hindi"),
    "pl":   ("波兰语",   "Polish"),
    "cs":   ("捷克语",   "Czech"),
    "nl":   ("荷兰语",   "Dutch"),
    "km":   ("高棉语",   "Khmer"),
    "my":   ("缅甸语",   "Burmese"),
    "fa":   ("波斯语",   "Persian"),
    "gu":   ("古吉拉特语", "Gujarati"),
    "ur":   ("乌尔都语", "Urdu"),
    "te":   ("泰卢固语", "Telugu"),
    "mr":   ("马拉地语", "Marathi"),
    "he":   ("希伯来语", "Hebrew"),
    "bn":   ("孟加拉语", "Bengali"),
    "ta":   ("泰米尔语", "Tamil"),
    "uk":   ("乌克兰语", "Ukrainian"),
    "bo":   ("藏语",     "Tibetan"),
    "kk":   ("哈萨克语", "Kazakh"),
    "mn":   ("蒙古语",   "Mongolian"),
    "ug":   ("维吾尔语", "Uyghur"),
    "yue":  ("粤语",     "Cantonese"),
    "zh-Hant": ("繁体中文", "Traditional Chinese"),
}


def _make_streamer(reply_parts, stop_flag):
    """创建 streamer callback（仅处理 Ctrl+C）。"""
    import select as _sel

    def cb(t):
        if stop_flag[0]:
            return True
        # 非阻塞检查 stdin 的 Ctrl+C
        if _sel.select([sys.stdin], [], [], 0)[0]:
            c = sys.stdin.read(1)
            if c == '\x03':
                stop_flag[0] = True
                return True
        reply_parts.append(t)
        sys.stdout.write(t)
        sys.stdout.flush()
        return False

    return cb


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




def _make_genai_config(temperature=0.7, top_p=0.9, top_k=40, max_tokens=1024, presence_penalty=None, reasoning=True, tokenizer=None):
    """创建 GenAI GenerationConfig。"""
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_tokens
    cfg.temperature = temperature
    cfg.top_p = top_p
    cfg.top_k = top_k
    cfg.do_sample = temperature >= 0.01
    if presence_penalty is not None:
        cfg.presence_penalty = presence_penalty
    # Reasoning budget: 不显示思考时用 budget=0 强制立即结束思考
    if not reasoning and tokenizer is not None:
        try:
            think_enc = tokenizer.encode("<think>", add_special_tokens=False)
            nothink_enc = tokenizer.encode("</think>", add_special_tokens=False)
            think_id = int(list(think_enc.input_ids.data)[0][0])
            nothink_id = int(list(nothink_enc.input_ids.data)[0][0])
            cfg.reasoning_budget_tokens = 0
            cfg.thinking_start_token_id = think_id
            cfg.thinking_end_token_id = nothink_id
        except Exception:
            pass
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
    return max(wcwidth(ch), 1)


def _total_width(chars, start=0, end=None):
    if end is None:
        end = len(chars)
    if start >= end:
        return 0
    return wcswidth("".join(chars[start:end]))


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
             image_path=None, reasoning=True):
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
        print("  //img PATH " + TR("加载图片", "load image"))
    print("  /temp 0.7   " + TR("设置温度", "set temperature"))
    print("  /system ... " + TR("设置系统提示词", "set system prompt"))
    print("  /clear      " + TR("清空上下文", "clear context"))
    print("  /help       " + TR("帮助", "help"))
    print("  /exit       " + TR("退出", "quit"))
    print("=" * 50)
    print()

    if ctx.get("optimum"):
        _run_chat_optimum(ctx, system, temperature, top_p, top_k, max_tokens, image_path, reasoning)
    else:
        _run_chat_genai(ctx, system, temperature, top_p, top_k, max_tokens, image_path, reasoning)


def _build_prompt(messages, tokenizer=None, enable_thinking=True):
    """将消息列表转为纯文本 prompt。

    优先使用模型的 chat template（通过 tokenizer.apply_chat_template），
    回退到手动构建。
    """
    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                extra_context={"enable_thinking": enable_thinking},
            )
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


def _run_chat_optimum(ctx, system, temperature, top_p, top_k, max_tokens, image_path=None, reasoning=True):
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

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template_kwargs={"enable_thinking": reasoning})
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

        thinking_filter = not reasoning  # True = 需要过滤思考内容
        in_think = [thinking_filter]  # 过滤时默认已在 think 块内
        reply_parts = []
        try:
            for t in streamer:
                if stop_flag[0]:
                    break
                if thinking_filter:
                    # 过滤 <think>...</think>
                    if in_think[0]:
                        if '</think>' in t:
                            idx = t.index('</think>')
                            after = t[idx + 8:]
                            if after:
                                reply_parts.append(after)
                                sys.stdout.write(after)
                                sys.stdout.flush()
                            in_think[0] = False
                        continue
                    if '<think>' in t:
                        idx = t.index('<think>')
                        before = t[:idx]
                        after = t[idx + 7:]
                        if before:
                            reply_parts.append(before)
                            sys.stdout.write(before)
                            sys.stdout.flush()
                        if after:
                            if '</think>' in after:
                                idx2 = after.index('</think>')
                                after_think = after[idx2 + 8:]
                                if after_think:
                                    reply_parts.append(after_think)
                                    sys.stdout.write(after_think)
                                    sys.stdout.flush()
                            else:
                                in_think[0] = True
                        continue
                    sys.stdout.write(t)
                    sys.stdout.flush()
                    reply_parts.append(t)
                else:
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


def _run_chat_genai(ctx, system, temperature, top_p, top_k, max_tokens, image_path=None, reasoning=True):
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

        pp = 1.5 if not reasoning else None
        gen_cfg = _make_genai_config(temperature, top_p, top_k, max_tokens, presence_penalty=pp, reasoning=reasoning, tokenizer=pipe.get_tokenizer())
        t0 = time.time()
        print(f"  {TR('回复', 'Reply')}:", end=" ", flush=True)

        reply_parts = []
        stop_flag = [False]
        streamer_callback = _make_streamer(reply_parts, stop_flag)

        old_handler = signal.signal(signal.SIGINT, lambda s, f: stop_flag.__setitem__(0, True))
        try:
            prompt = _build_prompt(messages, pipe.get_tokenizer(), reasoning)
            kwargs = {"generation_config": gen_cfg, "streamer": streamer_callback}
            if is_vlm:
                if current_image is not None:
                    kwargs["image"] = current_image
                pipe.generate(prompt, **kwargs)
            else:
                pipe.generate(prompt, gen_cfg, streamer_callback)
        except RuntimeError as e:
            err = str(e)
            if "reshape" in err:
                print(f"\n  ⚠ {TR('该模型不支持图像输入', 'This model does not support image input')}")
            else:
                print(f"\n  ⚠ {TR('生成失败', 'Generation failed')}: {err[:200]}")
        finally:
            signal.signal(signal.SIGINT, old_handler)
        reply_text = "".join(reply_parts)

        elapsed = time.time() - t0
        if stop_flag[0]:
            print(f"\n  ⚠ {TR('已中断', 'Interrupted')}")
            conv.append({"role": "assistant", "content": reply_text + TR(" [已中断]", " [Interrupted]")})
        else:
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
    import ov_cli as _ov
    from ov_cli import TR

    # 构建语言列表（按代码排序，常用语言排前）
    lang_codes = sorted(TRANSLATE_LANGS.keys(), key=lambda c: (0, c) if c in ("zh","en","ja","ko","fr","de","es","pt","ru","ar","it","tr","th","vi") else (1, c))
    lang_items = []
    for c in lang_codes:
        zh_name, en_name = TRANSLATE_LANGS[c]
        name = TR(zh_name, en_name)
        lang_items.append(f"{c}={name}")
    # 每行 4 列
    lang_lines = []
    for i in range(0, len(lang_items), 4):
        row = lang_items[i:i+4]
        lang_lines.append("  " + "  ".join(f"{item:16s}" for item in row))
    lang_display = "\n".join(lang_lines)

    print()
    print("        ██████╗ ██╗   ██╗     ██████╗██╗     ██╗")
    print("       ██╔═══██╗██║   ██║    ██╔════╝██║     ██║")
    print("       ██║   ██║██║   ██║    ██║     ██║     ██║")
    print("       ██║   ██║╚██╗ ██╔╝    ██║     ██║     ██║")
    print("       ╚██████╔╝ ╚████╔╝     ╚██████╗███████╗██║")
    print("        ╚═════╝   ╚═══╝       ╚═════╝╚══════╝╚═╝")
    print("=" * 50)
    print("  ov-cli " + TR("翻译终端", "Translation Terminal"))
    print(f"  {TR('设备', 'Device')}: {ctx['device']} | OpenVINO")
    print("=" * 50)
    print("  " + TR("直接输入文本 → 自动检测翻译方向", "Type text → auto detect language"))
    print("  //" + TR("语言代码 文本 → 指定目标语言", "lang_code text → force target language"))
    print(f"  " + TR("支持语言", "Supported codes") + f":\n{lang_display}")
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
            print("  //" + TR("语言代码 文本 → 指定目标语言", "lang_code text → force target language"))
            print("  " + TR("例如", "e.g.") + ": //ja おはよう, //fr Bonjour")
            print("  /exit " + TR("退出", "quit"))
            print()
            continue

        force_target = None
        if text.startswith("//") and len(text) > 2:
            # 尝试解析 //语言代码 文本
            space_pos = text.find(" ", 2)
            if space_pos > 2:
                code = text[2:space_pos]
                rest = text[space_pos+1:]
            elif text[2:].isalpha() and text[2:].isascii():
                code = text[2:]
                rest = ""
            else:
                code = None
                rest = text

            if code and code in TRANSLATE_LANGS:
                zh_name, en_name = TRANSLATE_LANGS[code]
                force_target = TR(zh_name, en_name)
                text = rest
            elif code and code.isalpha() and code.isascii():
                # 未知但看起来像语言代码，直接作为目标语言名
                force_target = code
                text = rest
            else:
                print("  ⚠ " + TR("未知语言代码。可用 /help 查看支持的语言", "Unknown language code. Use /help for supported codes"))
                continue
        elif text.startswith("//"):
            print("  ⚠ " + TR("未知指令", "Unknown command"))
            continue

        if force_target:
            target = force_target
        elif _ov._LANG == "en":
            # 英文界面：默认目标语言为英语
            _, en_name = TRANSLATE_LANGS["en"]
            target = TR(TRANSLATE_LANGS["en"][0], en_name)
        else:
            # 中文界面：默认目标语言为中文
            target = TR(TRANSLATE_LANGS["zh"][0], TRANSLATE_LANGS["zh"][1])

        prompt = t_zh.format(target=target, text=text) if has_chinese(text) else t_en.format(target=target, text=text)
        gen_cfg = _make_genai_config(temperature=0, max_tokens=max_tokens, reasoning=True, tokenizer=pipe.get_tokenizer())

        print(f"  → {target}", flush=True)
        t0 = time.time()
        sys.stdout.write("  ")
        sys.stdout.flush()
        reply_parts = []
        stop_flag = [False]
        streamer_callback = _make_streamer(reply_parts, stop_flag)

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
        _run_chat_optimum(ctx, None, 0, 0.9, 40, max_tokens, None, reasoning=False)
        return
    _run_translate_genai(ctx, max_tokens)
