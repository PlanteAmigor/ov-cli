"""
ov-cli chat: LLM 聊天/翻译终端。

支持两种官方格式:
  GenAI 格式 (optimum-cli 导出): 使用 openvino_genai LLMPipeline/VLMPipeline
  Optimum 格式 (optimum-intel OVModelForVisualCausalLM): Gemma-4 等
"""

import os, sys, time, json, re, signal, threading
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


def _make_streamer(reply_parts, stop_flag, on_first_token=None, thinking_filter=False):
    """创建 streamer callback。

    on_first_token: 首个 token 到达时的回调（用于停止进度指示器）。
    thinking_filter: 是否过滤 <think> 标签及思考内容。
    """
    import select as _sel
    _first = [True]
    # 启用 filter 时，初始假设在 think 块内（模型可能先输出思考才输出 </think>）
    in_think = [thinking_filter]

    def cb(t):
        if stop_flag[0]:
            return True
        if _first[0] and on_first_token:
            _first[0] = False
            on_first_token()
        # 非阻塞检查 stdin 的 Ctrl+C
        if _sel.select([sys.stdin], [], [], 0)[0]:
            c = sys.stdin.read(1)
            if c == '\x03':
                stop_flag[0] = True
                return True

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


# ── 管道模式 ────────────────────────────────────────────

def run_pipe(ctx, reasoning=True, max_tokens=1024, temperature=0.7):
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
            full = _build_prompt(conv, pipe.get_tokenizer(), enable_thinking=reasoning)

            cfg = ov_genai.GenerationConfig(max_new_tokens=max_tokens, temperature=temperature)
            cfg.do_sample = temperature >= 0.01

            if not reasoning:
                try:
                    tok = pipe.get_tokenizer()
                    think_id = int(list(tok.encode("<think>", add_special_tokens=False).input_ids.data)[0][0])
                    nothink_id = int(list(tok.encode("</think>", add_special_tokens=False).input_ids.data)[0][0])
                    cfg.reasoning_budget_tokens = 0
                    cfg.thinking_start_token_id = think_id
                    cfg.thinking_end_token_id = nothink_id
                except Exception:
                    pass

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
            if not reasoning:
                resp = re.sub(r'</?think>', '', resp).strip()
            print(_json.dumps({"text": resp, "time": round(elapsed, 1)}, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        pass


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
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
            except (ValueError, TypeError):
                from . import TR
                print(f"  {TR('交互式输入不支持管道，请使用 --mode once', 'Interactive input does not support pipe, use --mode once')}")
                break
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
    if ch == "\n":
        return 0
    return max(wcwidth(ch), 1)


def _total_width(chars, start=0, end=None):
    if end is None:
        end = len(chars)
    if start >= end:
        return 0
    # 只计算当前行（最后一个 \n 之后）的宽度
    s = "".join(chars[start:end])
    if "\n" in s:
        s = s.rsplit("\n", 1)[1]
    return wcswidth(s)


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
                # 取当前行（光标所在行，从上一个 \n 到光标）的内容
                before = "".join(buf[:char_pos])
                cur_line = before.rsplit("\n", 1)[1] if "\n" in before else before
                if cur_line:  # 当前行有内容 → 换行继续输入
                    buf.insert(char_pos, "\n")
                    widths.insert(char_pos, 0)
                    char_pos += 1
                    _sys.stdout.write("\r\n")
                    _sys.stdout.flush()
                    continue
                # 当前行无内容 → 提交
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
                    is_nl = buf[char_pos - 1] == "\n"
                    del buf[char_pos - 1]
                    del widths[char_pos - 1]
                    char_pos -= 1
                    if is_nl:
                        _sys.stdout.write("\033[A")  # 光标上移一行
                        # 清除当前行并重新绘制后续文本
                        import shutil
                        cols = shutil.get_terminal_size().columns
                        _sys.stdout.write("\r\033[K")
                        tail = "".join(buf[char_pos:])
                        _sys.stdout.write(tail)
                        _move_cursor(-_total_width(buf[char_pos:]))
                    else:
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

def run_once(ctx, prompt="", files=None, output=None,
             temperature=0.7, top_p=0.9, top_k=40, max_tokens=1024,
             reasoning=True, json_output=False):
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
                pages = _pdf_to_images(fpath)
                if pages:
                    all_pages.extend(pages)
                    print(f"  ✓ {TR('已加载 PDF', 'PDF loaded')}: {fpath} ({len(pages)} 页)")
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

    # 构建 prompt
    if ctx.get("optimum"):
        # Optimum 路径
        chat_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": reasoning})
        if all_pages and is_vlm:
            inputs = processor(text=[chat_prompt], images=all_pages, return_tensors="pt")
        else:
            inputs = processor(text=[chat_prompt], return_tensors="pt")

        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs, max_new_tokens=max_tokens,
            do_sample=temperature >= 0.01,
            temperature=temperature if temperature >= 0.01 else None,
            top_p=top_p, top_k=top_k,
            streamer=streamer,
        )

        reply_parts = []
        thread = Thread(target=ctx["model"].generate, kwargs=gen_kwargs)
        t0 = time.time()
        thread.start()
        for t in streamer:
            sys.stdout.write(t)
            sys.stdout.flush()
            reply_parts.append(t)
        thread.join()
        reply_text = "".join(reply_parts)
    else:
        # GenAI 路径
        img_tag = "<|vision_start|><|image_pad|><|vision_end|>\n"
        if all_pages:
            user_text = img_tag * len(all_pages) + user_text
            messages[0]["content"] = user_text

        tokenizer = pipe.get_tokenizer()
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                extra_context={"enable_thinking": reasoning})
        except Exception:
            prompt_text = f"<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n"

        gen_cfg = _make_genai_config(temperature, top_p, top_k, max_tokens,
                                     presence_penalty=None, reasoning=reasoning,
                                     tokenizer=tokenizer)

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
        streamer_cb = _make_streamer(reply_parts, stop_flag, on_first, thinking_filter=not reasoning)

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

    if not reasoning:
        reply_text = re.sub(r'</?think>', '', reply_text).strip()

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
        print("  //img PATH  " + TR("加载图片", "load image"))
        print("  //pdf PATH  " + TR("加载 PDF（全页转图片）", "load PDF (pages as images)"))
    print("  //txt PATH  " + TR("加载文本文件", "load text file"))
    print("  /temp N     " + TR("温度 (0-2)", "temperature"))
    print("  /system T   " + TR("系统提示词", "system prompt"))
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


def _load_image(path, max_pixels=448*448):
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
    max_pixels = 448 * 448
    dpi = 300
    px = 448
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


def _load_optimum(ov_path, device):
    """加载 Optimum 格式模型（OVModelForVisualCausalLM + AutoProcessor）。"""
    from . import TR

    from optimum.intel import OVModelForVisualCausalLM
    from transformers import AutoProcessor

    print(f"  加载 OVModelForVisualCausalLM ({device})...", end=" ", flush=True, file=sys.stderr)
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


# 每批最多发送的图片/PDF 页数
def _run_chat_optimum(ctx, system, temperature, top_p, top_k, max_tokens, image_path=None, reasoning=True):
    """Optimum 格式聊天模式（OVModelForVisualCausalLM + AutoProcessor）。"""
    model = ctx["model"]
    processor = ctx["processor"]
    is_vlm = ctx.get("is_vlm", False)
    from . import TR

    # 文件管理: {id, path, type, pages:[PIL]}
    loaded_files = []
    _next_id = 1

    # 预加载初始图片
    if image_path and is_vlm:
        if os.path.isfile(image_path):
            from PIL import Image
            loaded_files.append({"id": _next_id, "path": image_path, "type": "image", "pages": [Image.open(image_path).convert("RGB")]})
            print(f"  \u2713 {TR('\u5df2\u52a0\u8f7d\u56fe\u7247', 'Image loaded')}: {image_path}")
            _next_id += 1
        else:
            print(f"  \u26a0 {TR('\u627e\u4e0d\u5230\u56fe\u7247', 'Image not found')}: {image_path}")

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
                print("  //img PATH   " + TR("\u52a0\u8f7d\u56fe\u7247", "load image"))
                print("  //pdf PATH   " + TR("\u52a0\u8f7d PDF\uff08\u5168\u9875\u8f6c\u56fe\u7247\uff09", "load PDF (pages as images)"))
            print("  //txt PATH   " + TR("\u52a0\u8f7d\u6587\u672c\u6587\u4ef6", "load text file"))
            print("  /temp N      " + TR("\u6e29\u5ea6 (0-2)", "temperature"))
            print("  /system T    " + TR("\u7cfb\u7edf\u63d0\u793a\u8bcd", "system prompt"))
            print("  /help        " + TR("\u5e2e\u52a9", "help"))
            print("  /exit        " + TR("\u9000\u51fa", "quit"))
            print()
            continue
        if text.startswith("/temp "):
            try:
                temperature = max(0, min(2, float(text[6:])))
                print(f"  temperature = {temperature}")
            except:
                print("  \u26a0 /temp 0.7")
            print()
            continue
        if text.startswith("/system "):
            system = text[8:]
            print(f"  {TR('\u7cfb\u7edf\u63d0\u793a\u8bcd\u5df2\u66f4\u65b0', 'System prompt updated')}")
            print()
            continue
        if text.startswith("//img ") and is_vlm:
            import shlex
            paths = shlex.split(text[6:])
            if not paths:
                print(f"  \u26a0 {TR('\u7528\u6cd5', 'Usage')}: //img PATH1 [PATH2 ...]")
                print()
                continue
            loaded_any = 0
            for img_path in paths:
                if os.path.isfile(img_path):
                    loaded_files.append({"id": _next_id, "path": img_path, "type": "image", "pages": [_load_image(img_path)]})
                    print(f"  #{_next_id} \u2713 {TR('\u5df2\u52a0\u8f7d\u56fe\u7247', 'Image loaded')}: {img_path}")
                    _next_id += 1
                    loaded_any += 1
                else:
                    print(f"  \u26a0 {TR('\u627e\u4e0d\u5230\u56fe\u7247', 'Image not found')}: {img_path}")
            print()
            continue

        if text.startswith("//pdf ") and is_vlm:
            import shlex
            paths = shlex.split(text[6:])
            if not paths:
                print(f"  \u26a0 {TR('\u7528\u6cd5', 'Usage')}: //pdf PATH1 [PATH2 ...]")
                print()
                continue
            loaded_any = 0
            for pdf_path in paths:
                if os.path.isfile(pdf_path):
                    pages = _pdf_to_images(pdf_path)
                    if pages:
                        loaded_files.append({"id": _next_id, "path": pdf_path, "type": "pdf", "pages": pages})
                        print(f"  #{_next_id} \u2713 {TR('\u5df2\u52a0\u8f7d PDF', 'PDF loaded')}: {pdf_path}")
                        _next_id += 1
                        loaded_any += 1
                else:
                    print(f"  \u26a0 {TR('\u627e\u4e0d\u5230\u6587\u4ef6', 'File not found')}: {pdf_path}")
            print()
            continue

        if text.startswith("//txt "):
            import shlex
            paths = shlex.split(text[6:])
            if not paths:
                print(f"  \u26a0 {TR('\u7528\u6cd5', 'Usage')}: //txt PATH1 [PATH2 ...]")
                print()
                continue
            loaded_any = 0
            for txt_path in paths:
                if os.path.isfile(txt_path):
                    file_content = _load_text_file(txt_path)
                    loaded_files.append({"id": _next_id, "path": txt_path, "type": "text", "content": file_content})
                    print(f"  #{_next_id} \u2713 {TR('\u5df2\u52a0\u8f7d\u6587\u4ef6', 'File loaded')}: {txt_path}")
                    _next_id += 1
                    loaded_any += 1
                else:
                    print(f"  \u26a0 {TR('\u627e\u4e0d\u5230\u6587\u4ef6', 'File not found')}: {txt_path}")
            print()
            continue

        # \u6784\u5efa\u6d88\u606f\uff1a\u6240\u6709\u5df2\u52a0\u8f7d\u6587\u4ef6\u7684\u56fe\u7247 + \u6587\u672c
        all_pages = []
        txt_prefix = ""
        for f in loaded_files:
            if f["type"] == "text":
                fname = os.path.basename(f["path"])
                txt_prefix += f"[\u6587\u4ef6 {fname}]\n```\n{f['content']}\n```\n\n"
            else:
                all_pages.extend(f["pages"])
        if txt_prefix:
            text = txt_prefix + text
        loaded_files.clear()
        if all_pages and is_vlm:
            content = [{"type": "image", "image": img} for img in all_pages]
            content.append({"type": "text", "text": text})
            conv.append({"role": "user", "content": content})
        else:
            conv.append({"role": "user", "content": text})

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(conv)

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template_kwargs={"enable_thinking": reasoning})
        # VLM prefill 进度指示器
        n_vlm_pages = len(all_pages)
        progress_stop = threading.Event()
        if is_vlm and n_vlm_pages > 0:
            n_vis_tokens = n_vlm_pages * (max(img.width * img.height for img in all_pages) // (32 * 32))
            _t_prefill_start = time.time()
            def _on_first():
                progress_stop.set()
                pt = time.time() - _t_prefill_start
                print(f"\r  \u2713 {TR('\u89c6\u89c9\u7f16\u7801 + prefill \u5b8c\u6210', 'Vision + prefill done')} ({pt:.1f}s, ~{n_vis_tokens} tok)  ")
                print(f"  {TR('\u56de\u590d', 'Reply')}:", end=" ", flush=True)
            def _show_progress():
                while not progress_stop.is_set():
                    elapsed = time.time() - _t_prefill_start
                    print(f"\r  \u23f3 {TR('\u6b63\u5728\u5904\u7406', 'Processing')} {n_vlm_pages} {TR('\u9875', 'pages')}... ({elapsed:.0f}s)", end="", flush=True)
                    progress_stop.wait(1.0)
            threading.Thread(target=_show_progress, daemon=True).start()

        if all_pages and is_vlm:
            inputs = processor(text=[prompt], images=all_pages, return_tensors="pt")
        else:
            inputs = processor(text=[prompt], return_tensors="pt")

        t0 = time.time()
        if not (is_vlm and n_vlm_pages > 0):
            print(f"  {TR('\u56de\u590d', 'Reply')}:", end=" ", flush=True)

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

        thinking_filter = not reasoning
        in_think = [False]  # 初始不在 think 块内，由 <think> 标签触发
        reply_parts = []
        _opt_first = [True]
        try:
            for t in streamer:
                if _opt_first[0] and n_vlm_pages > 0:
                    _opt_first[0] = False
                    _on_first()
                if stop_flag[0]:
                    break
                if thinking_filter:
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
                    # 过滤孤立的 </think>（没有配对的 <think>）
                    if '</think>' in t:
                        idx = t.index('</think>')
                        after = t[idx + 8:]
                        if after:
                            reply_parts.append(after)
                            sys.stdout.write(after)
                            sys.stdout.flush()
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
            if not progress_stop.is_set():
                progress_stop.set()

        if stop_flag[0]:
            print(f"\n  \u26a0 {TR('\u5df2\u4e2d\u65ad', 'Interrupted')}")
            print()
            thread.join(timeout=5)
            continue

        thread.join()
        reply_text = "".join(reply_parts)
        elapsed = time.time() - t0
        conv.append({"role": "assistant", "content": reply_text})
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
                print("  //pdf PATH   " + TR("加载 PDF（全页转图片）", "load PDF (pages as images)"))
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
            import shlex
            paths = shlex.split(text[6:])
            if not paths:
                print(f"  ⚠ {TR('用法', 'Usage')}: //pdf PATH1 [PATH2 ...]")
                print()
                continue
            for pdf_path in paths:
                if os.path.isfile(pdf_path):
                    pages = _pdf_to_images(pdf_path)
                    if pages:
                        loaded_files.append({"id": _next_id, "path": pdf_path, "type": "pdf", "pages": pages})
                        print(f"  #{_next_id} {TR('已加载 PDF', 'PDF loaded')}: {pdf_path}")
                        _next_id += 1
                else:
                    print(f"  ⚠ {TR('找不到文件', 'File not found')}: {pdf_path}")
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

        pp = 1.5 if not reasoning else None
        gen_cfg = _make_genai_config(temperature, top_p, top_k, max_tokens, presence_penalty=pp, reasoning=reasoning, tokenizer=pipe.get_tokenizer())

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

        streamer_callback = _make_streamer(reply_parts, stop_flag, on_first_token, thinking_filter=not reasoning)

        t0 = time.time()
        old_handler = signal.signal(signal.SIGINT, lambda s, f: stop_flag.__setitem__(0, True))
        try:
            prompt = _build_prompt(messages, pipe.get_tokenizer(), reasoning)
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
        finally:
            signal.signal(signal.SIGINT, old_handler)
        reply_text = "".join(reply_parts)

        if not reasoning:
            reply_text = re.sub(r'</?think>', '', reply_text).strip()

        elapsed = time.time() - t0
        if stop_flag[0]:
            print(f"\n  ⚠ {TR('已中断', 'Interrupted')}")
            conv.append({"role": "assistant", "content": reply_text + TR(" [已中断]", " [Interrupted]")})
        else:
            conv.append({"role": "assistant", "content": reply_text})
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
        streamer_callback = _make_streamer(reply_parts, stop_flag, thinking_filter=not reasoning)

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
