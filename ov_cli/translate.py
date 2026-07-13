"""
ov-cli translate: 翻译终端。

使用 LLM/VLM 模型进行文本/图片翻译。
支持与 chat 相同的文件加载功能（//img, //txt, //pdf）。
"""

import os
import sys
import time
import threading
import re

import numpy as np
import openvino as ov
import openvino_genai as ov_genai

from . import TR
from .chat import (
    _make_streamer,
    _build_prompt,
    _make_genai_config,
    _load_image,
    _load_text_file,
    _count_tokens,
    readline,
)


# ── 翻译语言映射：代码 → (中文名, 英文名) ──────────────
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


# ── 翻译 prompt 模板 ──────────────────────────────────────
# 当 ctx 中未提供时使用这些默认值
_T_ZH = "将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
_T_EN = "Translate the following text into {target}. Note that you should only output the translated result without any additional explanation:\n\n{text}"


def run_translate(ctx, max_tokens=512):
    """翻译模式入口。"""
    pipe = ctx.get("pipe")
    is_vlm = ctx.get("is_vlm", False)
    t_zh = _T_ZH
    t_en = _T_EN

    # ── 语言列表（按代码排序，常用语言排前） ──────────
    lang_codes = sorted(TRANSLATE_LANGS.keys(),
                        key=lambda c: (0, c) if c in ("zh","en","ja","ko","fr","de","es","pt","ru","ar","it","tr","th","vi") else (1, c))
    lang_items = []
    for c in lang_codes:
        zh_name, en_name = TRANSLATE_LANGS[c]
        name = TR(zh_name, en_name)
        lang_items.append(f"{c}={name}")
    lang_lines = []
    for i in range(0, len(lang_items), 4):
        row = lang_items[i:i+4]
        lang_lines.append("  " + "  ".join(f"{item:16s}" for item in row))
    lang_display = "\n".join(lang_lines)

    # ── 横幅 ────────────────────────────────────────
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
    if is_vlm:
        print("  //img PATH  " + TR("加载图片", "load image"))
    print("  //txt PATH  " + TR("加载文本文件", "load text file"))
    print(f"  " + TR("支持语言", "Supported codes") + f":\n{lang_display}")
    print("  /temp N     " + TR("温度 (0-2)", "temperature"))
    print("  /help       " + TR("帮助", "help"))
    print("  /exit       " + TR("退出", "quit"))
    print("=" * 50)
    print()

    # ── 状态 ────────────────────────────────────────
    loaded_files = []  # {id, path, type, pages:[PIL]}
    _next_id = 1

    while True:
        try:
            text = readline()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue

        # ── 指令 ────────────────────────────────────
        if text in ("/exit", "exit", TR("退出", "exit")):
            break
        if text in ("/help", "help", TR("帮助", "help")):
            print("  //" + TR("语言代码 文本 → 指定目标语言", "lang_code text → force target language"))
            print("  " + TR("例如", "e.g.") + ": //ja おはよう, //fr Bonjour")
            if is_vlm:
                print("  //img PATH  " + TR("加载图片", "load image"))
            print("  //txt PATH  " + TR("加载文本文件", "load text file"))
            print("  /temp N     " + TR("温度 (0-2)", "temperature"))
            print("  /exit       " + TR("退出", "quit"))
            print()
            continue
        if text.startswith("/temp "):
            try:
                temperature = max(0, min(2, float(text[6:])))
                print(f"  temperature = {temperature}")
            except ValueError:
                print("  ⚠ /temp 0.7")
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
                    loaded_files.append({"id": _next_id, "path": img_path, "type": "image",
                                         "pages": [_load_image(img_path)]})
                    print(f"  #{_next_id} ✓ {TR('已加载图片', 'Image loaded')}: {img_path}")
                    _next_id += 1
                else:
                    print(f"  ⚠ {TR('找不到图片', 'Image not found')}: {img_path}")
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
                    content = _load_text_file(txt_path)
                    loaded_files.append({"id": _next_id, "path": txt_path, "type": "text", "content": content})
                    print(f"  #{_next_id} ✓ {TR('已加载文件', 'File loaded')}: {txt_path}")
                    _next_id += 1
                else:
                    print(f"  ⚠ {TR('找不到文件', 'File not found')}: {txt_path}")
            print()
            continue

        # ── 解析目标语言 ────────────────────────────
        temperature = 0.7  # 每次翻译重置 temperature
        force_target = None
        if text.startswith("//") and len(text) > 2:
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
                force_target = code
                text = rest
            else:
                print("  ⚠ " + TR("未知语言代码。可用 /help 查看支持的语言",
                                  "Unknown language code. Use /help for supported codes"))
                continue
        elif text.startswith("//"):
            print("  ⚠ " + TR("未知指令", "Unknown command"))
            continue

        if force_target:
            target_lang = force_target
        else:
            from ov_cli import _LANG
            if _LANG == "en":
                _, en_name = TRANSLATE_LANGS["en"]
                target_lang = TR(TRANSLATE_LANGS["en"][0], en_name)
            else:
                target_lang = TR(TRANSLATE_LANGS["zh"][0], TRANSLATE_LANGS["zh"][1])

        # ── 构建 prompt ─────────────────────────────
        all_pages = []
        txt_prefix = ""
        for f in loaded_files:
            if f["type"] == "text":
                fname = os.path.basename(f["path"])
                txt_prefix += f"[文件 {fname}]\n```\n{f['content']}\n```\n\n"
            else:
                all_pages.extend(f["pages"])
        loaded_files.clear()

        user_text = text
        if txt_prefix:
            user_text = txt_prefix + user_text

        def has_chinese(s):
            return any('\u4e00' <= c <= '\u9fff' for c in s[:30])

        if all_pages and is_vlm:
            # VLM 翻译：图片 + 文字一起发
            prompt_text = user_text
        else:
            # 纯文本翻译
            if has_chinese(user_text):
                prompt_text = t_zh.format(target=target_lang, text=user_text)
            else:
                prompt_text = t_en.format(target=target_lang, text=user_text)

        # ── 生成 ────────────────────────────────────
        gen_cfg = _make_genai_config(temperature=0, max_tokens=max_tokens)

        print(f"  → {target_lang}", flush=True)
        t0 = time.time()
        sys.stdout.write("  ")
        sys.stdout.flush()

        reply_parts = []
        stop_flag = [False]
        streamer_cb = _make_streamer(reply_parts, stop_flag, thinking_filter=True)

        try:
            if all_pages and is_vlm:
                # VLM 路径：图片 + 文字
                img_tag = "<|vision_start|><|image_pad|><|vision_end|>\n"
                final_prompt = img_tag * len(all_pages) + prompt_text
                image_tensors = [ov.Tensor(np.array(img)[None]) for img in all_pages]
                pipe.generate(final_prompt, generation_config=gen_cfg,
                              streamer=streamer_cb, images=image_tensors)
            else:
                # 纯文本路径
                messages = [{"role": "user", "content": prompt_text}]
                full_prompt = _build_prompt(messages, pipe.get_tokenizer(), enable_thinking=True)
                pipe.generate(full_prompt, gen_cfg, streamer_cb)
        except RuntimeError as e:
            print(f"\n  ⚠ {TR('生成失败', 'Generation failed')}: {str(e)[:200]}")

        elapsed = time.time() - t0
        reply_text = "".join(reply_parts)
        char_count = len(reply_text.replace(" ", ""))
        tok_count = _count_tokens(ctx, reply_text)
        print()
        print(f"  [{elapsed:.1f}s | {char_count} chars | {char_count/elapsed:.1f} ch/s | {tok_count/elapsed:.1f} tok/s]")
        print()


# ── 工具函数 ────────────────────────────────────────────────

def _has_chinese(text):
    return any('\u4e00' <= c <= '\u9fff' for c in text[:30])


def _build_translate_prompt(text, target_lang, ctx=None):
    """构建翻译 prompt。"""
    t_zh = _T_ZH
    t_en = _T_EN
    if _has_chinese(text):
        return t_zh.format(target=target_lang, text=text)
    else:
        return t_en.format(target=target_lang, text=text)


def _resolve_target_lang(lang_code):
    """将语言代码解析为完整语言名。"""
    if lang_code and lang_code in TRANSLATE_LANGS:
        zh_name, en_name = TRANSLATE_LANGS[lang_code]
        return TR(zh_name, en_name)
    return lang_code or ""


# ── Once 模式（单次翻译） ──────────────────────────────────

def run_once(ctx, prompt="", files=None, lang=None, output=None,
             max_tokens=512, json_output=False):
    """单次翻译。"""
    import json as _json
    pipe = ctx.get("pipe")
    is_vlm = ctx.get("is_vlm", False)

    target_lang = _resolve_target_lang(lang) if lang else _resolve_target_lang("zh" if _has_chinese(prompt) else "en")

    # 加载文件
    all_pages = []
    txt_prefix = ""
    if files:
        for fpath in files:
            fpath = os.path.abspath(fpath)
            if not os.path.isfile(fpath):
                print(f"  ⚠ {TR('找不到文件', 'File not found')}: {fpath}", file=sys.stderr)
                continue
            ext = os.path.splitext(fpath)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp") and is_vlm:
                all_pages.append(_load_image(fpath))
                print(f"  ✓ {TR('已加载图片', 'Image loaded')}: {fpath}", file=sys.stderr)
            elif ext in (".txt", ".md", ".json", ".py", ".c", ".cpp", ".h", ".hpp",
                         ".yaml", ".yml", ".toml", ".xml", ".csv", ".sh", ".env",
                         ".conf", ".cfg", ".ini", ".log", ".rst", ".tex", ".sql",
                         ".js", ".ts", ".tsx", ".jsx", ".vue", ".css", ".scss",
                         ".go", ".rs", ".java", ".kt", ".swift", ".rb", ".php",
                         ".pl", ".lua", ".r", ".m", ".mm"):
                txt_prefix += f"[{os.path.basename(fpath)}]\n```\n{_load_text_file(fpath)}\n```\n\n"
                print(f"  ✓ {TR('已加载文件', 'File loaded')}: {fpath}", file=sys.stderr)
            else:
                print(f"  ⚠ {TR('不支持的文件类型', 'Unsupported file type')}: {fpath}", file=sys.stderr)

    user_text = prompt
    if txt_prefix:
        user_text = txt_prefix + user_text

    # 构建翻译 prompt
    if all_pages and is_vlm:
        # VLM 翻译：文字直接作为指令，不需要翻译模板
        prompt_text = user_text
        if target_lang:
            prompt_text = f"Translate the text in the image into {target_lang}. Also translate any accompanying text:\n\n{user_text}"
    else:
        prompt_text = _build_translate_prompt(user_text, target_lang, ctx)

    gen_cfg = _make_genai_config(temperature=0, max_tokens=max_tokens)
    tokenizer = pipe.get_tokenizer()

    t0 = time.time()
    if all_pages and is_vlm:
        img_tag = "<|vision_start|><|image_pad|><|vision_end|>\n"
        final_prompt = img_tag * len(all_pages) + prompt_text
        image_tensors = [ov.Tensor(np.array(img)[None]) for img in all_pages]
        result = pipe.generate(final_prompt, generation_config=gen_cfg, images=image_tensors)
    else:
        messages = [{"role": "user", "content": prompt_text}]
        full_prompt = _build_prompt(messages, tokenizer, enable_thinking=True)
        result = pipe.generate(full_prompt, gen_cfg)

    elapsed = time.time() - t0
    reply_text = str(result).strip() if result else ""

    # 输出
    if output:
        out_path = output
        if os.path.isdir(out_path) or out_path.endswith(os.sep):
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_path, f"{ts}.md")
        meta = f"<!-- ov-cli translate | {time.strftime('%Y-%m-%d %H:%M:%S')} | lang: {lang or 'auto'} -->\n\n"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(meta + reply_text)
        print(f"  💾 {TR('已保存', 'Saved')}: {out_path}", file=sys.stderr)

    if json_output:
        print(_json.dumps({"text": reply_text, "time": round(elapsed, 1)}, ensure_ascii=False))
    else:
        print(reply_text)


# ── Pipe 模式（管道翻译） ─────────────────────────────────

def run_pipe(ctx, lang=None, max_tokens=512, temperature=0):
    """管道翻译：从 stdin 读文本，向 stdout 写 JSON 结果。"""
    import json as _json
    pipe = ctx.get("pipe")

    target_lang = _resolve_target_lang(lang) if lang else ""
    print(f"  🧪 {TR('翻译管道模式已启动', 'Translate pipe mode started')}", file=sys.stderr)
    if target_lang:
        print(f"  → {TR('目标语言', 'Target')}: {target_lang}", file=sys.stderr)

    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            text = line.strip()
            if not text:
                continue

            # 自动检测目标语言
            tl = target_lang if target_lang else _resolve_target_lang("zh" if _has_chinese(text) else "en")
            prompt_text = _build_translate_prompt(text, tl, ctx)
            gen_cfg = _make_genai_config(temperature=temperature, max_tokens=max_tokens)
            tokenizer = pipe.get_tokenizer()
            messages = [{"role": "user", "content": prompt_text}]
            full_prompt = _build_prompt(messages, tokenizer, enable_thinking=True)

            t0 = time.time()
            try:
                result = pipe.generate(full_prompt, gen_cfg)
            except RuntimeError as e:
                print(_json.dumps({"error": str(e)[:200]}, ensure_ascii=False), flush=True)
                continue

            elapsed = time.time() - t0
            resp = str(result).strip()
            print(_json.dumps({"text": resp, "time": round(elapsed, 1)}, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        pass
