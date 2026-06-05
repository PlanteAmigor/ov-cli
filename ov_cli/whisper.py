"""
ov-cli whisper: 语音转文字终端。

使用 OpenVINO GenAI WhisperPipeline 转录音频文件。
支持交互式多轮转录和单次模式。
"""

import os, sys, time
import openvino as ov
import openvino_genai as ov_genai
from ov_cli import TR
from ov_cli.chat import readline

# ── 支持的音频格式 ──

SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif", ".au", ".raw"}
FORMAT_HINT = ".wav .mp3 .flac .ogg .aiff"


def _is_audio_file(path):
    """检查文件是否是支持的音频格式。"""
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXTS, ext


def _load_audio(path):
    """加载音频文件，返回 16kHz float32 数组。"""
    import soundfile as sf
    data, sr = sf.read(path)
    if sr != 16000:
        import scipy.signal
        data = scipy.signal.resample(data, int(len(data) * 16000 / sr))
    return data


def _print_help():
    print("  //file PATH  " + TR("转录音频文件", "transcribe audio file"))
    print("  /lang CODE   " + TR("指定语言 (zh/en/ja/ko/fr/de...)", "set language (zh/en/ja/ko/fr/de...)"))
    print("  /help        " + TR("帮助", "help"))
    print("  /exit        " + TR("退出", "quit"))


# ── 加载模型 ──

def load_model(ov_path):
    """加载 WhisperPipeline。"""
    device = "GPU" if "GPU" in ov.Core().available_devices else "CPU"
    print(f"  {TR('加载 WhisperPipeline ({})...', 'Loading WhisperPipeline ({})...').format(device)}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()
    pipe = ov_genai.WhisperPipeline(ov_path, device)
    print(f"✓ ({time.time()-t0:.1f}s)", file=sys.stderr)
    return {"pipe": pipe, "device": device}


# ── 单次转录 ──

def run_once(ctx, file_path, lang=None, output=None, json_output=False):
    """单次转录，输出完自动退出。"""
    import json as _json
    pipe = ctx["pipe"]

    ok, ext = _is_audio_file(file_path)
    if not ok:
        print(f"  ❌ {TR('不支持的文件格式: {}', 'Unsupported format: {}').format(ext)}", file=sys.stderr)
        print(f"     {TR('支持的格式:', 'Supported formats:')} {FORMAT_HINT}", file=sys.stderr)
        return

    print(f"  {TR('⏳ 转录中...', '⏳ Transcribing...')}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()

    data = _load_audio(file_path)
    kwargs = {}
    if lang:
        kwargs["language"] = lang

    result = pipe.generate(data, **kwargs)
    text = result.texts[0] if result.texts else ""
    elapsed = time.time() - t0
    audio_sec = len(data) / 16000
    print(f"✓ ({elapsed:.1f}s, {audio_sec:.0f}s {TR('音频', 'audio')})", file=sys.stderr)

    # stdout: 纯结果
    if json_output:
        print(_json.dumps({"text": text, "time": round(elapsed, 1), "duration": round(audio_sec, 0)}, ensure_ascii=False))
    else:
        print(text)

    # 保存
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
            f.write(f"\n\n<!-- ov-cli whisper | {time.strftime('%Y-%m-%d %H:%M:%S')} | {file_path} -->\n")
        print(f"  {TR('已保存', 'Saved')}: {output}", file=sys.stderr)


# ── 交互模式 ──

def run_whisper(ctx, lang=None):
    """交互式转录终端。"""
    pipe = ctx["pipe"]
    current_lang = lang

    print()
    print("     ▄█▀▀▀▄█▀▀▀▀▄  ▄  ▄  ▄▀▀▀▄  ▄▀▀▀▄")
    print("    █       █    █  █  █  █    █  █    █")
    print("    █       █    █  █  █  █    █  █    █")
    print("    ▀▄▄▄▀  ▀▄▄▄▄▀  ▀▄▄▀  ▀▄▄▄▀  ▀▄▄▄▀")
    print("=" * 50)
    print("  ov-cli " + TR("语音转文字", "Speech to Text"))
    print(f"  {TR('设备', 'Device')}: {ctx['device']} | OpenVINO")
    if current_lang:
        print(f"  {TR('语言', 'Language')}: {current_lang}")
    print("=" * 50)
    _print_help()
    print("=" * 50)
    print()

    while True:
        try:
            line = readline().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line == "/exit":
            break
        elif line == "/help":
            print("=" * 50)
            _print_help()
            print("=" * 50)
            continue
        elif line.startswith("/lang "):
            current_lang = line.split("/lang ", 1)[1].strip()
            print(f"  {TR('语言已设置', 'Language set')}: {current_lang}")
            continue
        elif line.startswith("//file "):
            file_path = line.split("//file ", 1)[1].strip()
        elif line.startswith("/"):
            print(f"  ⚠ {TR('未知命令', 'Unknown command')}: {line}")
            continue
        else:
            # 直接输入路径
            file_path = line

        # 处理文件路径
        file_path = file_path.strip().strip("'\"")
        if not os.path.isfile(file_path):
            print(f"  ❌ {TR('文件不存在', 'File not found')}: {file_path}")
            continue

        ok, ext = _is_audio_file(file_path)
        if not ok:
            print(f"  ❌ {TR('不支持的文件格式: {}', 'Unsupported format: {}').format(ext)}")
            print(f"     {TR('支持的格式:', 'Supported formats:')} {FORMAT_HINT}")
            continue

        print(f"  {TR('📂 {}', '📂 {}').format(os.path.basename(file_path))}", end=" ", flush=True)
        t0 = time.time()

        try:
            data = _load_audio(file_path)
            kwargs = {}
            if current_lang:
                kwargs["language"] = current_lang

            audio_sec = len(data) / 16000
            print(f"({audio_sec:.0f}s {TR('音频', 'audio')})")
            print(f"  {TR('⏳ 转录中...', '⏳ Transcribing...')}", end=" ", flush=True)

            result = pipe.generate(data, **kwargs)
            text = result.texts[0] if result.texts else ""
            elapsed = time.time() - t0
            print(f"✓ ({elapsed:.1f}s)")
            print()
            print("─" * 50)
            print(text)
            print("─" * 50)
        except Exception as e:
            print(f"  ❌ {TR('转录失败', 'Transcription failed')}: {e}")
