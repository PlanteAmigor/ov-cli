"""
ov-cli asr: 语音转文字终端。

支持 Whisper (WhisperPipeline) 和 Qwen3-ASR (ASRPipeline)。
自动识别模型类型，无需手动指定。
"""

import os, sys, time, json
import openvino as ov
import openvino_genai as ov_genai
from ov_cli import TR, has_gpu
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

# Qwen3-ASR (ASRPipeline) 的 language 使用语言名，CLI 侧是短码，做映射
_LANG_MAP = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "en": "English",
    "en-us": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese",
    "ar": "Arabic",
    "hi": "Hindi",
}


def _map_lang(lang):
    """将 CLI 语言短码映射为 Qwen3-ASR 使用的语言名；未知则原样返回。"""
    if not lang:
        return None
    return _LANG_MAP.get(lang.lower(), lang)


def _detect_asr_type(ov_path):
    """检测 ASR 模型类型: 'whisper' | 'qwen3_asr'。"""
    cfg_path = os.path.join(ov_path, "config.json")
    if not os.path.isfile(cfg_path):
        return "whisper"
    with open(cfg_path) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])
    if any("Qwen3ASR" in a for a in archs):
        return "qwen3_asr"
    return "whisper"


def load_model(ov_path, device=None):
    """加载 ASR 模型，自动识别 Whisper / Qwen3-ASR。"""
    mtype = _detect_asr_type(ov_path)
    if mtype == "qwen3_asr":
        return _load_qwen3_asr(ov_path, device=device)
    return _load_whisper(ov_path, device=device)


def _load_whisper(ov_path, device=None):
    """加载 WhisperPipeline。"""
    if not device:
        device = "GPU" if has_gpu() else "CPU"
    print(f"  {TR('加载 WhisperPipeline ({})...', 'Loading WhisperPipeline ({})...').format(device)}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()
    pipe = ov_genai.WhisperPipeline(ov_path, device)
    print(f"✓ ({time.time()-t0:.1f}s)", file=sys.stderr)
    return {"pipe": pipe, "device": device, "asr_type": "whisper"}


def _load_qwen3_asr(ov_path, device=None):
    """加载 Qwen3-ASR 官方 ASRPipeline。"""
    if not device:
        device = "GPU" if has_gpu() else "CPU"
    print(f"  {TR('加载 Qwen3-ASR ASRPipeline ({})...', 'Loading Qwen3-ASR ASRPipeline ({})...').format(device)}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()
    pipe = ov_genai.ASRPipeline(ov_path, device)
    print(f"✓ ({time.time()-t0:.1f}s)", file=sys.stderr)
    return {"pipe": pipe, "device": device, "asr_type": "qwen3_asr"}


# ── 单次转录 ──

def _transcribe_whisper(ctx, file_path, lang):
    """Whisper 单次转录。"""
    pipe = ctx["pipe"]
    data = _load_audio(file_path)
    kwargs = {}
    if lang:
        kwargs["language"] = lang
    result = pipe.generate(data, **kwargs)
    return result.texts[0] if result.texts else ""


def _transcribe_qwen_asr(ctx, file_path, lang):
    """Qwen3-ASR 单次转录（ASRPipeline，输入 16k 归一化 float 列表）。"""
    pipe = ctx["pipe"]
    data = _load_audio(file_path)
    if data.ndim > 1:  # 立体声转单声道
        data = data.mean(axis=1)
    kwargs = {}
    mapped = _map_lang(lang)
    if mapped:
        kwargs["language"] = mapped
    result = pipe.generate(data.tolist(), **kwargs)
    return result.texts[0] if result.texts else ""


def run_once(ctx, file_path, lang=None, output=None, json_output=False):
    """单次转录，输出完自动退出。"""
    import json as _json

    ok, ext = _is_audio_file(file_path)
    if not ok:
        print(f"  ❌ {TR('不支持的文件格式: {}', 'Unsupported format: {}').format(ext)}", file=sys.stderr)
        print(f"     {TR('支持的格式:', 'Supported formats:')} {FORMAT_HINT}", file=sys.stderr)
        return

    print(f"  {TR('⏳ 转录中...', '⏳ Transcribing...')}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()

    try:
        if ctx.get("asr_type") == "qwen3_asr":
            text = _transcribe_qwen_asr(ctx, file_path, lang)
        else:
            text = _transcribe_whisper(ctx, file_path, lang)
    except Exception as e:
        print(f"✗", file=sys.stderr)
        print(f"  {TR('转录失败', 'Transcription failed')}: {str(e)[:200]}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t0
    audio_sec = 0
    if ctx.get("asr_type") == "qwen3_asr":
        # Qwen3-ASR 内部分帧，直接输出
        pass
    else:
        data = _load_audio(file_path)
        audio_sec = len(data) / 16000
    print(f"✓ ({elapsed:.1f}s)", file=sys.stderr)

    if json_output:
        print(_json.dumps({"text": text, "time": round(elapsed, 1), "duration": round(audio_sec, 0)},
                         ensure_ascii=False))
    else:
        print(text)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
            f.write(f"\n\n<!-- ov-cli asr | {time.strftime('%Y-%m-%d %H:%M:%S')} | {file_path} -->\n")
        print(f"  {TR('已保存', 'Saved')}: {output}", file=sys.stderr)


# ── 管道模式 ──

def run_pipe(ctx, lang=None):
    """管道模式：从 stdin 读音频路径，向 stdout 写 JSON 结果。
    模型常驻内存，每条结果 0.5s 内返回。

    用法:
      echo /path/to/audio.wav | ov-cli asr --model ./model --mode pipe
    """
    import json as _json
    asr_type = ctx.get("asr_type", "whisper")

    print(f"  🧪 {TR('管道模式已启动 (stdin/stdout)', 'Pipe mode started (stdin/stdout)')}", file=sys.stderr)
    while True:
            line = sys.stdin.readline()
            if not line:
                break
            path = line.strip()
            if not path:
                continue

            ok, _ = _is_audio_file(path)
            if not ok:
                print(_json.dumps({"error": f"unsupported format: {path}"}), flush=True)
                continue
            if not os.path.isfile(path):
                print(_json.dumps({"error": f"file not found: {path}"}), flush=True)
                continue

            t0 = time.time()
            try:
                if asr_type == "qwen3_asr":
                    text = _transcribe_qwen_asr(ctx, path, lang)
                else:
                    text = _transcribe_whisper(ctx, path, lang)
            except Exception as e:
                print(_json.dumps({"error": str(e)[:200]}), flush=True)
                continue

            elapsed = time.time() - t0
            print(_json.dumps({"text": text, "time": round(elapsed, 1)}, ensure_ascii=False), flush=True)


# ── 交互模式 ──

def run_whisper(ctx, lang=None):
    """交互式转录终端。"""
    pipe = ctx.get("pipe")  # Whisper / Qwen3-ASR 共用
    asr_type = ctx.get("asr_type", "whisper")
    current_lang = lang

    print()
    print("        ██████╗ ██╗   ██╗     ██████╗██╗     ██╗   ")
    print("       ██╔═══██╗██║   ██║ █  ██╔════╝██║     ██║ ")
    print("     █ ██║   ██║██║   ██║███ ██║     ██║     ██║█")
    print("   ███ ██║   ██║╚██╗ ██╔╝███ ██║     ██║     ██║███")
    print("██████ ╚██████╔╝ ╚████╔╝█████╚██████╗███████╗██║██████ ")
    print("        ╚═════╝   ╚═══╝       ╚═════╝╚══════╝╚═╝")
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
                file_path = line

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
                if asr_type == "qwen3_asr":
                    text = _transcribe_qwen_asr(ctx, file_path, current_lang)
                    elapsed = time.time() - t0
                    print(f"✓ ({elapsed:.1f}s)")
                else:
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
