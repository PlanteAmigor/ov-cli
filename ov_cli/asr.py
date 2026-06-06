"""
ov-cli asr: 语音转文字终端。

支持 Whisper (GenAI Pipeline) 和 Qwen3-ASR (自定义 OV)。
自动识别模型类型，无需手动指定。
"""

import os, sys, time, json, subprocess
from pathlib import Path
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

_ASR_TF_VERSION = "4.57.6"  # Qwen3-ASR 需要 transformers 4.x


def _pip_version(pkg):
    """用 pip list 查版本，不触发 import。"""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=columns"],
            timeout=10, text=True,
        )
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].lower().replace("-", "_") == pkg.lower().replace("-", "_"):
                return parts[1]
        return None
    except Exception:
        return None


def _ensure_qwen_asr_tf():
    """Qwen3-ASR 推理前确保 transformers 版本兼容，返回是否需恢复。"""
    cur = _pip_version("transformers")
    if cur and cur.startswith("4."):
        # 已经在 4.x，检查 qwen-asr / qwen-tts 是否兼容
        return False
    old_hf = _pip_version("huggingface_hub")
    print(f"  ⚡ Qwen3-ASR 需要 transformers 4.x（当前 {cur}），临时切换...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
         f"transformers=={_ASR_TF_VERSION}"],
        timeout=120,
    )
    # huggingface_hub 也要切到 0.x 以兼容 4.x transformers
    if old_hf and old_hf.startswith("1."):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
             "huggingface_hub==0.36.2"],
            timeout=60,
        )
    print(f"  ✓ 已切换至 transformers {_ASR_TF_VERSION}")
    return True


def _restore_tf(need_restore):
    """恢复 transformers 到最新版。"""
    if not need_restore:
        return
    print(f"  ⚡ 恢复 transformers...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
         "transformers"],
        timeout=120,
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
         "huggingface_hub"],
        timeout=60,
    )
    print(f"  ✓ 已恢复至 transformers {_pip_version('transformers')}")


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


def load_model(ov_path):
    """加载 ASR 模型，自动识别 Whisper / Qwen3-ASR。"""
    mtype = _detect_asr_type(ov_path)
    if mtype == "qwen3_asr":
        return _load_qwen3_asr(ov_path)
    return _load_whisper(ov_path)


def _load_whisper(ov_path):
    """加载 WhisperPipeline。"""
    device = "GPU" if "GPU" in ov.Core().available_devices else "CPU"
    print(f"  {TR('加载 WhisperPipeline ({})...', 'Loading WhisperPipeline ({})...').format(device)}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()
    pipe = ov_genai.WhisperPipeline(ov_path, device)
    print(f"✓ ({time.time()-t0:.1f}s)", file=sys.stderr)
    return {"pipe": pipe, "device": device, "asr_type": "whisper"}


def _load_qwen3_asr(ov_path):
    """加载 Qwen3-ASR OpenVINO 模型（自动切 transformers 版本）。"""
    need_restore = _ensure_qwen_asr_tf()
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "dlc"))
    from qwen_3_asr_helper import OVQwen3ASRModel

    device = "GPU" if "GPU" in ov.Core().available_devices else "CPU"
    print(f"  {TR('加载 Qwen3-ASR ({})...', 'Loading Qwen3-ASR ({})...').format(device)}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()
    try:
        model = OVQwen3ASRModel.from_pretrained(ov_path, device=device)
        print(f"✓ ({time.time()-t0:.1f}s)", file=sys.stderr)
        return {"model": model, "device": device, "asr_type": "qwen3_asr", "_need_restore": need_restore}
    except Exception:
        _restore_tf(need_restore)
        raise


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
    """Qwen3-ASR 单次转录。"""
    model = ctx["model"]
    results = model.transcribe(audio=file_path, language=lang)
    return results[0].text if results else ""


def run_once(ctx, file_path, lang=None, output=None, json_output=False):
    """单次转录，输出完自动退出。"""
    import json as _json
    need_restore = ctx.get("_need_restore", False)

    ok, ext = _is_audio_file(file_path)
    if not ok:
        print(f"  ❌ {TR('不支持的文件格式: {}', 'Unsupported format: {}').format(ext)}", file=sys.stderr)
        print(f"     {TR('支持的格式:', 'Supported formats:')} {FORMAT_HINT}", file=sys.stderr)
        _restore_tf(need_restore)
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
        _restore_tf(need_restore)
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

    _restore_tf(need_restore)


# ── 交互模式 ──

def run_whisper(ctx, lang=None):
    """交互式转录终端。"""
    pipe = ctx.get("pipe")  # Whisper only
    model = ctx.get("model")  # Qwen3-ASR only
    asr_type = ctx.get("asr_type", "whisper")
    current_lang = lang
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

    need_restore = ctx.get("_need_restore", False)
    try:
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
                    results = model.transcribe(audio=file_path, language=current_lang)
                    text = results[0].text if results else ""
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
    finally:
        _restore_tf(need_restore)
