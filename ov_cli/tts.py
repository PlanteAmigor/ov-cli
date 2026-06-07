"""
ov-cli tts: 语音合成终端。

支持 OpenVINO Qwen3-TTS 推理（CustomVoice / Base 声音克隆）。
"""

import os, sys, time, json
from pathlib import Path
from ov_cli import TR


# ── 设备选择 ──

def _choose_device():
    """自动选择设备: GPU 优先。"""
    import openvino as ov
    return "GPU" if "GPU" in ov.Core().available_devices else "CPU"


# ── 加载模型 ──

def load_model(ov_path, device=None):
    """加载 Qwen3-TTS OpenVINO 模型。"""
    if device is None:
        device = _choose_device()
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "dlc"))
    from qwen_tts_helper import OVQwen3TTSModel

    print(f"  {TR('加载 TTS 模型 ({})...', 'Loading TTS model ({})...').format(device)}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()
    model = OVQwen3TTSModel.from_pretrained(ov_path, device=device)
    print(f"✓ ({time.time()-t0:.1f}s)", file=sys.stderr)
    mtype = "custom_voice" if model.tts_model_type == "custom_voice" else "base"
    return {"model": model, "device": device, "model_type": mtype}


# ── 模型类型检测 ──

def detect_model_type(ov_path):
    """检测 TTS 模型类型: 'custom_voice' | 'base'。"""
    cfg_path = os.path.join(ov_path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])
    if not any("Qwen3TTS" in a for a in archs):
        return None
    return cfg.get("tts_model_type", "base")


# ── 预热 ──

def _warmup(model, mtype):
    """用短文本预热 TTS 模型。"""
    if mtype != "custom_voice":
        return  # Base 需要 ref_audio，无法预热
    print(f"  {TR('预热中...', 'Warming up...')}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()
    model.generate_custom_voice(text="测试", language="chinese", speaker=None, instruct=None)
    print(f"✓ ({time.time()-t0:.1f}s)", file=sys.stderr)


# ── 管道模式 ──

def run_pipe(ctx, speaker=None, language=None, instruct=None, ref_audio=None, warmup=True):
    """管道模式：从 stdin 读文本，向 stdout 写 JSON 结果（音频路径）。
    模型常驻内存，逐条合成语音。

    用法:
      echo '你好' | ov-cli tts --model ./0.6B-CV-ov --mode pipe --speaker Vivian
    """
    import json as _json, soundfile as sf
    model = ctx["model"]
    mtype = ctx["model_type"]

    if warmup:
        _warmup(model, mtype)

    print(f"  🧪 {TR('管道模式已启动 (stdin/stdout)', 'Pipe mode started (stdin/stdout)')}", file=sys.stderr)
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            text = line.strip()
            if not text:
                continue

            t0 = time.time()
            try:
                if mtype == "custom_voice":
                    if not speaker:
                        print(_json.dumps({"error": "CustomVoice 需要 --speaker"}), flush=True)
                        continue
                    wavs, sr = model.generate_custom_voice(
                        text=text, language=language or "auto",
                        speaker=speaker, instruct=instruct,
                    )
                else:
                    if not ref_audio:
                        print(_json.dumps({"error": "Base 模型需要 --ref-audio"}), flush=True)
                        continue
                    wavs, sr = model.generate_voice_clone(
                        text=text, language=language or "auto",
                        ref_audio=ref_audio, x_vector_only_mode=True,
                    )
            except Exception as e:
                print(_json.dumps({"error": str(e)[:200]}), flush=True)
                continue

            elapsed = time.time() - t0
            audio_len = len(wavs[0]) / sr if len(wavs[0]) > 0 else 0

            safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in text)[:40]
            fname = f"pipe_{time.strftime('%Y%m%d_%H%M%S')}_{safe}.wav"
            os.makedirs("outputs", exist_ok=True)
            path = os.path.join("outputs", fname)
            sf.write(path, wavs[0], sr)

            print(_json.dumps({
                "path": path, "text": text,
                "time": round(elapsed, 1), "duration": round(audio_len, 1),
            }, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        pass


# ── 单次推理 ──

def run_once(ctx, text, output=None, speaker=None, language=None, instruct=None,
             ref_audio=None, warmup=True, json_output=False):
    """单次 TTS 生成，输出完自动退出。"""
    import soundfile as sf
    model = ctx["model"]
    mtype = ctx["model_type"]

    # 预热
    if warmup:
        _warmup(model, mtype)

    print(f"  {TR('⏳ 生成语音中...', '⏳ Generating speech...')}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()

    try:
        if mtype == "custom_voice":
            if not speaker:
                print(f"\n  ⚠ {TR('CustomVoice 模型需要 --speaker 参数', 'CustomVoice model requires --speaker')}")
                speakers = model.get_supported_speakers()
                print(f"  {TR('可选声音', 'Available speakers')}: {', '.join(speakers)}")
                sys.exit(1)
            wavs, sr = model.generate_custom_voice(
                text=text, language=language or "auto",
                speaker=speaker, instruct=instruct,
            )
        else:  # base
            if not ref_audio:
                print(f"\n  ⚠ {TR('Base 模型需要 --ref-audio 参考音频', 'Base model requires --ref-audio')}")
                sys.exit(1)
            wavs, sr = model.generate_voice_clone(
                text=text, language=language or "auto",
                ref_audio=ref_audio, x_vector_only_mode=True,
            )
    except Exception as e:
        print(f"✗", file=sys.stderr)
        print(f"  {TR('语音生成失败', 'TTS failed')}: {str(e)[:200]}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t0
    audio_len = len(wavs[0]) / sr if len(wavs[0]) > 0 else 0

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        sf.write(output, wavs[0], sr)
        path = output
    else:
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in text)[:40]
        fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe}.wav"
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", fname)
        sf.write(path, wavs[0], sr)

    print(f"✓ ({elapsed:.1f}s, {audio_len:.1f}s)", file=sys.stderr)
    print(f"  {TR('💾 已保存', '💾 Saved')}: {path}", file=sys.stderr)

    if json_output:
        print(json.dumps({"path": path, "time": round(elapsed, 1), "duration": round(audio_len, 1)},
                         ensure_ascii=False))
    else:
        print(path)

    return wavs, sr
