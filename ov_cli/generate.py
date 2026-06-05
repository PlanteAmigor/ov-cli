"""
ov-cli generate: 生成终端。

支持文生图 (Text2ImagePipeline) 和语音合成 (OpenVINO Qwen3-TTS)。
自动识别模型类型，无需手动指定。
"""

import os, sys, time, json
from pathlib import Path
import openvino_genai as ov_genai
from PIL import Image
from ov_cli import TR
from ov_cli.chat import readline


# ── 默认参数 ──

_DEFAULT_WIDTH = 512
_DEFAULT_HEIGHT = 512
_DEFAULT_STEPS = 4
_DEFAULT_GUIDANCE = 0.0
_DEFAULT_SAVE_DIR = "outputs"


# ── 模型类型识别 ──

def _detect_model_type(ov_path):
    """识别模型类型: 'txt2img' | 'tts_cv' | 'tts_base'。"""
    cfg_path = os.path.join(ov_path, "config.json")
    if not os.path.isfile(cfg_path):
        return "txt2img"
    with open(cfg_path) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])
    if any("Qwen3TTS" in a for a in archs):
        tts_type = cfg.get("tts_model_type", "base")
        if tts_type == "custom_voice":
            return "tts_cv"
        return "tts_base"
    return "txt2img"


# ── 加载模型 ──

def _choose_device():
    """自动选择设备: GPU 优先。"""
    import openvino as ov
    return "GPU" if "GPU" in ov.Core().available_devices else "CPU"


def load_model(ov_path):
    """加载 Text2ImagePipeline。"""
    device = _choose_device()
    print(f"  {TR('加载 Text2ImagePipeline ({})...', 'Loading Text2ImagePipeline ({})...').format(device)}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()
    pipe = ov_genai.Text2ImagePipeline(ov_path, device)
    print(f"✓ ({time.time()-t0:.1f}s)", file=sys.stderr)
    return {"pipe": pipe, "device": device, "model_type": "txt2img"}


def load_tts_model(ov_path, device=None):
    """加载 OpenVINO Qwen3-TTS 模型。"""
    if device is None:
        device = _choose_device()
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "dlc"))
    from qwen_tts_helper import OVQwen3TTSModel

    print(f"  {TR('加载 TTS 模型 ({})...', 'Loading TTS model ({})...').format(device)}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()
    model = OVQwen3TTSModel.from_pretrained(ov_path, device=device)
    print(f"✓ ({time.time()-t0:.1f}s)", file=sys.stderr)
    mtype = "tts_cv" if model.tts_model_type == "custom_voice" else "tts_base"
    return {"model": model, "device": device, "model_type": mtype}


# ── TTS 单次推理 ──

def _warmup_tts(model, device):
    """用短文本预热 TTS 模型。"""
    print(f"  {TR('预热中...', 'Warming up...')}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()
    mtype = model.tts_model_type
    if mtype == "custom_voice":
        model.generate_custom_voice(text="测试", language="chinese", speaker=None, instruct=None)
    else:
        # Base 模型必须有 ref_audio 才能预热，跳过
        pass
    print(f"✓ ({time.time()-t0:.1f}s)", file=sys.stderr)


def run_tts_once(ctx, text, output=None, speaker=None, language=None, instruct=None,
                 ref_audio=None, warmup=True, json_output=False):
    """单次 TTS 生成，输出完自动退出。"""
    import soundfile as sf
    model = ctx["model"]
    mtype = ctx["model_type"]
    device = ctx["device"]

    # 预热
    if warmup and mtype == "tts_cv":
        _warmup_tts(model, device)

    # 执行推理
    print(f"  {TR('⏳ 生成语音中...', '⏳ Generating speech...')}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()

    try:
        if mtype == "tts_cv":
            if not speaker:
                print(f"\n  ⚠ {TR('CustomVoice 模型需要 --speaker 参数', 'CustomVoice model requires --speaker')}")
                speakers = model.get_supported_speakers()
                print(f"  {TR('可选声音', 'Available speakers')}: {', '.join(speakers)}")
                sys.exit(1)
            wavs, sr = model.generate_custom_voice(
                text=text, language=language or "auto",
                speaker=speaker, instruct=instruct,
            )
        else:  # tts_base
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

    # 保存
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


# ── 单次生图 ──

def run_once(ctx, prompt, output=None, width=_DEFAULT_WIDTH, height=_DEFAULT_HEIGHT,
             steps=_DEFAULT_STEPS, guidance=_DEFAULT_GUIDANCE, seed=None, json_output=False):
    """单次生图，输出完自动退出。"""
    import json as _json
    pipe = ctx["pipe"]
    print(f"  {TR('⏳ 生成中...', '⏳ Generating...')}", end=" ", flush=True, file=sys.stderr)
    t0 = time.time()

    kwargs = {"width": width, "height": height, "num_inference_steps": steps,
              "guidance_scale": guidance}
    if seed is not None:
        kwargs["rng_seed"] = seed

    try:
        result = pipe.generate(prompt, **kwargs)
    except Exception as e:
        print(f"✗", file=sys.stderr)
        print(f"  {TR('生图失败', 'Generation failed')}: {str(e)[:200]}", file=sys.stderr)
        sys.exit(1)

    img = Image.fromarray(result.data[0])
    elapsed = time.time() - t0

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        img.save(output)
        path = output
    else:
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:40]
        fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe}.png"
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", fname)
        img.save(path)

    print(f"✓ ({elapsed:.1f}s)", file=sys.stderr)
    print(f"  {TR('💾 已保存', '💾 Saved')}: {path}", file=sys.stderr)

    # stdout: 路径 / JSON
    if json_output:
        print(_json.dumps({"path": path, "time": round(elapsed, 1)}, ensure_ascii=False))
    else:
        print(path)

    return img


# ── 交互式生图 ──

def run_generate(ctx, width=_DEFAULT_WIDTH, height=_DEFAULT_HEIGHT,
                 steps=_DEFAULT_STEPS, guidance=_DEFAULT_GUIDANCE,
                 seed=None, save_dir=_DEFAULT_SAVE_DIR):
    """交互式生图终端。"""
    pipe = ctx["pipe"]
    os.makedirs(save_dir, exist_ok=True)
    history = []

    _logo_lines = [
        '        ██████╗ ██╗   ██╗     ██████╗██╗     ██╗',
        '       ██╔═══██╗██║   ██║    ██╔════╝██║     ██║',
        '       ██║   ██║██║   ██║    ██║     ██║     ██║',
        '       ██║   ██║╚██╗ ██╔╝    ██║     ██║     ██║',
        '       ╚██████╔╝ ╚████╔╝     ╚██████╗███████╗██║',
        '        ╚═════╝   ╚═══╝       ╚═════╝╚══════╝╚═╝',
    ]
    _pool = [17, 23, 30, 148, 226, 208, 218, 224]
    print()
    for i, line in enumerate(_logo_lines):
        n = len(line)
        seg = n // 4
        parts = [line[j:j+seg] for j in range(0, n, seg)]
        colored = ''
        for k, p in enumerate(parts):
            ci = min(i + k, len(_pool) - 1)
            colored += f'\033[38;5;{_pool[ci]}m{p}\033[0m'
        print(colored)
    print("=" * 50)
    print("  ov-cli " + TR("文生图终端", "Image Generation"))
    print(f"  {TR('设备', 'Device')}: {ctx['device']} | OpenVINO")
    print("=" * 50)
    _print_help()

    while True:
        try:
            line = readline().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            parts = line.split()
            cmd = parts[0]

            if cmd in ("/exit", "/quit"):
                break
            elif cmd == "/help":
                _print_help()
            elif cmd == "/size" and len(parts) >= 3:
                try:
                    width, height = int(parts[1]), int(parts[2])
                    print(f"  ✓ {TR('尺寸:', 'Size:')} {width}x{height}")
                except ValueError:
                    print(f"  {TR('格式:', 'Usage:')} /size W H")
            elif cmd == "/steps" and len(parts) >= 2:
                try:
                    steps = max(1, int(parts[1]))
                    print(f"  ✓ {TR('步数:', 'Steps:')} {steps}")
                except ValueError:
                    print(f"  {TR('格式:', 'Usage:')} /steps N")
            elif cmd == "/guidance" and len(parts) >= 2:
                try:
                    guidance = float(parts[1])
                    print(f"  ✓ guidance: {guidance}")
                except ValueError:
                    print(f"  {TR('格式:', 'Usage:')} /guidance F")
            elif cmd == "/seed":
                if len(parts) >= 2:
                    try:
                        seed = int(parts[1])
                        print(f"  ✓ seed: {seed}")
                    except ValueError:
                        print(f"  {TR('格式:', 'Usage:')} /seed [N]")
                else:
                    seed = None
                    print(f"  ✓ {TR('seed: random', 'seed: random')}")
            elif cmd == "/save" and len(parts) >= 2:
                save_dir = parts[1]
                os.makedirs(save_dir, exist_ok=True)
                print(f"  ✓ {TR('输出目录:', 'Output dir:')} {save_dir}")
            elif cmd == "/history":
                if not history:
                    print(f"  - {TR('暂无历史', 'No history')}")
                else:
                    for i, (p, f) in enumerate(history, 1):
                        print(f"  {i:>3}. {os.path.basename(f)}  ({p[:50]})")
            else:
                print(f"  ⚠ {TR('未知命令', 'Unknown command')}: {cmd}")
            continue

        # ── 生图 ──
        print(f"  ⏳ {width}x{height} x{steps} {TR('步', 'steps')}...", end=" ", flush=True)
        t0 = time.time()

        kwargs = {"width": width, "height": height, "num_inference_steps": steps,
                  "guidance_scale": guidance}
        if seed is not None:
            kwargs["rng_seed"] = seed

        try:
            result = pipe.generate(line, **kwargs)
            img = Image.fromarray(result.data[0])
            elapsed = time.time() - t0

            safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in line)[:40]
            fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe}.png"
            path = os.path.join(save_dir, fname)
            img.save(path)
            history.append((line, path))

            print(f"✓ ({elapsed:.1f}s)")
            print(f"  {TR('💾 已保存', '💾 Saved')}: {path}")
        except Exception as e:
            print(f"✗")
            print(f"  {TR('生图失败', 'Generation failed')}: {str(e)[:100]}")


def _print_help():
    print(f"  {TR('命令', 'Commands')}:")
    print(f"    /size W H              {TR('设置分辨率 (默认 512x512)', 'Set resolution (default 512x512)')}")
    print(f"    /steps N               {TR('推理步数 (默认 4)', 'Inference steps (default 4)')}")
    print(f"    /guidance F            guidance scale ({TR('默认', 'default')} 0.0)")
    print(f"    /seed [N]              {TR('设置/重置随机种子', 'Set/reset random seed')}")
    print(f"    /save DIR              {TR('设置输出目录', 'Set output directory')}")
    print(f"    /history               {TR('查看已生成的图片', 'View generated images')}")
    print(f"    /help                  {TR('显示本帮助', 'Show this help')}")
    print(f"    /exit                  {TR('退出', 'Exit')}")
    print()
