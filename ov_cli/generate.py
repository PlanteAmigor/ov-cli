"""
ov-cli generate: 文生图终端。

支持交互式多轮生图和单次生图。
使用 OpenVINO GenAI Text2ImagePipeline。
"""

import os, sys, time
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


# ── 加载模型 ──

def load_model(ov_path):
    """加载 Text2ImagePipeline。"""
    import openvino as ov
    device = "GPU" if "GPU" in ov.Core().available_devices else "CPU"
    print(f"  {TR('加载 Text2ImagePipeline ({})...', 'Loading Text2ImagePipeline ({})...').format(device)}", end=" ", flush=True)
    t0 = time.time()
    pipe = ov_genai.Text2ImagePipeline(ov_path, device)
    print(f"✓ ({time.time()-t0:.1f}s)")
    return {"pipe": pipe, "device": device}


# ── 单次生图 ──

def run_once(ctx, prompt, output=None, width=_DEFAULT_WIDTH, height=_DEFAULT_HEIGHT,
             steps=_DEFAULT_STEPS, guidance=_DEFAULT_GUIDANCE, seed=None):
    """单次生图，输出完自动退出。"""
    pipe = ctx["pipe"]
    print(f"  {TR('⏳ 生成中...', '⏳ Generating...')}", end=" ", flush=True)
    t0 = time.time()

    kwargs = {"width": width, "height": height, "num_inference_steps": steps,
              "guidance_scale": guidance}
    if seed is not None:
        kwargs["rng_seed"] = seed

    try:
        result = pipe.generate(prompt, **kwargs)
    except Exception as e:
        print(f"✗")
        print(f"  {TR('生图失败', 'Generation failed')}: {str(e)[:200]}")
        sys.exit(1)

    img = Image.fromarray(result.data[0])
    elapsed = time.time() - t0

    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        img.save(output)
        print(f"✓ ({elapsed:.1f}s)")
        print(f"  {TR('💾 已保存', '💾 Saved')}: {output}")
    else:
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:40]
        fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe}.png"
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", fname)
        img.save(path)
        print(f"✓ ({elapsed:.1f}s)")
        print(f"  {TR('💾 已保存', '💾 Saved')}: {path}")

    return img


# ── 交互式生图 ──

def run_generate(ctx, width=_DEFAULT_WIDTH, height=_DEFAULT_HEIGHT,
                 steps=_DEFAULT_STEPS, guidance=_DEFAULT_GUIDANCE,
                 seed=None, save_dir=_DEFAULT_SAVE_DIR):
    """交互式生图终端。"""
    pipe = ctx["pipe"]
    os.makedirs(save_dir, exist_ok=True)
    history = []

    print()
    print("        ██████╗ ██╗   ██╗     ██████╗██╗     ██╗")
    print("       ██╔═══██╗██║   ██║    ██╔════╝██║     ██║")
    print("       ██║   ██║██║   ██║    ██║     ██║     ██║")
    print("       ██║   ██║╚██╗ ██╔╝    ██║     ██║     ██║")
    print("       ╚██████╔╝ ╚████╔╝     ╚██████╗███████╗██║")
    print("        ╚═════╝   ╚═══╝       ╚═════╝╚══════╝╚═╝")
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
