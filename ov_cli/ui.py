"""
ov-cli ui: Gradio 网页界面。

自动检测模型类型，启动对应的交互界面。
支持 Chat / TTS / ASR / Image 四种模型。
"""

import os, sys, time, json
from pathlib import Path

# 清除 socks 代理避免 gradio/httpx 报错
for key in ["all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY",
            "https_proxy", "HTTPS_PROXY", "socks_proxy", "SOCKS_PROXY"]:
    os.environ.pop(key, None)

import gradio as gr
from ov_cli import TR

# 用于标记是否需要恢复 transformers 版本
_need_restore_tf = False


# ── 模型类型检测 ──

def _detect_model_type(ov_path):
    cfg_path = os.path.join(ov_path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if any("Qwen3TTS" in a for a in archs):
            return "tts"
        if any("Qwen3ASR" in a for a in archs):
            return "asr"
    # Diffusers 格式（FLUX/SD 等）
    model_idx = os.path.join(ov_path, "model_index.json")
    if os.path.isfile(model_idx):
        return "image"
    # 有 openvino_model.xml 或 openvino_config.json → LLM/VLM
    if os.path.isfile(os.path.join(ov_path, "openvino_model.xml")) or \
       os.path.isfile(os.path.join(ov_path, "openvino_config.json")):
        return "chat"
    # 尝试 Text2Image
    try:
        import openvino_genai as ov_genai
        _ = ov_genai.Text2ImagePipeline(ov_path, "CPU")
        return "image"
    except Exception:
        pass
    return "chat"


# ── Chat 界面 ──

def _build_chat_ui(model_path, device, reasoning=True):
    """LLM/VLM 聊天界面。"""
    from .chat import load_model as load_llm

    ctx = load_llm(model_path)
    pipe = ctx["pipe"]
    config = ctx.get("config", {})
    is_vlm = ctx.get("is_vlm", False)
    pending_images = []  # 上传的图片路径

    def add_file(file, history):
        """处理上传的图片。"""
        nonlocal pending_images
        if file is None:
            return history, None
        pending_images.append(file.name)
        msg = f"📷 {os.path.basename(file.name)}"
        return history + [{"role": "user", "content": msg}], None

    def respond(message, history):
        """流式聊天（支持图片）。"""
        import threading, queue, time as _time
        import numpy as np
        import openvino as ov
        from PIL import Image
        from openvino_genai import GenerationConfig

        nonlocal pending_images

        # 处理图片
        images = []
        for path in pending_images:
            try:
                img = Image.open(path).convert("RGB")
                images.append(ov.Tensor(np.array(img)[None]))
            except Exception:
                pass
        pending_images = []

        history = history or []
        prompt = ""
        for h in history:
            if h["role"] == "user":
                prompt += f"User: {h['content']}\n"
            elif h["role"] == "assistant":
                prompt += f"Assistant: {h['content']}\n"
        prompt += f"User: {message}\nAssistant: "

        gen_cfg = GenerationConfig(max_new_tokens=1024)
        gen_cfg.temperature = config.get("temperature", 0.7)

        if not reasoning:
            try:
                tok = pipe.get_tokenizer()
                think_id = int(list(tok.encode("<think>", add_special_tokens=False).input_ids.data)[0][0])
                nothink_id = int(list(tok.encode("</think>", add_special_tokens=False).input_ids.data)[0][0])
                gen_cfg.reasoning_budget_tokens = 0
                gen_cfg.thinking_start_token_id = think_id
                gen_cfg.thinking_end_token_id = nothink_id
            except Exception:
                pass

        q = queue.Queue()
        stop = [False]
        t0 = _time.time()
        tok_count = [0]

        class Streamer:
            def __call__(self, word):
                q.put(word)
                tok_count[0] += 1
                return stop[0]

        def run():
            try:
                kwargs = {}
                if is_vlm and images:
                    kwargs["images"] = images
                if is_vlm:
                    pipe.generate(prompt, generation_config=gen_cfg, streamer=Streamer(), **kwargs)
                else:
                    pipe.generate(prompt, gen_cfg, Streamer())
            except Exception:
                pass
            finally:
                q.put(None)

        t = threading.Thread(target=run, daemon=True)
        t.start()

        full = ""
        while True:
            word = q.get()
            if word is None:
                break
            full += word
            yield full

        elapsed = _time.time() - t0
        char_count = len(full.replace(" ", ""))
        print(f"  [{elapsed:.1f}s | {char_count} chars | {char_count/elapsed:.1f} ch/s | {tok_count[0]/elapsed:.1f} tok/s]", file=sys.stderr)

    with gr.Blocks(title="ov-cli Chat") as demo:
        gr.Markdown(f"# ov-cli Chat\n设备: {device}")
        chat_history = gr.State([])
        with gr.Row():
            new_btn = gr.Button("🆕 新建对话", size="sm")
            save_btn = gr.Button("💾 保存对话", size="sm")
            load_dropdown = gr.Dropdown(label="📂 加载对话", choices=[], value=None, interactive=True)
            load_btn = gr.Button("加载", size="sm")
            del_btn = gr.Button("🗑️ 删除", size="sm", variant="stop")

        chatbot = gr.Chatbot(label="对话", height=450)
        with gr.Row():
            txt = gr.Textbox(label="输入", scale=4, container=False)
            btn = gr.Button("发送", scale=1, variant="primary")

        file_input = gr.File(label="上传图片", file_count="single")

        file_input.upload(add_file, [file_input, chat_history], [chat_history, file_input])

        def new_chat():
            nonlocal pending_images
            pending_images = []
            return [], []

        def save_chat(history):
            import json, datetime
            os.makedirs("outputs", exist_ok=True)
            fname = f"outputs/chat_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            print(f"  💾 {TR('对话已保存', 'Chat saved')}: {fname}", file=sys.stderr)
            return history

        def load_chat(name):
            import json, glob
            if not name:
                return [], []
            # 根据显示名找文件
            files = sorted(glob.glob("outputs/chat_*.json"))
            for f in files:
                label = os.path.basename(f).replace("chat_", "").replace(".json", "")
                if label == name:
                    with open(f, "r", encoding="utf-8") as fp:
                        history = json.load(fp)
                    print(f"  📂 {TR('已加载', 'Loaded')}: {os.path.basename(f)}", file=sys.stderr)
                    return history, history
            return [], []

        def delete_chat(name):
            import glob, os
            if not name:
                return gr.Dropdown(choices=[])
            files = sorted(glob.glob("outputs/chat_*.json"))
            for f in files:
                label = os.path.basename(f).replace("chat_", "").replace(".json", "")
                if label == name:
                    os.remove(f)
                    print(f"  🗑️ {TR('已删除', 'Deleted')}: {os.path.basename(f)}", file=sys.stderr)
                    break
            remaining = sorted(glob.glob("outputs/chat_*.json"))
            return gr.Dropdown(choices=[os.path.basename(f).replace("chat_", "").replace(".json", "") for f in remaining])

        def _update_dropdown():
            import glob
            files = sorted(glob.glob("outputs/chat_*.json"))
            return gr.Dropdown(choices=[os.path.basename(f).replace("chat_", "").replace(".json", "") for f in files])

        new_btn.click(new_chat, None, [chatbot, chat_history])
        new_btn.click(_update_dropdown, None, load_dropdown)
        save_btn.click(save_chat, chat_history, chatbot)
        save_btn.click(_update_dropdown, None, load_dropdown)
        load_btn.click(load_chat, load_dropdown, [chatbot, chat_history])
        del_btn.click(delete_chat, load_dropdown, load_dropdown)
        demo.load(_update_dropdown, None, load_dropdown)

        def chat_fn(message, history):
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "⏳ 生成中..."})
            yield history, gr.Textbox(value="", interactive=False), gr.Button("生成中...", interactive=False), history
            for chunk in respond(message, history[:-1]):
                history[-1] = {"role": "assistant", "content": chunk}
                yield history, gr.Textbox(value="", interactive=False), gr.Button("生成中...", interactive=False), history
            yield history, gr.Textbox(value="", interactive=True), gr.Button("发送", interactive=True, variant="primary"), history

        btn.click(chat_fn, [txt, chat_history], [chatbot, txt, btn, chat_history])
        txt.submit(chat_fn, [txt, chat_history], [chatbot, txt, btn, chat_history])

    return demo


# ── TTS 界面 ──

def _build_tts_ui(model_path, device):
    """TTS 语音合成界面。"""
    global _need_restore_tf
    from .asr import _pip_version, _ensure_qwen_asr_tf as _ensure_tf, _restore_tf

    _ensure_tf()
    _need_restore_tf = True  # TTS 需要 4.x，退出后必须恢复
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "dlc"))
    from qwen_tts_helper import OVQwen3TTSModel

    print(f"  加载 TTS 模型...", file=sys.stderr)
    t0 = time.time()
    model = OVQwen3TTSModel.from_pretrained(model_path, device=device)
    print(f"  ✓ ({time.time()-t0:.1f}s)", file=sys.stderr)
    mtype = model.tts_model_type
    speakers = model.get_supported_speakers()

    def synthesize(text, speaker, language, instruct, progress=gr.Progress()):
        if not text:
            return None, "请输入文本"
        progress(0.2, desc="生成中...")
        try:
            if mtype == "custom_voice":
                if not speaker:
                    return None, f"请选择声音: {', '.join(speakers)}"
                wavs, sr = model.generate_custom_voice(
                    text=text, language=language or "auto",
                    speaker=speaker, instruct=instruct or None,
                )
            else:
                return None, "Base 模型请使用 CLI 命令行"
            progress(1.0)
            return (sr, wavs[0]), "生成完成"
        except Exception as e:
            return None, f"错误: {e}"

    with gr.Blocks(title="ov-cli TTS") as demo:
        gr.Markdown(f"# ov-cli TTS\n设备: {device} | 模型: {mtype}")
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(label="文本", lines=3, placeholder="输入要合成的文本...")
                with gr.Row():
                    lang = gr.Dropdown(
                        choices=["auto", "chinese", "english", "japanese", "korean",
                                 "french", "german", "spanish", "russian", "portuguese", "italian"],
                        value="auto", label="语言"
                    )
                    if mtype == "custom_voice" and speakers:
                        spk = gr.Dropdown(choices=speakers, value=speakers[0] if speakers else None, label="声音")
                    else:
                        spk = gr.Dropdown(choices=[], label="声音 (仅 CustomVoice)")
                instruct = gr.Textbox(label="语气指令 (可选)", placeholder="例如: 用温柔的语气说")
                btn = gr.Button("生成语音", variant="primary")
            with gr.Column(scale=2):
                audio = gr.Audio(label="结果", type="numpy")
                status = gr.Textbox(label="状态", interactive=False)

        btn.click(synthesize, [text, spk, lang, instruct], [audio, status])

    return demo


# ── ASR 界面 ──

def _build_asr_ui(model_path, device):
    """ASR 语音识别界面。"""
    global _need_restore_tf
    mtype = "qwen3_asr" if "Qwen3ASR" in open(os.path.join(model_path, "config.json")).read() else "whisper"

    if mtype == "qwen3_asr":
        from .asr import _ensure_qwen_asr_tf
        _ensure_qwen_asr_tf()
        _need_restore_tf = True  # Qwen3-ASR 需要 4.x，退出后必须恢复
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "dlc"))
        from qwen_3_asr_helper import OVQwen3ASRModel
        model = OVQwen3ASRModel.from_pretrained(model_path, device=device)

        def transcribe(audio, lang):
            if audio is None:
                return "", "请上传或录制音频"
            path = audio if isinstance(audio, str) else audio.name
            results = model.transcribe(audio=path, language=lang or None)
            text = results[0].text if results else ""
            return text, f"检测语种: {results[0].language if results else 'N/A'}"
    else:
        import openvino_genai as ov_genai
        pipe = ov_genai.WhisperPipeline(model_path, device)
        from .asr import _load_audio

        def transcribe(audio, lang):
            if audio is None:
                return "", "请上传或录制音频"
            path = audio if isinstance(audio, str) else audio.name
            data = _load_audio(path)
            kwargs = {}
            if lang:
                kwargs["language"] = lang
            result = pipe.generate(data, **kwargs)
            text = result.texts[0] if result.texts else ""
            return text, "转录完成"

    with gr.Blocks(title="ov-cli ASR") as demo:
        gr.Markdown(f"# ov-cli ASR\n设备: {device} | 模型: {mtype}")
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="音频")
                lang = gr.Dropdown(
                    choices=["", "Chinese", "English", "Japanese", "Korean",
                             "French", "German", "Spanish", "Portuguese",
                             "Italian", "Russian", "Vietnamese", "Thai",
                             "Arabic", "Indonesian", "Turkish", "Hindi",
                             "Malay", "Dutch", "Swedish", "Danish",
                             "Finnish", "Polish", "Czech", "Filipino",
                             "Persian", "Greek", "Romanian", "Hungarian",
                             "Macedonian", "Cantonese"],
                    value="", label="语言 (可选)"
                )
                btn = gr.Button("转录", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="转录结果", lines=8)
                status = gr.Textbox(label="状态")

        btn.click(transcribe, [audio_input, lang], [text_output, status])

    return demo


# ── Image 界面 ──

def _build_image_ui(model_path, device):
    """文生图界面。"""
    import openvino_genai as ov_genai
    pipe = ov_genai.Text2ImagePipeline(model_path, device)

    def generate(prompt, width, height, steps, guidance, seed):
        if not prompt:
            return None
        kwargs = {"width": width, "height": height, "num_inference_steps": steps,
                  "guidance_scale": guidance}
        if seed and seed != -1:
            kwargs["rng_seed"] = seed
        try:
            result = pipe.generate(prompt, **kwargs)
            from PIL import Image as PILImage
            return PILImage.fromarray(result.data[0])
        except Exception as e:
            raise gr.Error(f"生成失败: {e}")

    with gr.Blocks(title="ov-cli Image") as demo:
        gr.Markdown(f"# ov-cli Image\n设备: {device}")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="提示词", lines=3, placeholder="描述你想生成的图片...")
                with gr.Row():
                    width = gr.Slider(256, 1536, 512, step=64, label="宽度")
                    height = gr.Slider(256, 1536, 512, step=64, label="高度")
                with gr.Row():
                    steps = gr.Slider(1, 50, 4, step=1, label="步数")
                    guidance = gr.Slider(0, 20, 0, step=0.5, label="Guidance")
                seed = gr.Number(-1, label="随机种子 (-1=随机)", precision=0)
                btn = gr.Button("生成", variant="primary")
            with gr.Column():
                output = gr.Image(label="结果")

        btn.click(generate, [prompt, width, height, steps, guidance, seed], output)

    return demo


# ── 入口 ──

def launch_ui(model_path, device=None, port=7860, share=False, reasoning=True):
    """启动 Gradio 网页界面。"""
    import openvino as ov
    import signal as _signal
    if device is None:
        device = "GPU" if "GPU" in ov.Core().available_devices else "CPU"

    ov_path = os.path.abspath(model_path)
    if not os.path.isdir(ov_path):
        print(f"  ❌ {TR('模型目录不存在', 'Model path not found')}: {ov_path}")
        sys.exit(1)

    mtype = _detect_model_type(ov_path)
    print(f"  检测到模型类型: {mtype}", file=sys.stderr)

    builders = {
        "chat": _build_chat_ui,
        "tts": _build_tts_ui,
        "asr": _build_asr_ui,
        "image": _build_image_ui,
    }

    builder = builders.get(mtype)
    if builder is None:
        print(f"  ❌ {TR('不支持的模型类型', 'Unsupported model type')}: {mtype}")
        sys.exit(1)

    if mtype == "chat":
        demo = builder(ov_path, device, reasoning=reasoning)
    else:
        demo = builder(ov_path, device)
    print(f"  🌐 {TR('启动 Gradio 界面', 'Launching Gradio UI')}: http://localhost:{port}", file=sys.stderr)
    try:
        demo.launch(server_port=port, share=share)
    except KeyboardInterrupt:
        print("  ⏹️ 正在关闭...", file=sys.stderr)
        _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
    finally:
        demo.close()
        if _need_restore_tf:
            from .asr import _restore_tf
            _restore_tf(True)
        print("  ✅ 已退出", file=sys.stderr)
