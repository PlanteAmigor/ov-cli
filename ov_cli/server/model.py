"""
ov-cli server: 模型加载。
"""

import json
import os
import time

import openvino as ov
import openvino_genai as ov_genai

from .config import _log, _model_lock, _model_state


def load_model(model_path: str, device: str = "CPU") -> dict:
    """加载模型（线程安全，全局单例）。"""
    global _model_state
    with _model_lock:
        if _model_state and _model_state.get("model_path") == model_path:
            return _model_state

        is_vlm = os.path.isfile(os.path.join(model_path, "openvino_vision_embeddings_model.xml"))
        tag = "VLM" if is_vlm else "LLM"
        _log(f"  📦 加载 {tag}: {os.path.basename(model_path)} ({device})", end=" ")
        t0 = time.time()
        pipe = ov_genai.VLMPipeline(model_path, device) if is_vlm else ov_genai.LLMPipeline(model_path, device)
        _log(f"✓ {time.time()-t0:.1f}s")

        model_type = None
        model_name = os.path.basename(model_path)
        cfg_path = os.path.join(model_path, "config.json")
        n_ctx_train = 4096  # 默认
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type")
            model_name = cfg.get("_name_or_path", model_name)
            # 读取 max_position_embeddings（可能在外层或 text_config 内）
            n_ctx_train = cfg.get("max_position_embeddings") or 0
            if not n_ctx_train:
                text_cfg = cfg.get("text_config", {})
                n_ctx_train = text_cfg.get("max_position_embeddings", 4096)
            # 还可能是 sliding_window
            if not n_ctx_train:
                n_ctx_train = text_cfg.get("sliding_window", 4096)

        # Tokenizer 统计
        try:
            tok = pipe.get_tokenizer()
            test = tok.encode("Hello world")
            vocab = test.input_ids.shape[-1] if hasattr(test, 'input_ids') else 0
        except Exception:
            vocab = 0

        _log(f"  📋 模型: {model_name} | 类型: {model_type or tag} | 词表: {vocab} | 上下文: {n_ctx_train}")

        # 预热（使用接近真实请求的配置，避免首次请求触发懒编译卡死）
        try:
            _log(f"  ⚡ 编译预热...", end=" ")
            t_warm = time.time()
            warmup_cfg = ov_genai.GenerationConfig(
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                do_sample=True,
            )
            pipe.generate("hi", warmup_cfg)
            _log(f"✓ {time.time()-t_warm:.1f}s")
        except Exception:
            pass

        _model_state = {
            "pipe": pipe,
            "model_type": model_type,
            "is_vlm": is_vlm,
            "model_path": model_path,
            "model_name": model_name,
            "device": device,
            "vocab_size": vocab,
            "n_ctx_train": n_ctx_train,
        }
        return _model_state
