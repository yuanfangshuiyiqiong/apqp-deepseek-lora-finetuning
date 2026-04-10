import os
import sys
import json
import gc
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, BitsAndBytesConfig

# ==========================================
# 1. 环境配置 & Windows 乱码修复
# ==========================================
def setup_windows_console():
    """修复 Windows 终端显示中文乱码的问题"""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

setup_windows_console()

# ==========================================
# 2. 核心功能函数
# ==========================================

def load_tokenizer(model_dir: str):
    """
    从本地目录加载 Tokenizer。
    使用 PreTrainedTokenizerFast 避免部分环境下中文被吞掉的问题。
    """
    print(f"🔍 正在加载 Tokenizer: {model_dir}")
    tok_path = os.path.join(model_dir, "tokenizer.json")
    cfg_path = os.path.join(model_dir, "tokenizer_config.json")

    # 默认特殊 Token
    bos, eos, pad = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜end▁of▁sentence｜>"

    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            # 兼容处理复杂格式的 token 字段
            def _extract(v): return v.get("content") if isinstance(v, dict) else v
            bos = _extract(cfg.get("bos_token", bos))
            eos = _extract(cfg.get("eos_token", eos))
            pad = _extract(cfg.get("pad_token", pad))

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tok_path, 
        bos_token=bos, 
        eos_token=eos, 
        pad_token=pad
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def load_model(model_path: str):
    """
    优化后的加载函数：强制 GPU 部署，防止自动 CPU Offload。
    """
    print(f"🚀 正在加载模型: {model_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        # 移除可能引起冲突的冗余配置，保持核心量化参数
    )

    try:
        # 针对 RTX 3060 12G，强制使用第一块显卡
        # 使用 {"": 0} 而不是 "auto" 可以防止 transformers 自动做分片(Offload)
        forced_device_map = {"": 0} if torch.cuda.is_available() else None

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=forced_device_map, # 👈 关键点：强制全量入库显存
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            dtype=torch.float16 # 👈 修复 torch_dtype 警告
        )
        return model
    except Exception as e:
        print(f"⚠️ GPU 强制加载失败 (可能是显存碎片或配置冲突)，切换至 CPU 模式。")
        print(f"错误细节: {e}")
        
        # CPU 备选方案
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            dtype=torch.float32, 
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
import time

def clear_gpu(model):
    """
    深度清理显存，确保显存驱动彻底回收空间。
    """
    print("🧹 正在执行深度显存清理...")
    # 1. 尝试将模型移至 CPU (有时有助于释放)
    if model is not None:
        model.to("cpu")
    
    # 2. 删除引用
    del model
    
    # 3. 强制垃圾回收
    gc.collect()
    gc.collect() # 两次回收确保循环引用被打破
    
    # 4. 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 5. 短暂休眠，给显卡驱动响应时间
    time.sleep(1)
    print("✅ 显存清理完毕。")

def generate_response(model, tokenizer, prompt: str):
    """
    执行推理，增加 max_new_tokens 以防止回答截断。
    """
    # 构建 DeepSeek Chat 的 Prompt 模板（如果是 Chat 模型，建议加上模板引导）
    # 如果你的模型是 Chat 版，建议取消下面两行的注释：
    # prompt = f"User: {prompt}\n\nAssistant:"
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512, 
        padding=False
    )
    
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,        # 👈 从 64 增加到 512，确保回答完整
            do_sample=True,            # 开启采样，让语言更自然
            temperature=0.7,           # 适度的创造力
            top_p=0.9,
            repetition_penalty=1.1,    # 👈 防止生成重复内容
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # 获取输入序列的长度
    input_len = inputs["input_ids"].shape[1]
    
    # 仅解码生成的部分（不包含原始问题）
    # 这样可以清爽地对比 Base 和 Merge 的纯回答内容
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

def run_compare():
    # --- 配置区 ---
    BASE_PATH = "./deepseek_model/deepseek-ai/deepseek-llm-7b-chat"
    MERGE_PATH = "./output/merge_model2"
    
    questions = [
        "你是质量管理专家，请简述 APQP 的核心目的。",
        "供应商来料尺寸波动大且未通知变更，请分析其在 PPAP 流程中的违规点。",
        "某产品装配漏装螺钉，请从 PFMEA 角度提供一个改进建议。",
    ]

    # 存储结果
    results = {q: {"base": "", "merge": ""} for q in questions}

    # 第一阶段：运行 Base 模型
    print("\n" + "="*20 + " Phase 1: Base Model " + "="*20)
    tokenizer = load_tokenizer(BASE_PATH)
    model = load_model(BASE_PATH)
    model.eval()

    for q in questions:
        print(f"💬 正在处理问题: {q[:20]}...")
        results[q]["base"] = generate_response(model, tokenizer, q)
    
    clear_gpu(model)

    # 第二阶段：运行 Merge 模型
    print("\n" + "="*20 + " Phase 2: Merge Model " + "="*20)
    # 重新加载 merge 目录的 tokenizer（以防其有特殊 token）
    tokenizer = load_tokenizer(MERGE_PATH if os.path.exists(os.path.join(MERGE_PATH, "tokenizer.json")) else BASE_PATH)
    model = load_model(MERGE_PATH)
    model.eval()

    for q in questions:
        print(f"💬 正在处理问题: {q[:20]}...")
        results[q]["merge"] = generate_response(model, tokenizer, q)
    
    clear_gpu(model)

    # 第三阶段：输出对比结果
    print("\n" + "█"*60)
    print("📋 推理对比报告")
    print("█"*60)
    
    for i, q in enumerate(questions, 1):
        print(f"\n【问题 {i}】：{q}")
        print("-" * 30)
        print(f"【原模型】\n{results[q]['base']}")
        print("-" * 15)
        print(f"【Merge模型】\n{results[q]['merge']}")
        print("\n" + "="*50)

if __name__ == "__main__":
    try:
        run_compare()
    except KeyboardInterrupt:
        print("\n🛑 用户停止运行。")
    except Exception as e:
        print(f"\n❌ 程序崩溃: {e}")