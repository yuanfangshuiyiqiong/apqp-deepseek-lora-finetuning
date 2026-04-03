import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ===== 路径配置 =====
base_model_path = r"E:\liworkplace\lora - 副本\deepseek-llm-7B-chat-lora-ft\model\deepseek-ai\deepseek-llm-7b-chat"
lora_path = r"E:\liworkplace\lora - 副本\deepseek-llm-7B-chat-lora-ft\output\deepseek-mutil-test-2"
save_path = r"E:\liworkplace\lora - 副本\deepseek-llm-7B-chat-lora-ft\output\merge_model-2"

os.makedirs(save_path, exist_ok=True)


def merge_lora():
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

    print("加载基础模型（CPU）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",              # ✅ 强制 CPU
        torch_dtype=torch.float32      # ✅ CPU 必须 float32
    )

    print("加载 LoRA...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        device_map="cpu"               # ✅ 同样 CPU
    )

    print("开始合并 LoRA...")
    model = model.merge_and_unload()   # 🔥 核心步骤

    print("保存模型...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("✅ 合并完成！保存路径：", save_path)


if __name__ == "__main__":
    merge_lora()