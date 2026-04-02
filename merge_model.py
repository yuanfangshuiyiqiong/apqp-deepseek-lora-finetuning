import os
import shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def copy_files_not_in_B(A_path, B_path):
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    files_in_A = os.listdir(A_path)
    files_in_A = set([f for f in files_in_A if not (".bin" in f or "safetensors" in f)])
    files_in_B = set(os.listdir(B_path))

    for file in files_in_A - files_in_B:
        src = os.path.join(A_path, file)
        dst = os.path.join(B_path, file)

        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

def merge_lora():
    base_model_path = r"E:\liworkplace\lora - 副本\deepseek-llm-7B-chat-lora-ft\model\deepseek-ai\deepseek-llm-7b-chat"
    lora_path = r"E:\liworkplace\lora - 副本\deepseek-llm-7B-chat-lora-ft\output\deepseek-mutil-test-2"
    save_path = r"E:\liworkplace\lora - 副本\deepseek-llm-7B-chat-lora-ft\output\merge_model"

    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print("加载 base 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("加载 LoRA...")
    model = PeftModel.from_pretrained(model, lora_path)

    print("开始 merge...")
    merged_model = model.merge_and_unload()

    print("保存模型...")
    tokenizer.save_pretrained(save_path)
    merged_model.save_pretrained(save_path)

    copy_files_not_in_B(base_model_path, save_path)

    print(f"✅ 合并完成，保存路径：{save_path}")

if __name__ == "__main__":
    merge_lora()