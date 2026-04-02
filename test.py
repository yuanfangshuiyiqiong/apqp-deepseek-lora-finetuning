import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 路径
base_model_path = r"E:\liworkplace\lora - 副本\deepseek-llm-7B-chat-lora-ft\model\deepseek-ai\deepseek-llm-7b-chat"
merge_path = r"E:\liworkplace\lora - 副本\deepseek-llm-7B-chat-lora-ft\output\merge_model"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

print("加载原模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

print("加载merge模型...")
merge_model = AutoModelForCausalLM.from_pretrained(
    merge_path,
    device_map="auto",
    torch_dtype=torch.float16
)

def generate(model, prompt):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

q = "APQP流程有哪些阶段？"

print("\n【原模型】")
print(generate(base_model, q))

print("\n【Merge模型】")
print(generate(merge_model, q))