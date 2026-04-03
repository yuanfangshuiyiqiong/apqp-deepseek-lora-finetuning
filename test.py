import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 路径
base_model_path = r"E:\liworkplace\lora - 副本\deepseek-llm-7B-chat-lora-ft\model\deepseek-ai\deepseek-llm-7b-chat"
merge_path = r"E:\liworkplace\lora - 副本\deepseek-llm-7B-chat-lora-ft\output\merge_model-2"

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

# 👉 强制使用 CPU
device = torch.device("cpu")

print("加载原模型（CPU）...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cpu",
    torch_dtype=torch.float32   # ⚠️ CPU 必须 float32
)

print("加载 merge 模型（CPU）...")
merge_model = AutoModelForCausalLM.from_pretrained(
    merge_path,
    device_map="cpu",
    torch_dtype=torch.float32
)

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,     # ⚠️ CPU建议小一点
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 测试问题（建议多问几个）
questions = [
    "你是APQP质量管理专家，请根据输入内容进行专业分析，并严格按照格式输出：\n【主题】：...\n【问题分析】：...\n【风险】：...\n【改进建议】：...",
    "某项目风险应对计划表中，对于“设备故障”风险，应对措施为“购买备件”，但未明确备件清单、采购周期或维护计划，且责任人为设备部（未指定具体人员）。"
    "什么是PPAP?"
]

for q in questions:
    print("\n" + "="*50)
    print(f"问题：{q}")

    print("\n【原模型】")
    print(generate(base_model, q))

    print("\n【Merge模型】")
    print(generate(merge_model, q))