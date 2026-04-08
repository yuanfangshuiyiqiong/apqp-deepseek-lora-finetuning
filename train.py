# =========================
# 1️⃣ 加载模型 + 分词器
# =========================
import gc
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./deepseek_model/deepseek-ai/deepseek-llm-7b-chat"
data_path = "./data/apqp_high_quality.json"
output_dir = "./output/deepseek-mutil-test"

# 优先使用 8bit；也提供可选方案：使用 fp16（当 bitsandbytes 不可用/加载失败时自动切换）
QUANT_MODE = os.getenv("QUANT_MODE", "8bit").lower()  # "8bit" or "fp16"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
if tokenizer.pad_token is None:
    # DeepSeek 类模型通常没有显式 pad_token，用 eos_token 作为 padding 以避免 collator/padding 报错
    tokenizer.pad_token = tokenizer.eos_token

print("分词器加载完成")

# =========================
# 2️⃣ 数据处理函数（必须在前）
# =========================
MAX_LENGTH = 768

def process_data(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text.strip():
        prompt = f"### 指令：{instruction}\n### 输入：{input_text}\n### 输出："
    else:
        prompt = f"### 指令：{instruction}\n### 输出："

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False
    )["input_ids"]

    full = tokenizer(
        prompt + output,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    # 只训练答案部分
    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), MAX_LENGTH)
    labels[:prompt_len] = [-100] * prompt_len
    # 把 padding 的 token 也 mask 掉，避免无意义梯度/影响损失
    labels = [(-100 if mask == 0 else token_id) for token_id, mask in zip(labels, attention_mask)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# =========================
# 3️⃣ 加载数据
# =========================
from datasets import load_dataset

train_ds = load_dataset(
    "json",
    data_files=data_path,
    split="train"
)

print("数据字段：", train_ds.column_names)
print("样本示例：", train_ds[0])

train_dataset = train_ds.map(
    process_data,
    remove_columns=train_ds.column_names
)

# =========================
# 4️⃣ LoRA
# =========================
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    r=32,  # 提升表达能力（在 3060 12GB 上追求速度/效果折中）
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)

# =========================
# 5️⃣ 训练参数
# =========================
from transformers import TrainingArguments

def make_train_args(per_device_train_batch_size: int, gradient_accumulation_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=200,
        learning_rate=2e-4,
        fp16=True,  # 保持 fp16 以提升计算吞吐
        gradient_checkpointing=False,  # 不使用省显存手段，优先速度
        optim="adamw_torch",  # 替换原来的 paged_adamw_8bit
        report_to="none",
        remove_unused_columns=False,
    )

# =========================
# 6️⃣ Trainer
# =========================
from transformers import Trainer, DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
)

def load_base_model():
    # 保留 device_map="auto" 与 trust_remote_code=True
    if QUANT_MODE == "8bit":
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[WARN] 8bit 加载失败，将自动切换为 fp16。错误: {e}")

    # 可选方案：fp16 半精度（当不走 8bit 时使用）
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,  # transformers 新版推荐使用 dtype（避免 torch_dtype 弃用警告）
        device_map="auto",
        trust_remote_code=True,
    )

def build_peft_model():
    base_model = load_base_model()
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model

# 优先使用更激进配置；若 CUDA 显存不足，fallback 到更保守的配置避免直接报错中断
train_attempts = [
    {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 4},
    {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 8},  # fallback：更稳
]

last_err = None
for attempt_idx, attempt_cfg in enumerate(train_attempts, start=1):
    try:
        print(f"\n===== 训练尝试 {attempt_idx}: batch={attempt_cfg['per_device_train_batch_size']}, grad_acc={attempt_cfg['gradient_accumulation_steps']} =====")
        model = build_peft_model()
        train_args = make_train_args(**attempt_cfg)

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(output_dir)
        print("训练完成 ✅")
        last_err = None
        break
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg and "memory" in msg:
            last_err = e
            print("[WARN] 检测到显存不足，启用 fallback 配置重试...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            continue
        raise

# =========================
# 7️⃣ 保存模型
# =========================
if last_err is not None:
    raise last_err