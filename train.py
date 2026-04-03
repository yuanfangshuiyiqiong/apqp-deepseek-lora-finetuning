# =========================
# 1️⃣ 加载模型 + 分词器
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

model_path = "./model/deepseek-ai/deepseek-llm-7b-chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",   # ✅ 必须auto（不要None）
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

print("模型加载完成")

# =========================
# 2️⃣ 数据处理函数（必须在前）
# =========================
MAX_LENGTH = 512

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
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
    labels = labels[:MAX_LENGTH]

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
    data_files="./data/apqp_high_quality.json",
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
    r=16,
    lora_alpha=32,
    lora_dropout=0.02,
    bias="none",
    target_modules=['q_proj', 'v_proj'],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 5️⃣ 训练参数
# =========================
from transformers import TrainingArguments

output_dir = "./output/deepseek-mutil-test-2"

train_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    logging_steps=2,
    save_steps=200,
    learning_rate=1e-4,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
)

# =========================
# 6️⃣ Trainer
# =========================
from transformers import Trainer, DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()

# =========================
# 7️⃣ 保存模型
# =========================
trainer.save_model(output_dir)

print("训练完成 ✅")