# 🚀 APQP DeepSeek LoRA Fine-Tuning

## 📌 Project Overview

This project focuses on **domain-specific fine-tuning of a Large Language Model (LLM)** for **APQP (Advanced Product Quality Planning)** scenarios using **LoRA (Low-Rank Adaptation)**.

The goal is to build an **intelligent quality management assistant** that can:

* Understand APQP-related problems
* Perform structured analysis
* Provide professional improvement suggestions

---

## 🧠 Key Features

* ✅ Domain adaptation for APQP knowledge
* ✅ Structured output generation (analysis + risk + suggestions)
* ✅ Lightweight fine-tuning with LoRA
* ✅ Optimized for limited GPU resources (RTX 3060)

---

## 🏗️ Model Architecture

* Base Model: DeepSeek-LLM-7B-Chat
* Fine-Tuning Method: LoRA
* Quantization: 4-bit (QLoRA supported)
* Frameworks:

  * Transformers
  * PEFT
  * Accelerate

---

## 📂 Project Structure

```
.
├── data/
│   └── apqp_high_quality.json   # Training dataset
├── output/
│   ├── lora_model/              # LoRA weights
│   └── merged_model/            # Merged full model
├── train.py                     # Training script
├── merge.py                     # Merge LoRA with base model
├── test.py                      # Inference & comparison
└── README.md
```

---

## 📊 Training Data

The dataset is designed with **enterprise-level structured outputs**:

```text
【Topic】
【Problem Analysis】
【Risk】
【Improvement Suggestions】
```

This structure significantly improves:

* Model stability
* Professional reasoning ability
* Output consistency

---

## ⚙️ Training Configuration

Recommended setup for RTX 3060 (12GB):

```python
load_in_4bit = True
per_device_train_batch_size = 1
gradient_accumulation_steps = 8

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
```

---

## 🚀 How to Run

### 1️⃣ Train LoRA

```bash
python train.py
```

---

### 2️⃣ Merge Model

```bash
python merge.py
```

---

### 3️⃣ Run Inference

```bash
python test.py
```

---

## 🔍 Example Output

### Input

```
APQP流程有哪些阶段？
```

### Output

```
【主题】：APQP流程管理
【问题分析】：APQP流程涉及多个阶段，需要系统化管理...
【风险】：流程不规范可能导致质量问题...
【改进建议】：建立标准化流程与阶段评审机制...
```

---

## 📈 Results

After fine-tuning:

* ✔ More structured outputs
* ✔ Better domain understanding
* ✔ Improved professional tone

---

## 💡 Future Work

* 🔹 Upgrade to Qwen2.5-7B-Instruct
* 🔹 Integrate RAG (Retrieval-Augmented Generation)
* 🔹 Build web-based QA system
* 🔹 Deploy as API service

---

## 🧑‍💻 Author

David Li

---

## ⭐ Acknowledgements

* DeepSeek LLM
* Hugging Face Transformers
* PEFT (LoRA)
* Open-source AI community

---

## 📜 License

MIT License

---

## Local JSONL RAG v1

This project includes a minimal offline RAG pipeline for manually prepared enterprise knowledge files.

1. Put cleaned `.txt`, `.md`, or `.json` files under `knowledge_docs/`.
2. Build the local JSONL vector store:

```bash
python -m back.rag.build_vector_store --input-dir knowledge_docs --output knowledge_base/vector_store.jsonl
```

3. Start the FastAPI backend. On startup, the backend loads `knowledge_base/vector_store.jsonl` when it exists.
4. Chat requests still use `/chat/completions`; before generation, the backend retrieves relevant chunks and injects them into the prompt.

The JSONL record format is:

```json
{
  "id": "doc_001_chunk_0001",
  "text": "knowledge chunk text",
  "embedding": [0.01, 0.02],
  "metadata": {
    "source": "apqp_rag_sample.md",
    "chunk_index": 1,
    "category": "APQP"
  }
}
```

The default embedding backend is a deterministic local hashing embedder, so it works without downloading an embedding model. It can be replaced later with a stronger local embedding model behind the same JSONL format.
