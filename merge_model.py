import os
import json
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# ===== 路径配置（与 train.py 工程内路径一致）=====
base_model_path = "./deepseek_model/deepseek-ai/deepseek-llm-7b-chat"
lora_path = "./output/deepseek-mutil-test"
save_path = "./output/merge_model2"

os.makedirs(save_path, exist_ok=True)

def load_tokenizer(model_dir: str):
    """
    重要：避免 AutoTokenizer 在当前环境下吞中文，改为直接从 tokenizer.json 构造 fast tokenizer。
    """
    from transformers import PreTrainedTokenizerFast

    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    tok_path = os.path.join(model_dir, "tokenizer.json")

    bos = "<｜begin▁of▁sentence｜>"
    eos = "<｜end▁of▁sentence｜>"
    pad = "<｜end▁of▁sentence｜>"
    def _tok_content(v, default: str) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return v.get("content") or default
        return default
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        bos = _tok_content(cfg.get("bos_token"), bos)
        eos = _tok_content(cfg.get("eos_token"), eos)
        pad = _tok_content(cfg.get("pad_token"), pad)

    tok = PreTrainedTokenizerFast(tokenizer_file=tok_path, bos_token=bos, eos_token=eos, pad_token=pad)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _fix_no_split_modules(model):
    """PEFT 在部分层在 CPU 上会走 accelerate.get_balanced_memory；若 _no_split_modules 是 set，
    会被误包成 [set]，进而 set(...) 触发 TypeError: unhashable type: 'set'。"""
    ns = getattr(model, "_no_split_modules", None)
    if isinstance(ns, set):
        model._no_split_modules = list(ns)


def merge_lora():
    print("加载 tokenizer...")
    tokenizer = load_tokenizer(base_model_path)

    use_cuda = torch.cuda.is_available()
    base_model = None

    if use_cuda:
        # 优先 8bit + 单卡：7B 约能放进 12GB，避免 device_map=auto 把层卸到 CPU 触发 PEFT/accelerate 的 bug
        try:
            from transformers import BitsAndBytesConfig

            print("加载基础模型（GPU / 8bit / cuda:0）...")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map={"": "cuda:0"},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"[WARN] 8bit 单卡加载失败，回退到 fp16 + device_map=auto。原因: {e}")
            print("加载基础模型（GPU / fp16 / device_map=auto）...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
    else:
        print("未检测到 CUDA，使用 CPU + float32...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="cpu",
            dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    _fix_no_split_modules(base_model)

    print("加载 LoRA...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("开始合并 LoRA...")
    model = model.merge_and_unload()

    print("保存模型...")
    try:
        model.save_pretrained(save_path, safe_serialization=True)
    except RuntimeError as e:
        if use_cuda and "out of memory" in str(e).lower():
            print("[WARN] 保存时显存不足，改为在 CPU 上保存...")
            torch.cuda.empty_cache()
            model = model.to("cpu")
            model.save_pretrained(save_path, safe_serialization=True)
        else:
            raise

    tokenizer.save_pretrained(save_path)

    print("✅ 合并完成！保存路径：", save_path)


if __name__ == "__main__":
    merge_lora()
