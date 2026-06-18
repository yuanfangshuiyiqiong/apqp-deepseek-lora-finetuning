"""
FastAPI 推理后端 - 基于 transformers 的本地大模型对话服务
支持 4-bit/8-bit 量化，OpenAI 格式的接口
"""

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from back.rag import HashingEmbeddingModel, JsonlVectorStore, build_rag_prompt
except ImportError:
    from rag import HashingEmbeddingModel, JsonlVectorStore, build_rag_prompt

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 配置参数 ====================
# 获取脚本所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.getenv(
    "VECTOR_STORE_PATH",
    os.path.join(BASE_DIR, "..", "knowledge_base", "vector_store.jsonl"),
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.05"))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
MODEL_PATH = os.path.join(BASE_DIR, "..", "output", "merge_model2")  # 模型路径
QUANTIZATION = "8bit"  # 量化方式: "4bit", "8bit", 或 None (根据你的模型配置)

# ==================== 全局变量 ====================
app = FastAPI(title="AI Chat API", version="1.0.0")
model = None
tokenizer = None
rag_store = JsonlVectorStore([])
embedding_model = HashingEmbeddingModel(dimension=EMBEDDING_DIMENSION)

# ==================== CORS 配置 ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================
class Message(BaseModel):
    """单条消息"""
    role: str = Field(..., description="角色: system, user, assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """聊天请求"""
    messages: List[Message] = Field(..., description="消息列表 (OpenAI 格式)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_new_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成长度")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p 采样")
    stream: bool = Field(default=False, description="是否流式输出")
    model: Optional[str] = Field(default=None, description="模型名称 (可选)")


class ChatResponse(BaseModel):
    """聊天响应"""
    id: str = "chatcmpl-001"
    object: str = "chat.completion"
    created: int = 0
    model: str = "local-model"
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


# ==================== 模型加载 ====================
def load_model():
    """加载模型和分词器"""
    global model, tokenizer
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型路径不存在: {MODEL_PATH}")
    
    logger.info(f"正在加载模型: {MODEL_PATH}")
    logger.info(f"量化方式: {QUANTIZATION}")
    
    # 量化配置
    quantization_config = None
    if QUANTIZATION == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif QUANTIZATION == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )
    
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model_kwargs = {
        "pretrained_model_name_or_path": MODEL_PATH,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    
    logger.info("模型加载完成!")


def build_prompt(messages: List[Message]) -> str:
    """构建对话 prompt"""
    prompt = ""
    
    for msg in messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n\n"
    
    prompt += "Assistant: "
    return prompt


def load_rag_store():
    """Load the local JSONL vector store if it exists."""
    global rag_store
    rag_store = JsonlVectorStore.load(VECTOR_STORE_PATH)
    if rag_store.records:
        logger.info(f"RAG vector store loaded: {VECTOR_STORE_PATH} ({len(rag_store.records)} chunks)")
    else:
        logger.info(f"RAG vector store not found or empty: {VECTOR_STORE_PATH}")


def enrich_messages_with_rag(messages: List[Message]) -> List[Message]:
    """Inject retrieved knowledge into the latest user message."""
    if not rag_store.records:
        return messages

    latest_user_index = None
    latest_question = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            latest_user_index = index
            latest_question = messages[index].content
            break

    if latest_user_index is None or not latest_question:
        return messages

    query_embedding = embedding_model.embed(latest_question)
    results = rag_store.search(query_embedding, top_k=RAG_TOP_K, min_score=RAG_MIN_SCORE)
    contexts = [
        {
            "text": result.record.text,
            "metadata": result.record.metadata,
            "score": result.score,
        }
        for result in results
    ]
    rag_content = build_rag_prompt(latest_question, contexts)
    enriched = list(messages)
    enriched[latest_user_index] = Message(role="user", content=rag_content)
    return enriched


def generate_response(
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float
) -> str:
    """生成回复"""
    if model is None or tokenizer is None:
        raise RuntimeError("模型未加载")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


async def generate_stream(
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float
):
    """流式生成回复"""
    if model is None or tokenizer is None:
        raise RuntimeError("模型未加载")
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    from transformers import TextIteratorStreamer
    from threading import Thread
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for text in streamer:
        yield f"data: {text}\n\n"
    
    yield "data: [DONE]\n\n"
    thread.join()


# ==================== API 端点 ====================
@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    try:
        load_rag_store()
        load_model()
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise e


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "ok",
        "message": "AI Chat API 运行中",
        "model": MODEL_PATH,
        "quantization": QUANTIZATION,
        "rag": {
            "vector_store": VECTOR_STORE_PATH,
            "chunks": len(rag_store.records),
            "top_k": RAG_TOP_K,
            "min_score": RAG_MIN_SCORE,
        }
    }


@app.get("/models")
async def list_models():
    """获取可用模型列表"""
    return {
        "object": "list",
        "data": [{
            "id": "local-model",
            "object": "model",
            "created": 1700000000,
            "owned_by": "local",
            "permission": [],
            "root": "local-model",
            "parent": None
        }]
    }


@app.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """
    聊天补全接口 (OpenAI 兼容)
    
    请求示例:
    {
        "messages": [
            {"role": "user", "content": "你好"}
        ],
        "temperature": 0.7,
        "max_new_tokens": 512,
        "top_p": 0.9
    }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载，请检查服务状态")
    
    try:
        # 构建 prompt
        prompt = build_prompt(enrich_messages_with_rag(request.messages))
        
        if request.stream:
            # 流式输出
            return StreamingResponse(
                generate_stream(prompt, request.temperature, request.max_new_tokens, request.top_p),
                media_type="text/event-stream"
            )
        
        # 非流式输出
        response_text = generate_response(
            prompt,
            request.temperature,
            request.max_new_tokens,
            request.top_p
        )
        
        import time
        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model="local-model",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(tokenizer(prompt)["input_ids"]),
                "completion_tokens": len(tokenizer(response_text)["input_ids"]),
                "total_tokens": len(tokenizer(prompt)["input_ids"]) + len(tokenizer(response_text)["input_ids"])
            }
        )
        
    except Exception as e:
        logger.error(f"生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 启动命令 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
