from typing import Any, Dict, Iterable


NO_EVIDENCE_RULE = "只有在未检索到相关资料时，才说明“当前知识库没有找到明确依据”，不要编造。"


def build_rag_prompt(user_question: str, contexts: Iterable[Dict[str, Any]]) -> str:
    context_blocks = []
    for index, context in enumerate(contexts, start=1):
        metadata = context.get("metadata") or {}
        source = metadata.get("source", "unknown")
        chunk_index = metadata.get("chunk_index", "?")
        score = context.get("score", 0.0)
        text = context.get("text", "")
        context_blocks.append(
            f"[资料{index}] source={source}, chunk={chunk_index}, score={score:.4f}\n{text}"
        )

    context_text = "\n\n".join(context_blocks) if context_blocks else "未检索到相关资料。"
    return (
        "你是企业内部 APQP 和项目质量管理助手。请优先依据下方知识库资料回答用户问题。\n"
        f"{NO_EVIDENCE_RULE}\n"
        "回答应清晰、专业，并在最后列出使用到的资料来源。\n\n"
        f"【知识库资料】\n{context_text}\n\n"
        f"【用户问题】\n{user_question}"
    )
