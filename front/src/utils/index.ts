import { ChatRequest, ChatResponse } from "../types";

// API 基础 URL
const API_BASE_URL = "http://localhost:8000";

// 生成唯一 ID
export function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// 格式化时间戳
export function formatTime(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  if (diff < 60000) return "刚刚";
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`;
  if (diff < 604800000) return `${Math.floor(diff / 86400000)} 天前`;

  return date.toLocaleDateString("zh-CN");
}

// 发送聊天请求
export async function sendChatMessage(
  messages: { role: string; content: string }[],
  _onChunk?: (text: string) => void,
): Promise<string> {
  const request: ChatRequest = {
    messages: messages as ChatRequest["messages"],
    temperature: 0.7,
    max_new_tokens: 1024,
    top_p: 0.9,
    stream: false,
  };

  try {
    const response = await fetch(`${API_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`请求失败: ${response.status}`);
    }

    const data: ChatResponse = await response.json();
    return data.choices[0].message.content;
  } catch (error) {
    console.error("API 错误:", error);
    throw error;
  }
}

// 保存对话到本地存储
export function saveConversations(conversations: any[]): void {
  try {
    localStorage.setItem(
      "ai-chat-conversations",
      JSON.stringify(conversations),
    );
  } catch (error) {
    console.error("保存失败:", error);
  }
}

// 从本地存储加载对话
export function loadConversations(): any[] {
  try {
    const data = localStorage.getItem("ai-chat-conversations");
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
}

// 生成对话标题
export function generateTitle(messages: { role?: string; content: string }[]): string {
  const firstUserMessage = messages.find((m) => m.role === "user");
  if (firstUserMessage) {
    const content = firstUserMessage.content;
    return content.length > 30 ? content.substring(0, 30) + "..." : content;
  }
  return "新对话";
}
