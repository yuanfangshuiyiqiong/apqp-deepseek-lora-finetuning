import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Moon, Sun, User, Menu, Trash2, Plus } from 'lucide-react';
import { Message, Conversation } from '../types';
import { sendChatMessage, generateId, generateTitle, saveConversations, loadConversations } from '../utils';
import { useTheme } from '../hooks/useTheme';
import MarkdownRenderer from './MarkdownRenderer';

const MODEL_NAME = 'DeepSeek Local Model';

export default function ChatInterface() {
  const { theme, toggleTheme } = useTheme();
  const [conversations, setConversations] = useState<Conversation[]>(() => loadConversations());
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const currentConversation = conversations.find(c => c.id === currentConversationId);
  const messages = currentConversation?.messages || [];

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // 保存对话
  useEffect(() => {
    saveConversations(conversations);
  }, [conversations]);

  // 创建新对话
  const createNewConversation = () => {
    const newConversation: Conversation = {
      id: generateId(),
      title: '新对话',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    };
    setConversations(prev => [newConversation, ...prev]);
    setCurrentConversationId(newConversation.id);
    setInput('');
    if (window.innerWidth < 768) {
      setSidebarOpen(false);
    }
  };

  // 选择对话
  const selectConversation = (id: string) => {
    setCurrentConversationId(id);
    if (window.innerWidth < 768) {
      setSidebarOpen(false);
    }
  };

  // 删除对话
  const deleteConversation = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setConversations(prev => prev.filter(c => c.id !== id));
    if (currentConversationId === id) {
      setCurrentConversationId(null);
    }
  };

  // 发送消息
  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: input.trim(),
      timestamp: Date.now()
    };

    // 确保当前有对话
    let convId = currentConversationId;
    if (!convId) {
      const newConversation: Conversation = {
        id: generateId(),
        title: '新对话',
        messages: [],
        createdAt: Date.now(),
        updatedAt: Date.now()
      };
      setConversations(prev => [newConversation, ...prev]);
      convId = newConversation.id;
      setCurrentConversationId(convId);
    }

    // 添加用户消息
    setConversations(prev => prev.map(c => {
      if (c.id === convId) {
        return {
          ...c,
          messages: [...c.messages, userMessage],
          title: c.messages.length === 0 ? generateTitle([...c.messages, userMessage]) : c.title,
          updatedAt: Date.now()
        };
      }
      return c;
    }));

    setInput('');
    setIsLoading(true);

    try {
      const allMessages = conversations.find(c => c.id === convId)?.messages || [];
      const apiMessages = [...allMessages, userMessage].map(m => ({
        role: m.role,
        content: m.content
      }));

      const response = await sendChatMessage(apiMessages);

      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: response,
        timestamp: Date.now()
      };

      setConversations(prev => prev.map(c => {
        if (c.id === convId) {
          return {
            ...c,
            messages: [...c.messages, assistantMessage],
            updatedAt: Date.now()
          };
        }
        return c;
      }));
    } catch (error) {
      const errorMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: `抱歉，发生了错误：${error instanceof Error ? error.message : '未知错误'}`,
        timestamp: Date.now()
      };

      setConversations(prev => prev.map(c => {
        if (c.id === convId) {
          return {
            ...c,
            messages: [...c.messages, errorMessage],
            updatedAt: Date.now()
          };
        }
        return c;
      }));
    } finally {
      setIsLoading(false);
    }
  };

  // 自动调整输入框高度
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  // 处理键盘事件
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen bg-background">
      {/* 侧边栏遮罩 (移动端) */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* 侧边栏 */}
      <aside className={`
        fixed md:static inset-y-0 left-0 z-50
        w-72 bg-card border-r border-border
        flex flex-col
        transform transition-transform duration-300 ease-in-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        {/* 侧边栏头部 */}
        <div className="p-4 border-b border-border">
          <button
            onClick={createNewConversation}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 
                     bg-primary text-primary-foreground rounded-lg
                     hover:bg-primary/90 transition-colors font-medium"
          >
            <Plus size={18} />
            <span>新对话</span>
          </button>
        </div>

        {/* 对话列表 */}
        <div className="flex-1 overflow-y-auto scrollbar-thin p-2">
          {conversations.map(conv => (
            <div
              key={conv.id}
              onClick={() => selectConversation(conv.id)}
              className={`
                group flex items-center gap-2 px-3 py-3 rounded-lg cursor-pointer
                transition-colors mb-1
                ${conv.id === currentConversationId 
                  ? 'bg-accent text-accent-foreground' 
                  : 'hover:bg-accent/50 text-muted-foreground hover:text-foreground'}
              `}
            >
              <div className="flex-1 truncate text-sm font-medium">
                {conv.title}
              </div>
              <button
                onClick={(e) => deleteConversation(e, conv.id)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 
                         transition-opacity"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        {/* 侧边栏底部 */}
        <div className="p-4 border-t border-border space-y-2">
          <button
            onClick={toggleTheme}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg
                     hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
          >
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            <span className="text-sm">{theme === 'dark' ? '浅色模式' : '深色模式'}</span>
          </button>
          <div className="flex items-center gap-3 px-3 py-2">
            <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
              <User size={16} className="text-primary-foreground" />
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium">用户</div>
              <div className="text-xs text-muted-foreground">本地部署</div>
            </div>
          </div>
        </div>
      </aside>

      {/* 主内容区 */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* 顶部栏 */}
        <header className="flex items-center gap-4 p-4 border-b border-border bg-card">
          <button
            onClick={() => setSidebarOpen(true)}
            className="md:hidden p-2 hover:bg-accent rounded-lg transition-colors"
          >
            <Menu size={20} />
          </button>
          <div className="flex-1">
            <h1 className="font-semibold text-lg">{MODEL_NAME}</h1>
          </div>
          <button
            onClick={toggleTheme}
            className="p-2 hover:bg-accent rounded-lg transition-colors hidden md:block"
          >
            {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </header>

        {/* 消息区域 */}
        <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
                <span className="text-3xl">💬</span>
              </div>
              <h2 className="text-xl font-semibold mb-2">开始新对话</h2>
              <p className="text-muted-foreground max-w-md">
                输入你的问题，AI 助手将为你解答。支持 Markdown 格式，包括代码高亮。
              </p>
            </div>
          ) : (
            messages.map(message => (
              <div
                key={message.id}
                className={`flex gap-4 animate-fade-in ${
                  message.role === 'user' ? 'flex-row-reverse' : ''
                }`}
              >
                {/* 头像 */}
                <div className={`
                  flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
                  ${message.role === 'user' 
                    ? 'bg-primary text-primary-foreground' 
                    : 'bg-secondary text-secondary-foreground'}
                `}>
                  {message.role === 'user' ? <User size={18} /> : <span className="text-sm">AI</span>}
                </div>

                {/* 消息内容 */}
                <div className={`flex-1 max-w-3xl ${message.role === 'user' ? 'text-right' : ''}`}>
                  <div className={`
                    inline-block px-4 py-3 rounded-2xl text-left
                    ${message.role === 'user'
                      ? 'bg-primary text-primary-foreground rounded-tr-md'
                      : 'bg-muted rounded-tl-md'}
                  `}>
                    <MarkdownRenderer content={message.content} />
                  </div>
                  <div className="text-xs text-muted-foreground mt-1 px-1">
                    {new Date(message.timestamp).toLocaleTimeString('zh-CN', { 
                      hour: '2-digit', 
                      minute: '2-digit' 
                    })}
                  </div>
                </div>
              </div>
            ))
          )}
          
          {/* 加载指示器 */}
          {isLoading && (
            <div className="flex gap-4 animate-fade-in">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-secondary flex items-center justify-center">
                <span className="text-sm">AI</span>
              </div>
              <div className="flex-1 max-w-3xl">
                <div className="inline-block px-4 py-3 rounded-2xl rounded-tl-md bg-muted">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 size={18} className="animate-spin" />
                    <span>正在思考...</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* 输入区域 */}
        <div className="p-4 border-t border-border bg-card">
          <div className="flex gap-3 items-end max-w-4xl mx-auto">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入你的问题... (Shift+Enter 换行，Enter 发送)"
                className="w-full px-4 py-3 pr-12 bg-background border border-input rounded-xl
                         resize-none focus:outline-none focus:ring-2 focus:ring-ring/50
                         placeholder:text-muted-foreground min-h-[48px] max-h-[200px]"
                rows={1}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className={`
                flex-shrink-0 p-3 rounded-xl transition-all
                ${input.trim() && !isLoading
                  ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                  : 'bg-muted text-muted-foreground cursor-not-allowed'}
              `}
            >
              {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
            </button>
          </div>
          <p className="text-xs text-center text-muted-foreground mt-2">
            AI 助手可能会产生不准确的信息，请保持批判性思维
          </p>
        </div>
      </main>
    </div>
  );
}
