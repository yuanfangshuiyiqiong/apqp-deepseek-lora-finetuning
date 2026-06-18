import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { CSSProperties } from 'react';
import { useTheme } from '../hooks/useTheme';
import { useState } from 'react';
import { Copy, Check } from 'lucide-react';

interface MarkdownRendererProps {
  content: string;
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const { theme } = useTheme();
  const [copiedIndex, setCopiedIndex] = useState<string | null>(null);
  const syntaxTheme = (theme === 'dark' ? oneDark : oneLight) as { [key: string]: CSSProperties };

  const handleCopy = async (code: string, index: string) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedIndex(index);
      setTimeout(() => setCopiedIndex(null), 2000);
    } catch (err) {
      console.error('复制失败:', err);
    }
  };

  return (
    <ReactMarkdown
      className="prose prose-sm max-w-none dark:prose-invert prose-p:my-2 prose-headings:my-3 prose-ul:my-2 prose-ol:my-2 prose-li:my-1"
      components={{
        code({ className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          const codeString = String(children).replace(/\n$/, '');
          const index = `${match?.[1] || 'text'}-${codeString.slice(0, 20)}`;

          const isInline = !match && !codeString.includes('\n');

          if (isInline) {
            return (
              <code
                className="px-1.5 py-0.5 bg-muted-foreground/10 rounded text-sm font-mono"
                {...props}
              >
                {children}
              </code>
            );
          }

          return (
            <div className="relative group my-3 rounded-lg overflow-hidden">
              {/* 代码头部 */}
              <div className="flex items-center justify-between px-4 py-2 bg-muted border-b border-border">
                <span className="text-xs text-muted-foreground font-medium">
                  {match?.[1] || '代码'}
                </span>
                <button
                  onClick={() => handleCopy(codeString, index)}
                  className="flex items-center gap-1 text-xs text-muted-foreground 
                           hover:text-foreground transition-colors opacity-0 group-hover:opacity-100"
                >
                  {copiedIndex === index ? (
                    <>
                      <Check size={14} />
                      <span>已复制</span>
                    </>
                  ) : (
                    <>
                      <Copy size={14} />
                      <span>复制</span>
                    </>
                  )}
                </button>
              </div>

              {/* 代码内容 */}
              <SyntaxHighlighter
                style={syntaxTheme}
                language={match?.[1] || 'text'}
                PreTag="div"
                customStyle={{
                  margin: 0,
                  padding: '1rem',
                  background: theme === 'dark' ? '#1a1b26' : '#fafafa',
                  fontSize: '0.875rem',
                  lineHeight: '1.5'
                }}
              >
                {codeString}
              </SyntaxHighlighter>
            </div>
          );
        },
        p({ children }) {
          return <p className="my-2 leading-relaxed">{children}</p>;
        },
        h1({ children }) {
          return <h1 className="text-2xl font-bold my-4">{children}</h1>;
        },
        h2({ children }) {
          return <h2 className="text-xl font-bold my-3">{children}</h2>;
        },
        h3({ children }) {
          return <h3 className="text-lg font-semibold my-2">{children}</h3>;
        },
        ul({ children }) {
          return <ul className="list-disc list-inside my-2 space-y-1">{children}</ul>;
        },
        ol({ children }) {
          return <ol className="list-decimal list-inside my-2 space-y-1">{children}</ol>;
        },
        blockquote({ children }) {
          return (
            <blockquote className="border-l-4 border-primary pl-4 my-3 italic text-muted-foreground">
              {children}
            </blockquote>
          );
        },
        a({ href, children }) {
          return (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              {children}
            </a>
          );
        },
        table({ children }) {
          return (
            <div className="overflow-x-auto my-3">
              <table className="min-w-full border border-border rounded-lg overflow-hidden">
                {children}
              </table>
            </div>
          );
        },
        th({ children }) {
          return (
            <th className="px-4 py-2 bg-muted text-left font-semibold border-b border-border">
              {children}
            </th>
          );
        },
        td({ children }) {
          return (
            <td className="px-4 py-2 border-b border-border">
              {children}
            </td>
          );
        },
        hr() {
          return <hr className="my-4 border-border" />;
        },
        img({ src, alt }) {
          return (
            <img
              src={src}
              alt={alt}
              className="max-w-full h-auto rounded-lg my-3"
            />
          );
        }
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
