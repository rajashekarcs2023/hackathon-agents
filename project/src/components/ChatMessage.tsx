"use client";

import React, { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { CodeBlock } from "./CodeBlock";
import { CitationList } from "./CitationList";
import CopyToClipboard from "react-copy-to-clipboard";
import { Button } from "@/components/ui/button";
import { Copy, Check } from "lucide-react";

type Message = {
  id: string;
  role: "user" | "assistant" | "system" | "data";
  content: string;
};

type ChatMessageProps = {
  message: Message;
};

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  // Extract citations and clean content
  const { cleanContent, citations } = useMemo(() => {
    if (isUser) {
      return { cleanContent: message.content, citations: [] };
    }

    const citationRegex = /\[Source:\s*(https?:\/\/[^\]]+)\]/g;
    const extractedCitations: { url: string }[] = [];
    const matches = message.content.matchAll(citationRegex);

    for (const match of matches) {
      const url = match[1].trim();
      if (!extractedCitations.find((c) => c.url === url)) {
        extractedCitations.push({ url });
      }
    }

    // Remove inline citations from content
    const cleaned = message.content.replace(citationRegex, "");

    return {
      cleanContent: cleaned,
      citations: extractedCitations,
    };
  }, [message.content, isUser]);

  const [copiedStates, setCopiedStates] = React.useState<{
    [key: string]: boolean;
  }>({});

  const handleInlineCodeCopy = (text: string) => {
    setCopiedStates({ ...copiedStates, [text]: true });
    setTimeout(() => {
      setCopiedStates((prev) => ({ ...prev, [text]: false }));
    }, 2000);
  };

  // Determine if inline code should be copyable or just styled
  const shouldBeCopyable = (text: string): boolean => {
    const copyablePatterns = [
      // Commands
      /^(npm|yarn|bun|git|cd|mkdir|ls|cp|mv|rm|chmod|curl|wget|pip|python|node|docker|kubectl)\s/i,
      // URLs and addresses
      /^https?:\/\/|^localhost:|^127\.0\.0\.1:|^0\.0\.0\.0:/i,
      // Installation commands
      /install|init|create|build|dev|start|test|deploy/i,
      // Code snippets with special characters
      /[(){}\[\]<>=!&|+\-*/%]/,
      // Environment variables
      /^[A-Z_]+=/,
    ];

    const nonCopyablePatterns = [
      // File extensions
      /\.(js|jsx|ts|tsx|py|html|css|scss|json|md|mdx|txt|yml|yaml|toml|env|gitignore)$/i,
      // Simple UI elements and settings
      /^(Settings|Dashboard|Profile|Account|Home|Login|Signup|Menu|Sidebar|Header|Footer|Navigation)$/i,
      // Single words without special characters (likely just styled terms)
      /^[a-zA-Z][a-zA-Z0-9_-]*$/,
    ];

    // Check non-copyable patterns first (more specific)
    if (nonCopyablePatterns.some((pattern) => pattern.test(text.trim()))) {
      return false;
    }

    // Check copyable patterns
    if (copyablePatterns.some((pattern) => pattern.test(text.trim()))) {
      return true;
    }

    // Default to copyable for URLs and longer technical strings
    return text.includes("://") || text.length > 20 || /[.:/\\-]/.test(text);
  };

  return (
    <div
      className={`flex max-w-[480px] pr-3 ${
        isUser ? "justify-end pr-4" : "justify-center"
      }`}
    >
      <div
        className={`${
          isUser ? "max-w-[75%]" : "max-w-[90%] pr-1"
        } rounded-lg ${
          isUser
            ? "bg-[#f0efef] text-primary-foreground p-3"
            : "bg-white border border-slate-200 p-4"
        }`}
        style={{ wordBreak: "break-word", overflowWrap: "break-word" }}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // Code blocks with syntax highlighting
            // @ts-expect-error - react-markdown types issue
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || "");
              const codeString = String(children).replace(/\n$/, "");

              if (!inline && match) {
                return <CodeBlock language={match[1]} value={codeString} />;
              }

              // Inline code with conditional copy functionality
              const isCopyable = !isUser && shouldBeCopyable(codeString);

              return (
                <span className="relative inline-flex items-center group">
                  <code
                    className={`
                      px-1.5 py-0.5 rounded text-xs font-mono
                      ${
                        isUser
                          ? "bg-primary-foreground/20 text-black text-wrap"
                          : "bg-[#f0efef] text-slate-800 font-medium"
                      }
                    `}
                    {...props}
                  >
                    {children}
                  </code>
                  {isCopyable && (
                    <CopyToClipboard
                      text={codeString}
                      onCopy={() => handleInlineCodeCopy(codeString)}
                    >
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 ml-2 opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        {copiedStates[codeString] ? (
                          <Check className="h-3 w-3 text-green-500" />
                        ) : (
                          <Copy className="h-3 w-3 text-slate-400" />
                        )}
                      </Button>
                    </CopyToClipboard>
                  )}
                </span>
              );
            },
            // Links
            a({ href, children }) {
              return (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`
                    underline underline-offset-2 transition-colors
                    ${
                      isUser
                        ? "text-primary-foreground/90 hover:text-primary-foreground"
                        : "text-blue-600 hover:text-blue-800"
                    }
                  `}
                >
                  {children}
                </a>
              );
            },
            // Lists
            ul({ children }) {
              return (
                <ul
                  className={`list-disc list-outside ml-5 space-y-1 my-3 leading-relaxed ${
                    isUser ? "text-slate-900 font-medium" : ""
                  }`}
                >
                  {children}
                </ul>
              );
            },
            ol({ children }) {
              return (
                <ol
                  className={`list-decimal list-outside ml-5 space-y-1 my-3 leading-relaxed ${
                    isUser ? "text-slate-900 font-medium" : ""
                  }`}
                >
                  {children}
                </ol>
              );
            },
            li({ children }) {
              return <li className="text-sm text-slate-700">{children}</li>;
            },
            // Paragraphs
            p({ children }) {
              return (
                <p
                  className={`mb-3 last:mb-0 leading-relaxed text-sm ${
                    isUser ? "text-slate-900 font-medium" : "text-slate-800"
                  }`}
                >
                  {children}
                </p>
              );
            },
            // Headers
            h1({ children }) {
              return (
                <h1 className="text-xl font-bold mb-3 mt-4 first:mt-0 text-slate-900">
                  {children}
                </h1>
              );
            },
            h2({ children }) {
              return (
                <h2 className="text-lg font-semibold mb-2 mt-4 first:mt-0 text-slate-900">
                  {children}
                </h2>
              );
            },
            h3({ children }) {
              return (
                <h3 className="text-base font-semibold mb-2 mt-3 first:mt-0 text-slate-900">
                  {children}
                </h3>
              );
            },
            // Blockquotes
            blockquote({ children }) {
              return (
                <blockquote className="border-l-3 border-blue-500 bg-blue-50 pl-4 py-2 my-3 rounded-r-md">
                  <div className="text-sm text-slate-700 italic font-medium">
                    {children}
                  </div>
                </blockquote>
              );
            },
            // Tables
            table({ children }) {
              return (
                <div className="overflow-x-auto my-6">
                  <table className="min-w-full divide-y divide-slate-200 border border-slate-200 rounded-lg">
                    {children}
                  </table>
                </div>
              );
            },
            thead({ children }) {
              return <thead className="bg-slate-50">{children}</thead>;
            },
            th({ children }) {
              return (
                <th className="px-3 py-2 text-left text-xs font-semibold text-slate-900">
                  {children}
                </th>
              );
            },
            td({ children }) {
              return (
                <td className="px-3 py-2 text-xs text-slate-700">{children}</td>
              );
            },
          }}
        >
          {cleanContent}
        </ReactMarkdown>
        {!isUser && citations.length > 0 && (
          <CitationList citations={citations} />
        )}
      </div>
    </div>
  );
}
