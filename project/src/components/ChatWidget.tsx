"use client";

import React, { useState, useEffect, useRef } from "react";
import { useChat } from "ai/react";
import { AnimatePresence, motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare, X, RotateCcw, Sparkles } from "lucide-react";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { LoadingIndicator } from "./LoadingIndicator";

interface ChatWidgetProps {
  apiUrl?: string;
}

export function ChatWidget({ apiUrl = "/api/chat" }: ChatWidgetProps) {
  const [isOpen, setIsOpen] = useState(false);

  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
    setMessages,
    status,
    error,
  } = useChat({
    api: apiUrl,
    onError: (err) => {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: "Sorry, an error occurred. Please try again.",
        },
      ]);
    },
  });

  const handleClear = () => {
    setMessages([]);
  };

  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollAreaRef.current) {
      const viewport = scrollAreaRef.current.querySelector(
        '[data-slot="scroll-area-viewport"]'
      );
      if (viewport) {
        viewport.scrollTop = viewport.scrollHeight;
      }
    }
  }, [messages]);

  return (
    <>
      {!isOpen && (
        <Button
          onClick={() => setIsOpen(true)}
          className="font-medium cursor-pointer"
          variant="default"
        >
          <Sparkles className="w-6 h-6" />
          Ask AI
        </Button>
      )}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 50 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 50 }}
            className="fixed bottom-4 right-4 left-4 top-4 md:left-auto md:top-auto md:bottom-10 md:right-4 w-auto md:w-[400px] lg:w-[500px] h-auto md:h-[500px] lg:h-[600px] bg-background border rounded-lg shadow-xl flex flex-col overflow-hidden z-10"
          >
            <div className="p-4 border-b flex items-center justify-between">
              <h3 className="font-semibold text-lg flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-blue-900" />
                Assistant
              </h3>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleClear}
                  title="Clear chat"
                >
                  <RotateCcw className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsOpen(false)}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>
            <ScrollArea
              className="flex-1 md:h-[330px] lg:h-[430px] overflow-hidden"
              ref={scrollAreaRef}
            >
              <div className="p-4 flex flex-col space-y-4">
                {messages.map((message) => (
                  <ChatMessage key={message.id} message={message} />
                ))}
                {isLoading && status !== "streaming" && <LoadingIndicator />}
              </div>
            </ScrollArea>
            <ChatInput
              input={input}
              handleInputChange={handleInputChange}
              handleSubmit={handleSubmit}
              isLoading={isLoading}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
