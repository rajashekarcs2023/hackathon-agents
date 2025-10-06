import React from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";

type ChatInputProps = {
  input: string;
  handleInputChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  isLoading: boolean;
};

export function ChatInput({
  input,
  handleInputChange,
  handleSubmit,
  isLoading,
}: ChatInputProps) {
  return (
    <>
    <form onSubmit={handleSubmit} className="p-4 border-t flex space-x-2">
      <Textarea
        value={input}
        onChange={handleInputChange}
        placeholder="Ask a question..."
        className="flex-1 min-h-[40px]"
        disabled={isLoading}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e as unknown as React.FormEvent<HTMLFormElement>);
          }
        }}
      />
      <Button type="submit" disabled={isLoading}>
        <Send className="w-4 h-4" />
        </Button>
      </form>
      <div className="text-xs text-gray-400 text-center pb-3">
        <p>
          This model can make mistakes. Check important info
        </p>
      </div>
    </>
  );
}
