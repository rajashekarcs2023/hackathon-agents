"use client";

import { Grid } from "ldrs/react";
import 'ldrs/react/Grid.css'
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Bot } from "lucide-react";

export function LoadingIndicator() {
  return (
    <div className="flex items-start gap-3 mb-4">
      <Avatar className="w-8 h-8">
        <AvatarFallback>
          <Bot className="w-5 h-5" />
        </AvatarFallback>
      </Avatar>
      <Grid
        size="32"
        speed="1.5"
        color="currentColor"
      />
    </div>
  );
}