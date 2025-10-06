import React from "react";
import { createRoot } from "react-dom/client";
import { ChatWidget } from "@/components/ChatWidget";

interface ChatWidgetConfig {
  containerId: string;
  apiUrl?: string;
}

export default function initChatWidget(config: string | ChatWidgetConfig) {
  const { containerId, apiUrl } = typeof config === 'string' 
    ? { containerId: config, apiUrl: undefined }
    : config;
    
  const container = document.getElementById(containerId);
  if (container) {
    createRoot(container).render(
      <React.StrictMode>
        <ChatWidget apiUrl={apiUrl} />
      </React.StrictMode>
    );
  } else {
    console.error(`Container with id ${containerId} not found`);
  }
}
