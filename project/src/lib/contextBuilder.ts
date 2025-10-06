type Message = {
  id: string;
  role: "user" | "assistant" | "system" | "data";
  content: string;
};


export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// check if message is a follow up
function isFollowUpMessage(message: string): boolean {
  const followUpPatterns = [
    // Question words that suggest continuation
    /^(what about|how do i|how can i|can you also|is there|are there)/i,
    // Reference words
    /\b(it|that|this|the code|the setup|the function|the method|the agent)\b/i,
    // Continuation phrases
    /\b(also|additionally|furthermore|and|but|however)\b/i,
    // Pronouns that reference previous context
    /\b(they|them|these|those)\b/i,
  ];

  return followUpPatterns.some(pattern => pattern.test(message));
}

// check if message is a topic change
function isTopicChange(message: string): boolean {
  const topicChangePatterns = [
    // Greetings and conversation starters
    /^(hi|hello|hey|good morning|good afternoon|good evening)/i,
    // New topic indicators
    /^(now|next|let's|i want to|i need to|can you help me with)/i,
    // Complete subject changes (very different from Fetch.ai context)
    /\b(weather|food|sports|politics|news)\b/i,
  ];

  return topicChangePatterns.some(pattern => pattern.test(message));
}

// build context from recent messages
export function buildContextFromMessages(
  messages: Message[],
  maxTokens: number = 7000
): string {
  if (messages.length === 0) {
    return "";
  }

  // Always include the last message (current query)
  const lastMessage = messages[messages.length - 1];
  
  // If it's not a user message, just return its content
  if (lastMessage.role !== "user") {
    return lastMessage.content;
  }

  const contextMessages: Message[] = [lastMessage];
  let totalTokens = estimateTokens(lastMessage.content);

  // Look back through recent messages (max 5 total messages)
  for (let i = messages.length - 2; i >= 0 && contextMessages.length < 5; i--) {
    const currentMessage = messages[i];
    
    // Only consider user messages for context building
    if (currentMessage.role !== "user") {
      continue;
    }

    const messageTokens = estimateTokens(currentMessage.content);
    
    // Check if adding this message would exceed token limit
    if (totalTokens + messageTokens > maxTokens) {
      break;
    }

    // Check for topic change - stop if we hit a clear topic boundary
    if (isTopicChange(currentMessage.content)) {
      break;
    }

    // Check if this message or the message after it suggests continuity
    const nextMessage = messages[i + 1];
    const isRelevant = 
      isFollowUpMessage(nextMessage?.content || "") ||
      isFollowUpMessage(currentMessage.content);

    if (isRelevant) {
      contextMessages.unshift(currentMessage);
      totalTokens += messageTokens;
    } else {
      // If we already have multiple messages and this one doesn't seem related, stop
      if (contextMessages.length > 1) {
        break;
      }
    }
  }

  // If we only have one message, return it as-is
  if (contextMessages.length === 1) {
    return lastMessage.content;
  }

  // Combine messages with clear delineation for better retrieval
  const contextString = contextMessages
    .map((msg, index) => {
      if (index === contextMessages.length - 1) {
        return `Current query: ${msg.content}`;
      }
      return `Previous context: ${msg.content}`;
    })
    .join("\n\n");

  return contextString;
}