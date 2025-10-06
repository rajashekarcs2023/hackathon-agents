import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";
import { getContext } from "@/lib/rag";
import { buildContextFromMessages } from "@/lib/contextBuilder";
import { Redis } from "@upstash/redis";
import { Ratelimit } from "@upstash/ratelimit";

export async function OPTIONS(req: Request) {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    },
  });
}

export async function POST(req: Request) {
  const ip = req.headers.get("x-forwarded-for") ?? "anonymous";
  const redis = new Redis({
    url: process.env.UPSTASH_REDIS_REST_URL!,
    token: process.env.UPSTASH_REDIS_REST_TOKEN!,
  });
  const ratelimit = new Ratelimit({
    redis,
    limiter: Ratelimit.slidingWindow(10, "60 s"),
    analytics: true,
  });
  const { success, limit, remaining, reset } = await ratelimit.limit(ip);
  if (!success) {
    return new Response("Rate limit exceeded", {
      status: 429,
      headers: {
        "X-RateLimit-Limit": limit.toString(),
        "X-RateLimit-Remaining": remaining.toString(),
        "X-RateLimit-Reset": reset.toString(),
      },
    });
  }
  const { messages } = await req.json();
  const lastMessage = messages[messages.length - 1].content;

  // Build smart context from recent messages for better retrieval
  const queryContext = buildContextFromMessages(messages);
  const context = await getContext(queryContext);
  const prompt = `
You are a precise AI assistant specialized in Fetch.ai documentation. Your sole source of truth is the provided context from official docs—do not use external knowledge or make up information. Answer the user's query concisely and helpfully, grounding every claim in the context. If you're unable to answer the question due to a lack of info in the context, let the user know

For citations:
- Include inline citations as ()[Source: URL] directly after the relevant sentence or fact.
- Only cite if it directly supports your answer; avoid over-citing.

Markdown formatting guidelines for clean, professional documentation:

**Text Formatting:**
- Use **bold** for important concepts, key terms, and emphasis
- Use *italics* for parameters, variables, or field names
- Use ### headings to organize different sections clearly
- Use numbered lists (1. 2. 3.) for step-by-step processes
- Use bullet points (- or *) for feature lists or options

**Code and Technical Elements:**
- Use \`backticks\` for copyable elements: commands, URLs, code snippets, installation instructions
  - Examples: \`npm install -g node\`, \`localhost:3000\`, \`git clone\`, \`https://example.com\`
- Use \`backticks\` for file names and UI elements (these will be styled but not copyable)
  - Examples: \`index.mdx\`, \`Settings\`, \`.env\`, \`package.json\`
- Use code blocks with language specification for multi-line code:
  \`\`\`python
  # Multi-line code example
  \`\`\`

**Organization:**
- Start responses with clear section headers when appropriate
- Use > blockquotes for important notes, warnings, or tips
- Use | tables | when | displaying | structured | data |
- Keep content well-spaced and organized like professional documentation
- Group related information under clear headings

**Style Goals:**
- Use consistent formatting throughout the response
- Prioritize readability and scanability
- Include proper spacing between sections

Context (retrieved chunks from docs):
${context}

User Query:
${lastMessage}

Think step-by-step:
1. Analyze the query.
2. Identify matching info from context.
3. Formulate a clear response using appropriate markdown formatting.`;

  const result = await streamText({
    model: openai("gpt-4o-2024-08-06"),
    prompt,
  });

  const response = result.toDataStreamResponse();
  response.headers.set("X-RateLimit-Limit", limit.toString());
  response.headers.set("X-RateLimit-Remaining", remaining.toString());
  response.headers.set("X-RateLimit-Reset", reset.toString());

  response.headers.set("Access-Control-Allow-Origin", "*");
  response.headers.set("Access-Control-Allow-Methods", "POST, OPTIONS");
  response.headers.set(
    "Access-Control-Allow-Headers",
    "Content-Type, Authorization"
  );

  return response;
}
