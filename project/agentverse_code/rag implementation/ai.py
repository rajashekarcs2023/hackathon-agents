import os
import json
import requests

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}

def get_completion(context: str, query: str) -> str:
    """
    Fetches a completion from the OpenAI API using provided context and query.

    Args:
        context (str): The context from Fetch.ai documentation.
        query (str): The user's query to be answered.

    Returns:
        str: The API response content or an error message if the request fails.
    """
    body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise AI assistant specialized in Fetch.ai documentation. "
                    "Your sole source of truth is the provided context from official docs—do not use external knowledge or make up information. "
                    "Answer the user's query concisely and helpfully, grounding every claim in the context. "
                    "If you're unable to answer the question due to a lack of info in the context, let the user know.\n\n"
                    "For citations:\n"
                    "- Include inline citations as ()[Source: URL] directly after the relevant sentence or fact.\n"
                    "- Only cite if it directly supports your answer; avoid over-citing.\n\n"
                    "Markdown formatting guidelines for clean, professional documentation:\n\n"
                    "**Text Formatting:**\n"
                    "- Use **bold** for important concepts, key terms, and emphasis\n"
                    "- Use *italics* for parameters, variables, or field names\n"
                    "- Use ### headings to organize different sections clearly\n"
                    "- Use numbered lists (1. 2. 3.) for step-by-step processes\n"
                    "- Use bullet points (- or *) for feature lists or options\n\n"
                    "**Code and Technical Elements:**\n"
                    "- Use `backticks` for copyable elements: commands, URLs, code snippets, installation instructions\n"
                    "- Examples: `npm install -g node`, `localhost:3000`, `git clone`, `https://example.com`\n"
                    "- Use `backticks` for file names and UI elements (these will be styled but not copyable)\n"
                    "- Examples: `index.mdx`, `Settings`, `.env`, `package.json`\n"
                    "- Use code blocks with language specification for multi-line code:\n"
                    "```python\n"
                    "# Multi-line code example\n"
                    "```\n\n"
                    "**Organization:**\n"
                    "- Start responses with clear section headers when appropriate\n"
                    "- Use > blockquotes for important notes, warnings, or tips\n"
                    "- Use | tables | when | displaying | structured | data |\n"
                    "- Keep content well-spaced and organized like professional documentation\n"
                    "- Group related information under clear headings\n\n"
                    "**Style Goals:**\n"
                    "- Use consistent formatting throughout the response\n"
                    "- Prioritize readability and scanability\n"
                    "- Include proper spacing between sections\n\n"
                    f"Context (retrieved chunks from docs):\n{context}\n\n"
                    f"User Query:\n{query}\n\n"
                    "Think step-by-step:\n"
                    "1. Analyze the query.\n"
                    "2. Identify matching info from context.\n"
                    "3. Formulate a clear response using appropriate markdown formatting."
                )
            },
            {"role": "user", "content": query},
        ],
        "max_tokens": MAX_TOKENS,
    }

    try:
        response = requests.post(OPENAI_URL, headers=HEADERS, data=json.dumps(body), timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        return f"LLM error: {e}"