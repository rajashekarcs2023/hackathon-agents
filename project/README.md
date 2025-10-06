# Fetch.ai Documentation Chat Assistant

A powerful **Retrieval-Augmented Generation (RAG)** powered chat assistant that enables developers to interactively query Fetch.ai documentation. Built with Next.js, OpenAI, and vector search technology to provide accurate, real-time answers grounded in official documentation.

## Features

- **Semantic Search** - Advanced vector similarity search using OpenAI embeddings
- **Real-time Chat** - Streaming responses with AI-powered conversation
- **Documentation Grounding** - All answers sourced exclusively from official Fetch.ai docs
- **Inline Citations** - Transparent source attribution for every answer
- **Embeddable Widget** - Lightweight chat widget for any website

## Tech Stack

**Frontend:**

- [Next.js 15](https://nextjs.org/) - React framework with App Router
- [TypeScript](https://www.typescriptlang.org/) - Type-safe development
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first styling
- [Framer Motion](https://www.framer.com/motion/) - Smooth animations

**Backend & AI:**

- [OpenAI GPT-4o](https://openai.com/) - Language model for responses
- [Vercel AI SDK](<[https://openai.com/](https://ai-sdk.dev/docs/introduction)>) - AI Response Streaming
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) - Text vectorization
- [LangChain](https://langchain.com/) - AI application framework
- [Pinecone](https://www.pinecone.io/) - Vector databases

**Infrastructure:**

- [Upstash Redis](https://upstash.com/) - Rate limiting and caching
- [Vercel](https://vercel.com/) - Deployment platform

## Cron: refresh docs

- API route: `src/app/api/cron/refresh/route.ts`
- Uses `sitefetch` to crawl docs and upsert into Pinecone with OpenAI embeddings
- Scheduled via `vercel.json` every 1st and 15th at 00:00 UTC

Manual trigger (local): start dev and hit `GET /api/cron/refresh`
