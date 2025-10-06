import { NextRequest } from "next/server";
import { fetchSite } from "sitefetch";
import { pinecone, indexName, indexHost } from "@/lib/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const DOC_SITES: { url: string; match?: string[]; contentSelector?: string }[] =
  [
    { url: "https://docs.agentverse.ai" },
    { url: "https://network.fetch.ai/docs" },
    { url: "https://uagents.fetch.ai" },
    { url: "https://docs.asi1.ai/docs" },
    { url: "https://docs.flockx.io" },
  ];

export const maxDuration = 300; // allow up to 5 minutes on Vercel cron

export async function GET(_req: NextRequest) {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return new Response("Missing OPENAI_API_KEY", { status: 500 });
    }

    const index = pinecone.index(indexName, indexHost);
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 800,
      chunkOverlap: 150,
    });
    const embeddings = new OpenAIEmbeddings({
      modelName: "text-embedding-3-small",
    });

    // 1) Clear all vectors from the default namespace
    await index.namespace("").deleteAll();

    let totalChunks = 0;
    let totalPages = 0;

    // 2) Re-crawl and upload everything to the default namespace
    for (const site of DOC_SITES) {
      const pagesMap = await fetchSite(site.url, {
        concurrency: 8,
        match: site.match,
        contentSelector: site.contentSelector,
      });

      for (const page of pagesMap.values()) {
        totalPages++;
        const content = page.content ?? "";
        if (!content) continue;

        const chunks = await splitter.splitText(content);
        if (chunks.length === 0) continue;

        const vectors = await embeddings.embedDocuments(chunks);
        const records = chunks.map((chunk, i) => ({
          id: `${page.url}#${i}`,
          values: vectors[i],
          metadata: {
            text: chunk,
            source: page.url,
            title: page.title ?? "",
            pageIndex: i,
          },
        }));

        await index.upsert(records);
        totalChunks += chunks.length;
      }
    }

    return new Response(
      JSON.stringify({ status: "ok", totalPages, totalChunks }),
      {
        status: 200,
        headers: { "content-type": "application/json" },
      }
    );
  } catch (err: unknown) {
    console.error(err);
    const message = err instanceof Error ? err.message : "unknown";
    return new Response(JSON.stringify({ error: message }), { status: 500 });
  }
}
