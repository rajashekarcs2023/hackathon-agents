// scripts/ingest-docs.ts
import fs from "fs/promises";
import { XMLParser } from "fast-xml-parser";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { v4 as uuidv4 } from "uuid";

// Config
const CHUNK_SIZE = 800; // Tokens-ish
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "text-embedding-3-small",
});
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY || "" });
const index = pinecone.index(process.env.PINECONE_INDEX_NAME || "", process.env.PINECONE_INDEX_HOST || "");

const seen = new Set<string>();

function escapeXmlEntitiesInContent(text: string): string {
  return text.replace(/<content>([\s\S]*?)<\/content>/g, (match, content) => {
    const escapedContent = content
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&apos;");
    return `<content>${escapedContent}</content>`;
  });
}

async function ingest(filePath: string) {
  const rawData = await fs.readFile(filePath, "utf-8");

  // Escape XML entities in content but preserve XML structure
  const cleanedData = escapeXmlEntitiesInContent(rawData);

  const wrapped = `<pages>${cleanedData}</pages>`;

  const parser = new XMLParser();
  const parsed = parser.parse(wrapped);

  const pages = parsed.pages.page || [];

  for (const page of pages) {
    const title = page.title || "Untitled";
    const url = page.url || "";
    const content = page.content || "";

    const contentHash = Buffer.from(content).toString("base64");
    if (seen.has(contentHash)) continue;
    seen.add(contentHash);

    // Chunk
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: CHUNK_SIZE,
      chunkOverlap: 150,
    });
    const chunks = await splitter.splitText(content);

    // Embed chunks
    const vectors = await embeddings.embedDocuments(chunks);

    // metadata
    const records = chunks.map((chunk, i) => ({
      id: `${uuidv4()}-${i}`,
      values: vectors[i],
      metadata: { text: chunk, source: url, title, pageIndex: i },
    }));

    // Upsert to DB
    await index.upsert(records);
    console.log(`Ingested ${chunks.length} chunks from ${url}`);
  }

  console.log("Ingestion complete!");
}

for (const filePath of [
  "docs/agentverse.txt",
  "docs/ASI.txt",
  "docs/flockx.txt",
  "docs/flockx-api-reference.txt",
  "docs/innovationlab.txt",
  "docs/uagents.txt",
  "docs/uagents-api-reference.txt"
]) {
  ingest(filePath).catch(console.error);
}
