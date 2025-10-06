import { Pinecone } from "@pinecone-database/pinecone";

if (!process.env.PINECONE_API_KEY) {
  throw new Error("PINECONE_API_KEY not set");
}

export const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

export const indexName = process.env.PINECONE_INDEX_NAME || "";

export const indexHost = process.env.PINECONE_INDEX_HOST || "";
