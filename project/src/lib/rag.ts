import { PineconeStore } from "@langchain/pinecone";
import { pinecone, indexName, indexHost } from "./pinecone";
import { embeddings } from "./embeddings";

export async function getContext(query: string) {
  const index = pinecone.index(indexName, indexHost);
  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex: index,
  });
  const results = await vectorStore.similaritySearch(query, 8);
  return results
    .map((r) => {
      return `Source: ${r.metadata.source}\n\n Content: ${r.pageContent}`;
    })
    .join("\n\n");
}
