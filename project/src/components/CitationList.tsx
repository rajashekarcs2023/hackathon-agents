import React from "react";
import { ExternalLink } from "lucide-react";

interface Citation {
  url: string;
  title?: string;
}

interface CitationListProps {
  citations: Citation[];
}

export function CitationList({ citations }: CitationListProps) {
  if (citations.length === 0) return null;

  return (
    <div className="mt-6 pt-4 border-t border-slate-200 dark:border-slate-700">
      <p className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-3">Sources:</p>
      <div className="flex flex-wrap gap-2">
        {citations.map((citation, index) => (
          <a
            key={index}
            href={citation.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-sm bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 px-3 py-2 rounded-lg transition-colors text-slate-700 dark:text-slate-300"
          >
            <ExternalLink className="w-3 h-3" />
            <span className="max-w-[250px] truncate">
              {citation.title || new URL(citation.url).hostname}
            </span>
          </a>
        ))}
      </div>
    </div>
  );
}