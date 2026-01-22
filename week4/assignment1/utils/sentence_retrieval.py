import re
import math
from collections import defaultdict, Counter
from typing import List, Tuple

# -----------------------------
# Helpers
# -----------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def split_sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())


# -----------------------------
# Retriever
# -----------------------------
class ContextRetriever:
    def __init__(self):
        self.documents = {}
        self.doc_term_freqs = {}
        self.doc_freqs = defaultdict(int)
        self.total_docs = 0

    def add_document(self, doc_id: str, text: str):
        self.total_docs += 1
        self.documents[doc_id] = text

        tokens = tokenize(text)
        tf = Counter(tokens)
        self.doc_term_freqs[doc_id] = tf

        for term in tf:
            self.doc_freqs[term] += 1

    def retrieve_docs(self, query: str, top_k: int = 3):
        query_tokens = tokenize(query)
        scores = defaultdict(float)

        for doc_id, tf in self.doc_term_freqs.items():
            for term in query_tokens:
                if term in tf:
                    idf = math.log(self.total_docs / (1 + self.doc_freqs[term]))
                    scores[doc_id] += tf[term] * idf

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def extract_sentences(
        self,
        query: str,
        doc_ids: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:

        query_tokens = set(tokenize(query))
        sentence_scores = []

        for doc_id in doc_ids:
            sentences = split_sentences(self.documents[doc_id])

            for sent in sentences:
                sent_tokens = tokenize(sent)
                overlap = sum(1 for t in sent_tokens if t in query_tokens)

                if overlap > 0:
                    sentence_scores.append((sent, overlap))

        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return sentence_scores[:top_k]
