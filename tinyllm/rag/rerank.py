from typing import List, Dict

from smartpy.utility.ai_util import get_top_n_diverse_texts
from tinyllm.rag.document.document import Document


class ReRanker:

    def __init__(self,
                 docs = [],
                 scores = []):
        self.docs = docs
        self.scores = scores

    def add_doc(self,
                doc: Document,
                scores: float):
        self.docs.append(doc)
        self.scores.append(scores)

    def rerank(self,
               top_n: int) -> List[Document]:
        # Normalize scores
        # Average scores for each doc
        # Sort by score
        # Implement MMR based
        texts = [doc.content for doc in self.docs]
        embeddings = [doc.embeddings for doc in self.docs]
        top_n_texts = get_top_n_diverse_texts(texts=texts,
                                              embeddings=embeddings,
                                              top_n=top_n)
        top_n_docs = [doc for doc in self.docs if doc.content in top_n_texts]
        return top_n_docs