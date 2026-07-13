from rank_bm25 import BM25Okapi


class BM25Search:
    """BM25 keyword search. Same interface as TfidfSearch so both
    can be swapped into run_test_queries() without changes."""

    def __init__(self, docs):
        self.docs = docs
        self.tokenized_docs = [doc.lower().split() for doc in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs) # build vocabulary from all docs (same idea as TfidfVectorizer's feature names)

    def get_scores(self, query):
        """Return a BM25 score for every document, given one query."""
        tokenized_query = query.lower().split()
        return self.bm25.get_scores(tokenized_query)