from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfSearch:
    """TF-IDF keyword search. Same interface as BM25Search so both
    can be swapped into run_test_queries() without changes."""

    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(docs)# find word importance scores

    def get_scores(self, query):
        """Return a similarity score for every document, given one query."""
        query_vec = self.vectorizer.transform([query])
        scores = (self.doc_matrix @ query_vec.T).toarray().flatten()
        return scores