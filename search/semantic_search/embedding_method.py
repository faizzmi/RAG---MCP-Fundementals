from sentence_transformers import SentenceTransformer, util


class SemanticSearch:
    def __init__(self, docs, model_name="all-MiniLM-L6-v2"):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(docs, convert_to_tensor=True)

    def get_scores(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.doc_embeddings)
        return scores[0].cpu().numpy()