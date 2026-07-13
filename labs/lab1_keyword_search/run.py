from labs.lab1_keyword_search.bm25_method import BM25Search
from labs.lab1_keyword_search.tfidf_method import TfidfSearch
from shared.utils import load_policies, run_test_queries

titles, docs = load_policies("data/policies.json")

tfidf = TfidfSearch(docs)
bm25 = BM25Search(docs)

test_queries = [
    "gym membership reimbursement",     # keyword-favoring, low overlap with title
    "money back for working from home", # semantic-favoring, no exact keyword match
    "how do I get reimbursed",          # ambiguous, appears in most docs
]

print("=" * 70)
print("TF-IDF RESULTS")
print("=" * 70)
run_test_queries("TF-IDF", tfidf.get_scores, test_queries, titles)

print("\n" + "=" * 70)
print("BM25 RESULTS")
print("=" * 70)
run_test_queries("BM25", bm25.get_scores, test_queries, titles)