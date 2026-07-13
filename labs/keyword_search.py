"""
Lab 1: Basic Search and Keyword Limitations
Compares TF-IDF and BM25 keyword search against the same policy dataset,
using the same test queries, so results can be compared side by side.
"""

from search.keyword_search.bm25_method import BM25Search
from search.keyword_search.tidf_method import TfidfSearch
from search.keyword_search.utils import load_policies, run_test_queries


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