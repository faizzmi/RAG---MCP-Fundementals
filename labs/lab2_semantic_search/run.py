from shared.utils import load_policies, evaluate_accuracy
from labs.lab1_keyword_search.tfidf_method import TfidfSearch
from labs.lab1_keyword_search.bm25_method import BM25Search
from labs.lab2_semantic_search.embedding_method import SemanticSearch

titles, docs = load_policies("data/policies.json")

tfidf = TfidfSearch(docs)
bm25 = BM25Search(docs)
semantic = SemanticSearch(docs)

# (query, expected_doc_index) — index matches order in policies.json
test_cases = [
    ("gym membership reimbursement", 2),        # Health & Wellness Policy
    ("money back for working from home", 0),    # Work From Home Reimbursement
    ("can I get money for gym", 2),              # Health & Wellness Policy, zero exact overlap
    ("annual budget for courses", 1),            # Learning & Development Budget
    ("adjusting my schedule for childcare", 4),  # Flexible Work Hours
]

methods = [
    ("TF-IDF", tfidf.get_scores),
    ("BM25", bm25.get_scores),
    ("Semantic", semantic.get_scores),
]

results = {}
for name, search_fn in methods:
    results[name] = evaluate_accuracy(name, search_fn, test_cases, titles)

print("\n" + "=" * 40)
print("SUMMARY")
print("=" * 40)
for name, acc in results.items():
    print(f"  {name:10} {acc:.0f}%")
