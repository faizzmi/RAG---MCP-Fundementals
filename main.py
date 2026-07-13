from search.utils import load_policies, print_results, evaluate_accuracy
from search.keyword_search.tfidf_method import TfidfSearch
from search.keyword_search.bm25_method import BM25Search
from search.semantic_search.embedding_method import SemanticSearch
from labs.lab3_chunking_vector_db.chunking import chunk_document
from labs.query_monitor import check_chunk_sizes, check_overlap_preserved, log_query
from labs.lab3_chunking_vector_db.chromadb import get_fresh_collection, add_chunks, query_collection


def choose(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input("> ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("invalid, try again")


def get_chunks():
    chunk_size = int(input("Chunk size (default 400): ") or 400)
    overlap = int(input("Overlap (default 75): ") or 75)
    with open("data/policies_large.txt", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_document(text, chunk_size, overlap)

    size_issues = check_chunk_sizes(chunks)
    overlap_issues = check_overlap_preserved(chunks, overlap)
    if size_issues:
        print("\nSize issues:")
        for i in size_issues:
            print(" -", i)
    if overlap_issues:
        print("\nOverlap issues:")
        for i in overlap_issues:
            print(" -", i)

    titles = [f"chunk_{i}" for i in range(len(chunks))]
    return titles, chunks


def run_single_query():
    source = choose("Data source:", [
        "policies.json (5 docs, no chunking)",
        "policies_large.txt (chunked)",
    ])
    method = choose("Search method:", [
        "TF-IDF", "BM25", "Semantic (sentence-transformers)", "Chroma vector DB",
    ])
    query = input("\nQuery: ").strip()

    use_chunks = "large.txt" in source or method == "Chroma vector DB"
    titles, docs = get_chunks() if use_chunks else load_policies("data/policies.json")

    if method == "Chroma vector DB":
        collection = get_fresh_collection()
        add_chunks(collection, docs)
        results = query_collection(collection, query, n_results=3)
        entry = log_query(query, results)
        print(f"\nQuery: {entry['query']!r}")
        for r in entry["results"]:
            flag = "  <-- LOW CONFIDENCE" if r["low_confidence"] else ""
            print(f"  [{r['id']}] dist={r['distance']:.3f}{flag}")
            print(f"      {r['chunk_preview']}...")
        return

    engine = {
        "TF-IDF": TfidfSearch,
        "BM25": BM25Search,
        "Semantic (sentence-transformers)": SemanticSearch,
    }[method](docs)

    scores = engine.get_scores(query)
    print_results(method, query, scores, titles, top_k=3)


def run_keyword_comparison():
    """Lab 1: TF-IDF vs BM25 side by side, canned test queries."""
    titles, docs = load_policies("data/policies.json")
    tfidf = TfidfSearch(docs)
    bm25 = BM25Search(docs)

    test_queries = [
        "gym membership reimbursement",
        "money back for working from home",
        "how do I get reimbursed",
    ]

    for name, fn in [("TF-IDF", tfidf.get_scores), ("BM25", bm25.get_scores)]:
        print("\n" + "=" * 60)
        print(f"{name} RESULTS")
        print("=" * 60)
        for query in test_queries:
            scores = fn(query)
            print_results(name, query, scores, titles)


def run_accuracy_comparison():
    """Lab 2: TF-IDF vs BM25 vs Semantic, labeled accuracy %."""
    titles, docs = load_policies("data/policies.json")
    tfidf = TfidfSearch(docs)
    bm25 = BM25Search(docs)
    semantic = SemanticSearch(docs)

    test_cases = [
        ("gym membership reimbursement", 2),
        ("money back for working from home", 0),
        ("can I get money for gym", 2),
        ("annual budget for courses", 1),
        ("adjusting my schedule for childcare", 4),
    ]

    results = {}
    for name, fn in [("TF-IDF", tfidf.get_scores), ("BM25", bm25.get_scores), ("Semantic", semantic.get_scores)]:
        results[name] = evaluate_accuracy(name, fn, test_cases, titles)

    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    for name, acc in results.items():
        print(f"  {name:10} {acc:.0f}%")


def main():
    mode = choose("What do you want to run?", [
        "Single query (pick source / method / chunk params)",
        "Lab 1: Keyword comparison (TF-IDF vs BM25, canned queries)",
        "Lab 2: Accuracy comparison (TF-IDF vs BM25 vs Semantic, labeled)",
    ])
    if mode.startswith("Single"):
        run_single_query()
    elif mode.startswith("Lab 1"):
        run_keyword_comparison()
    else:
        run_accuracy_comparison()


if __name__ == "__main__":
    main()
