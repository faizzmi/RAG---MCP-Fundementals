"""Shared helpers used by every search method (TF-IDF, BM25, and later
semantic/vector search) so we don't repeat loading or printing logic."""

import json
import numpy as np


def load_policies(path="data/policies.json"):
    """Load policy documents. Returns (titles, docs) as parallel lists."""
    with open(path, encoding="utf-8") as f:
        policies = json.load(f)

    titles = [p["title"] for p in policies]
    docs = [p["text"] for p in policies]
    return titles, docs


def print_results(method_name, query, scores, titles, top_k=3):
    """Print the top_k ranked results for one query, in a consistent
    format regardless of which search method produced the scores."""
    ranked = np.argsort(scores)[::-1][:top_k]

    print(f"\n[{method_name}] Query: {query!r}")
    for rank, doc_id in enumerate(ranked, start=1):
        print(f"  {rank}. {titles[doc_id]:35s} score={scores[doc_id]:.3f}")


def run_test_queries(method_name, search_fn, queries, titles):
    """Run a list of queries through a search_fn(query) -> scores array,
    and print results for each using the shared formatter."""
    for query in queries:
        scores = search_fn(query)
        print_results(method_name, query, scores, titles)

def evaluate_accuracy(method_name, search_fn, test_cases, titles):
    """Run test_cases (query + expected doc id) through search_fn,
    print each result, and report overall accuracy for this method."""
    correct = 0

    print(f"\n[{method_name}] Accuracy check")
    for query, expected_id in test_cases:
        scores = search_fn(query)
        predicted_id = int(np.argmax(scores))
        is_correct = predicted_id == expected_id
        correct += is_correct

        status = "✓" if is_correct else "✗"
        print(f"  {status} {query!r:45} predicted={titles[predicted_id]:30} expected={titles[expected_id]}")

    accuracy = correct / len(test_cases) * 100
    print(f"  Accuracy: {correct}/{len(test_cases)} ({accuracy:.0f}%)")
    return accuracy
