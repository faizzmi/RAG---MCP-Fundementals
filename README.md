# RAG & MCP Fundamentals

Personal project following the freeCodeCamp "RAG & MCP Fundamentals" crash
course. Testing keyword search (TF-IDF, BM25) vs semantic search on a small
set of HR policy documents, before moving on to vector databases, chunking,
a full RAG pipeline, and MCP.

## Project structure

```
.
├── data/
│   └── policies.json          # 5 HR policy docs (id, title, text)
├── search/
│   ├── tfidf_search.py        # TfidfSearch class
│   ├── bm25_search.py         # BM25Search class
│   └── utils.py               # shared: load_policies(), run_test_queries()
├── lab1_keyword_search.py     # Lab 1: compares TF-IDF vs BM25
├── requirements.txt
└── README.md
```

Both search classes expose the same interface, `get_scores(query)` returns
a similarity/relevance score for every document. That means any new search
method (e.g. semantic search in Lab 2) only needs to implement `get_scores`
to plug into the same `run_test_queries()` output function, no duplicate
printing/formatting code per method.

## Setup

```powershell
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
py lab1_keyword_search.py
```

## Test queries used

- `"gym membership reimbursement"` — keyword-favoring, low title overlap
- `"money back for working from home"` — semantic-favoring, no exact keyword match
- `"how do I get reimbursed"` — ambiguous, appears across most docs

## Notes / findings

- TF-IDF and BM25 mostly agree on top results, but disagree on ambiguous
  queries where multiple docs share vocabulary (e.g. "reimbursed" appears
  in 4 of 5 policies).
- Neither method handles queries with zero keyword overlap well (e.g. a
  future query like "can I get money for gym" wouldn't necessarily surface
  "Health & Wellness Policy" if the wording diverges further) — this is
  the gap Lab 2 (semantic search / embeddings) is meant to close.

## Roadmap (per course structure)

- [x] Lab 1: Keyword search (TF-IDF, BM25)
- [ ] Lab 2: Semantic search with embeddings
- [ ] Lab 3: Vector database (Chroma)
- [ ] Lab 4: Document chunking
- [ ] Lab 5: Full RAG pipeline
- [ ] Labs 6–8: MCP server/client