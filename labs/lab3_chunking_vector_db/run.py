from labs.lab3_chunking_vector_db.chunking import chunk_document
from labs.lab3_chunking_vector_db.chromadb import get_fresh_collection, add_chunks, query_collection

# chunking best practices (enforced below):
# size       - 200-500 characters, 50-100 character overlap
# boundary   - split on sentence end, never mid-word
# quality    - see query_monitor.py for real-query testing + result monitoring
CHUNK_SIZE = 400
OVERLAP = 75

with open("data/policies_large.txt", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_document(text, CHUNK_SIZE, OVERLAP)
print(f"Chunked into {len(chunks)} chunks")

# print("\n--- All chunks ---")
# for i, chunk in enumerate(chunks):
#     print(f"\n[chunk_{i}] ({len(chunk)} chars)")
#     print(chunk)

collection = get_fresh_collection()
add_chunks(collection, chunks)

query = "Can I bring my cats to work?"
results = query_collection(collection, query, n_results=2)

ids = results["ids"][0]
docs = results["documents"][0]
distances = results["distances"][0]

print(f"\nQuery: {query!r}")
for doc_id, doc, dist in zip(ids, docs, distances):
    preview = doc[:150].replace("\n", " ")
    print("Answer:")
    print(f"- [{doc_id}] dist={dist:.3f}")
    print(f"- {preview}...")