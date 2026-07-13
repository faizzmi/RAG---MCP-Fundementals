from labs.lab3_chunking_vector_db.chunking import chunk_document
from labs.lab3_chunking_vector_db.chromadb import get_fresh_collection, add_chunks, query_collection
from labs.query_monitor import check_chunk_sizes, check_overlap_preserved, log_query

def ingest(text, chunk_size=400, overlap=75):
    chunks = chunk_document(text, chunk_size, overlap)
    size_issues = check_chunk_sizes(chunks)
    overlap_issues = check_overlap_preserved(chunks, overlap)

    collection = get_fresh_collection()
    add_chunks(collection, chunks)

    return {
        "collection": collection,
        "chunk_count": len(chunks),
        "size_issues": size_issues,
        "overlap_issues": overlap_issues,
        "chunks": chunks
    }

def ask(collection, query, n_results=2):
    results = query_collection(collection, query, n_results)
    return log_query(query, results)
