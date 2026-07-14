import os
import json
import hashlib
import chromadb
from chromadb.utils import embedding_functions

class VectorDBError(Exception):
    pass

class LLMError(Exception):
    pass

class EmbeddingError(Exception):
    pass

CHROMA_PATH = "./chroma_store"
COLLECTION_NAME = "hr_policies"

client = chromadb.PersistentClient(path=CHROMA_PATH)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def get_collection():
    try:
        return client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    except Exception as e:
        raise VectorDBError(f"collection not found: {e}")

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest(folder_path):
    collection = client.get_or_create_collection(COLLECTION_NAME, embedding_function=embed_fn)
    all_chunks = []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(folder_path, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{fname}_{i}".encode()).hexdigest()
            collection.add(documents=[chunk], ids=[chunk_id], metadatas=[{"source": fname}])
            all_chunks.append(chunk)
    return {"chunks": all_chunks, "count": len(all_chunks)}

def embed_query(query):
    try:
        vectors = embed_fn([query])
        return vectors[0]
    except Exception as e:
        raise EmbeddingError(f"embed fail: {e}")

def semantic_search(query, top_k=3):
    collection = get_collection()
    embed_query(query)
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    if not docs:
        raise VectorDBError("no results found")
    return docs

def keyword_search(query, top_k=3):
    try:
        collection = get_collection()
        all_docs = collection.get()["documents"]
    except Exception as e:
        raise VectorDBError(f"keyword search fail: {e}")
    query_words = set(query.lower().split())
    scored = []
    for doc in all_docs:
        doc_words = set(doc.lower().split())
        score = len(query_words & doc_words)
        if score > 0:
            scored.append((score, doc))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:top_k]]

def text_search(query, top_k=3):
    try:
        collection = get_collection()
        all_docs = collection.get()["documents"]
    except Exception as e:
        raise VectorDBError(f"text search fail: {e}")
    matches = [doc for doc in all_docs if query.lower() in doc.lower()]
    return matches[:top_k] if matches else ["No matching text found."]

def call_llm(query, context_chunks):
    if not context_chunks:
        raise LLMError("no context to send")
    context = "\n\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer based on context above."
    raise LLMError("LLM API not configured in this example")

def format_retrieved_chunks(chunks):
    if not chunks:
        return "No relevant policy information found."
    formatted = "\n\n---\n\n".join(chunks)
    return f"Here relevant policy excerpts:\n\n{formatted}"

def run_full_pipeline(query):
    chunks = semantic_search(query)
    return call_llm(query, chunks)

def get_policy_answer(query):
    try:
        return run_full_pipeline(query)
    except VectorDBError:
        results = keyword_search(query)
        return format_retrieved_chunks(results)
    except LLMError:
        results = semantic_search(query)
        return format_retrieved_chunks(results)
    except EmbeddingError:
        results = text_search(query)
        return format_retrieved_chunks(results)
    except Exception:
        return "Service temporarily unavailable. Please try again later."

if __name__ == "__main__":
    docs_folder = "./hr_policies"
    if os.path.exists(docs_folder):
        result = ingest(docs_folder)
        print(f"ingested {result['count']} chunks")
    query = "How many annual leave days do I get?"
    answer = get_policy_answer(query)
    print(answer)
    