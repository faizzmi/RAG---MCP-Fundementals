import chromadb

client = chromadb.Client()

def get_fresh_collection(name="policies"):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name)


def add_chunks(collection, chunks):
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"chunk_id": i, "length": len(chunk)}],
            ids=[f"chunk_{i}"],
        )


def query_collection(collection, query_text, n_results=2):
    return collection.query(query_texts=[query_text], n_results=n_results)