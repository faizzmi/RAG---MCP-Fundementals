# chunking best practices (enforced below):
# size       - 200-500 characters, 50-100 character overlap
# boundary   - split on sentence end, never mid-word
# quality    - see query_monitor.py for real-query testing + result monitoring

def chunk_document(text, chunk_size=400, overlap=75):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    text = text.strip()
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            search_from = start + int(chunk_size * 0.7)
            window = text[search_from:end]
            best = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
            if best != -1:
                end = search_from + best + 1
            else:
                fallback = end
                while fallback > start and not text[fallback].isspace():
                    fallback -= 1
                end = fallback if fallback > start else end
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_len:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
