import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# test docs
sentences = [
    "Dogs are allowed in the office on Fridays.",
    "Pets can come to work on Furry Fridays.",
    "Remote work policy allows 3 days from home.",
]

embeddings = model.encode(sentences, normalize_embeddings=True)

# calc similarity
sm_1_2 = np.dot(embeddings[0], embeddings[1])
sm_1_3 = np.dot(embeddings[0], embeddings[2])
sm_2_3 = np.dot(embeddings[1], embeddings[2])

print(f"Similarity between sentence 1 and 2: {sm_1_2*100:.1f}")
print(f"Similarity between sentence 1 and 3: {sm_1_3*100:.1f}")
print(f"Similarity between sentence 2 and 3: {sm_2_3*100:.1f}")