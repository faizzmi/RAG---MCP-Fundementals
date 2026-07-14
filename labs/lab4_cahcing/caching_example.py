import redis
import json
import hashlib

cache = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_response(quey, context):
    # Create a unique key based on the query and context
    key = hashlib.sha256(f"{quey}:{context}".encode()).hexdigest()
    
    # Check if the response is already cached
    cached_response = cache.get(key)
    if cached_response:
        return json.loads(cached_response)
    
    # If not cached, generate the response (this is just a placeholder)
    response = generate_rag_response(quey, context)
    
    # Cache the response for future use
    cache.set(key, json.dumps(response), ex=3600)  # Cache for 1 hour
    
    return response


def generate_rag_response(query, context):
    # Placeholder implementation for generating a response
    return {"query": query, "context": context, "response": "Generated response"}
