import time
from typing import Optional

# Import the core Glia components from the public surface
from glia import GliaManager, CacheInvalidator, CacheWatcher, VectorDBAdapter

# Import your custom infrastructure wrappers
from embedder import RealEmbedder
from db_wrapper import ChromaGliaWrapper

# ==============================================================================
# 1. Initialize Infrastructure
# ==============================================================================
print("Initializing Embedder and ChromaDB...")
embedder = RealEmbedder()
chroma_wrapper = ChromaGliaWrapper(host="localhost", port=8000)

collection_name = "products"
try:
    product_collection = chroma_wrapper.client.create_collection(collection_name)
except Exception:
    # Collection already exists
    product_collection = chroma_wrapper.client.get_collection(collection_name)

# ==============================================================================
# 2. Initialize Glia Caching System
# ==============================================================================
print("Initializing Glia...")

# IMPORTANT: all-MiniLM-L6-v2 outputs 384 dimensions. We MUST tell Glia this, 
# otherwise RediSearch will throw an error expecting the default 768.
manager = GliaManager(
    vectorizer=embedder,
    redis_url="redis://localhost:6379",
    index_name="ecommerce_cache",
    vector_dims=384 
)

invalidator = CacheInvalidator(cache_manager=manager)

# Configure the adapter to poll our Chroma wrapper every 5 seconds
adapter = VectorDBAdapter(
    client=chroma_wrapper,
    collection=collection_name,
    timestamp_field="updated_at",
    mode="polling",
    source_id_field="id", # This maps to Chroma's document ID
    poll_interval=5.0
)

# Start the background polling thread
watcher = CacheWatcher(invalidator=invalidator, adapters=[adapter])
watcher.start()

# ==============================================================================
# 3. The RAG Chatbot Function
# ==============================================================================
def ask_chatbot(question: str, product_id: str) -> str:
    print(f"\n[USER]: {question}")
    
    # --- Step A: Check Glia Cache ---
    cached_answer: Optional[str] = manager.check(prompt=question)
    if cached_answer:
        print("🟢 [SYSTEM]: CACHE HIT! Returning fast, cheap answer.")
        return cached_answer
        
    print("🔴 [SYSTEM]: CACHE MISS! Querying ChromaDB and simulating LLM...")
    
    # --- Step B: Simulate RAG Retrieval ---
    db_result = product_collection.get(ids=[product_id])
    if not db_result['ids']:
        return "Product not found."
    
    price = db_result['metadatas'][0]['price']
    
    # --- Step C: Simulate LLM Generation ---
    generated_answer = f"According to our live catalog, it costs ${price:.2f}."
    print(f"🤖 [LLM]: Generating answer: {generated_answer}")
    
    # --- Step D: Store in Glia Cache ---
    manager.store(prompt=question, response=generated_answer, source_id=product_id)
    return generated_answer

# ==============================================================================
# 4. The Test Execution Flow
# ==============================================================================
if __name__ == "__main__":
    try:
        print("\n--- PHASE 1: SEEDING CHROMA DB ---")
        product_collection.upsert(
            ids=["prod_5932"],
            documents=["The SuperWidget is our flagship product."],
            metadatas=[{"price": 10.00, "updated_at": int(time.time())}]
        )
        
        # Query 1: Cache is empty. Should be a MISS, generate $10, and store.
        ask_chatbot("How much is the SuperWidget?", "prod_5932")
        
        # Query 2: Cache is primed. Should be a HIT, returning $10 immediately.
        ask_chatbot("How much is the SuperWidget?", "prod_5932")
        
        print("\n--- PHASE 2: BACKGROUND MUTATION (PRICE CHANGE) ---")
        new_timestamp = int(time.time()) + 10
        product_collection.update(
            ids=["prod_5932"],
            metadatas=[{"price": 15.00, "updated_at": new_timestamp}]
        )
        print("Database Updated: SuperWidget price is now $15.00.")
        
        print("\nWaiting for Glia CacheWatcher to poll (approx. 6 seconds)...")
        # Sleep slightly longer than poll_interval (5.0s) so the background thread 
        # has time to fetch the update and invalidate the Redis cache.
        time.sleep(6) 
        
        print("\n--- PHASE 3: VERIFICATION ---")
        # Query 3: Glia should have deleted the stale $10 answer. 
        # This should be a MISS, and generate the new $15 answer!
        ask_chatbot("How much is the SuperWidget?", "prod_5932")
        
    finally:
        # Always clean up the background threads
        print("\nStopping Glia Watcher...")
        watcher.stop()
        print("Test Complete.")