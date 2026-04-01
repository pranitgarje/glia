import chromadb
from typing import Any, List, Dict, Optional

class ChromaGliaWrapper:
    def __init__(self, host: str = "localhost", port: int = 8000):
        # Connect to the ChromaDB running in our Docker container
        self.client = chromadb.HttpClient(host=host, port=port)
        
    def get_collection(self, collection: str):
        return self.client.get_collection(collection)

    def fetch_updated(
        self, 
        collection: str, 
        since: Optional[Any], 
        timestamp_field: str
    ) -> List[Dict[str, Any]]:
        col = self.client.get_collection(collection)
        
        if since is None:
            results = col.get()
        else:
            results = col.get(where={timestamp_field: {"$gt": since}})
            
        records: List[Dict[str, Any]] = []
        
        if results and results.get("ids"):
            for i in range(len(results["ids"])):
                metadata = results["metadatas"][i] if results.get("metadatas") else {}
                record = {
                    "id": results["ids"][i], 
                    timestamp_field: metadata.get(timestamp_field),
                    "document": results["documents"][i] if results.get("documents") else ""
                }
                record.update(metadata)
                records.append(record)
                
        return records

# import chromadb
# from typing import Any, List, Dict, Optional

# class ChromaGliaWrapper:
#     """
#     Wraps a ChromaDB client to provide the polling interface expected by Glia.
    
#     Glia's VectorDBAdapter expects the client to expose:
#     1. A connection probe: get_collection() or describe_collection()
#     2. A polling method: fetch_updated()
#     """
    
#     def __init__(self, path: str = "./mock_chroma_db"):
#         # We use a persistent client so data survives between runs if needed,
#         # but you could also use chromadb.Client() for an in-memory ephemeral DB.
#         self.client = chromadb.PersistentClient(path=path)
        
#     def get_collection(self, collection: str):
#         """
#         Connection probe used by Glia's VectorDBAdapter.connect() 
#         to ensure the database is reachable before starting the watcher thread.
#         """
#         return self.client.get_collection(collection)

#     def fetch_updated(
#         self, 
#         collection: str, 
#         since: Optional[Any], 
#         timestamp_field: str
#     ) -> List[Dict[str, Any]]:
#         """
#         Translates Glia's polling request into a ChromaDB query.
        
#         Parameters:
#         - collection: The name of the Chroma collection.
#         - since: The last high-water-mark timestamp seen by Glia (None on first run).
#         - timestamp_field: The metadata field used for tracking updates (e.g., 'updated_at').
#         """
#         col = self.client.get_collection(collection)
        
#         # If 'since' is None, this is Glia's bootstrap run; fetch everything to establish a baseline.
#         if since is None:
#             results = col.get()
#         else:
#             # Use ChromaDB's metadata filtering to find only records newer than the cursor
#             results = col.get(
#                 where={timestamp_field: {"$gt": since}}
#             )
            
#         # ChromaDB returns a dictionary of lists: {'ids': [...], 'metadatas': [...], 'documents': [...]}
#         # Glia expects a flat list of dictionaries representing individual records. We map that here.
#         records: List[Dict[str, Any]] = []
        
#         if results and results.get("ids"):
#             for i in range(len(results["ids"])):
#                 metadata = results["metadatas"][i] if results.get("metadatas") else {}
                
#                 # Build the flat record dict
#                 record = {
#                     "id": results["ids"][i], # We will map this to Glia's source_id_field
#                     timestamp_field: metadata.get(timestamp_field),
#                     "document": results["documents"][i] if results.get("documents") else ""
#                 }
                
#                 # Merge the rest of the metadata in case we need it later
#                 record.update(metadata)
#                 records.append(record)
                
#         return records