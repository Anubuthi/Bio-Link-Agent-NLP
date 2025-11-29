import chromadb
from sentence_transformers import SentenceTransformer
import os
import shutil

class TrialVectorStore:
    def __init__(self):
        """
        Initialize the Vector Engine.
        Uses a Fresh Persistent Client to avoid SQLite in-memory errors on macOS/Python 3.12.
        """
        print("🧠 Loading Embedding Model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define a specific path for this app's database
        self.db_path = "./data/chroma_temp"
        
        # CRITICAL FIX: Force a fresh start every time.
        # This deletes the old database folder if it exists, ensuring 
        # ChromaDB creates new tables from scratch (fixing 'no such table' errors).
        if os.path.exists(self.db_path):
            try:
                shutil.rmtree(self.db_path)
                print("🧹 Cleared old database cache.")
            except Exception as e:
                print(f"⚠️ Warning: Could not clear cache: {e}")

        # Initialize standard PersistentClient (More stable than Ephemeral on Mac)
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Create/Get collection
        self.collection = self.client.get_or_create_collection(name="trials_cache")

    def index_trials(self, trials_list):
        """
        Takes the raw list of trials from the API and pushes them into the Vector DB.
        """
        if not trials_list:
            print("⚠️ No trials to index.")
            return 0
            
        # --- ROBUST DATA PREPARATION START ---
        ids = []
        documents = []
        metadatas = []

        for t in trials_list:
            # 1. Safely get ID (Support both 'nct_id' and 'id' keys)
            trial_id = t.get('nct_id') or t.get('id')
            if not trial_id:
                continue 
            
            ids.append(str(trial_id))
            
            # 2. Prepare Text for Embedding (Title + Criteria)
            title = t.get('title', 'No Title')
            criteria = t.get('criteria', '')
            documents.append(f"{title} \n {criteria}")
            
            # 3. Metadata for display
            metadatas.append({"title": title})
        # --- ROBUST DATA PREPARATION END ---
        
        if not ids:
            return 0

        print(f"🔄 Vectorizing {len(ids)} trials...")
        embeddings = self.model.encode(documents).tolist()
        
        # Store in DB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print("✅ Indexing Complete.")
        return len(ids)

    def search(self, patient_query, n_results=3):
        """
        Performs Cosine Similarity Search.
        """
        query_vec = self.model.encode([patient_query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=n_results
        )
        
        matches = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                matches.append({
                    "id": results['ids'][0][i],
                    "title": results['metadatas'][0][i]['title'],
                    "score": 1 - results['distances'][0][i], 
                    "snippet": results['documents'][0][i][:300] + "..."
                })
                
        return matches