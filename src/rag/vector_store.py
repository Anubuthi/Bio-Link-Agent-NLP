import chromadb
from sentence_transformers import SentenceTransformer
import os
import shutil
import uuid
import time

class TrialVectorStore:
    def __init__(self):
        """
        Initialize the Vector Engine.
        Uses a UNIQUE path every run to avoid 'readonly database' locks.
        """
        print("🧠 Loading Embedding Model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 1. Garbage Collection: Try to clean up OLD sessions from previous runs
        data_root = "./data"
        if os.path.exists(data_root):
            for item in os.listdir(data_root):
                item_path = os.path.join(data_root, item)
                # Only delete folders that look like our temp sessions
                if os.path.isdir(item_path) and item.startswith("chroma_session_"):
                    try:
                        shutil.rmtree(item_path)
                        print(f"🧹 Deleted old session: {item}")
                    except:
                        # If it's locked, just skip it. It's not our problem right now.
                        pass

        # 2. Create a NEW Unique Path for THIS session
        # This guarantees 100% fresh permissions
        session_id = str(uuid.uuid4())[:8]
        self.db_path = f"./data/chroma_session_{session_id}"
        
        print(f"📂 Creating fresh Vector DB at: {self.db_path}")

        # 3. Initialize Client
        # We don't need special settings anymore because the path is unique
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        self.collection = self.client.get_or_create_collection(name="trials_cache")

    def index_trials(self, trials_list):
        if not trials_list:
            print("⚠️ No trials to index.")
            return 0
            
        ids = []
        documents = []
        metadatas = []

        for t in trials_list:
            # Robust ID handling
            trial_id = t.get('nct_id') or t.get('id')
            if not trial_id: continue 
            
            ids.append(str(trial_id))
            
            # Contextual embedding
            title = t.get('title', 'No Title')
            criteria = t.get('criteria', '')
            documents.append(f"{title} \n {criteria}")
            
            metadatas.append({"title": title})
        
        if not ids: return 0

        print(f"🔄 Vectorizing {len(ids)} trials...")
        embeddings = self.model.encode(documents).tolist()
        
        # Upsert
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print("✅ Indexing Complete.")
        return len(ids)

    def search(self, patient_query, n_results=3):
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