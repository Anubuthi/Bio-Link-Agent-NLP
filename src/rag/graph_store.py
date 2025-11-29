import os
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
load_dotenv()

class KnowledgeGraphEngine:
    def __init__(self):
        # 1. Connect to Neo4j
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        print("🔌 Connecting to Neo4j...")
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("✅ Neo4j Connected.")
        except Exception as e:
            print(f"❌ Neo4j Failed: {e}")

        # 2. Load NLP Models (The "A+ Grade" Features)
        # We use a lighter model for speed, but you can swap for 'd4data/biomedical-ner-all'
        print("🧠 Loading Biomedical NER Model...")
        self.ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
        
        print("🧠 Loading PICO Classifier (SafeTensors)...")
        self.pico_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def close(self):
        if self.driver:
            self.driver.close()

    def analyze_text(self, text):
        """
        Runs NLP pipeline:
        1. Classifies text into PICO (Population, Intervention, Outcome).
        2. Extracts Entities (Chemicals, Diseases).
        """
        if not text or len(text) < 10:
            return {"pico": "Other", "entities": []}

        # Step A: PICO Classification (Is this an Outcome sentence?)
        # We classify the whole abstract for simplicity, or you could split by sentence.
        pico_labels = ["Outcome", "Intervention", "Population", "Background"]
        pico_result = self.pico_pipeline(text[:512], candidate_labels=pico_labels)
        top_pico = pico_result['labels'][0] # Top predicted label

        # Step B: Biomedical NER (Find Drugs/Diseases)
        # Truncate to 512 tokens for BERT stability
        ner_results = self.ner_pipeline(text[:512])
        
        entities = []
        for entity in ner_results:
            # Map BERT labels to our Graph labels
            # B-Chemical -> Chemical, B-Disease -> Disease
            label_map = {
                "Chemical": "Chemical",
                "Disease_disorder": "Disease",
                "Medication": "Chemical",
                "Diagnostic_procedure": "Intervention"
            }
            # Only keep high confidence entities
            if entity['score'] > 0.60:
                clean_type = label_map.get(entity['entity_group'], "Concept")
                entities.append({"name": entity['word'], "type": clean_type})

        return {"pico": top_pico, "entities": entities}

    def build_graph(self, papers, trials):
        """
        Ingest data into Neo4j with Semantic Relationships.
        """
        with self.driver.session() as session:
            # 1. Clear Graph (Optional)
            session.run("MATCH (n) DETACH DELETE n")
            
            # 2. Process Papers (Retrospective Evidence)
            print(f"📚 Analyzing {len(papers)} Papers...")
            for p in papers:
                # Run NLP
                nlp_data = self.analyze_text(p['abstract'])
                pico_context = nlp_data['pico'] # e.g., "Outcome"
                
                # Create Paper Node
                session.run("""
                    MERGE (p:Paper {id: $id})
                    SET p.title = $title, p.abstract = $abstract, p.pico = $pico
                """, id=p['id'], title=p['title'], abstract=p['abstract'], pico=pico_context)
                
                # Link Entities based on PICO Context
                for ent in nlp_data['entities']:
                    # Cypher logic: Create specific node types (Chemical, Disease)
                    # And link them. If PICO is 'Outcome', the relationship is stronger.
                    rel_type = "HAS_OUTCOME" if pico_context == "Outcome" else "MENTIONS"
                    
                    query = f"""
                        MATCH (p:Paper {{id: $id}})
                        MERGE (e:{ent['type']} {{name: $name}})
                        MERGE (p)-[:{rel_type}]->(e)
                    """
                    session.run(query, id=p['id'], name=ent['name'])

            # 3. Process Trials (Prospective Action)
            print(f"🏥 Analyzing {len(trials)} Trials...")
            for t in trials:
                # We assume trials are "Interventions" by default
                session.run("""
                    MERGE (t:Trial {id: $id})
                    SET t.title = $title, t.criteria = $criteria
                """, id=t['nct_id'], title=t['title'], criteria=t['criteria'])
                
                # Use simple extraction for trials (faster) or full NER
                # Let's use the same NER pipeline for consistency
                nlp_data = self.analyze_text(t['title'] + " " + t['criteria'])
                
                for ent in nlp_data['entities']:
                    # Logic: Trials "RECRUIT" for Diseases and "TEST" Chemicals
                    rel_type = "RECRUITS_FOR" if ent['type'] == "Disease" else "TESTS"
                    
                    query = f"""
                        MATCH (t:Trial {{id: $id}})
                        MERGE (e:{ent['type']} {{name: $name}})
                        MERGE (t)-[:{rel_type}]->(e)
                    """
                    session.run(query, id=t['nct_id'], name=ent['name'])
                    
        print("✅ Knowledge Graph Built Successfully.")

    def get_visualization_data(self):
        """
        Fetch graph for Streamlit with strict deduplication to prevent UI errors.
        """
        nodes = []
        edges = []
        seen_nodes = set() # Track IDs to prevent duplicates
        
        with self.driver.session() as session:
            # 1. Fetch Nodes
            # We explicitly return labels to color code correctly
            result = session.run("MATCH (n) RETURN n.id, n.title, n.name, labels(n) as types")
            for record in result:
                node_types = record['types']
                # Determine Label (Title for Papers, Name for Concepts)
                label = record['n.title'] or record['n.name'] or "Unknown"
                
                # Determine ID (Robust check)
                nid = record['n.id'] if record['n.id'] else record['n.name']
                
                # --- DEDUPLICATION FIX ---
                if not nid: continue # Skip broken nodes
                nid = str(nid) # Ensure string format
                
                if nid in seen_nodes:
                    continue # Skip if we already added this ID
                seen_nodes.add(nid)
                # -------------------------

                # Dynamic Coloring
                if "Paper" in node_types: color = "#ff4b4b"     # Red
                elif "Trial" in node_types: color = "#00c853"   # Green
                elif "Chemical" in node_types: color = "#29b6f6" # Blue
                elif "Disease" in node_types: color = "#ffa726"  # Orange
                else: color = "#eeeeee"
                
                nodes.append({
                    "id": nid, 
                    "label": label[:20], 
                    "color": color, 
                    "size": 20
                })

            # 2. Fetch Edges
            # We verify that source/target actually exist in our seen_nodes to prevent dangling edges
            result = session.run("MATCH (a)-[r]->(b) RETURN a.id, a.name, b.id, b.name, type(r) as rel")
            for record in result:
                source = str(record['a.id'] or record['a.name'])
                target = str(record['b.id'] or record['b.name'])
                
                # Only add edge if both nodes are valid
                if source in seen_nodes and target in seen_nodes:
                    edges.append({
                        "source": source, 
                        "target": target, 
                        "label": record['rel']
                    })
                
        return nodes, edges