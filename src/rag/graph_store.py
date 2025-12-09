'''import os
import re
import sys
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

class KnowledgeGraphEngine:
    def __init__(self):
        # 1. Connect to Neo4j
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        try:
            if not uri or not user or not password:
                raise ValueError("Missing NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD")
        except ValueError as ve:
            print(f" Configuration Error: {ve}")
            sys.exit(1) 
        
        print(" Connecting to Neo4j...")
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("Neo4j Connected.")
        except Exception as e:
            print(f" Neo4j Failed: {e}")

        # 2. Load NLP Models (The "A+ Grade" Features)
        # We use a lighter model for speed, but you can swap for 'd4data/biomedical-ner-all'
        print(" Loading Biomedical NER Model...")
        self.ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
        
        print(" Loading PICO Classifier (SafeTensors)...")
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
            print(f" Analyzing {len(papers)} Papers...")
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
            print(f" Analyzing {len(trials)} Trials...")
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
                    
        print(" Knowledge Graph Built Successfully.")

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
        '''

import os
import re
import sys
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
from transformers import pipeline

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

class KnowledgeGraphEngine:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not user or not password:
            print("❌ Error: Missing Neo4j credentials")
            sys.exit(1)
            
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("✅ Neo4j Connected.")
        except Exception as e:
            print(f"❌ Neo4j Failed: {e}")

        print("🧠 Loading NLP Models...")
        self.ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
        self.pico_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def close(self):
        if self.driver: self.driver.close()

    def analyze_text(self, text):
        if not text or len(text) < 10:
            return {"pico": "Background", "entities": []}

        short_text = text[:512] 

        # PICO
        try:
            res = self.pico_pipeline(short_text, candidate_labels=["Outcome", "Intervention", "Population", "Background"])
            pico = res['labels'][0]
        except:
            pico = "Unknown"

        # NER
        entities = []
        try:
            ner_res = self.ner_pipeline(short_text)
            
            banned_words = {
                "study", "group", "data", "analysis", "results", "using", "between", 
                "associated", "clinical", "patient", "year", "years", "time", "high",
                "during", "after", "before", "treatment", "control", "placebo"
            }
            
            for ent in ner_res:
                if ent['score'] < 0.75: continue
                word = ent['word'].strip()
                if word.startswith("##"): continue
                if len(word) < 3: continue
                if word.lower() in banned_words: continue
                if not re.match(r'^[a-zA-Z0-9\s\-\.]+$', word): continue

                grp = ent['entity_group']
                etype = "Concept"
                if "Chemical" in grp or "Medication" in grp: etype = "Chemical"
                elif "Disease" in grp or "Diagnosis" in grp: etype = "Disease"
                elif "Gene" in grp or "Protein" in grp: etype = "Target"
                
                if not any(e['name'].lower() == word.lower() for e in entities):
                    entities.append({"name": word, "type": etype})
        except: pass

        return {"pico": pico, "entities": entities}

    def build_graph(self, papers, trials):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n") 
            
            print(f"📚 Processing {len(papers)} Papers...")
            for p in papers:
                nlp = self.analyze_text(p.get('abstract', ''))
                
                session.run("""
                    MERGE (p:Paper {id: $id})
                    SET p.title = $title, p.date = $date, p.journal = $journal, p.pico = $pico, p.abstract = $abstract
                """, id=p['id'], title=p['title'], date=p.get('date'), journal=p.get('journal'), pico=nlp['pico'], abstract=p.get('abstract'))
                
                for ent in nlp['entities']:
                    session.run(f"""
                        MATCH (p:Paper {{id: $id}})
                        MERGE (e:{ent['type']} {{name: $name}})
                        MERGE (p)-[r:MENTIONS]->(e)
                        SET r.context = $pico
                    """, id=p['id'], name=ent['name'], pico=nlp['pico'])

            print(f"🏥 Processing {len(trials)} Trials...")
            for t in trials:
                session.run("""
                    MERGE (t:Trial {id: $id})
                    SET t.title = $title, t.phase = $phase, t.status = $status, t.criteria = $criteria
                """, id=t['nct_id'], title=t['title'], phase=t.get('phase'), status=t.get('status'), criteria=t.get('criteria'))
                
                nlp = self.analyze_text(f"{t['title']} {t.get('criteria', '')}")
                
                for ent in nlp['entities']:
                    rel = "INVESTIGATES" if ent['type'] == "Chemical" else "RECRUITS_FOR"
                    session.run(f"""
                        MATCH (t:Trial {{id: $id}})
                        MERGE (e:{ent['type']} {{name: $name}})
                        MERGE (t)-[:{rel}]->(e)
                    """, id=t['nct_id'], name=ent['name'])
        
        print("✅ Graph Built.")

    def get_visualization_data(self):
        """
        Retrieves graph data formatted for visual clarity.
        - Adds 'title' (tooltip) containing full text.
        - Adds 'label' to edges.
        """
        nodes = []
        edges = []
        seen = set()
        
        with self.driver.session() as session:
            # 1. Fetch Nodes + Properties for Tooltips
            # We fetch everything we need to construct a helpful hover text
            result = session.run("""
                MATCH (n) 
                RETURN n.id, n.title, n.name, labels(n) as types, 
                       n.abstract, n.criteria, n.date, n.journal, n.phase, n.status
            """)
            
            for r in result:
                lbls = r['types']
                nid = r['n.id'] or r['n.name']
                label_text = r['n.title'] or r['n.name'] or "Node"
                
                if not nid or nid in seen: continue
                seen.add(nid)
                
                # --- Construct Rich Tooltip ---
                tooltip = ""
                if "Paper" in lbls:
                    color = "#ef5350" # Red
                    # Tooltip shows Journal, Date, and Abstract
                    tooltip = f"📄 PAPER: {label_text}\n\n📅 {r['n.date']} | 📖 {r['n.journal']}\n\n📝 ABSTRACT:\n{r['n.abstract'][:400]}..."
                elif "Trial" in lbls:
                    color = "#66bb6a" # Green
                    # Tooltip shows Status, Phase, and Criteria
                    tooltip = f"🏥 TRIAL: {label_text}\n\n🚦 {r['n.status']} | 📊 {r['n.phase']}\n\n📋 CRITERIA:\n{r['n.criteria'][:400]}..."
                elif "Chemical" in lbls:
                    color = "#42a5f5" # Blue
                    tooltip = f"💊 DRUG: {label_text}"
                elif "Disease" in lbls:
                    color = "#ffa726" # Orange
                    tooltip = f"🦠 DISEASE: {label_text}"
                else:
                    color = "#bdbdbd"
                    tooltip = f"Concept: {label_text}"
                
                nodes.append({
                    "id": nid, 
                    "label": label_text[:15] + "...", # Keep visual label short
                    "title": tooltip,                 # Hover text has full detail
                    "color": color, 
                    "size": 25 if "Concept" not in str(lbls) else 15
                })

            # 2. Fetch Edges with Labels
            result = session.run("MATCH (a)-[r]->(b) RETURN a.id, a.name, b.id, b.name, type(r) as rel")
            for r in result:
                src = r['a.id'] or r['a.name']
                tgt = r['b.id'] or r['b.name']
                if src in seen and tgt in seen:
                    edges.append({
                        "source": src, 
                        "target": tgt, 
                        "label": r['rel'] # "INVESTIGATES", "MENTIONS", etc.
                    })
                    
        return nodes, edges
    def query_graph(self, query: str) -> str:
        """
        Retrieves related entities and relationships from the graph 
        based on keywords in the user query.
        """
        # 1. Extract potential entities from the query using your existing NLP pipeline
        nlp_data = self.analyze_text(query)
        entities = [e['name'] for e in nlp_data['entities']]
        
        # Fallback: If no entities found, use the whole query as a keyword
        if not entities:
            keywords = [query]
        else:
            keywords = entities

        context_lines = []
        with self.driver.session() as session:
            for term in keywords:
                # Cypher query: Find nodes matching the term and their immediate neighbors
                # We limit to 10 relationships per term to prevent context window overflow
                cypher = """
                MATCH (n)-[r]-(m)
                WHERE n.name =~ '(?i).*' + $term + '.*' OR n.title =~ '(?i).*' + $term + '.*'
                RETURN n.name AS source, type(r) AS rel, m.name AS target, m.title AS target_title
                LIMIT 10
                """
                result = session.run(cypher, term=term)
                
                for record in result:
                    src = record['source'] or "Unknown"
                    rel = record['rel']
                    # Use title if it's a paper/trial, otherwise name
                    tgt = record['target_title'] or record['target'] or "Unknown"
                    context_lines.append(f"- {src} {rel} {tgt}")

        if not context_lines:
            return "No specific graph knowledge found for this query."
            
        # Deduplicate and return
        return "Knowledge Graph Relationships:\n" + "\n".join(list(set(context_lines)))