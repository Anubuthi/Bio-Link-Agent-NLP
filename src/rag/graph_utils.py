# src/rag/graph_utils.py

from typing import List, Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt

# You already have your NER & outcome classifiers elsewhere
# Assume each paper/trial is already processed into:
# {
#   "drugs": [...],
#   "targets": [...],
#   "diseases": [...],
#   "pathways": [...],
#   "outcomes": ["Positive" / "Negative" / "Neutral"],
#   "side_effects": [...],
#   "biomarkers": [...],
#   "subgroups": [...],  # e.g. ["Age ≥ 65", "Female", "Location: US"]
# }

def add_node(G: nx.Graph, name: str, ntype: str):
    """Add node with type + human-readable label if not exists."""
    if name not in G:
        label = f"{ntype.capitalize()}: {name}"
        G.add_node(name, type=ntype.upper(), label=label)


def build_clinical_kg(
    processed_papers: List[Dict],
    processed_trials: List[Dict],
) -> nx.Graph:
    """
    Build a knowledge graph from structured info extracted from
    papers and trials.
    """
    G = nx.Graph()

    # ----- From papers -----
    for p in processed_papers:
        drugs = p.get("drugs", [])
        targets = p.get("targets", [])
        diseases = p.get("diseases", [])
        pathways = p.get("pathways", [])
        outcomes = p.get("outcomes", [])  # e.g. ["Positive"] or ["Negative"]
        side_effects = p.get("side_effects", [])
        biomarkers = p.get("biomarkers", [])

        for d in drugs:
            add_node(G, d, "DRUG")
        for t in targets:
            add_node(G, t, "TARGET")
        for dz in diseases:
            add_node(G, dz, "DISEASE")
        for pw in pathways:
            add_node(G, pw, "PATHWAY")
        for se in side_effects:
            add_node(G, se, "SIDE_EFFECT")
        for bm in biomarkers:
            add_node(G, bm, "BIOMARKER")
        for oc in outcomes:
            add_node(G, oc, "OUTCOME")  # e.g. "Positive", "Negative"

        # Relationships
        for d in drugs:
            for t in targets:
                G.add_edge(d, t, relation="TARGETS")
            for dz in diseases:
                G.add_edge(d, dz, relation="TREATS_OR_TESTED_FOR")
            for pw in pathways:
                G.add_edge(d, pw, relation="MODULATES_PATHWAY")
            for oc in outcomes:
                G.add_edge(d, oc, relation="HAS_OUTCOME")
            for se in side_effects:
                G.add_edge(d, se, relation="CAUSES_SIDE_EFFECT")
            for bm in biomarkers:
                G.add_edge(d, bm, relation="ASSOCIATED_BIOMARKER")

        for t in targets:
            for pw in pathways:
                G.add_edge(t, pw, relation="PART_OF_PATHWAY")
            for dz in diseases:
                G.add_edge(t, dz, relation="ASSOCIATED_WITH_DISEASE")

        for pw in pathways:
            for dz in diseases:
                G.add_edge(pw, dz, relation="ASSOCIATED_WITH_DISEASE")

    # ----- From trials -----
    for tr in processed_trials:
        trial_id = tr.get("id") or tr.get("nct_id")
        if not trial_id:
            continue

        trial_name = f"NCT{trial_id}" if not str(trial_id).startswith("NCT") else trial_id
        add_node(G, trial_name, "TRIAL")

        drugs = tr.get("drugs", [])
        diseases = tr.get("diseases", []) or [tr.get("condition", "")]
        biomarkers = tr.get("biomarkers", [])
        side_effects = tr.get("side_effects", [])
        subgroups = tr.get("subgroups", [])

        for d in drugs:
            add_node(G, d, "DRUG")
            G.add_edge(trial_name, d, relation="TESTS_DRUG")

        for dz in diseases:
            if dz:
                add_node(G, dz, "DISEASE")
                G.add_edge(trial_name, dz, relation="FOR_DISEASE")

        for bm in biomarkers:
            add_node(G, bm, "BIOMARKER")
            G.add_edge(trial_name, bm, relation="USES_BIOMARKER")

        for se in side_effects:
            add_node(G, se, "SIDE_EFFECT")
            G.add_edge(trial_name, se, relation="REPORTS_SIDE_EFFECT")

        for sg in subgroups:
            add_node(G, sg, "SUBGROUP")
            G.add_edge(trial_name, sg, relation="TARGETS_SUBGROUP")

    return G


def graph_to_json(G: nx.Graph) -> Dict:
    """Convert to nodes/edges lists for front-end or MCP."""
    nodes = []
    edges = []

    for n, data in G.nodes(data=True):
        nodes.append({
            "id": n,
            "type": data.get("type", "UNKNOWN"),
            "label": data.get("label", n),
        })

    for u, v, data in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "relation": data.get("relation", "RELATED"),
        })

    return {"nodes": nodes, "edges": edges}


def plot_graph(
    G: nx.Graph,
    out_path: str = "clinical_kg.png",
    max_nodes: int = 40,
):
    """Nice human-readable plot for demo/slide."""

    if len(G.nodes) > max_nodes:
        sub_nodes = list(G.nodes)[:max_nodes]
        G = G.subgraph(sub_nodes).copy()

    pos = nx.spring_layout(G, k=0.6, seed=42)

    # Color by node type
    color_map = {
        "DRUG": "tab:red",
        "TARGET": "tab:green",
        "DISEASE": "tab:blue",
        "PATHWAY": "tab:orange",
        "OUTCOME": "tab:purple",
        "SIDE_EFFECT": "tab:pink",
        "TRIAL": "tab:gray",
        "BIOMARKER": "tab:brown",
        "SUBGROUP": "tab:olive",
    }

    node_colors = []
    labels = {}
    for n, data in G.nodes(data=True):
        ntype = data.get("type", "UNKNOWN")
        node_colors.append(color_map.get(ntype, "tab:gray"))
        labels[n] = data.get("label", n)

    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos,
        labels=labels,
        node_color=node_colors,
        node_size=600,
        font_size=7,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
