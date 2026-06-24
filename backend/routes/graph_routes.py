# backend/routes/graph_routes.py
"""
Knowledge Graph routes: exports NetworkX graph data (nodes, links) for frontend visualizations.
"""

import pickle
import os
import networkx as nx
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from backend.database import get_db, User, UserActivity
from backend.auth import get_current_user
from sqlalchemy.orm import Session

router = APIRouter(prefix="/graph", tags=["Knowledge Graph"])

# ═══════════════════════════════════════════════════════════════════════
# GRAPH LOADER
# ═══════════════════════════════════════════════════════════════════════

G = None

def get_graph():
    global G
    if G is not None:
        return G
    if os.path.exists('models/knowledge_graph.pkl'):
        try:
            with open('models/knowledge_graph.pkl', 'rb') as f:
                G = pickle.load(f)
            print("Loaded knowledge graph from pickle.")
            return G
        except Exception as e:
            print("Error loading graph:", e)
            
    # Inline fallback if pickle missing
    print("Pickle missing. Creating empty network fallback.")
    G = nx.Graph()
    return G

# ═══════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════════════

class NodeSchema(BaseModel):
    id: str
    type: str

class LinkSchema(BaseModel):
    source: str
    target: str
    relation: str

class GraphResponse(BaseModel):
    nodes: List[NodeSchema]
    links: List[LinkSchema]

# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

@router.get("/subgraph", response_model=GraphResponse)
def get_local_subgraph(
    disease: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Returns local neighborhood nodes and links around a specific disease."""
    graph = get_graph()
    if graph.number_of_nodes() == 0:
        return {"nodes": [], "links": []}
        
    # Default to a random disease if none selected
    disease_title = disease.strip().title() if disease else "Fungal Infection"
    
    # Check if disease node exists
    if disease_title not in graph:
        # Fallback search if title case mismatches
        all_diseases = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'Disease']
        if not all_diseases:
            raise HTTPException(status_code=404, detail="No diseases exist in the knowledge graph")
        disease_title = all_diseases[0]
        
    # Extract neighbors
    neighbors = list(graph.neighbors(disease_title))
    subgraph_nodes = [disease_title] + neighbors
    sub = graph.subgraph(subgraph_nodes)
    
    # Format nodes
    nodes = []
    for n in sub.nodes():
        nodes.append({
            "id": n,
            "type": graph.nodes[n].get("type", "Symptom")
        })
        
    # Format links
    links = []
    for u, v in sub.edges():
        links.append({
            "source": u,
            "target": v,
            "relation": graph.edges[u, v].get("relation", "connected")
        })
        
    # Log graph view activity
    log = UserActivity(
        user_id=current_user.id,
        activity_type="view_graph",
        details=f"Viewed subgraph for: {disease_title}"
    )
    db.add(log)
    db.commit()
    
    return {"nodes": nodes, "links": links}

@router.get("/all", response_model=GraphResponse)
def get_full_graph(current_user: User = Depends(get_current_user)):
    """Returns nodes and links for the entire knowledge graph."""
    graph = get_graph()
    
    nodes = []
    for n in graph.nodes():
        nodes.append({
            "id": n,
            "type": graph.nodes[n].get("type", "Symptom")
        })
        
    links = []
    for u, v in graph.edges():
        links.append({
            "source": u,
            "target": v,
            "relation": graph.edges[u, v].get("relation", "connected")
        })
        
    return {"nodes": nodes, "links": links}
