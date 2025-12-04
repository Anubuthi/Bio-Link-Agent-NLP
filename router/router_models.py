from typing import Any, Dict, Literal
from pydantic import BaseModel, Field


class ToolSelection(BaseModel):
    """
    Schema describing the router model's decision.

    This is what we ask the small LLM (via Ollama) to output as JSON.
    """

    tool_name: Literal[
        "search_medical_data",
        "search_pubmed",
        "search_trials",
        "match_trials_semantic",
        "build_knowledge_graph",
    ] = Field(
        ...,
        description="Name of the single best tool to call for this query.",
    )

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON parameters to pass to the chosen tool.",
    )

    confidence_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="How confident you are that this is the right tool + params.",
    )

    reasoning: str = Field(
        "",
        description="One or two sentences about why this tool and parameters were chosen.",
    )
