import json
from typing import Any, Dict

from ollama import chat

from .router_models import ToolSelection
from .router_prompt import ROUTER_SYSTEM_PROMPT
from .answer_prompt import ANSWER_SYSTEM_PROMPT

from src.server import (
    search_medical_data,
    search_pubmed,
    search_trials,
    match_trials_semantic,
    build_knowledge_graph,
    search_knowledge_graph,
)


def _call_ollama_router(
    user_query: str,
    model_name: str = "qwen2.5:3b",
) -> Dict[str, Any]:
    """
    Low-level helper that calls the small LLM via Ollama and
    returns the raw JSON object produced by the model.
    """
    response = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        # Ask Ollama to format output according to our Pydantic schema
        format=ToolSelection.model_json_schema(),
        options={
            "temperature": 0.0,
        },
    )

    content = response["message"]["content"]

    # content is usually a JSON string; handle both str and dict just in case
    if isinstance(content, str):
        data = json.loads(content)
    else:
        data = content

    return data


def route_query(
    user_query: str,
    model_name: str = "qwen2.5:3b",
) -> ToolSelection:
    """
    High-level function:
    - Sends the user query to the router LLM.
    - Parses the JSON into a ToolSelection object.
    """
    data = _call_ollama_router(user_query, model_name=model_name)
    return ToolSelection(**data)


def execute_tool(
    selection: ToolSelection,
    original_query: str | None = None,
) -> str:
    """
    Execute the chosen tool with the given parameters.

    Adds defensive fallbacks: if the router forgot to include a required
    parameter (like 'query' or 'condition'), we fall back to the original
    user query when available.
    """
    tool_map = {
        "search_medical_data": search_medical_data,
        "search_pubmed": search_pubmed,
        "search_trials": search_trials,
        "match_trials_semantic": match_trials_semantic,
        "build_knowledge_graph": build_knowledge_graph,
        "search_knowledge_graph": search_knowledge_graph,
    }

    if selection.tool_name not in tool_map:
        raise ValueError(f"Unknown tool_name: {selection.tool_name}")

    tool_fn = tool_map[selection.tool_name]

    # Start with the router-provided parameters
    params: Dict[str, Any] = dict(selection.parameters or {})

    # ---------- Defensive fallbacks by tool ----------
    if selection.tool_name == "search_medical_data":
        # Ensure 'query' is present
        if "query" not in params:
            if original_query is None:
                raise ValueError("Router did not provide 'query' for search_medical_data.")
            params["query"] = original_query

    elif selection.tool_name == "search_pubmed":
        if "query" not in params:
            if original_query is None:
                raise ValueError("Router did not provide 'query' for search_pubmed.")
            params["query"] = original_query
        # Default max_results if missing
        params.setdefault("max_results", 5)

    elif selection.tool_name == "search_trials":
        if "condition" not in params:
            if original_query is None:
                raise ValueError("Router did not provide 'condition' for search_trials.")
            # Fallback: treat entire query as condition string
            params["condition"] = original_query
        params.setdefault("limit", 5)

    elif selection.tool_name == "match_trials_semantic":
        # condition required
        if "condition" not in params:
            if original_query is None:
                raise ValueError("Router did not provide 'condition' for match_trials_semantic.")
            params["condition"] = original_query

        # patient_note required
        if "patient_note" not in params:
            if original_query is None:
                raise ValueError("Router did not provide 'patient_note' for match_trials_semantic.")
            params["patient_note"] = original_query

        params.setdefault("limit", 5)

    elif selection.tool_name == "build_knowledge_graph":
        if "topic" not in params:
            if original_query is None:
                raise ValueError("Router did not provide 'topic' for build_knowledge_graph.")
            params["topic"] = original_query
        params.setdefault("max_papers", 10)
        params.setdefault("max_trials", 10)

    print(f"\n🚀 [DEBUG] ROUTER SELECTED: {selection.tool_name}")
    print(f"📦 [DEBUG] PARAMS: {params}\n")

    # Finally, call the underlying tool function
    return tool_fn(**params)


def route_and_execute(
    user_query: str,
    model_name: str = "qwen2.5:3b",
) -> Dict[str, Any]:
    """
    Convenience helper that:
    - Routes the query
    - Executes the chosen tool
    - Returns both the selection metadata and the raw tool result
    """
    selection = route_query(user_query, model_name=model_name)
    result = execute_tool(selection, original_query=user_query)

    return {
        "selection": selection.model_dump(),
        "result": result,
    }

def generate_final_answer(
    user_query: str,
    selection: ToolSelection,
    tool_result: str,
    model_name: str = "qwen2.5:3b",  # or "llama3:8b" or even same as router if you want
) -> str:
    """
    Use a (possibly larger) LLM to turn the raw tool_result into a
    coherent, user-facing answer.

    - user_query: original natural language question
    - selection: ToolSelection (tool_name + params + reasoning)
    - tool_result: string returned by the tool (JSON or text)
    """
    # We'll keep the tool_result as-is; if it's huge later we can truncate it.
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "User question:\n"
                f"{user_query}\n\n"
                "Chosen tool:\n"
                f"{selection.tool_name}\n\n"
                "Tool parameters:\n"
                f"{json.dumps(selection.parameters, indent=2)}\n\n"
                "Raw tool result (JSON or text):\n"
                f"{tool_result}\n\n"
                "Please write a clear, coherent answer for the user based ONLY on this data."
            ),
        },
    ]

    response = chat(
        model=model_name,
        messages=messages,
        options={"temperature": 0.2},
    )

    return response["message"]["content"]


if __name__ == "__main__":
    # Simple CLI loop for local testing
    print(" Bio-Link Router (using Ollama)")
    print("Model: qwen2.5:3b")
    print("Type Ctrl+C to exit.\n")

    try:
        while True:
            user_q = input("User: ").strip()
            if not user_q:
                continue

            try:
                selection = route_query(user_q)
                print("\n[Router Decision]")
                print(f"  Tool:       {selection.tool_name}")
                print(f"  Parameters: {selection.parameters}")
                print(f"  Confidence: {selection.confidence_score:.2f}")
                print(f"  Reasoning:  {selection.reasoning}\n")

                result = execute_tool(selection, original_query=user_q)
                print("----- RAW TOOL RESULT (debug) -----")
                print(result)
                print("-----------------------------------\n")

                final_answer = generate_final_answer(
                    user_query=user_q,
                    selection=selection,
                    tool_result=result,
                    model_name="qwen2.5:3b",  # or any other Ollama model you like
                )

                print("===== FINAL ANSWER =====")
                print(final_answer)
                print("========================\n")


            except Exception as e:
                print(f"[ERROR] {e}\n")

    except KeyboardInterrupt:
        print("\nExiting router CLI.")
