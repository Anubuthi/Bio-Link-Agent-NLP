import json
import sys
from typing import TypedDict, Any, Dict

import requests
from ollama import chat
from langgraph.graph import StateGraph, END

# Assuming these are Pydantic models for routing
from router_models import ToolSelection
from router_prompt import ROUTER_SYSTEM_PROMPT
from answer_prompt import ANSWER_SYSTEM_PROMPT

# --- Configuration ---
MCP_SERVER_URL = "http://127.0.0.1:8005"
MCP_TOOL_ENDPOINT = f"{MCP_SERVER_URL}/tool/execute"
ROUTER_MODEL = "qwen2.5:3b"
ANSWER_MODEL = "qwen2.5:3b"
# ---------------------


# --- State Definition ---
class GraphState(TypedDict):
    """State object that gets passed between nodes in the graph"""
    user_query: str
    tool_name: str | None
    tool_parameters: Dict[str, Any] | None
    confidence_score: float | None
    reasoning: str | None
    tool_result: str | None
    final_answer: str | None
    error: str | None


# --- Node 1: Router Node ---
def router_node(state: GraphState) -> GraphState:
    """
    Routes the user query using a small LLM to determine which tool to call.
    Updates state with tool selection details.
    """
    user_query = state["user_query"]
    
    try:
        # Call Ollama with router prompt
        response = chat(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            format=ToolSelection.model_json_schema(),
            options={"temperature": 0.0},
        )
        
        content = response["message"]["content"]
        
        # Parse JSON response
        if isinstance(content, str):
            data = json.loads(content)
        else:
            data = content
        
        # Parse into ToolSelection object
        selection = ToolSelection(**data)
        
        # Update state
        return {
            **state,
            "tool_name": selection.tool_name,
            "tool_parameters": selection.parameters or {},
            "confidence_score": selection.confidence_score,
            "reasoning": selection.reasoning,
        }
        
    except Exception as e:
        return {
            **state,
            "error": f"Router node error: {str(e)}",
            "final_answer": f"Error in routing: {str(e)}"
        }


# --- Node 2: Tool Execution Node ---
def tool_execution_node(state: GraphState) -> GraphState:
    """
    Executes the selected tool by calling the MCP server via HTTP.
    Applies defensive parameter fallbacks before calling the server.
    """
    # Check if there was an error in previous node
    if state.get("error"):
        return state
    
    tool_name = state["tool_name"]
    params = dict(state["tool_parameters"] or {})
    original_query = state["user_query"]
    
    try:
        # --- Defensive fallbacks by tool ---
        if tool_name in ["search_medical_data", "search_pubmed"]:
            if "query" not in params:
                params["query"] = original_query
            if tool_name == "search_pubmed":
                params.setdefault("max_results", 5)
        
        elif tool_name == "search_trials":
            if "condition" not in params:
                params["condition"] = original_query
            params.setdefault("limit", 5)
        
        elif tool_name == "match_trials_semantic":
            if "condition" not in params:
                params["condition"] = original_query
            if "patient_note" not in params:
                params["patient_note"] = original_query
            params.setdefault("limit", 5)
        
        elif tool_name == "build_knowledge_graph":
            if "topic" not in params:
                params["topic"] = original_query
            params.setdefault("max_papers", 10)
            params.setdefault("max_trials", 10)
        
        # --- HTTP Call to MCP Server ---
        payload = {
            "tool_name": tool_name,
            "parameters": params
        }
        
        response = requests.post(
            MCP_TOOL_ENDPOINT,
            json=payload,
            timeout=300  # 5 minutes for long operations
        )
        response.raise_for_status()
        
        # Parse response
        response_data = response.json()
        tool_result = response_data.get("result", f"Error: No 'result' key in server response: {response_data}")
        
        return {
            **state,
            "tool_result": tool_result,
        }
        
    except requests.exceptions.ConnectionError:
        error_msg = (
            f"Could not connect to MCP server at {MCP_SERVER_URL}. "
            "Please ensure the MCP server is running."
        )
        return {
            **state,
            "error": error_msg,
            "final_answer": error_msg
        }
    
    except requests.exceptions.HTTPError as err:
        error_msg = f"HTTP error executing tool: {err}\nServer Response: {response.text}"
        return {
            **state,
            "error": error_msg,
            "final_answer": error_msg
        }
    
    except Exception as e:
        error_msg = f"Unexpected error during tool execution: {str(e)}"
        return {
            **state,
            "error": error_msg,
            "final_answer": error_msg
        }


# --- Node 3: Answer Generation Node ---
def answer_generation_node(state: GraphState) -> GraphState:
    """
    Generates a user-friendly answer from the raw tool result using an LLM.
    """
    # Check if there was an error in previous nodes
    if state.get("error"):
        return state
    
    user_query = state["user_query"]
    tool_name = state["tool_name"]
    tool_parameters = state["tool_parameters"]
    tool_result = state["tool_result"]
    
    try:
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "User question:\n"
                    f"{user_query}\n\n"
                    "Chosen tool:\n"
                    f"{tool_name}\n\n"
                    "Tool parameters:\n"
                    f"{json.dumps(tool_parameters, indent=2)}\n\n"
                    "Raw tool result (JSON or text):\n"
                    f"{tool_result}\n\n"
                    "Please write a clear, coherent answer for the user based ONLY on this data."
                ),
            },
        ]
        
        response = chat(
            model=ANSWER_MODEL,
            messages=messages,
            options={"temperature": 0.2},
        )
        
        final_answer = response["message"]["content"]
        
        return {
            **state,
            "final_answer": final_answer,
        }
        
    except Exception as e:
        error_msg = f"Error generating final answer: {str(e)}"
        return {
            **state,
            "error": error_msg,
            "final_answer": error_msg
        }


# --- Graph Construction ---
def create_graph() -> StateGraph:
    """
    Creates and compiles the LangGraph workflow.
    
    Flow: START → Router → Tool Execution → Answer Generation → END
    """
    # Initialize the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("answer_generation", answer_generation_node)
    
    # Define edges (linear flow)
    workflow.set_entry_point("router")
    workflow.add_edge("router", "tool_execution")
    workflow.add_edge("tool_execution", "answer_generation")
    workflow.add_edge("answer_generation", END)
    
    # Compile the graph
    return workflow.compile()


# --- Main Execution ---
if __name__ == "__main__":
    print("="*60)
    print("🧬 Bio-Link Router (LangGraph + Ollama)")
    print("="*60)
    print(f"MCP Server URL: {MCP_SERVER_URL}")
    print(f"Router Model: {ROUTER_MODEL}")
    print(f"Answer Model: {ANSWER_MODEL}")
    print("Type 'quit' or Ctrl+C to exit.\n")
    
    try:
        # Check MCP server connectivity at startup
        try:
            requests.get(MCP_SERVER_URL, timeout=5).raise_for_status()
            print("🟢 MCP Server check: Success.\n")
        except requests.exceptions.RequestException:
            print(f"🔴 MCP Server check: FAILED. Ensure server is running at {MCP_SERVER_URL}\n")
            sys.exit(1)
        
        # Create the LangGraph workflow
        graph = create_graph()
        print("✅ LangGraph workflow compiled successfully.\n")
        print("="*60 + "\n")
        
        # CLI loop
        while True:
            user_q = input("User: ").strip()
            
            if not user_q:
                continue
            
            if user_q.lower() in ["quit", "exit", "q"]:
                print("Exiting router CLI.")
                break
            
            try:
                # Invoke the graph with initial state
                initial_state = {
                    "user_query": user_q,
                    "tool_name": None,
                    "tool_parameters": None,
                    "confidence_score": None,
                    "reasoning": None,
                    "tool_result": None,
                    "final_answer": None,
                    "error": None,
                }
                
                # Run the graph
                result = graph.invoke(initial_state)
                
                # Display results
                print("\n" + "="*60)
                print("[ROUTER DECISION]")
                print("="*60)
                print(f"Tool:       {result.get('tool_name', 'N/A')}")
                print(f"Parameters: {json.dumps(result.get('tool_parameters', {}), indent=2)}")
                print(f"Confidence: {result.get('confidence_score', 0):.2f}")
                print(f"Reasoning:  {result.get('reasoning', 'N/A')}")
                print("="*60 + "\n")
                
                # Show raw tool result (debug)
                print("----- RAW TOOL RESULT (debug) -----")
                print(result.get("tool_result", "No result"))
                print("-----------------------------------\n")
                
                # Show final answer
                print("="*60)
                print("[FINAL ANSWER]")
                print("="*60)
                print(result.get("final_answer", "No answer generated"))
                print("="*60 + "\n")
                
                # Show error if any
                if result.get("error"):
                    print(f"⚠️  Error occurred: {result['error']}\n")
                
            except Exception as e:
                print(f"\n❌ [ERROR] {e}\n")
    
    except KeyboardInterrupt:
        print("\n\nExiting router CLI.")