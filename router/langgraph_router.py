"""
LangGraph Agentic Workflow with proper MCP tool calling.
Supports multi-tool execution, dynamic routing, and reflection.
"""
import json
import sys
from typing import TypedDict, Annotated, Literal
from operator import add

import requests
from ollama import chat
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- Configuration ---
MCP_SERVER_URL = "http://127.0.0.1:8005"
MCP_TOOLS_LIST_ENDPOINT = f"{MCP_SERVER_URL}/tools/list"
MCP_TOOL_CALL_ENDPOINT = f"{MCP_SERVER_URL}/tools/call"
AGENT_MODEL = "qwen2.5:3b"  # Larger model for better reasoning
# ---------------------


# --- State Definition ---
class AgentState(TypedDict):
    """
    Agentic state with message history and tool call tracking.
    """
    messages: Annotated[list[dict], add]  # Conversation history
    user_query: str  # Original query
    available_tools: list[dict]  # MCP tool schemas
    next_action: Literal["call_tool", "respond", "end"]  # Agent decision
    tool_calls: list[dict]  # Pending tool calls
    tool_results: list[dict]  # Completed tool results
    iteration: int  # Iteration counter
    max_iterations: int  # Safety limit


# --- Agent System Prompt ---
AGENT_SYSTEM_PROMPT = """You are Bio-Link Agent, an expert biomedical research assistant with access to specialized tools.

AVAILABLE TOOLS:
{tools_description}

YOUR WORKFLOW:
1. Analyze the user's query carefully
2. Decide if you need to call tool(s) or can answer directly
3. You can call MULTIPLE tools if needed (e.g., search papers AND trials)
4. After getting tool results, synthesize a comprehensive answer
5. If results are insufficient, you can call more tools

TOOL CALLING FORMAT:
When you need to call tools, respond with JSON:
{{
  "action": "call_tool",
  "tool_calls": [
    {{
      "tool": "search_pubmed",
      "arguments": {{"query": "EGFR mutations in lung cancer", "max_results": 5}}
    }},
    {{
      "tool": "search_clinical_trials",
      "arguments": {{"condition": "EGFR-positive NSCLC", "limit": 5}}
    }}
  ],
  "reasoning": "Need both research and trial data to answer comprehensively"
}}

RESPONDING FORMAT:
When you're ready to answer, respond with JSON:
{{
  "action": "respond",
  "answer": "Based on the research papers and clinical trials...",
  "confidence": 0.9
}}

ANSWER FORMATTING GUIDELINES:
When writing your "answer" field, follow these rules:
- Start with a short high-level summary (2-4 sentences)
- Then add bullet points with key findings:
  - Important drugs, targets, pathways from papers
  - Notable trials with NCT IDs, phase, intervention, population
  - Important limitations or uncertainties
- Use Markdown formatting:
  - "### Summary" for overview
  - "### Key Papers" for research findings
  - "### Notable Trials" for clinical trials
- ONLY use information that appears in the tool results - DO NOT hallucinate
- Cite sources: PMIDs for papers (e.g., PMID: 12345678), NCT IDs for trials (e.g., NCT12345678)
- If data is sparse or limited, say so clearly
- Keep it concise but informative

GUIDELINES:
- Use specific, detailed queries for tools (not vague terms)
- For patient matching, use match_patient_to_trials with full clinical details
- For research mechanisms, use search_pubmed
- For trial availability, use search_clinical_trials
- For comprehensive overviews, call BOTH search_pubmed and search_clinical_trials
- For knowledge graphs, use build_knowledge_graph when explicitly requested
- For querying EXISTING graphs, use search_knowledge_graph  
- Always explain your reasoning
- Be concise but thorough
- Cite sources when possible (PMIDs, NCT IDs)

Current conversation context:
{conversation_history}
"""


# --- Helper Functions ---
def get_mcp_tools() -> list[dict]:
    """Fetch available tools from MCP server."""
    try:
        response = requests.get(MCP_TOOLS_LIST_ENDPOINT, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("tools", [])
    except Exception as e:
        sys.stderr.write(f"⚠️  Failed to fetch MCP tools: {e}\n")
        return []


def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a single tool call on MCP server."""
    try:
        payload = {
            "name": tool_name,
            "arguments": arguments
        }
        response = requests.post(
            MCP_TOOL_CALL_ENDPOINT,
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "error": f"MCP tool call failed: {str(e)}",
            "tool": tool_name,
            "arguments": arguments
        }


def format_tools_description(tools: list[dict]) -> str:
    """Format tool schemas for prompt."""
    lines = []
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        params = tool.get("inputSchema", {}).get("properties", {})
        
        lines.append(f"\n{name}:")
        lines.append(f"  Description: {desc}")
        lines.append(f"  Parameters: {list(params.keys())}")
    
    return "\n".join(lines)


def format_conversation_history(messages: list[dict]) -> str:
    """Format message history for context."""
    lines = []
    for msg in messages[-6:]:  # Last 6 messages for context
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            preview = content[:200] + "..." if len(content) > 200 else content
            lines.append(f"{role.upper()}: {preview}")
    return "\n".join(lines)


# --- Agent Node ---
def agent_node(state: AgentState) -> AgentState:
    """
    Main agent reasoning node. Decides whether to call tools or respond.
    """
    messages = state["messages"]
    available_tools = state["available_tools"]
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 5)
    
    # Check iteration limit
    if iteration >= max_iterations:
        return {
            **state,
            "next_action": "respond",
            "messages": messages + [{
                "role": "assistant",
                "content": json.dumps({
                    "action": "respond",
                    "answer": "I've reached my reasoning limit. Based on what I've gathered, here's my response...",
                    "confidence": 0.5
                })
            }],
            "iteration": iteration + 1
        }
    
    try:
        # Build agent prompt
        tools_desc = format_tools_description(available_tools)
        conv_history = format_conversation_history(messages)
        
        system_prompt = AGENT_SYSTEM_PROMPT.format(
            tools_description=tools_desc,
            conversation_history=conv_history
        )
        
        # Call LLM
        response = chat(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                *messages
            ],
            options={"temperature": 0.1}
        )
        
        content = response["message"]["content"]
        
        # Try to parse JSON response
        try:
            if isinstance(content, str):
                # Extract JSON from markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                decision = json.loads(content)
            else:
                decision = content
        except json.JSONDecodeError:
            # Fallback: treat as direct response
            decision = {
                "action": "respond",
                "answer": content,
                "confidence": 0.7
            }
        
        action = decision.get("action", "respond")
        
        # Add agent's decision to messages
        new_messages = messages + [{
            "role": "assistant",
            "content": json.dumps(decision) if isinstance(decision, dict) else content
        }]
        
        if action == "call_tool":
            return {
                **state,
                "next_action": "call_tool",
                "tool_calls": decision.get("tool_calls", []),
                "messages": new_messages,
                "iteration": iteration + 1
            }
        else:
            return {
                **state,
                "next_action": "respond",
                "messages": new_messages,
                "iteration": iteration + 1
            }
    
    except Exception as e:
        error_msg = f"Agent reasoning error: {str(e)}"
        sys.stderr.write(f"❌ {error_msg}\n")
        
        return {
            **state,
            "next_action": "respond",
            "messages": messages + [{
                "role": "assistant",
                "content": json.dumps({
                    "action": "respond",
                    "answer": f"I encountered an error: {error_msg}",
                    "confidence": 0.0
                })
            }],
            "iteration": iteration + 1
        }


# --- Tool Execution Node ---
def tool_execution_node(state: AgentState) -> AgentState:
    """
    Execute tool calls in parallel and add results to state.
    """
    tool_calls = state.get("tool_calls", [])
    messages = state["messages"]
    
    if not tool_calls:
        return {
            **state,
            "next_action": "respond"
        }
    
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("tool")
        arguments = tool_call.get("arguments", {})
        
        sys.stderr.write(f"🔧 Calling tool: {tool_name} with {arguments}\n")
        
        result = call_mcp_tool(tool_name, arguments)
        results.append({
            "tool": tool_name,
            "arguments": arguments,
            "result": result
        })
    
    # Add tool results to messages
    tool_message = {
        "role": "tool",
        "content": json.dumps(results, indent=2)
    }
    
    return {
        **state,
        "tool_results": results,
        "messages": messages + [tool_message],
        "next_action": "call_tool",  # Return to agent for synthesis
        "tool_calls": []  # Clear pending calls
    }


# --- Response Node ---
def response_node(state: AgentState) -> AgentState:
    """
    Extract final answer from agent's response.
    """
    messages = state["messages"]
    
    # Get last assistant message
    last_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_msg = msg
            break
    
    if not last_msg:
        return {
            **state,
            "next_action": "end"
        }
    
    content = last_msg.get("content", "")
    
    try:
        if isinstance(content, str):
            data = json.loads(content)
        else:
            data = content
        
        final_answer = data.get("answer", content)
    except:
        final_answer = content
    
    # Add final answer as user-facing message
    return {
        **state,
        "messages": messages + [{
            "role": "final_answer",
            "content": final_answer
        }],
        "next_action": "end"
    }


# --- Routing Logic ---
def should_continue(state: AgentState) -> Literal["agent", "tools", "respond", "end"]:
    """
    Determine next node based on agent's decision.
    """
    next_action = state.get("next_action", "end")
    
    if next_action == "call_tool":
        # Check if we have pending tool calls
        if state.get("tool_calls"):
            return "tools"
        else:
            # Tool results ready, return to agent
            return "agent"
    elif next_action == "respond":
        return "respond"
    else:
        return "end"


# --- Graph Construction ---
def create_agentic_graph() -> StateGraph:
    """
    Create the agentic LangGraph workflow.
    
    Flow:
      START → Agent (decide) → Tools (execute) → Agent (synthesize) → Response → END
                ↓                                      ↑
                └──────── (if needs more tools) ──────┘
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)
    workflow.add_node("respond", response_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "agent": "agent",
            "tools": "tools",
            "respond": "respond",
            "end": END
        }
    )
    
    # Tools always return to agent for synthesis
    workflow.add_edge("tools", "agent")
    
    # Response goes to END
    workflow.add_edge("respond", END)
    
    return workflow.compile()


# --- Main Execution ---
def main():
    print("="*60)
    print("🧬 Bio-Link Agentic System (LangGraph + MCP)")
    print("="*60)
    print(f"MCP Server: {MCP_SERVER_URL}")
    print(f"Agent Model: {AGENT_MODEL}")
    print("Type 'quit' to exit.\n")
    
    # Fetch available tools
    print("📡 Fetching MCP tools...")
    available_tools = get_mcp_tools()
    
    if not available_tools:
        print("❌ No tools available. Ensure MCP server is running.")
        sys.exit(1)
    
    print(f"✅ Loaded {len(available_tools)} tools:")
    for tool in available_tools:
        print(f"   - {tool.get('name')}")
    print()
    
    # Create graph
    graph = create_agentic_graph()
    print("✅ LangGraph compiled successfully.\n")
    print("="*60 + "\n")
    
    # Interactive loop
    while True:
        user_input = input("User: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break
        
        try:
            # Initialize state
            initial_state = {
                "messages": [{
                    "role": "user",
                    "content": user_input
                }],
                "user_query": user_input,
                "available_tools": available_tools,
                "next_action": "call_tool",
                "tool_calls": [],
                "tool_results": [],
                "iteration": 0,
                "max_iterations": 5
            }
            
            # Run graph
            print("\n🤖 Agent thinking...\n")
            result = graph.invoke(initial_state)
            
            # Extract final answer
            final_answer = None
            for msg in reversed(result["messages"]):
                if msg.get("role") == "final_answer":
                    final_answer = msg.get("content")
                    break
            
            if not final_answer:
                # Fallback: get last assistant message
                for msg in reversed(result["messages"]):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        try:
                            data = json.loads(content) if isinstance(content, str) else content
                            final_answer = data.get("answer", content)
                        except:
                            final_answer = content
                        break
            
            # Display answer
            print("="*60)
            print("[AGENT RESPONSE]")
            print("="*60)
            print(final_answer or "No response generated")
            print("="*60)
            print(f"\n📊 Iterations: {result.get('iteration', 0)}")
            print(f"🔧 Tool calls made: {len(result.get('tool_results', []))}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()