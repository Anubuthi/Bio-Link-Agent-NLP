"""
Configuration file for Bio-Link Agentic System.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- MCP Server Configuration ---
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8005"))
MCP_SERVER_URL = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}"

# MCP Endpoints
MCP_ENDPOINTS = {
    "tools_list": f"{MCP_SERVER_URL}/tools/list",
    "tool_call": f"{MCP_SERVER_URL}/tools/call",
    "health": f"{MCP_SERVER_URL}/health",
    "sse": f"{MCP_SERVER_URL}/sse"
}

# --- LLM Configuration ---
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen2.5:7b")
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))

# --- API Keys ---
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "test@example.com")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Timeouts ---
TOOL_CALL_TIMEOUT = int(os.getenv("TOOL_CALL_TIMEOUT", "300"))  # 5 minutes
MCP_HEALTH_CHECK_TIMEOUT = int(os.getenv("MCP_HEALTH_CHECK_TIMEOUT", "5"))

# --- RAG Configuration ---
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "./logs/biolink.log")

# Create log directory if it doesn't exist
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)