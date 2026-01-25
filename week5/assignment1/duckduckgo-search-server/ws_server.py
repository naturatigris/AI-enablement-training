from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize MCP server
mcp = FastMCP(name="duckduckgo-search-server")

# Initialize LangChain DuckDuckGo tool
ddg_search = DuckDuckGoSearchRun()

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query
        max_results: Number of results to return

    Returns:
        Search results as text
    """
    try:
        result = ddg_search.run(query)
        return {
            "query": query,
            "results": result[: max_results * 500]  # soft limit
        }
    except Exception as e:
        return {
            "query": query,
            "error": str(e)
        }

if __name__ == "__main__":
    print("DuckDuckGo MCP Server started")
    mcp.run()
