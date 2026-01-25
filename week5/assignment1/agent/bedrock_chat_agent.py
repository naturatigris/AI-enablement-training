from langchain_aws import ChatBedrockConverse
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain_core.messages import SystemMessage

# Global clients to keep them alive
_mcp_clients = []

async def bedrock_chat_agent(enable_tools: bool = True):
    global _mcp_clients
    
    model = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.1,
        max_tokens=1000,
        region_name="us-east-1"
    )
    
    if enable_tools:
        # Create and initialize MCP clients
        client1 = MultiServerMCPClient(
            {
                "filesystem": {
                    "transport": "stdio",
                    "command": "python",
                    "args": [
                        "/Users/sandhyaanand/Documents/projects/AI enablement training/week5/assignment1/file-system-mcp-server/file-system-mcp-server/fs_server.py"
                    ],
                }
            }
        )
        client2 = MultiServerMCPClient(
            {
                "web_search_system": {
                    "transport": "stdio",
                    "command": "python",
                    "args": [
                        "/Users/sandhyaanand/Documents/projects/AI enablement training/week5/assignment1/duckduckgo-search-server/ws_server.py"
                    ],
                }
            }
        )

        # Enter the async context managers
        await client1.__aenter__()
        await client2.__aenter__()
        
        # Store clients globally to keep them alive
        _mcp_clients.extend([client1, client2])
        
        # Get tools from both clients
        tools1 = client1.get_tools()
        tools2 = client2.get_tools()
        tools = tools1 + tools2

        agent = create_agent(model=model, tools=tools, debug=True)
    else:
        agent = create_agent(model=model, debug=True)
        
    return agent

async def cleanup_mcp_clients():
    """Call this function when shutting down to properly close MCP clients"""
    global _mcp_clients
    for client in _mcp_clients:
        try:
            await client.__aexit__(None, None, None)
        except Exception as e:
            print(f"Error closing MCP client: {e}")
    _mcp_clients.clear()
