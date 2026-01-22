from langchain_aws import ChatBedrockConverse
from langchain.agents import create_agent
from mcp_server.server import google_docs_query
from mcp_server.server import rag_query
from langchain.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient  


# from mcp.server import web_search
async def bedrock_chat_agent():
    model = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.1,
        max_tokens=1000,
        region_name="us-east-1"
    )

    prompt = SystemMessage("system",
        """You are an enterprise AI assistant for Presidio.

    You have access to the following tools:
    {tools}

    Tool names:
    {tool_names}

    Rules:
    - Always reason step by step
    - If the question is about HR, insurance, benefits, or policy → use GoogleDocs first
    - If GoogleDocs returns no answer → use RAG
    - If still unavailable → clearly say information is not available

    Use the following format EXACTLY:

    Thought: your reasoning
    Action: one of [{tool_names}]
    Action Input: input to the tool

    OR

    Thought: final reasoning
    Final Answer: your answer"""
        )

    # tools = [google_docs_query,rag_query
    # ]
    client = MultiServerMCPClient(  
        {
            "My MCP Server": {
                "transport": "http",  # HTTP-based remote server
                # Ensure you start your weather server on port 8000
                "url": "http://localhost:8000/mcp",
            }
        }
    )

    tools = await client.get_tools()  


    agent=create_agent(model=model, tools=tools,system_prompt=prompt)
    return agent

