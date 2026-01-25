from langchain_aws import ChatBedrockConverse
from langchain.agents import create_agent
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

    prompt_string = SystemMessage("system","""
        You are a HR agent capable of answering employee's queries on various HR policies.

        Available tools:
        {tools}

        If you don't have the necessary information to answer a question, please say that you don't have necessary information to answer the question. DO NOT answer questions which are not related to HR policies or Presidio.
    """
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


    agent=create_agent(model=model, tools=tools,debug=True,system_prompt=prompt_string)
    return agent

