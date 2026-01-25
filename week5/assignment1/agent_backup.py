from langchain_aws import ChatBedrockConverse
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain.messages import SystemMessage

import asyncio




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

    async with client1:
        tools1 = client1.get_tools()           

    async with client2:
        tools2 = client2.get_tools()
    tools = tools1 + tools2

    agent=create_agent(model=model, tools=tools,debug=True,system_prompt=prompt_string)

asyncio.run(bedrock_chat_agent())
