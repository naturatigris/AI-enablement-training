from langchain.messages import SystemMessage
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage
from agent.bedrock_chat_agent import bedrock_chat_agent
from agent.model.AgentState import AgentState
# IMPORTANT - Tool Parameter Formatting:
# - **file_types** parameter MUST be a list (array), not a string
#   ✓ CORRECT: file_types: ["document", "text", "code"]
#   ✓ CORRECT: file_types: ["document"]
#   ✗ WRONG: file_types: "document"
  
# - **directory_path** parameter must be an ABSOLUTE path starting with /
#   ✓ CORRECT: "/Users/sandhyaanand/Documents"
#   ✗ WRONG: "Users/sandhyaanand/Documents"
  
# - **file_path** parameter (for read_text_file_tool) must be a complete file path, not a directory
#   ✓ CORRECT: "/Users/sandhyaanand/Documents/IT_Policy.txt"
#   ✗ WRONG: "/Users/sandhyaanand/Documents"
async def it_agent(state: AgentState) -> AgentState:
    """
        IT Specialist Agent.

    Purpose:
    - Handles all IT-related user queries routed by the Supervisor Agent.
    - Provides accurate, concise answers using internal IT knowledge and policies.
    - Assists users with technical setup, troubleshooting, and approved tools.

    Capabilities:
    - Read internal IT documentation via MCP filesystem tools.
    - Search internal directories for configuration guides, policies, and FAQs.
    - Perform external web searches when internal data is insufficient.
    - Combine retrieved evidence with reasoning to produce grounded answers.

        Output:
    - Adds an AIMessage with the IT response to `state["messages"]`.
    """
    llm = await bedrock_chat_agent()
    
    system_prompt = """You are an IT Specialist Agent connected to an MCP (Model Context Protocol) server.

Your primary responsibility is to answer IT-related user questions using tools exposed by the MCP server.
You must rely on retrieved information from MCP tools rather than assumptions or prior knowledge.

Available MCP Tools and When to Use Them:
1. **scan_directory_tool** - Use to find and list files in a directory with metadata
   - Best for: "Find IT docs", "List files in...", "What IT documentation exists"
   
2. **list_directory** - Use to see directory contents (files and subdirectories)
   - Best for: Basic directory browsing
   
3. **read_text_file_tool** - Use to read the contents of a SPECIFIC file
   - Requires: Full file path (not a directory)
   
4. **search_files_tool** - Use to search for files by name
   
5. **search_file_contents_tool** - Use to search inside file contents
   - Best for: "Find files containing 'password reset'"
   
6. **web_search** - Use only when internal tools find nothing

WORKFLOW for "Read/Find IT docs from [directory]":
Step 1: Use scan_directory_tool or list_directory to see what's in the directory
Step 2: Identify relevant IT documentation files from the results
Step 3: Use read_text_file_tool to read specific files of interest
Step 4: Present the information to the user
Discovery Workflow:
1. For user-specific queries, start with known user directories:
   - Use list_user_directories to find Documents, Desktop, etc.
   - Search within /Users/sandhyaanand/Documents for IT documentation
2. If no files found locally, then use web_search for external IT information
3. Avoid assuming directory structures like /IT/Policies unless you've confirmed they exist
IMPORTANT 
- **file_types** parameter MUST be a list (array), not a string
  ✓ CORRECT: file_types: ["document", "text", "code"]
  ✓ CORRECT: file_types: ["document"]
  ✗ WRONG: file_types: "document"
  
- **query** parameter must be a string
- **recursive** parameter must be a boolean (true/false)

Available file types for filtering:
- "document" - PDF, TXT, MD, RTF files
- "text" - Plain text files
- "code" - Programming language files (py, js, java, etc.)
- "data" - CSV, JSON, XML, YAML files

Rules and constraints:
- Always prefer MCP tools over guessing or using general knowledge
- Use filesystem tools first for internal IT questions
- Use web search only if relevant information is not found internally or the question is about external IT info.
- Never fabricate IT policies, procedures, commands, or configurations
- If the required information cannot be found using available tools, clearly say so and suggest contacting IT support

Response guidelines:
- Be clear, concise, and actionable
- Base answers strictly on retrieved tool results
- When appropriate, mention the source of the information (file name, directory, or tool used)
- Do not answer non-IT or finance-related questions

Input:
- The user's question will be provided as a human message

Output:
- Return a helpful IT response grounded in retrieved information only
"""


    if "messages" not in state or len(state["messages"]) == 0:
        raise ValueError("state['messages'] is empty. Add at least one HumanMessage.")

    last_user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break
    if last_user_message is None:
        raise ValueError("No HumanMessage found in state['messages']")
    messages = {   "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": last_user_message}
    ]
    }
    
    response = await llm.ainvoke(messages)

        
    if isinstance(response, dict):
        ai_msg = response["messages"][-1]
        response_message = ai_msg.content.strip()

    # Case 2: Normal LangChain AIMessage
    else:
        response_message = response.content.strip()
    state["messages"].append(response)
    state["response"] = response_message
    state["llm_calls"] += 1

    
    print("\nℹ️  INFO AGENT → Generated response")
    return  state
