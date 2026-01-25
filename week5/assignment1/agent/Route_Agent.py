from langchain.messages import SystemMessage
from langchain_core.messages import SystemMessage, HumanMessage
from agent.bedrock_chat_agent import bedrock_chat_agent
from agent.model.AgentState import AgentState
from agent.model.RouteDecision import RouteDecision

async def router_agent(state: AgentState) -> AgentState:
    """
    Supervisor router agent.

    Purpose:
    - Classifies user queries into IT-related or Finance-related.
    - Routes the query to the appropriate specialist agent (IT Agent or Finance Agent).

    Classification Categories:
    - IT: Technical support, software, hardware, access, VPN, systems, tools
    - Finance: Payroll, reimbursement, budget, invoices, expenses, payments

    Output:
    - Returns the routing decision only (IT, Finance, or Unclear)
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
    print("Routing agent initiation.....")
    llm = await bedrock_chat_agent(enable_tools=False)
    llm=llm.bind(response_format=RouteDecision)
   
    
    system_prompt = """
        You are a Supervisor Agent responsible for routing employee queries
        to the correct specialist agent.

        Analyze the user's message and classify it into ONE of the following categories:

        1. **IT**
        - Use when the query is related to technology, systems, software, hardware, or IT policies.
        - Topics include: VPN, laptops, software installation, access issues, credentials, approved tools.
        - Examples:
            - "How do I set up VPN?"
            - "What software is approved for use?"
            - "How can I request a new laptop?"

        2. **Finance**
        - Use when the query is related to financial processes, payroll, expenses, or budgets.
        - Topics include: reimbursement, payroll dates, invoices, budget reports, payments.
        - Examples:
            - "How do I file a reimbursement?"
            - "When is payroll processed?"
            - "Where can I find last month's budget report?"

        Return ONLY the classification label: IT or Finance.
        """

    messages = {   "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": last_user_message}
    ]
    }
    

    # decision: RouteDecision = await llm.ainvoke(messages)
    response = await llm.ainvoke(messages)
    
    if isinstance(response, dict):
        ai_msg = response["messages"][-1]
        route_text = ai_msg.content.strip()

    # Case 2: Normal LangChain AIMessage
    else:
        route_text = response.content.strip().upper()

    
    # Create RouteDecision object
    decision = RouteDecision(route=route_text)

    state["route"] = decision.route
    state["llm_calls"] += 1
    # state["messages"].append(HumanMessage(content=last_user_message))  # original user
    state["messages"].append(SystemMessage(content=f"Router decision: {decision.route}"))
    
    print(f"\n🧭 ROUTER → Routing to: {decision.route}")
    return state

