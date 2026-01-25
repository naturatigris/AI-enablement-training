from agent.model.AgentState import AgentState
from typing import Literal
from langgraph.graph import StateGraph, END, START
from agent.Finance_Agent import finance_agent
from agent.IT_Agent import it_agent
from agent.Route_Agent import router_agent

def route_to_agent(state: AgentState) -> Literal["IT","Finance"]:
    """
    Conditional edge function that determines which agent to invoke.
    """
    return state["route"]


def create_agent_graph():
    """
    Builds the LangGraph workflow with LLM-powered routing.
    """
    
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("router", router_agent)
    workflow.add_node("IT", it_agent)
    workflow.add_node("Finance", finance_agent)
    
    # Start with router
    workflow.add_edge(START, "router")
    
    # Conditional routing from router to agents
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "IT": "IT",
            "Finance": "Finance"        
        }
    )
    
    # All agents end the workflow
    workflow.add_edge("Finance", END)
    workflow.add_edge("IT", END)
    
    return workflow.compile()
