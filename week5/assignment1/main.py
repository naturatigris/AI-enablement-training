from agent.multiagent import create_agent_graph 
import asyncio
from langchain_core.messages import HumanMessage,AIMessage
from agent.model.AgentState import AgentState

async def chat():
    print("=" * 60)
    print("Welcome to Presidio HR Assistant!")
    print("=" * 60)
    print("Ask me about insurance, benefits, and HR policies!")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            agent = create_agent_graph()

            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\nThank you for using Presidio HR Assistant. Goodbye!")
                break
            
            # For CompiledStateGraph, invoke returns a dict with 'messages' key
            initial_state: AgentState = {
                "messages": [HumanMessage(content=user_input)],
                "llm_calls": 0,
                "route": "",
                "response": ""
                    }
            result = await agent.ainvoke(initial_state)
            
            response = result
            response = extract_final_response(result)
            
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try rephrasing your question.\n")
def extract_final_response(state: dict) -> str:
    """
    Robustly extract the final assistant response from a LangGraph state.
    Priority:
    1. state["response"] (if set by graph)
    2. Last AIMessage found anywhere (even nested)
    """

    # ✅ 1. Best case: graph already set response
    if isinstance(state, dict):
        response = state.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()

    # ✅ 2. Walk messages (including nested dicts)
    def walk(messages):
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content
            if isinstance(msg, dict) and "messages" in msg:
                found = walk(msg["messages"])
                if found:
                    return found
        return None

    messages = state.get("messages", [])
    found = walk(messages)

    return found or "No response generated."

if __name__ == "__main__":
    asyncio.run(chat())
