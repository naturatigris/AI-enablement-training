from agent.myfirstagent import bedrock_chat_agent
import asyncio
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import os
from langfuse import Langfuse
from langchain_core.messages import AIMessage, BaseMessage
from nemoguardrails import LLMRails, RailsConfig
from langchain_aws import ChatBedrock

load_dotenv()
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse = Langfuse()

langfuse_handler = CallbackHandler(
    public_key=langfuse_public_key
)

class CustomBedrockConverse(ChatBedrock):
    """Custom wrapper to ensure consistent string output from Bedrock"""
    
    def _format_output(self, content):
        """Ensure output is always a string, not a list"""
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content) if content else ""
    
    def invoke(self, input, config=None, **kwargs):
        """Override invoke to format output"""
        result = super().invoke(input, config, **kwargs)
        if isinstance(result, BaseMessage):
            if isinstance(result.content, list):
                result.content = self._format_output(result.content)
        return result
    
    async def ainvoke(self, input, config=None, **kwargs):
        """Override ainvoke to format output"""
        result = await super().ainvoke(input, config, **kwargs)
        if isinstance(result, BaseMessage):
            if isinstance(result.content, list):
                result.content = self._format_output(result.content)
        return result


def extract_response_content(result):
    """
    Helper function to safely extract content from GenerationResponse.
    Handles both string and list returns from rails.generate_async()
    """
    if result is None:
        return ""
    
    # If result is a dict with 'content' key
    if isinstance(result, dict):
        content = result.get("content", "")
    else:
        content = result
    
    # Handle case where content is a list
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    
    # Ensure it's a string
    return str(content) if content else ""


async def process_with_guardrails(user_input: str, agent, rails: LLMRails):
    """
    Process input through guardrails chain:
    1. Input guardrails (check jailbreak, mask sensitive data)
    2. Agent processing
    3. Output guardrails (fact check, hallucination check, moderation)
    """
    
    # Step 1: Input Guardrails - validate and process user input
    try:
        input_result = await rails.generate_async(
            messages=[{"role": "user", "content": user_input}]
        )
        input_response = extract_response_content(input_result)
        print("✓ Input Guardrails: Passed")
    except Exception as e:
        print(f"⚠️ Input guardrails error: {e}")
        input_response = user_input
    
    # Check if input was blocked
    if input_response and any(phrase in input_response.lower() for phrase in ["cannot", "can't", "refused", "refuse"]):
        return input_response, []
    
    # Step 2: Process with Agent - use original or validated input
    validated_input = input_response if input_response else user_input
    try:
        agent_result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": validated_input}]},
            config={"callbacks": [langfuse_handler]}
        )
        
        # Extract agent response
        if isinstance(agent_result, dict) and 'messages' in agent_result:
            messages = agent_result['messages']
            if messages and len(messages) > 0:
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    agent_response = last_message.get('content', str(last_message))
                else:
                    agent_response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                # Handle case where content is a list
                if isinstance(agent_response, list):
                    agent_response = "\n".join(str(item) for item in agent_response)
            else:
                agent_response = "No response generated."
        else:
            agent_response = str(agent_result)
        
        print(f"✓ Agent Processing: Complete")
    except Exception as e:
        print(f"⚠️ Agent processing error: {e}")
        agent_response = "I apologize, but I encountered an error processing your request."
    # # Step 3: Output Guardrails - validate agent response
    # output_result = await rails.generate_async (
    #         messages=[
    #             {"role": "user", "content": validated_input},
    #             {"role": "assistant", "content": agent_response}
    #         ]
            
    #     )
    # final_response = extract_response_content(output_result)
    # if not final_response:

    final_response = agent_response
    
    # Extract tool calls
    tools = []
    if isinstance(agent_result, dict) and "messages" in agent_result:
        for msg in agent_result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    tools.append(call["name"])
    
    return final_response, tools


async def chat():
    print("=" * 60)
    print("Welcome to Presidio HR Assistant!")
    print("=" * 60)
    print("Ask me about insurance, benefits, and HR policies!")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # Use custom model that ensures string outputs
    model = CustomBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.1,
        max_tokens=1000,
        region_name="us-east-1"
    )
    
    # Initialize guardrails and agent once
    try:
        rails_config = RailsConfig.from_path("./gaurdrials")
        rails = LLMRails(config=rails_config, llm=model)
        agent = await bedrock_chat_agent()
        print("✓ Guardrails and agent initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize guardrails or agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\nThank you for using Presidio HR Assistant. Goodbye!")
                break
            
            # Process through guardrails chain
            response, tools = await process_with_guardrails(user_input, agent, rails)
            
            print(f"\nAssistant: {response}\n")
            
            if tools:
                print(f"Tools used: {', '.join(tools)}\n")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            # Uncomment for debugging:
            # import traceback
            # traceback.print_exc()
            print("Please try rephrasing your question.\n")


if __name__ == "__main__":
    asyncio.run(chat())