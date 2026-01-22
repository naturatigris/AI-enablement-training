from agent.myfirstagent import bedrock_chat_agent 
import asyncio

async def chat():
    print("=" * 60)
    print("Welcome to Presidio HR Assistant!")
    print("=" * 60)
    print("Ask me about insurance, benefits, and HR policies!")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            agent = await bedrock_chat_agent()

            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\nThank you for using Presidio HR Assistant. Goodbye!")
                break
            
            # For CompiledStateGraph, invoke returns a dict with 'messages' key
            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            
            # Extract the assistant's response from messages
            if isinstance(result, dict) and 'messages' in result:
                messages = result['messages']
                # Get the last message (assistant's response)
                if messages and len(messages) > 0:
                    last_message = messages[-1]
                    if isinstance(last_message, dict):
                        response = last_message.get('content', str(last_message))
                    else:
                        response = last_message.content if hasattr(last_message, 'content') else str(last_message)
                else:
                    response = "No response generated."
            else:
                response = str(result)
            
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try rephrasing your question.\n")

if __name__ == "__main__":
    asyncio.run(chat())
