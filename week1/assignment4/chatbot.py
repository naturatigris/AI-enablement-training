import requests
from chroma_utils import semantic_search,get_context_with_sources
from session import contextualize_query,format_history_for_prompt,add_message
from openai import AzureOpenAI

 
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:7b"

def call_gpt(prompt: str) -> str:
    client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint="https://azure-openai-101.openai.azure.com/",
        api_key="api key",
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

def call_llama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]
 
def get_prompt(context, conversation_history, query):
  prompt = f"""Based on the following context and conversation history, please provide a relevant and contextual response.
    If the answer cannot be derived from the context, only use the conversation history or say "I cannot answer this based on the provided information."

    Context from documents:
    {context}

    Previous conversation:
    {conversation_history}

    Human: {query}

    Assistant:"""
  return prompt
 
 
# Updated generate response function with conversation history also passed for Chatbot Memory
def generate_response(query: str, context: str, conversation_history: str = "",model:str=""):
    """Generate a response using Ollama (DeepSeek-R1) with conversation history"""

    prompt = get_prompt(context, conversation_history, query)

    try:
        if model == "gpt":
            return call_gpt(prompt)
        else:
            response = call_llama(prompt)
            return response.strip()


    except Exception as e:
        print(f"Error generating response: {e}")
        return "I encountered an error while generating the response."
def conversational_rag_query(
    collection,
    query: str,
    session_id: str,
    n_chunks: int = 3,
    model:str=''
):
    """Perform RAG query with conversation history"""
    # Get conversation history
    conversation_history = format_history_for_prompt(session_id)

    # Handle follo up questions
    print('model_type:',model)
    query = contextualize_query(query, conversation_history,model)
    # print("Contextualized Query:", query)

    # Get relevant chunks
    context, sources = get_context_with_sources(
        semantic_search(collection, query, n_chunks)
    )
    # print("Context:", context)
    # print("Sources:", sources)


    response = generate_response(query, context, conversation_history,model)

    # Add to conversation history
    add_message(session_id, "user", query)
    add_message(session_id, "assistant", response)

    return response, sources
 
