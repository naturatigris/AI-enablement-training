import uuid
from datetime import datetime
import json
import ollama
from openai import AzureOpenAI


# In-memory conversation store
conversations = {}

def create_session():
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    conversations[session_id] = []
    return session_id
def add_message(session_id: str, role: str, content: str):
    """Add a message to the conversation history"""
    if session_id not in conversations:
        conversations[session_id] = []

    conversations[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

def get_conversation_history(session_id: str, max_messages: int = None):
    """Get conversation history for a session"""
    if session_id not in conversations:
        return []

    history = conversations[session_id]
    if max_messages:
        history = history[-max_messages:]

    return history
def format_history_for_prompt(session_id: str, max_messages: int = 5):
    """Format conversation history for inclusion in prompts"""
    history = get_conversation_history(session_id, max_messages)
    formatted_history = ""

    for msg in history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n\n"

    return formatted_history.strip()
def contextualize_query(query: str, conversation_history: str, model: str) -> str:
    prompt = f"""
    Given a chat history and the latest user question which might reference context
    in the chat history, formulate a standalone question which can be understood
    without the chat history. Do NOT answer the question.

    Chat history:
    {conversation_history}

    Question:
    {query}
    """

    try:
        if model=='gpt':
            endpoint = "https://azure-openai-101.openai.azure.com/"
            deployment = "gpt-4.1"
            api_version = "2024-12-01-preview"
            subscription_key = "api key"

            # Create Azure OpenAI client
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=subscription_key,
            )
            response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_completion_tokens=13107,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        else:

            response = ollama.chat(
                model="deepseek-r1:7b",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.0}
            )

        if model == "gpt":
            return response.choices[0].message.content.strip()
        else:
            return response["message"]["content"].strip()

    except Exception as e:
        print(f"Error contextualizing query: {e}")
        return query
