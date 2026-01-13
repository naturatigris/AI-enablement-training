from chroma_utils import collection
from session import create_session
from chatbot import conversational_rag_query
from chroma_utils import process_and_add_documents
# Create a new conversation session
session_id = create_session()
source_file='Source'
process_and_add_documents(collection,source_file)
model=''
print("Enter the model you want to use. Choose an option:\n1. Ollama\n2. GPT-4o")
option=input()
while True:
    try:
        if int(option)==1:
            model='ollama'
            break
        elif int(option)==2:
            model='gpt'
            break
        else:
            print("enter valid input")
            option=input()
    except Exception as e:
        print("enter valid input")
        option=input()


while True:
    print("enter your question or enter exit to end the session:")
    query=input("chat: ")
    if query.lower().strip()=='exit':
        break
    response, sources = conversational_rag_query(
                    collection,
                    query,
                    session_id,
                    model=model
        )

    # query = "When was GreenGrow Innovations founded?"

    # print(response)
    # # query = "Where is it located?"
    # response, sources = conversational_rag_query(
    #             collection,
    #             query,
    #             session_id
    # )

    print(response)
