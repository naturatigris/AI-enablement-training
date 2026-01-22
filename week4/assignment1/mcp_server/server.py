# server.py
# server.py
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_classic.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import GoogleDriveLoader
from fastmcp import FastMCP



app = FastMCP("My MCP Server")

@app.tool()
def google_docs_query(query: str) -> str:
    """
    Connects to Google Sheets/Docs to answer insurance-related queries for Presidio.
    """
    try:
        # If loading a specific document
        loader = GoogleDriveLoader(
            credentials_path="./credentials.json",
            token_path="./token.json",
            document_ids=["1nRMrpZNYBNcmYVXN--Dp5o4SzGl66zivcIcRCrvs-5g"]  
        )
        
     
        
        docs = loader.load()
        print(f"Loaded {len(docs)} documents")
        print("Content:\n", docs)
        return docs


    except Exception as e:
        return f"Error connecting to Google Docs: {str(e)}"


@app.tool
def rag_query(query: str) -> str:
    """
    Search HR Policy documents using vector embeddings.
    """
    try:
        build_vectorstore()
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            region_name="us-east-1"
        )
        

        vectordb = Chroma(
            persist_directory="vector_store",
            embedding_function=embeddings
        )

        results = vectordb.similarity_search(query, k=3)

        if not results:
            return "No relevant HR policy found."

        return "\n\n".join([doc.page_content for doc in results])

    except Exception as e:
        return f"Error querying vector store: {str(e)}"


# @tool
# def web_search(query: str) -> str:
#     """
#     Fetch industry benchmarks, trends, and regulatory updates.
#     """
#     try:
#         # First, install the library: pip install googlesearch-python

#         query = "web search using python"
#         for url in search(query, num_results=5):
#             print(url)

#         if results:
#             return "\n\n".join(results)
#         return "No web results found."
        
#     except Exception as e:
#         return f"Error fetching web data: {str(e)}"
def load_documents():
    docs = []
    loaders = [
        PyPDFLoader("mcp/data/Hybrid Work Policy 2026.pdf")
    ]

    for loader in loaders:
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    return splitter.split_documents(docs)
def build_vectorstore():
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )

    documents = load_documents()

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="vector_store"
    )

    vectordb.persist()
    print("Vector store created and persisted.")
if __name__ == "__main__":
    app.run(transport="streamable-http")
