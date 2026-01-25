# server.py
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_classic.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import GoogleDriveLoader
from fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun
import os
import re
from dotenv import load_dotenv
from utils.sentence_retrieval import ContextRetriever

load_dotenv()
api_key = os.getenv("SERPAPI_API_KEY") 
app = FastMCP("My MCP Server")

@app.tool(name="googleDocs")
async def google_docs_query(query: str) -> str:
    """
   Search and retrieve content from Presidio's insurance policy documents stored in Google Docs.
    
    Use this tool for questions about:
    - Group insurance plans
    - Benefits policies
    - Coverage limitations
    - Authority limitations
    - Insurance procedures
    
    Args:
        query: The user's question about insurance or benefits
        
    Returns:
        str: The relevant document content
            """
    try:
        # If loading a specific document
        loader = GoogleDriveLoader(
            credentials_path="./credentials.json",
            token_path="./token.json",
            document_ids=["1nRMrpZNYBNcmYVXN--Dp5o4SzGl66zivcIcRCrvs-5g"]  
        )
    
        docs = loader.load()
        # print(f"Loaded {len(docs)} documents")
       
        content = "\n\n".join(doc.page_content for doc in docs)
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        engine= ContextRetriever()
        engine.add_document('doc',content)
        docs = engine.retrieve_docs(query)
        doc_ids = [doc_id for doc_id, _ in docs]

        sentences = engine.extract_sentences(query, doc_ids)
        context = "\n\n".join(
            f"{i+1}. {sent.strip()}"
            for i, (sent, _) in enumerate(sentences)
        )
        
        return context


    except Exception as e:
        return f"Error connecting to Google Docs: {str(e)}"


@app.tool(name="rag")
async def rag_query(query: str) -> str:
    """
    Search internal HR policy documents using vector embeddings and semantic search.
    
    Use this tool for questions about:
    - Hybrid work policies
    - Remote work guidelines
    - Office attendance requirements
    - Work from home procedures
    - Company HR policies
    - Employee policy documents
    - Internal company guidelines
    
    This tool searches through PDF documents stored in the vector database
    and returns the most relevant sections based on semantic similarity.
    
    Args:
        query: The user's question about HR policies or company guidelines
        
    Returns:
        str: Relevant excerpts from HR policy documents
    """
    try:
        # build_vectorstore()
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


@app.tool(name="web_search_serpapi")
async def web_search(query: str,max_results: int = 5) -> str:
    """
        Search the web for current information, industry benchmarks, trends, and regulatory updates.
    
    Use this tool for questions about:
    - Industry benchmarks and standards
    - Market trends and analysis
    - Regulatory updates and compliance
    - Current news and developments
    - External research and data
    - Competitor information
    - Public information not in internal documents
    - Recent changes in insurance/HR regulations
    
    This tool performs a Google search and retrieves content from the top result.
    
    Args:
        query: Search query for finding external information
        
    Returns:
        str: Content from the top search result with source URL

    """
    ddg_search = DuckDuckGoSearchRun()

    try:
        result = ddg_search.run(query)
        return result[: max_results * 500]  # soft limit
        
    except Exception as e:
        return {
            "query": query,
            "error": str(e)
        }

        
    except Exception as e:
        return f"Error fetching web data: {str(e)}"
def load_documents():
    docs = []
    loaders = [
        PyPDFLoader("mcp_server/data/Hybrid Work Policy 2026.pdf")
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
