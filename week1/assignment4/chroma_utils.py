import chromadb
import os
from chromadb.utils import embedding_functions
from doc_processor import read_document,split_text

# Initialize ChromaDB client with persistence
client = chromadb.PersistentClient(path="chroma_db")

# Configure sentence transformer embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or get existing collection
collection = client.get_or_create_collection(
    name="documents_collection",
    embedding_function=sentence_transformer_ef
)
def process_document(file_path: str):
    """Process a single document and prepare it for ChromaDB"""
    try:
        # Read the document
        content = read_document(file_path)

        # Split into chunks
        chunks = split_text(content)

        # Prepare metadata
        file_name = os.path.basename(file_path)
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

        return ids, chunks, metadatas
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []
def add_to_collection(collection, ids, texts, metadatas):
    """Add documents to collection in batches"""
    if not texts:
        return

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        collection.add(
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )

def process_and_add_documents(collection, folder_path: str):
    """Process all documents in a folder and add to collection"""
    files = [os.path.join(folder_path, file) 
             for file in os.listdir(folder_path) 
             if os.path.isfile(os.path.join(folder_path, file))]

    for file_path in files:
        print(f"Processing {os.path.basename(file_path)}...")
        ids, texts, metadatas = process_document(file_path)
        add_to_collection(collection, ids, texts, metadatas)
        print(f"Added {len(texts)} chunks to collection")
def print_search_results(results):
    """Print formatted search results"""
    print("\nSearch Results:\n" + "-" * 50)

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i]

        print(f"\nResult {i + 1}")
        print(f"Source: {meta['source']}, Chunk {meta['chunk']}")
        print(f"Distance: {distance}")
        print(f"Content: {doc}\n")
def semantic_search(collection, query: str, n_results: int = 10):
    """Perform semantic search on the collection"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    docs = list(zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
))
    ranked = []

    for doc, dist, meta in docs:
        kw_score = keyword_overlap_score(doc, query)
        score = hybrid_score(dist, kw_score)
        ranked.append((doc, score, meta))

    ranked.sort(key=lambda x: x[1], reverse=True)
    top_k_docs = ranked[:3]

    return top_k_docs
 
def get_context_with_sources(results):
    """Extract context and source information from search results"""
    # Combine document chunks into a single context
    docs_col_0 = [item[0] for item in results]
    meta_col_2 = [item[2] for item in results]
    context = "\n\n".join(docs_col_0)
 
    # Format sources with metadata
    sources = [
        f"{meta['source']} (chunk {meta['chunk']})"
        for meta in meta_col_2
    ]
 
    return context, sources

def hybrid_score(distance, keyword_hits):
    return (1 - distance) * 0.7 + keyword_hits * 0.3
def keyword_overlap_score(doc, query):
    return len(set(doc.lower().split()) & set(query.lower().split()))
