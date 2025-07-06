import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

def inspect_vector_db():
    print("=" * 60)
    print("🔍 ChromaDB Vector Database Inspector")
    print("=" * 60)
    
    # Connect to your existing ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_simple_db")
    collection = chroma_client.get_or_create_collection("pdf_recipes")
    
    # Get basic info
    total_count = collection.count()
    print(f"📊 Total documents in database: {total_count}")
    
    if total_count == 0:
        print("❌ Database is empty. Run your main script first!")
        return
    
    # Get top 5 documents
    print("\n🔍 Top 5 Documents in Database:")
    print("-" * 60)
    
    # Get all data (limited to first 5)
    results = collection.get(limit=5, include=['documents', 'metadatas', 'embeddings'])
    
    for i, (doc_id, document, metadata, embedding) in enumerate(zip(
        results['ids'], 
        results['documents'], 
        results['metadatas'], 
        results['embeddings']
    )):
        print(f"\n📄 Document {i+1}:")
        print(f"   ID: {doc_id}")
        print(f"   Page: {metadata['page']}, Chunk: {metadata['chunk']}")
        print(f"   Text Preview: {document[:100]}...")
        print(f"   Embedding Shape: {len(embedding)} dimensions")
        print(f"   Embedding Sample: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
    
    print("\n" + "=" * 60)
    print("🧮 Vector Database Statistics")
    print("=" * 60)
    
    # Calculate some statistics
    all_embeddings = np.array(results['embeddings'])
    print(f"📊 Embedding dimensions: {all_embeddings.shape[1]}")
    print(f"📊 Mean embedding value: {np.mean(all_embeddings):.4f}")
    print(f"📊 Std embedding value: {np.std(all_embeddings):.4f}")
    print(f"📊 Min embedding value: {np.min(all_embeddings):.4f}")
    print(f"📊 Max embedding value: {np.max(all_embeddings):.4f}")
    
    # Show similarity between first two documents
    if len(results['embeddings']) >= 2:
        emb1 = np.array(results['embeddings'][0])
        emb2 = np.array(results['embeddings'][1])
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"📊 Similarity between doc 1 & 2: {similarity:.4f}")

def search_and_inspect(query="Thai curry"):
    print("\n" + "=" * 60)
    print(f"🔍 Searching for: '{query}'")
    print("=" * 60)
    
    # Initialize embedder (same as main script)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_simple_db")
    collection = chroma_client.get_or_create_collection("pdf_recipes")
    
    # Generate query embedding
    query_embedding = embedder.encode(query).tolist()
    print(f"🧮 Query embedding shape: {len(query_embedding)} dimensions")
    print(f"🧮 Query embedding sample: [{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, {query_embedding[2]:.4f}, ...]")
    
    # Search similar documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=['documents', 'metadatas', 'distances']
    )
    
    print(f"\n📋 Top 3 similar documents:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        print(f"\n🔍 Result {i+1}:")
        print(f"   Similarity Score: {1 - distance:.4f}")
        print(f"   Distance: {distance:.4f}")
        print(f"   Page: {metadata['page']}, Chunk: {metadata['chunk']}")
        print(f"   Text: {doc[:200]}...")

def show_all_documents():
    print("\n" + "=" * 60)
    print("📚 All Documents in Database")
    print("=" * 60)
    
    chroma_client = chromadb.PersistentClient(path="./chroma_simple_db")
    collection = chroma_client.get_or_create_collection("pdf_recipes")
    
    # Get all documents
    results = collection.get(include=['documents', 'metadatas'])
    
    print(f"📊 Total documents: {len(results['documents'])}")
    
    # Group by page
    pages = {}
    for doc_id, doc, metadata in zip(results['ids'], results['documents'], results['metadatas']):
        page = metadata['page']
        if page not in pages:
            pages[page] = []
        pages[page].append({
            'id': doc_id,
            'text': doc,
            'chunk': metadata['chunk']
        })
    
    for page_num in sorted(pages.keys()):
        print(f"\n📄 Page {page_num}: {len(pages[page_num])} chunks")
        for chunk in pages[page_num][:2]:  # Show first 2 chunks per page
            print(f"   📝 {chunk['text'][:80]}...")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Inspect top 5 vectors")
    print("2. Search and inspect similarity")
    print("3. Show all documents by page")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        inspect_vector_db()
    elif choice == "2":
        query = input("Enter search query (or press Enter for 'Thai curry'): ").strip()
        search_and_inspect(query if query else "Thai curry")
    elif choice == "3":
        show_all_documents()
    elif choice == "4":
        inspect_vector_db()
        search_and_inspect()
        show_all_documents()
    else:
        print("Invalid choice. Running basic inspection...")
        inspect_vector_db()