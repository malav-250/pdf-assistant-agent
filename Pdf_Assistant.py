import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from PyPDF2 import PdfReader
from groq import Groq
import numpy as np

load_dotenv()

if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

embedder = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="./chroma_simple_db")
collection = chroma_client.get_or_create_collection("pdf_recipes")

def download_and_process_pdf():
    pdf_url = "https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
    
    print("📥 Downloading PDF...")
    response = requests.get(pdf_url)
    
    with open("temp_recipes.pdf", "wb") as f:
        f.write(response.content)
    
    print("📄 Processing PDF...")
    reader = PdfReader("temp_recipes.pdf")
    
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            chunks = text.split('\n\n')
            for j, chunk in enumerate(chunks):
                if chunk.strip() and len(chunk.strip()) > 50:
                    texts.append({
                        'id': f"page_{i}_chunk_{j}",
                        'text': chunk.strip(),
                        'metadata': {'page': i, 'chunk': j}
                    })
    
    print(f"📊 Found {len(texts)} text chunks")
    return texts

def setup_vector_db():
    try:
        count = collection.count()
        if count > 0:
            print(f"✅ Vector DB already has {count} documents")
            return
    except:
        pass
    
    print("🔧 Setting up vector database...")
    texts = download_and_process_pdf()
    
    documents = []
    embeddings = []
    ids = []
    metadatas = []
    
    for item in texts:
        embedding = embedder.encode(item['text']).tolist()
        
        documents.append(item['text'])
        embeddings.append(embedding)
        ids.append(item['id'])
        metadatas.append(item['metadata'])
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"✅ Added {len(documents)} documents to vector DB")

def search_knowledge(query, n_results=3):
    query_embedding = embedder.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results['documents'][0] if results['documents'] else []

def chat_with_groq(user_message, context_docs):
    context = "\n\n".join(context_docs) if context_docs else "No relevant context found."
    
    prompt = f"""You are a helpful assistant that answers questions about Thai recipes based on the provided context.

Context from Thai Recipes PDF:
{context}

User Question: {user_message}

Please answer the question based on the context provided. If the context doesn't contain relevant information, say so and provide a general helpful response."""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=1000
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"Error calling Groq: {e}"

def main():
    print("=" * 60)
    print("🤖 Simple PDF Assistant")
    print("📄 Knowledge: Thai Recipes")
    print("🧠 Model: Groq Llama3-70B")
    print("📁 Vector DB: Chroma")
    print("=" * 60)
    
    print("🔧 Setting up...")
    setup_vector_db()
    
    try:
        test_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello!"}],
            model="llama3-70b-8192",
            max_tokens=50
        )
        print("✅ Groq connection successful")
    except Exception as e:
        print(f"❌ Groq connection failed: {e}")
        return
    
    print("\n🚀 Ready! Ask me about Thai recipes!")
    print("💡 Try: 'What are some popular Thai dishes?'")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("😎 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("🔍 Searching knowledge base...")
            relevant_docs = search_knowledge(user_input)
            
            print("🤖 Generating response...")
            response = chat_with_groq(user_input, relevant_docs)
            
            print(f"\n🤖 Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()