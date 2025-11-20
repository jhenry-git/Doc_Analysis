import os
import sys
import warnings
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Suppress Python 3.9 Deprecation Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Load Environment ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY not found in .env file.")
    sys.exit(1)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-2.5-flash")

# --- Chroma / Embeddings Config ---
DB_PATH = "./chroma_db_storage"
COLLECTION_NAME = "pdf_rag_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    # 1. Connect to ChromaDB
    if not os.path.exists(DB_PATH):
        print(f"❌ Database path not found: {DB_PATH}")
        print("   Run 'ingest_pdf.py' first!")
        return

    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Failed to load Chroma collection: {e}")
        return

    print(f"[INFO] Loading Embedding Model: {EMBEDDING_MODEL}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\n" + "=" * 60)
    print(" 🤖 AI Analyst Ready! Type 'exit' to quit.")
    print("=" * 60 + "\n")

    # 2. Chat Loop
    while True:
        user_query = input("\nYou: ").strip()

        if user_query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not user_query:
            continue

        print("   🔍 Searching document...")

        # A. Encode user query
        query_vector = embed_model.encode(user_query).tolist()

        # B. Retrieve from Chroma
        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=5,
                include=["documents"]
            )
        except Exception as e:
            print(f"❌ Chroma Query Error: {e}")
            continue

        docs_nested = results.get("documents", [])

        if not docs_nested or not docs_nested[0]:
            print("   ⚠️ No relevant information found.")
            continue

        # C. Flatten documents
        flattened_docs = docs_nested[0]
        context_text = "\n\n".join(flattened_docs)

        # Optional safety trimming
        context_text = context_text[:20000]

        # D. Generate Answer with Gemini
        prompt = f"""
        You are a helpful assistant answering questions strictly based on the provided PDF content.

        RULES:
        1. Only answer using the provided CONTEXT.
        2. If the answer is not in the context, say exactly:
           "I cannot find that information in the document."
        3. Keep answers concise and professional.

        CONTEXT:
        {context_text}

        QUESTION:
        {user_query}
        """

        print("   🧠 Synthesizing answer...")

        try:
            response = llm_model.generate_content(prompt)
            output = response.text.strip()
            print(f"\nGemini:\n{'-'*60}\n{output}\n{'-'*60}")
        except Exception as e:
            print(f"❌ Gemini API Error: {e}")


if __name__ == "__main__":
    main()
