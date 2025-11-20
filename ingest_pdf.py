import os
import gc
import itertools
from typing import Iterator, List, Generator

# Integration Imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import chromadb

# --- Configuration ---
FILE_PATH = "tsla-20231231-gen.pdf"  # Target PDF
DB_PATH = "./chroma_db_storage"    # Persistence directory
COLLECTION_NAME = "pdf_rag_collection"
BATCH_SIZE = 100                   # Chunks per batch (Controls RAM usage)
CHUNK_SIZE = 1000                  # Characters per chunk
CHUNK_OVERLAP = 200                # Character overlap
MODEL_NAME = "all-MiniLM-L6-v2"    # Low memory embedding model

def load_pdf_lazy(file_path: str) -> Iterator:
    """
    Lazily loads PDF pages.
    Critically, this yields one page at a time, preventing the entire
    PDF from being loaded into RAM at once.
    """
    print(f"[INFO] Initializing Lazy Loader for: {file_path}")
    loader = PyMuPDFLoader(file_path)
    #.lazy_load() is the key method here vs.load()
    return loader.lazy_load()

def split_text_generator(
    doc_iterator: Iterator,
    chunk_size: int,
    chunk_overlap: int
) -> Generator:
    """
    Consumes pages from the lazy loader and yields chunks immediately.
    This ensures we don't accumulate a massive list of splits in memory.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for doc in doc_iterator:
        # Split only the current page
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            yield chunk

def batch_generator(iterable: Iterator, n: int) -> Iterator[List]:
    """
    Helper to group items from a stream into fixed-size batches.
    Standard itertools pattern for memory efficiency.
    """
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch

def main():
    # 1. Initialize Vector DB (Chroma)
    print("[INFO] Setting up ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # 2. Initialize Embedding Model (Lightweight)
    print(f"[INFO] Loading Embedding Model: {MODEL_NAME}...")
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    # Optimize for CPU/Low-RAM if needed
    # embedding_model.to('cpu')

    # 3. Pipeline Setup
    # Create the chain of generators: Loader -> Splitter -> Batcher
    page_stream = load_pdf_lazy(FILE_PATH)
    chunk_stream = split_text_generator(page_stream, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_batches = batch_generator(chunk_stream, BATCH_SIZE)

    print("[INFO] Starting Pipeline Processing...")
    total_chunks = 0

    # 4. Process Stream
    for i, batch in enumerate(chunk_batches):
        # Extract text content for embedding
        batch_texts = [doc.page_content for doc in batch]
        
        # Extract metadata and ensure IDs are unique
        batch_metadatas = [doc.metadata for doc in batch]
        batch_ids = [f"id_{total_chunks + j}" for j in range(len(batch))]

        # Generate Embeddings (Only for this batch)
        # encode() handles this small batch efficiently in RAM
        embeddings = embedding_model.encode(batch_texts, show_progress_bar=False)

        # Insert into Vector DB
        collection.add(
            documents=batch_texts,
            embeddings=embeddings.tolist(),
            metadatas=batch_metadatas,
            ids=batch_ids
        )

        count = len(batch)
        total_chunks += count
        print(f"   -> Processed Batch {i+1} ({count} chunks). Total: {total_chunks}")

        # explicit memory cleanup (Optional but recommended for tight limits)
        del batch_texts, batch_metadatas, batch_ids, embeddings
        gc.collect()

    print(f" Ingestion Complete. Total Chunks: {total_chunks}")

if __name__ == "__main__":
    main()
