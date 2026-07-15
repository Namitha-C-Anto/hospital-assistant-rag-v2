from config import (
    DOCS_PATH, 
    CHUNKS_PATH,)

from rag.loader import load_documents_from_folder
from rag.splitter import split_documents
from rag.storage import save_chunks
from rag.vectorstore import (
    create_vectorstore, 
    save_vectorstore,)
from utils.logger import logger

def build_vector_database():
    
    """
    Load documents, split them into chunks, save the chunks,
    create a FAISS vector database, and persist it to disk.
    """

    try:
        # Step 1: Load documents
        logger.info("Loading documents...")
        documents = load_documents_from_folder(DOCS_PATH)
        logger.info(f"Loaded {len(documents)} documents.")

        # Step 2: Split into chunks
        logger.info("Splitting documents...")
        chunks = split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks.")

        # Save chunks for BM25 / hybrid retrieval.
        logger.info("Saving chunks...")
        save_chunks(chunks, CHUNKS_PATH)

        # Step 3: Create vector store
        logger.info("Creating vector database...")
        vectorstore = create_vectorstore(chunks)

        # Step 4: Save vector DB
        logger.info("Saving vector database...")
        save_vectorstore(vectorstore)

        logger.info("Vector database created successfully.")
        return vectorstore

    except Exception:
        logger.exception("Failed to build vector database.")
        raise

