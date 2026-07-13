import os
import fitz
import re
import hashlib
from langchain_core.documents import Document
from utils.logger import logger

def load_documents_from_folder(
    folder_path: str
) -> list[Document]:

    """
    Load PDF documents from a folder.

    Each page is extracted, normalized, and converted into a
    LangChain Document with metadata for the source PDF,
    page number, and document identifier.

    Args:
        folder_path: Directory containing PDF documents.

    Returns:
        List of loaded LangChain documents.
    """

    documents = []
    
    for file in sorted(os.listdir(folder_path)):

        if not file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder_path, file)

        try:
            with fitz.open(pdf_path) as pdf:

                for page_number, page in enumerate(pdf):

                    text = page.get_text(sort=True)
                    # Normalize whitespace extracted from the PDF.
                    text = re.sub(r"\s+", " ", text).strip()

                    doc_id = hashlib.md5(
                        f"{file}_{page_number}_{text}".encode("utf-8")
                        ).hexdigest()

                    if not text:
                        logger.warning(
                            f"No text found in {file}, page {page_number + 1}"
                            )
                        continue

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file,
                                "page": page_number + 1,     # Store pages using 1-based numbering.
                                "doc_id": doc_id
                                }
                            )
                    )
        except Exception as e:
            logger.exception(f"Failed to process '{file}'.")

    
    logger.info(f"Loaded {len(documents)} pages from {folder_path}")
    return documents