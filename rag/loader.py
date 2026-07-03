import os
import fitz
import re
import hashlib
from langchain_core.documents import Document
from typing import List
import logging
logger = logging.getLogger(__name__)


def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Load PDF documents from a folder.

    Each PDF page is converted into a LangChain Document with
    normalized text and metadata.
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
                                "page": page_number + 1,          # human-friendly numbering 
                                "doc_id": doc_id
                                }
                            )
                    )
        except Exception as e:
            logger.error(f"Failed to process {file}: {e}")

    
    logger.info(f"Loaded {len(documents)} pages from {folder_path}")
    return documents