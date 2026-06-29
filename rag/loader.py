import os
import fitz

from langchain_core.documents import Document


def load_documents_from_folder(folder_path):
    documents = []

    for file in os.listdir(folder_path):

        if not file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder_path, file)

        pdf = fitz.open(pdf_path)

        for page_number, page in enumerate(pdf):

            text = page.get_text(sort=True)

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file,
                        "page": page_number
                    }
                )
            )

    return documents