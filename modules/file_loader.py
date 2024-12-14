import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .vector_store_utils import ChromaDBUtils

class PDFLoader:
    def __init__(self):
        """Initialize the PDF loader with the ChromaDB utility."""
        self.vector_utils = ChromaDBUtils()

    def load_pdf_to_vector_store(self, file_path: str):
        """
        Loads a single PDF file into the ChromaDB vector store.

        Args:
            file_path (str): Path to the PDF file.
        """
        try:
            # Load PDF and extract text
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Split text into chunks for better indexing
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)

            # Add documents to the vector store
            self.vector_utils.add_documents(docs)

            print(f"Successfully loaded and stored: {file_path}")

        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")

# Usage Example
if __name__ == "__main__":
    pdf_loader = PDFLoader()
    pdf_path = "example.pdf"  # Replace with your PDF file path

    if os.path.exists(pdf_path):
        pdf_loader.load_pdf_to_vector_store(pdf_path)
    else:
        print(f"File not found: {pdf_path}")
