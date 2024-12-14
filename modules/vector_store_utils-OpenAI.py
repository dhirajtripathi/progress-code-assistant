import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings  # Replace with Gemini embeddings if available
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ChromaDB storage path
CHROMADB_STORAGE_PATH = os.getenv("CHROMADB_STORAGE_PATH", "./vector_store/chromadb_storage")

class ChromaDBUtils:
    def __init__(self):
        """Initialize the ChromaDB vector store and embeddings."""
        self.embedding_model = OpenAIEmbeddings()  # Replace with Gemini embeddings if needed
        self.vector_store = Chroma(
            collection_name="progress_4gl_docs",
            embedding_function=self.embedding_model,
            persist_directory=CHROMADB_STORAGE_PATH,
        )

    def add_documents(self, documents):
        """
        Add a list of documents to the ChromaDB vector store.

        Args:
            documents (list): List of LangChain document objects.
        """
        try:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            print("Documents successfully added to the vector store.")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")

    def create_retrieval_chain(self):
        """
        Create a retrieval-based QA chain using the ChromaDB vector store.

        Returns:
            RetrievalQA: A LangChain RetrievalQA chain.
        """
        try:
            retriever = self.vector_store.as_retriever()
            retrieval_chain = RetrievalQA.from_chain_type(
                retriever=retriever, chain_type="stuff", return_source_documents=True
            )
            return retrieval_chain
        except Exception as e:
            print(f"Error creating retrieval chain: {e}")
            return None

    def query_vector_store(self, query: str):
        """
        Query the vector store using a LangChain retrieval-based agent.

        Args:
            query (str): The user's query.

        Returns:
            str: The agent's response.
        """
        try:
            retrieval_chain = self.create_retrieval_chain()

            if retrieval_chain is None:
                raise ValueError("Retrieval chain could not be initialized.")

            tools = [
                Tool(
                    name="Retrieve Document",
                    func=retrieval_chain.run,
                    description="Use this tool to retrieve information from the vector store.",
                )
            ]

            agent = initialize_agent(tools, llm=self.embedding_model, agent="zero-shot-react-description", verbose=True)
            response = agent.run(query)
            return response

        except Exception as e:
            print(f"Error querying vector store: {e}")
            return None

    def list_collections(self):
        """
        List all collections in the ChromaDB vector store.

        Returns:
            list: A list of collection names.
        """
        try:
            if hasattr(self.vector_store, "list_collections"):
                collections = self.vector_store.list_collections()
                return [col["name"] for col in collections]
            return []
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []

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
