import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
import tempfile
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize LLM (Gemini)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=G_API_KEY,temperature=0)

# Streamlit UI
st.title("🔍 Progress OpenEdge ABL Coding Assistant")
st.sidebar.header("Upload ABL Documentation")

uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_loader = PyPDFLoader(tmp_file.name)
            documents.extend(pdf_loader.load())

# **✅ Fix: Check if documents exist before processing**
if not documents:
    st.warning("⚠️ No documents loaded. Please upload a valid PDF.")
    st.stop()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# **✅ Fix: Check if chunks exist before embedding**
if not chunks:
    st.warning("⚠️ No text chunks created. The document may be empty or not readable.")
    st.stop()

# Generate embeddings
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
doc_texts = [chunk.page_content for chunk in chunks]  # Extract text from chunks
doc_embeddings = embedding_model.embed_documents(doc_texts)  # Generate embeddings

# **✅ Fix: Check if embeddings were generated**
if not doc_embeddings:
    st.error("❌ Embedding generation failed. Check if the document has text.")
    st.stop()

# Create FAISS index
embedding_dim = len(doc_embeddings[0])  # Get embedding size
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(doc_embeddings))

# Properly map document IDs
docstore = InMemoryDocstore()
index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

#for i, chunk in enumerate(chunks):
#    docstore[str(i)] = chunk  # Store document chunk
documents_dict = {str(i): chunk for i, chunk in enumerate(chunks)}
docstore.add(documents_dict)

vectorstore = FAISS(
    embedding_function=embedding_model, 
    index=index, 
    docstore=docstore, 
    index_to_docstore_id=index_to_docstore_id
)

# Set up retriever
retriever = vectorstore.as_retriever()

# Ensure chain is correctly initialized
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

# Query input
query = st.text_input("Ask about Progress OpenEdge ABL:")

if query:
    with st.spinner("Searching..."):
        response = qa_chain.invoke({"question": query, "chat_history": []})
    st.subheader("💡 Answer:")
    st.write(response["answer"])
