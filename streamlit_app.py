import sys
import os
import streamlit as st
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Add the current directory to PYTHONPATH
sys.path.append(os.path.dirname(__file__))

# Import custom modules
from modules.file_loader import PDFLoader
from modules.code_generator import CodeGenerator
from modules.code_reviewer import CodeReviewer
from modules.vector_store_utils import ChromaDBUtils

# Initialize components
pdf_loader = PDFLoader()
vector_utils = ChromaDBUtils()
code_generator = CodeGenerator()
code_reviewer = CodeReviewer()

# Streamlit App Configuration
st.set_page_config(page_title="Progress 4GL GenAI Assistant", layout="wide")
st.title("Progress 4GL GenAI Assistant")

# Tab setup
tab1, tab2, tab3 = st.tabs(["📄 Upload Documentation", "🛠️ Generate Code", "🔍 Review Code"])

# Tab 1: Upload Documentation
with tab1:
    st.header("Upload Progress 4GL Documentation")
    uploaded_files = st.file_uploader(
        "Upload your Progress 4GL-related PDF files", type="pdf", accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            st.write(f"Processing file: {file.name}")
            try:
                pdf_loader.load_pdf_to_vector_store(file)
            except Exception as e:
                logging.error(f"Error processing file {file.name}: {e}")
                st.error(f"Failed to process file {file.name}: {e}")
            else:
                st.success(f"File {file.name} uploaded and added to vector store successfully!")

# Tab 2: Generate Code
with tab2:
    st.header("Generate Progress 4GL Code")
    requirement = st.text_area(
        "Enter your requirement (e.g., 'Fetch customer data from the database.')", height=200
    )
    if st.button("Generate Code"):
        if not requirement:
            st.error("Please enter a requirement to generate code.")
        else:
            with st.spinner("Generating code..."):
                try:
                    logging.debug(f"Generating code for requirement: {requirement}")
                    generated_code = code_generator.generate_code(requirement)
                    if not generated_code or generated_code.strip() == "":
                        raise ValueError("No response received from the LLM.")
                    st.code(generated_code, language="plaintext")
                    st.success("Code generated successfully!")
                except Exception as e:
                    logging.error(f"Error generating code: {e}")
                    st.error(f"Failed to generate code: {e}")

# Tab 3: Review Code
with tab3:
    st.header("Review Generated Code")
    existing_code = st.text_area(
        "Paste the generated code here for review.", height=200
    )
    if st.button("Review Code"):
        if not existing_code:
            st.error("Please provide code to review.")
        else:
            with st.spinner("Reviewing code..."):
                try:
                    logging.debug(f"Reviewing code: {existing_code}")
                    review_comments = code_reviewer.review_code(existing_code)
                    if not review_comments or review_comments.strip() == "":
                        raise ValueError("No response received from the code reviewer.")
                    st.text_area("Code Review Comments:", value=review_comments, height=300)
                    st.success("Code reviewed successfully!")
                except Exception as e:
                    logging.error(f"Error reviewing code: {e}")
                    st.error(f"Failed to review code: {e}")

# Footer
st.markdown(
    """
    ---
    **Progress 4GL GenAI Assistant** | Powered by LangChain, Streamlit, and Gemini Flash 1.5
    """
)
