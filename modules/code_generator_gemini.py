import os
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # Replace with Gemini LLM wrapper
from langchain.chains import RetrievalQA
from .vector_store_utils import ChromaDBUtils
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Access the API key from the environment variable
api_key = os.environ.get("GEMINI_API_KEY")
import google.generativeai as genai


#EXPORT G_API_KEY="AIzaSyBQmEldQBkdJ9QDszc-NdFYvl_3VGAEsjs"

genai.configure(api_key="AIzaSyBQmEldQBkdJ9QDszc-NdFYvl_3VGAEsjs")


#model = genai.GenerativeModel("gemini-1.5-flash")
#response = model.generate_content("Write a short story about a magic backpack.")
#print(response.text)

class CodeGenerator:
    def __init__(self):
        """
        Initialize the Code Generator with the Gemini LLM and ChromaDB integration.
        """
        #self.llm = ChatGoogleGenerativeAI(model="gemini-flash-1.5", temperature=0, api_key=api_key)  # Use Gemini Flash 1.5 model
        self.llm = genai.GenerativeModel("gemini-1.5-flash")
        self.template = """
            You are an expert Progress 4GL/ABL/OpenEdge programmer.
            Use the documentation provided and the user's requirement to generate code.
            Ensure the code follows best practices and is well-documented.

            Requirement:
            {requirement}

            Relevant Documentation:
            {documentation}

            Output:
        """
        self.prompt = PromptTemplate(input_variables=["requirement", "documentation"], template=self.template)
        self.vector_utils = ChromaDBUtils()

    def retrieve_documentation(self, query: str) -> str:
        """
        Retrieve relevant documentation from ChromaDB for the given query.

        Args:
            query (str): The query to retrieve relevant documentation.

        Returns:
            str: Concatenated relevant documentation.
        """
        try:
            retrieval_chain = self.vector_utils.create_retrieval_chain()
            response = retrieval_chain.run(query)
            return response
        except Exception as e:
            print(f"Error retrieving documentation: {e}")
            return "No relevant documentation found."

    def generate_code(self, requirement: str) -> str:
        """
        Generate code based on the given requirement and retrieved documentation.

        Args:
            requirement (str): The user's programming requirement.

        Returns:
            str: Generated code.
        """
        try:
            # Retrieve relevant documentation
            documentation = self.retrieve_documentation(requirement)

            # Format the prompt
            formatted_prompt = self.prompt.format(requirement=requirement, documentation=documentation)

            # Define the code generation tool
            tools = [
                Tool(
                    name="Generate Progress 4GL Code",
                    func=self._code_generation_tool,
                    description="Generates Progress 4GL/ABL/OpenEdge code using the provided requirement and documentation.",
                )
            ]

            # Initialize the agent
            agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True)
            response = agent.run(requirement)
            return response

        except Exception as e:
            print(f"Error generating code: {e}")
            return "Error generating code."

    def _code_generation_tool(self, requirement: str) -> str:
        """
        Tool function to generate code using the LLM with the enriched prompt.

        Args:
            requirement (str): The user's programming requirement.

        Returns:
            str: Generated code.
        """
        try:
            # Retrieve relevant documentation for the requirement
            documentation = self.retrieve_documentation(requirement)

            # Format the prompt with requirement and documentation
            prompt = self.prompt.format(requirement=requirement, documentation=documentation)
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error in code generation tool: {e}")
            return "Error in code generation."

# Usage Example
if __name__ == "__main__":
    code_gen = CodeGenerator()
    requirement = "Create a Progress 4GL program that retrieves customer data from the database and displays it in a tabular format."
    generated_code = code_gen.generate_code(requirement)
    print("Generated Code:\n", generated_code)