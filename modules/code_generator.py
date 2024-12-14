from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()

# Access the API key from the environment variable
api_key = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=api_key)


class LangChainGenaiLLM(LLM):
    """
    A custom wrapper for `genai.GenerativeModel` to make it compatible with LangChain.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self._model_name = model_name
        self._model = genai.GenerativeModel(model_name)

    @property
    def _llm_type(self) -> str:
        return "genai"

    def _call(self, prompt: str, stop: list = None) -> str:
        """
        Calls the generative model to get a response.

        Args:
            prompt (str): The input prompt.
            stop (list, optional): List of stop sequences. Defaults to None.

        Returns:
            str: The generated text.
        """
        response = self._model.generate_content(prompt)
        return response.text.strip()


class CodeGenerator:
    def __init__(self):
        """
        Initialize the Code Generator with the Gemini LLM and ChromaDB integration.
        """
        self.llm = LangChainGenaiLLM("gemini-1.5-flash")
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

    def generate_code(self, requirement: str) -> str:
        """
        Generate code based on the given requirement and retrieved documentation.

        Args:
            requirement (str): The user's programming requirement.

        Returns:
            str: Generated code.
        """
        try:
            # Mocking documentation retrieval
            documentation = "Sample relevant documentation from ChromaDB."

            # Format the prompt
            formatted_prompt = self.prompt.format(requirement=requirement, documentation=documentation)

            # Define the code generation tool
            tools = [
                Tool(
                    name="Generate Progress 4GL Code",
                    func=lambda prompt: self.llm.predict(prompt),  # Use the predict method of LangChainGenaiLLM
                    description="Generates Progress 4GL/ABL/OpenEdge code using the provided requirement and documentation.",
                )
            ]

            # Initialize the agent
            agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)
            response = agent.run(requirement)
            return response
        except Exception as e:
            print(f"Error generating code: {e}")
            return "Error generating code."


# Usage Example
if __name__ == "__main__":
    code_gen = CodeGenerator()
    requirement = "Create a Progress 4GL program that retrieves customer data from the database and displays it in a tabular format."
    generated_code = code_gen.generate_code(requirement)
    print("Generated Code:\n", generated_code)
