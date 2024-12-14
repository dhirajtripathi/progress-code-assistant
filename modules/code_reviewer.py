import os
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI  # Replace with Gemini Flash 1.5 wrapper
from langchain_google_genai import ChatGoogleGenerativeAI  # Replace with Gemini LLM wrapper
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.llms.base import LLM
import google.generativeai as genai


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


class CodeReviewer:
    def __init__(self):
        """
        Initialize the Code Reviewer with the Gemini LLM.
        """
        #self.llm = ChatGoogleGenerativeAI(model="gemini-flash-1.5", temperature=0, api_key=api_key)  # Use Gemini Flash 1.5 model
        self.llm = LangChainGenaiLLM("gemini-1.5-flash")
        self.template = """
        You are an expert in reviewing Progress 4GL/ABL/OpenEdge code.
        Analyze the provided code for best practices, performance issues, maintainability, and adherence to standards.
        Provide a detailed review with actionable suggestions and comments.

        Code to Review:
        {code}

        Review:
        """
        self.prompt = PromptTemplate(input_variables=["code"], template=self.template)

    def review_code(self, code: str) -> str:
        """
        Review the generated code using the LLM.

        Args:
            code (str): The code to review.

        Returns:
            str: Code review comments and suggestions.
        """
        try:
            # Format the prompt
            formatted_prompt = self.prompt.format(code=code)            
            tools = [
                Tool(
                    name="Code Review Tool",
                    func=lambda prompt: self.llm.predict(prompt),
                    description="Reviews Progress 4GL/ABL/OpenEdge code for best practices, performance, and maintainability.",
                )
            ]

            agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)
            response = agent.run(code)
            return response

        except Exception as e:
            print(f"Error reviewing code: {e}")
            return "Error reviewing code."

# Usage Example
if __name__ == "__main__":
    code_reviewer = CodeReviewer()
    generated_code = """
    /* Sample Progress 4GL Code */
    FOR EACH Customer NO-LOCK:
        DISPLAY Customer.CustomerID Customer.CustomerName Customer.Balance.
    END.
    """
    review = code_reviewer.review_code(generated_code)
    print("Code Review:\n", review)
