import os
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI  # Replace with Gemini Flash 1.5 wrapper
from langchain_google_genai import ChatGoogleGenerativeAI  # Replace with Gemini LLM wrapper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

class CodeReviewer:
    def __init__(self):
        """
        Initialize the Code Reviewer with the Gemini LLM.
        """
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-1.5", temperature=0, api_key=api_key)  # Use Gemini Flash 1.5 model
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
            tools = [
                Tool(
                    name="Code Review Tool",
                    func=self._code_review_tool,
                    description="Reviews Progress 4GL/ABL/OpenEdge code for best practices, performance, and maintainability.",
                )
            ]

            agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True)
            response = agent.run(code)
            return response

        except Exception as e:
            print(f"Error reviewing code: {e}")
            return "Error reviewing code."

    def _code_review_tool(self, code: str) -> str:
        """
        Tool function to review the code using the LLM.

        Args:
            code (str): The code to review.

        Returns:
            str: Code review comments.
        """
        try:
            prompt = self.prompt.format(code=code)
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error in code review tool: {e}")
            return "Error in code review."

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
