import os
import time
from langchain.prompts import FewShotPromptTemplate,PromptTemplate
from langchain_openai import AzureChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from esql_utils import gen_token
import pyperclip
import shutil

#c8dd620e94045298943ccf83604982
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
token = gen_token()

#Function to load multiple fewshot examples from example folder

def load_few_shot_examples(example_folder):
    few_shot_examples = []
    # look for set of incoming, outgoing and esql code files
    for filename in os.listdir(example_folder):
        if filename.endswith(('.esql')):
            base_name = filename.split("_")[0]
            incoming_file_path = os.path.join(example_folder, f"{base_name}_incoming.json")
            outgoing_file_path = os.path.join(example_folder, f"{base_name}_outgoing.json")
            esql_file_path = os.path.join(example_folder, f"{base_name}_esqlcode.esql")

            if os.path.exists(incoming_file_path) and os.path.exists(outgoing_file_path):
                with open(incoming_file_path, 'r') as incoming_file:
                    incoming_msg = incoming_file.read()
                    incoming_msg = incoming_msg.replace("{", "{{").replace("}", "}}")

                with open(outgoing_file_path, 'r') as outgoing_file:
                    outgoing_msg = outgoing_file.read()
                    outgoing_msg = outgoing_msg.replace("{", "{{").replace("}", "}}")

                with open(esql_file_path, 'r') as esql_file:
                    esql_msg = esql_file.read()
                # add to few_shot_examples    
                few_shot_examples.append({
                    "incoming": incoming_msg, 
                    "outgoing": outgoing_msg,
                    "esqlmsg": esql_msg
                })
    return few_shot_examples

 # function to generate ESQL code from incoming and outgoing messages using Azure Chat OpenAI
def generate_esql_code(model, incoming_file, outgoing_file, mapping_rules, best_practices, example_folder):
    if model != "Azure OpenAI":
        st.error("Model not supported")
        return ""

    # load few shot examples
    few_shot_examples = load_few_shot_examples(example_folder)

    prompt_file = "devprompt.txt"

    with open(os.path.join("prompts", prompt_file), 'r') as file:
        prompt_msg = file.read()

    incoming_msg = incoming_file
    outgoing_msg = outgoing_file
    
    # Read incoming and outgoing files selected by user
    with open(os.path.join("tempDir", incoming_file), 'r') as f:
        incoming_msg = f.read()
        incoming_msg = incoming_msg.replace("{", "{{").replace("}", "}}")
        
    with open(os.path.join("tempDir", outgoing_file), 'r') as f:
        outgoing_msg = f.read()
        outgoing_msg = outgoing_msg.replace("{", "{{").replace("}", "}}")

    example_prompt = PromptTemplate(
        input_variables= ["incoming", "outgoing", "esql_msg "],
        template="""
        Incoming message: 
        {incoming}
        
        Outgoing message:
        {outgoing}
        
        ESQL code:
        {esql_msg}

        """
    )

    #construct the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=few_shot_examples,
        example_prompt=example_prompt,
        prefix=f"""
        {prompt_msg}
        """,
        suffix=f"""
        Incoming message:
        {incoming_msg}

        Outgoing message:
        {outgoing_msg}

        Mapping rules:
        {mapping_rules}

        Best practices:
        {best_practices}

        """,
        input_variables= ["incoming", "outgoing", "esql_msg "]
    )

    # generate the prompt based on the few shot prompt template
    prompt = few_shot_prompt_template.format(
        incoming_msg=incoming_msg,
        outgoing_msg=outgoing_msg,
        mapping_rules=mapping_rules,
        best_practices=best_practices
    )

    llm = AzureChatOpenAI(
        model='gpt-4o',
        azure_endpoint=AZURE_ENDPOINT,
        azure_ad_token=token,
        azure_deployment="exai-gpt-4o",
        api_version="2024-02-15-preview",
        temperature=0.6
        )

    print("prompt to LLM: ", prompt)  

    response = llm.invoke(prompt)

    #Extract the ESQL code from the response
    esql_code = response.content

    return esql_code

 #function to review the generated ESQL code
def review_esql_code(model, generated_code, mapping_rules, best_practices):
    if model != "Azure OpenAI":
        st.error("Model not supported")
        return ""

    # Review Prompt
    review_prompt = f"""
    You are an expert in IBM ACE and ESQL code review. Given the following ESQL code, please provide detailed review on the code quality, adherence to mapping rules, and best practices.:

    ESQL code:
    {generated_code}

    Mapping rules:
    {mapping_rules}

    Best practices:
    {best_practices}

    Review Comments:
    """

    llm_review = AzureChatOpenAI(
        model='gpt-4o',
        azure_endpoint=AZURE_ENDPOINT,
        azure_ad_token=token,
        azure_deployment="exai-gpt-4o",
        api_version="2024-02-15-preview",
        temperature=0.6
    )

    response_review = llm_review.invoke(review_prompt)
    review_feedback = response_review.content
    return review_feedback

#function to download the generated ESQL code to a file 
def download_esql_code(esql_code, file_path):
    with open(file_path, 'w') as f:
        if esql_code:
            esql_code_only = esql_code.split("'''esql")[1].split("'''")[0]
            f.write(esql_code_only)
        f.close()

#function to copy the generated ESQL code to clipboard
def esql2clipboard(esql_code):
    esql_code_only = esql_code
    pyperclip.copy(esql_code_only)
    print("ESQL code copied to clipboard:", pyperclip.paste())
    #st.success("ESQL code copied to clipboard")    

#fucntion to download feedback
def download_feedback(userfeed, file_path):
    with open(file_path, 'w') as f:
        if userfeed:
            f.write(userfeed)
        f.close()

# Main Streamlit App UI Logic
def main():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #737CA9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("**:blue[ESQL Code Generation App]**")

    #Sidebar for model selection and input data
    st.sidebar.header("Configuration")

    llm_model = st.sidebar.selectbox("Select Model", ["Azure OpenAI"])

    incoming_file = st.sidebar.file_uploader("Upload Incoming Message File", type=None)
    outgoing_file = st.sidebar.file_uploader("Upload Outgoing Message File", type=None)

    #Save files to tempDir
    if incoming_file is not None:
        file_details = {"FileName":incoming_file.name,"FileType":incoming_file.type,"FileSize":incoming_file.size}
        with open(os.path.join("tempDir", incoming_file.name), 'wb') as f:
            f.write(incoming_file.getbuffer())
        f.close()
        st.success("Incoming message file saved successfully")

    if outgoing_file is not None:
        file_details = {"FileName":outgoing_file.name,"FileType":outgoing_file.type,"FileSize":outgoing_file.size}
        with open(os.path.join("tempDir", outgoing_file.name), 'wb') as f:
            f.write(outgoing_file.getbuffer())
        f.close()
        st.success("Outgoing message file saved successfully")    

    #end file save
    # 
    data_folder = "./data"

    mapping_rules = st.sidebar.text_area("Enter Mapping Rules") 
    best_practices = st.sidebar.text_area("Enter Best Practices")

    bpfile = "./bestpractice/best_practices.txt"   

    if best_practices:
        try:
            with open(bpfile, 'r') as f:
                best_practices = best_practices + "\n" + f.read()
        except FileNotFoundError:
            print("Best Practices File not found")  

    else:
        try:
            with open(bpfile, 'r') as f:
                best_practices = f.read()
        except FileNotFoundError:
            print("Best Practices File not found") 

    #Button to generate ESQL code and review
    if st.sidebar.button("Generate ESQL Code"):
        #Generate ESQL code
        esql_code = generate_esql_code(llm_model, incoming_file.name, outgoing_file.name, mapping_rules, best_practices, "examples")
        st.session_state.generated_code = esql_code
        ifn = incoming_file.name.__str__()
        ofn = outgoing_file.name.__str__()
        
        shutil.move(f"tempDir/{ifn}", "Archive/"+ifn+time.strftime("%Y%m%d-%H%M%S"))
        shutil.move(f"tempDir/{ofn}", "Archive/"+ofn+time.strftime("%Y%m%d-%H%M%S"))

        if os.path.exists("tempDir/"+ifn):
            shutil.remove("tempDir/"+ifn)

        if os.path.exists("tempDir/"+ofn):
            shutil.remove("tempDir/"+ofn)

    if st.sidebar.button("Review ESQL Code"):
        review_feedback = review_esql_code(llm_model, st.session_state.get("generated_code", ""), mapping_rules, best_practices) 
        st.session_state.review_feedback = review_feedback

    #Main Section to display ESQL code and review feedback
    st.header("")               
    generated_code = st.text_area("Generated ESQL Code", st.session_state.get("generated_code", ""), height=500)

    if generated_code:
        if st.button("Download ESQL Code"):
            download_esql_code(generated_code, 'result/esqlcode_' + time.strftime("%Y%m%d-%H%M%S") + '.esql')
            st.success("ESQL code downloaded successfully")

        if st.button("Copy ESQL Code to Clipboard"):
            esql2clipboard(generated_code)
            st.success("ESQL code copied to clipboard")

    user_feedback = st.text_area("Provide Feedback", st.session_state.get("user_feedback", ""), height=500)     

    if "review_feedback" in st.session_state:
        st.header("Code Review Feedback")
        st.text_area("Review Feedback", st.session_state.get("review_feedback", ""), height=500)

    if st.button("Save Feedback"):
        download_feedback(user_feedback, 'result/feedback_' + time.strftime("%Y%m%d-%H%M%S") + '.txt')  
        st.success("Feedback saved successfully")

if __name__ == "__main__":
    main()
