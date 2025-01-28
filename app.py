from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
import time
import streamlit as st
from streamlit_chat import message

dico = {"Barbara J. Aehlert - ACLS Study Guide-Elsevier (2021)": 11, 
        "David Taggart, Yasir Abu-Omar - Core Concepts in Cardiac Surgery-Oxford University Press (2018)":11, 
        "Florian Falter, Albert C. Perrino, Jr., Robert A. Baker - Cardiopulmonary Bypass-Cambridge University Press (2022)":11,
        "Gregory S. Matte - Perfusion for Congenital Heart Surgery_ Notes on Cardiopulmonary Bypass for a Complex Patient Population-Wiley-Blackwell (2015)":14,
        "John Englert,  Clifton Marschel,  Kelly D. Hedlund -The Manual of Clinical Perfusion 3rd Edition" : 11,
        "Bryan V. Lich,  D. Mark Brown  -The Manual of Clinical Perfusion 2nd Edition": 7,
        "Karen Whalen, Carinda Feild, Rajan Radhakrishnan - Lippincott Illustrated Reviews_ Pharmacology-Wolters Kluwer (2019)" : 13,
        "Graeme MacLaren, Daniel Brodie, Roberto Lorusso, Giles Peek, Ravi Thiagarajan, Leen Vercaemst - Extracorporeal Life Support-The ELSO Red Book, 6e-Extracorporeal Life Support Organization (2022)" : 35,
        "Mulroney, Susan E._Myers, Adam K._Netter, Frank Henry - Netter's essential physiology-Elsevier (2016)" : 20}

load_dotenv()
api_key_1 = "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"
api_key_2 = "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"

# Streamed response emulator
def response_generator(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.05)



embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

db = FAISS.load_local("Question_Setting", embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 20})

llm_1 = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key = api_key_1 )

llm_2 = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key = api_key_2)

system_prompt = (
    """
    You are an AI designed to create multiple-choice exam questions based on a given context. Your task is to generate questions that are relevant to the specified categories and subcategories. Each question must:
    Include four answer options (labeled A, B, C, and D).
    Be a mix of 1st order (recall and comprehension) and 2nd order (application and analysis) types of questions.
    Ensure that one of the answer options is correct.
    Randomly position the correct answer among the four options. The correct answer should always be bolded.
    Align the content of the question with the specified category and subcategory.

    Categories:
        A. Clinical Application
        B. Basic Science
    Subcategories:
        Anatomy and Physiology
        Pharmacology
        Pathology
        Laboratory Analysis
        Quality Assurance
        Devices and Equipment
        Clinical Management
        Special Patient Groups
        Special Procedures and Techniques
        Catastrophic Events and Device Failure
        Patient Monitoring

    User Input Format: “Create [number] [Category] questions in the category of [Subcategory].”

    Example Input: “Create 4 Clinical Application questions in the category of Patient Monitoring.”

    Response Format: [Question Text]
    A. [Option]
    B. [Option]
    C. [Option]
    D. [Option]

    Note:

    Ensure the questions are balanced in terms of difficulty (both easy and challenging).
    Ensure the questions are closely tied to the specified category and subcategory.""" 
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)



def response_generate(query):
    try:
        question_answer_chain = create_stuff_documents_chain(llm_1, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        result = rag_chain.invoke({"input": query})
        result_answer = result['answer']
    except:
        question_answer_chain = create_stuff_documents_chain(llm_2, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        result = rag_chain.invoke({"input": query})
        result_answer = result['answer']

    formatted_response = f"{result_answer}\n\nRelevant Sources:\n"

    for i in range(3):
        src = result['context'][i].metadata['source'].split('/')[-1].strip(".pdf")

        if src == "Cardiopulmonary Bypass and Mechanical Support- Principles and Practice":
            src = "Glenn P. Gravlee, Richard F. Davis, John W. Hammon, Barry D. Kussman -\
                Cardiopulmonary Bypass and Mechanical Support- Principles and Practice"

        elif src == "- Extracorporeal Life Support-The ELSO Red Book, 6e-Extracorporeal Life Support Organization (2022)":
            src = "Graeme MacLaren, Daniel Brodie, Roberto Lorusso, Giles Peek, Ravi Thiagarajan, Leen Vercaemst - Extracorporeal Life Support-The ELSO Red Book, 6e-Extracorporeal Life Support Organization (2022)"

        page = result['context'][i].metadata['page']
        page = int(page)+1
        if src in dico:
            page = page - dico[src]
        page = str(page)
        src = f"*{src}*"
        formatted_response += f"- Source: {src}, Page: {page}\n"

    return formatted_response, result_answer

def  main():
    st.set_page_config(layout="wide")

    # if "history" not in st.session_state:
    #     st.session_state.history = "" 
    
    
    st.markdown("<h1 style='text-align: center; color: navy;'> Perfusion Question Bot </h1>", unsafe_allow_html=True)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Question?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Update sales history in session_state
        # st.session_state.history += "\nUser: " + prompt

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            formatted_response, result_answer = response_generate(prompt)
            output = formatted_response.lstrip("\n")
            response = st.write_stream(response_generator(output))
            

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Update sales history with assistant response
        # st.session_state.history += "\nAI: " + result_answer

from streamlit.web import cli as stcli
from streamlit import runtime
import sys

if __name__ == '__main__':
    main()      