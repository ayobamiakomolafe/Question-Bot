__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAI, ChatOpenAI
from dotenv import load_dotenv
import os
import time
import streamlit as st
from streamlit_chat import message


os.environ["OPENAI_API_KEY"] = st.secrets["API_KEY"]

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

db = FAISS.load_local("Question_Setting", embeddings, allow_dangerous_deserialization=True)

PROMPT_TEMPLATE = """
You are an Assistant that Generates examination worthy multiple-choice question set based on the provided context, with the following specifications: 
    Include four answer options (A, B, C, and D) with a randomly positioned correct answer. 
    Always Ensure the correct answer is bolded. 
    Mix 1st-order (recall and comprehension) and 2nd-order (application and analysis) question types. 
    Balance difficulty levels between easy and challenging questions. 
    Ensure questions generated are not repeated.

    Question Text 
    A. [Option] 
    B. [Option]
    C. [Option]
    D. [Option].


{context}

---

{question} based on the above context but don't add details like 'Based on the above text or context' to questions.
"""


model = ChatOpenAI(model_name="gpt-4o", temperature=0.5, max_tokens = None)
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

def response_generate(query_text):
    results = db.similarity_search_with_relevance_scores(query_text, k=30)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = prompt_template.format(context=context_text, question=query_text)
    prompt = prompt.replace("Chapter", "")
    response_text = model.predict(prompt)
    return response_text


def  main():
    st.set_page_config(layout="wide")
        
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
        
        

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            formatted_response = response_generate(prompt)
            output = formatted_response.lstrip("\n")
            st.markdown(output)
            

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": output})
        

from streamlit.web import cli as stcli
from streamlit import runtime
import sys

if __name__ == '__main__':
    main()      
