import streamlit as st
from langchain_utils import get_response, create_vector_db, get_qa_chain

st.set_page_config(page_title="FAQ Bot")

st.title("FAQBot")

# Button to create the knowledge base
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

# User input for the question
question = st.text_input("Question: ")

if question:
    # Get the response
    answer = get_response(question=question)

    # Display the answer
    st.header("Answer: ")
    st.write(answer)
