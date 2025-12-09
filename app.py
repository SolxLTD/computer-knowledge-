import streamlit as st
from chatbot import load_corpus, preprocess_sentences, get_best_answer

st.title("💻 Computer Knowledge Chatbot")

st.write("Ask any question about computers. The bot will answer from the text file you provided.")

sentences = load_corpus()
cleaned_sentences = preprocess_sentences(sentences)

user_input = st.text_input("Ask a question:")

if user_input:
    answer = get_best_answer(user_input, sentences, cleaned_sentences)
    st.write("### Answer:")
    st.write(answer)
