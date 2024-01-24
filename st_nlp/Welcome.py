import streamlit as st
from cache_func import punkt

st.set_page_config(layout="wide")
punkt()

st.title("NLP Final Project")

st.markdown("Authors: Maryam DOLLET, DE TRENTINIAN Guillaume, ESILV")

st.markdown("#### Welcome to our NLP project")

st.markdown("We're delighted to present our work on Trustpilot, a company whose aim is to introduce customers to companies ranked and rated by other customers in various sectors.")
st.markdown("In the course of our work, we scraped the data from their site, cleaned it up and did a Topic Modeling.")
st.markdown("We performed embedding on the terms found in the company descriptions and clustered the embeddings with HDBSCAN to view the differend groups.")
st.markdown("We also used Doc2Vec for each company description to implement Semantic Search with keywords")
st.markdown("We tried out the new Chatbot function of Streamlit with the Semantic Search, to make the user interface more pleasant.")
st.markdown("Finally, our project ends with three Supervised Leaning tasks:")
st.markdown("- In the first part we tried to see what performance we would get by using Scikit-Learn Classification models to predict the categories")
st.markdown("- In the second part, we finetuned a BERT model with our company description data to predict the categories as well.")
st.markdown("- the third part, uses a Multi class classification model with LSTM to predict the scores given by users to brands based on the reviews they leave on the website.")

st.markdown("### Encountered Complications")
st.markdown("During our work we encountered a problem with the LFS file system and had reached the limit of the bandwidth. In consequence we could not continue to push our code and had to make another repository for the rest of our project.")

st.markdown("The first GitHub repository contains the first parts of the project: Web Scraping, Tokenization, Topic Modeling, Embedding and Semantic Search notebooks. The repository contains the data files used in those notebooks")
st.write("https://github.com/Maryam-Dollet/NLP_Project")

st.markdown("The second GitHub repository contains the rest of our work: Supervised Learning Notebooks up to date, BERT and Scikit-Learn. It is also the GitHub which is used to host the Streamlit Web Application.")
st.write("https://github.com/Maryam-Dollet/NLP-project-final")

st.write("In each page of the Streamlit App we will give you a link to the dedicated notebooks for you to see our coding process, in hope to make it easier.")