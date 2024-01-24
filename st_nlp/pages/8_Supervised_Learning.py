import streamlit as st
import plotly.express as px
import pandas as pd
from cache_func import load_company_tagged, load_doc2vec, supervised_sklearn

d2v = load_doc2vec()

df = load_company_tagged()
category_count = df["category"].value_counts().reset_index()

st.header("Supervised Learning")

st.write("In this section we try to predict using our trustpilot company description data to predict the category of a given description.")
st.write("if we import the transformed data, we can see that there is a class imbalance:")

fig = px.bar(category_count, x="category", y="count", title="Number of Companies per Category", height=500, width=800)
st.plotly_chart(fig)

st.write("So we decides to take categories which have at least 400 company descriptions. These are the categories which answer the condition.")

category_list = list(category_count[category_count["count"] >= 400]["category"])
st.write(category_list)

company_sample = df[df["category"].isin(category_list)].dropna()
company_sample = company_sample.groupby('category').apply(lambda x: x.sample(n=406, random_state=42)).reset_index(drop=True)

tags = company_sample["tag"]
labels = company_sample["company_name"]
category = company_sample["category"]
vectors = [d2v.dv[tag] for tag in tags]

vectors_df = pd.DataFrame(vectors)
vectors_df["category"] = category

st.subheader("Classification Machine Learning Using Scikit-Learn")
st.write("file: supervised_sklearn.ipynb")
st.write("https://github.com/Maryam-Dollet/NLP-project-final/blob/main/BERT_classification.ipynb")

st.write("We tried out different Classification Models which were available in scikit-learn library to make a classification algorithm which uses the Doc2Vec Embedding introduced in the Semantic Search to classify company description in the existing categories")

models = supervised_sklearn(vectors_df)
keysList = list(models.keys())

selected_model = st.selectbox("Choose ML Classification Model", keysList)

st.dataframe(models[selected_model], height=600, use_container_width=True)


st.subheader("Category Classification using BERT Model")
st.write("file: BERT_classification.ipynb")
st.write("https://github.com/Maryam-Dollet/NLP-project-final/blob/main/supervised_sklearn.ipynb")

st.write("We tried to finetune a BERT model with Pytorch with our company description data to also predict the categories of a description. It took a lot of time with colab so we only trained it for one epoch which tool several hours.")
st.write("Here are the accuracies per category")

st.image("img/bert_results.png")

st.write("If we look closely there are categories which accuracy is quite good, for some like business_services or events_entertainment the results are bad, this can be due to the lack of variety of word usage which makes those categories not destinguishable from others.")
st.write("Another Hypothesis would be the quality of data that is present on Trustpilot. Indeed, there are companies which could be from several companies, but are only listed in one. For example, Shin Sekai a store which supplies japanese products like food but also leisure items, is categorized in food_beverages_tabacco.")
st.write("To improve this model we can try to train it for more that one epoch to see if the accuracy improves, or we can do a data augmentation for the other categories.")