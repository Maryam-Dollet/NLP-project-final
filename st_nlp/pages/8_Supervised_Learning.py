import streamlit as st
import plotly.express as px
import pandas as pd
from cache_func import load_company_tagged, load_doc2vec, supervised_sklearn

d2v = load_doc2vec()

df = load_company_tagged()
category_count = df["category"].value_counts().reset_index()

st.header("Supervised Learning")

st.write("In this first section we try to predict using our trustpilot company description data to predict the category of a given description.")
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

st.subheader("Prediction of the number of stars using LSTM")

st.markdown("#### Selecting a part of the data:")
st.write("The first decision was to keep only the reviews of the first 1000 companies with the most reviews, to be able to get a good representation of the popularity of a company through its reviews.")
st.write("The language:")
st.write("The reviews being in majority in french (we had a few english or foreign languages reviews too), we decided to only keep the french ones for a better analysis of our dataset. The computational cost of translating all the reviews in english was too high for this option to be considered.")
st.write("For this function, we used wordpunct_tokenize and stopwords to create a new column in the dataset indicating the language having the most words in the review column.")

st.markdown("#### Balancing the dataset :")
st.write("We observed that we had an unbalanced dataset as follows :")
st.image("img/score_count.png")

st.write("We used Under-sampling and decided to keep 1114 random reviews of each class of notation ")

st.markdown("#### One-hot encoding:")
st.write("We one-hot encoded the number of stars of the ratings (our labels) as follows:")
st.write("1 => [1,0,0,0,0]")
st.write("4 => [0,0,0,1,0]")

st.markdown("#### Tokenizer and padding (/truncating) :")
st.write("For the input data to be a vector of the same length for each review")

st.markdown("#### The model:")
st.write("It is a sequential neural network with an embedding layer,a spatial dropout to help prevent overfitting, and an LSTM layer for sequence processing. It predicts scores for reviews through a softmax activation (for multi-class classification).")

st.markdown("#### Results of this model:")
st.write("We are not exactly sure about the reason why we have such bad results. It could be a lack of data, but we rather think there is a problem in the model we created or with the format of data used. However, we found this approach interesting enough to put it in this streamlit.")

