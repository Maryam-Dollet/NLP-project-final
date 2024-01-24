import streamlit as st
from cache_func import load_reviews_sample2, load_companies_translated, load_reviews_sample, load_company_tagged, get_freq, get_wordcloud, load_corpus, get_ngrams

df = load_reviews_sample2()
df2 = load_companies_translated()
df3 = load_reviews_sample()
df4 = load_company_tagged()

pipe = load_corpus()

st.title("Data Processing")

st.write("file name: data_cleaning.ipynb")
st.write("https://github.com/Maryam-Dollet/NLP_Project/blob/main/data_cleaning.ipynb")

st.subheader("First Cleaning")

st.write("Descriptions and comments contain emojis and other punctuation marks that render the data impure. That's why we have to clean them up.")
st.write("To do this, we used the clean-text library: https://pypi.org/project/clean-text/")

st.write()
st.write("After cleaning the columns company description and reviews we chose to translate them.")

st.dataframe(df[["company_name", "description", "score"]], use_container_width=True)

st.subheader("Translation: Company Description")

st.write("The library used to translate the texts: https://pypi.org/project/deep-translator/")
st.write("First we translated the french company description to english, we took the unique company names from the reviews data. In result, we had to translate a maximum of 12996 descriptions without taking into account the pages where the description is lacking. To translate without errors we had to drop the rows without description data")

st.dataframe(df2)

st.subheader("Translation: Company reviews")
st.write("There are in total 235503 reviews without taking into account the None values. There is also an imbalance in the number of reviews per category. To reduce the problem we randomly chose the same number of reviews per category")

st.dataframe(df3)

st.title("Data Tokenization")
st.write("file name: data_tokenizaation.ipynb")
st.write("https://github.com/Maryam-Dollet/NLP_Project/blob/main/data_tokenization.ipynb")

st.markdown("##### General Procedure")
st.write("- Remove punctuation")
st.write("- Lower the characters of tokens")
st.write("- Remove tokens whose length is below 2")
st.write("- Apply lemmatization of each token")

st.markdown("### Words that appear the most in all descriptions")

# st.dataframe(df4)

desc_token = df4["tokenized_desc"].apply(lambda x: x.split())
# st.dataframe(desc_token)

freq = get_freq(desc_token)

wc = get_wordcloud(freq)

st.image(wc.to_array(), width=800)

st.markdown("### Words that appear the most for each category")

category = st.selectbox("Select Category", df4["category"].unique())

filtered_df = df4[df4["category"] == category]

filtered_desc = filtered_df["tokenized_desc"].apply(lambda x: x.split())

cat_freq = get_freq(filtered_desc)

wc = get_wordcloud(cat_freq)

st.image(wc.to_array(), width=800)

st.markdown("### N-grams per category")

ng = st.selectbox("N-grams", [2, 3, 4])

n = get_ngrams(ng, df2, pipe)

d1 = dict(list(n.items())[len(n)//2:])
d2 = dict(list(n.items())[:len(n)//2])
col1, col2 = st.columns(2)

with col1:
    for x, y in d2.items():
        st.markdown(f"### {x}")
        st.markdown(y)

with col2:
    for x, y in d1.items():
        st.markdown(f"### {x}")
        st.markdown(y)
