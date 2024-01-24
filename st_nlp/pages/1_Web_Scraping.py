import streamlit as st
import pandas as pd
from cache_func import load_companies_translated, load_reviews_sample, load_category

df = load_companies_translated() 
df2 = load_reviews_sample()

# st.dataframe(df)

st.title("Data Gathering: Web Scraping French Companies on Trustpilot")

st.write("files: scrap_trust_pilot.ipynb and scrap_trust_pilot_2.ipynb")
st.write("https://github.com/Maryam-Dollet/NLP_Project/blob/main/scrap_trust_pilot.ipynb")
st.write("https://github.com/Maryam-Dollet/NLP_Project/blob/main/scrap_trust_pilot_2.ipynb")

st.write("Using Beautiful Soup to scrap the website first we extracted the links to the categories listed in this link:")

st.write("https://fr.trustpilot.com/categories")

st.markdown("#### Here are the categories extracted")
a = pd.concat([pd.DataFrame(df["category"].unique(), columns=["category"]), load_category()], axis=1)

st.dataframe(a, use_container_width=True, column_config={"url":st.column_config.LinkColumn("App URL")})

st.write("Next, we used the links and number of pages extracted to extract all the links of each company as well as the number of pages of the reviews")

st.write("Then, we extracted the data on each company site using the links extracted previously. In the end we had 235503 rows of reviews. For this we had to set checkpoint in order to not lose data in case the program crashes. So, for each tenth iteration, we saved the extracted data and if we reexecture the program by indicating the right iteration, the program continues. This process took time, because we had to set up a sleep(2) because the website will strike us for requesting too much and too fast. This made the extraction take a lot of time.")

st.markdown("#### Sample of the reviews we extracted")

st.dataframe(df2.iloc[:, :-2])
