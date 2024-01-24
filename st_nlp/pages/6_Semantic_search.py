import streamlit as st
import plotly.express as px
from cache_func import load_company_tagged, load_doc2vec, find_similar_doc, get_UMAP_d2v, hdbscan_cluster_2
df = load_company_tagged()
d2v = load_doc2vec()

st.title("Semantic Search Using Doc2Vec")
st.write("file: semantic_search2.ipynb")
st.write("https://github.com/Maryam-Dollet/NLP_Project/blob/main/semantic_search2.ipynb")

st.write("To be able to make a similarity search of documents with a sentence, we trained a Doc2Vec with each description of a company having a tag. This enables us to search the company that matches the best the request")

st.dataframe(df)

st.markdown("### Get Matching Documents")

sentence = st.text_input('Request to Search')
if st.button("Get similar doc:"):
    best_match = find_similar_doc(d2v, sentence, df)
    best_match = [int(x) for x in best_match]
    st.dataframe(df[df['tag'].isin(best_match[:5])])

st.markdown("### UMAP Visualization")

df_umap = get_UMAP_d2v(d2v, df)

st.dataframe(df_umap)

cluster_df = hdbscan_cluster_2(df_umap)

cluster_df['category'] = cluster_df['category'].replace('-1' ,'outlier')
st.write(f"Number of Ouliers Detected: {len(cluster_df[cluster_df['category'] == 'outlier'])} out of {len(cluster_df)}")

fig_3d = px.scatter_3d(
    cluster_df, x="x", y="y", z="z", hover_data=cluster_df[["word", "cat"]], color="category"
)
fig_3d.update_layout(width=1300 ,height=1000)
fig_3d.update_traces(marker_size=3)
fig_3d.update_traces(visible="legendonly", selector=lambda t: not t.name in cluster_df["category"].unique())
st.plotly_chart(fig_3d)

st.markdown("### Analysis of the UMAP DBSCAN Clustering:")
st.write("If we remove the outliers on the graph we can observe by looking at the points in the same cluster that there are different categories of companies which are close to one another. \
         This could mean that the data is not representative enough for the Doc2Vec to be more precise. We can also hypothesize that it is the lack of meaningful variety that could contribute to the limitation")