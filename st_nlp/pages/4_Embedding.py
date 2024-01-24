import streamlit as st
import plotly_express as px
import plotly.graph_objects as go

from cache_func import load_model, get_similarity, get_PCA, get_TSNE, get_UMAP

w2v = load_model('models/w2v_company_desc_model')
glove = load_model('models/glove_transfer')

st.title("Embeddings")
st.write("file: embedding.ipynb")
st.write("https://github.com/Maryam-Dollet/NLP_Project/blob/main/embedding.ipynb")
st.subheader("Word2Vec Training")

st.write("We used the gensim library to train a Word2Vec on our tokenized description tokens.")

st.subheader("GloVe Model Augmentation")

st.write("We used a pretrained GloVe model to add the weights to our trained Word2Vec to better it.")

st.subheader("Test the Similarity of the Two Models for Comparison")

request1 = st.text_input('token')
if st.button("Get similarity for Word2Vec"):
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.write("Word2Vec")
            st.write(get_similarity(w2v, request1))
        except:
            st.write("An error has occured, the word is not in the model")
    with col2:
        try:
            st.write("Word2Vec + GloVe")
            st.write(get_similarity(glove, request1))
        except:
            st.write("An error has occured, the word is not in the model")

st.subheader("PCA Visualisation")

st.markdown("#### PCA Word2Vec")

result_df1 = get_PCA(w2v)

fig_3d = px.scatter_3d(
    result_df1, x="x", y="y", z="z", hover_name="word"
)
fig_3d.update_layout(width=1300 ,height=1000)
fig_3d.update_traces(marker_size=2)
st.plotly_chart(fig_3d)

st.markdown("#### PCA Augmented Model")

result_df2 = get_PCA(glove)

fig_3d = px.scatter_3d(
    result_df2, x="x", y="y", z="z", hover_name="word"
)
fig_3d.update_layout(width=1300 ,height=1000)
fig_3d.update_traces(marker_size=2)
st.plotly_chart(fig_3d)

st.subheader("TSNE Visualisation")

st.markdown("#### TSNE Word2Vec")

result_df3 = get_TSNE(w2v)

fig_3d = px.scatter_3d(
    result_df3, x="x", y="y", z="z", hover_name="word"
)
fig_3d.update_layout(width=1300 ,height=1000)
fig_3d.update_traces(marker_size=2)
st.plotly_chart(fig_3d)

st.markdown("#### TSNE Augmented Model")

result_df4 = get_TSNE(glove)

fig_3d = px.scatter_3d(
    result_df4, x="x", y="y", z="z", hover_name="word"
)
fig_3d.update_layout(width=1300 ,height=1000)
fig_3d.update_traces(marker_size=2)
st.plotly_chart(fig_3d)

st.subheader("UMAP")

st.markdown("#### Word2Vec")

result5 = get_UMAP(w2v)
# st.dataframe(result5)

fig_3d = px.scatter_3d(
    result5, x="x", y="y", z="z", hover_name="word"
)
fig_3d.update_layout(width=1300 ,height=1000)
fig_3d.update_traces(marker_size=2)
st.plotly_chart(fig_3d)

st.markdown("#### Augmented Model")

result6 = get_UMAP(glove)

fig_3d = px.scatter_3d(
    result6, x="x", y="y", z="z", hover_name="word"
)
fig_3d.update_layout(width=1300 ,height=1000)
fig_3d.update_traces(marker_size=2)

st.plotly_chart(fig_3d)

st.subheader("Tensorboard")

st.write("We loaded the tsv metadata and vector files in: https://projector.tensorflow.org/")
st.write("We used UMAP to visualise the closest points of each model")

st.markdown("#### Word2Vec")

st.image("img/w2v_tensorboard.png")

st.markdown("#### Augmented Model")

st.image("img/glove_tensorboard.png")

st.subheader("Analysis")