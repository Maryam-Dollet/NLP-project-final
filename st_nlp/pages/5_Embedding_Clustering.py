import streamlit as st
import plotly.express as px

from cache_func import load_model, get_UMAP, hdbscan_cluster

w2v = load_model('models/w2v_company_desc_model')
glove = load_model('models/glove_transfer')

result5 = get_UMAP(w2v)
result6 = get_UMAP(glove)

st.header("Using UMAP and HDBScan for Clustering")

hdbscan_df1 = hdbscan_cluster(result5)
hdbscan_df1['category'] = hdbscan_df1['category'].replace('-1' ,'outlier')
# st.dataframe(hdbscan_df1)
st.write(f"Number of Ouliers Detected: {len(hdbscan_df1[hdbscan_df1['category'] == 'outlier'])} out of {len(hdbscan_df1)}")

fig_3d = px.scatter_3d(
    hdbscan_df1, x="x", y="y", z="z", hover_name="word", color="category"
)
fig_3d.update_layout(width=1300 ,height=1000)
fig_3d.update_traces(marker_size=3)
fig_3d.update_traces(visible="legendonly", selector=lambda t: not t.name in hdbscan_df1["category"].unique()[1:])
st.plotly_chart(fig_3d)

hdbscan_df2 = hdbscan_cluster(result6)
# st.dataframe(hdbscan_df2)
hdbscan_df2['category'] = hdbscan_df2['category'].replace('-1' ,'outlier')
st.write(f"Number of Ouliers Detected: {len(hdbscan_df2[hdbscan_df2['category'] == 'outlier'])} out of {len(hdbscan_df2)}")
# st.write(hdbscan_df2["category"].unique())

fig_3d = px.scatter_3d(
    hdbscan_df2, x="x", y="y", z="z", hover_name="word", color="category"
)
fig_3d.update_layout(width=1300 ,height=1000)
fig_3d.update_traces(marker_size=3)
fig_3d.update_traces(visible="legendonly", selector=lambda t: not t.name in hdbscan_df2["category"].unique()[1:])
st.plotly_chart(fig_3d)