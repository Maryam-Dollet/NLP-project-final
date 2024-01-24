import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict  # For word frequency
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models import LdaModel
from gensim import corpora
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from umap import UMAP
from hdbscan import HDBSCAN
import pyLDAvis
import pyLDAvis.gensim
from wordcloud import WordCloud

# Load Data
@st.cache_data
def load_companies_translated():
    return pd.read_csv("data/company_desc_translated.csv", sep=";").dropna(ignore_index=True)


@st.cache_data
def load_reviews_sample():
    return pd.read_csv("data/reviews_sample_translated.csv", sep=";").dropna(ignore_index=True)


@st.cache_data
def load_category():
    return pd.read_csv("data/category_data.csv", sep=";").dropna(ignore_index=True)


@st.cache_data
def load_reviews_sample2():
    return pd.read_csv("data/reviews_sample.csv")

# Embeddings
@st.cache_data
def load_model(path: str):
    model = Word2Vec.load(path)
    return model


def get_similarity(model, token):
    return model.wv.most_similar(positive=[token])


@st.cache_data
def get_PCA(_model):
    labels = list(_model.wv.key_to_index.keys())
    vectors = _model.wv[_model.wv.key_to_index.keys()]

    pca = PCA(n_components=3)
    result = pca.fit_transform(vectors)

    result_df = pd.DataFrame(result, columns=["x", "y", "z"])
    result_df["word"] = labels
    return result_df


@st.cache_data
def get_TSNE(_model):
    labels = list(_model.wv.key_to_index.keys())
    vectors = _model.wv[_model.wv.key_to_index.keys()]

    tsne = TSNE(n_components=3, verbose=1,n_iter=1000,random_state=1)
    tsne_results = tsne.fit_transform(vectors)

    result_df = pd.DataFrame(tsne_results, columns=["x", "y", "z"])
    result_df["word"] = labels

    return result_df


@st.cache_data
def get_UMAP(_model):
    labels = list(_model.wv.key_to_index.keys())
    vectors = _model.wv[_model.wv.key_to_index.keys()]

    umap_3d = UMAP(n_components=3, init='random', random_state=0)
    proj_3d = umap_3d.fit_transform(vectors)

    result_df = pd.DataFrame(proj_3d, columns=["x", "y", "z"])
    result_df["word"] = labels

    return result_df

# Clustering
def hdbscan_cluster(df):
    clusterable_embedding = list(df[["x", "y", "z"]].values)
    labels = HDBSCAN(min_samples=10,min_cluster_size=20,).fit_predict(clusterable_embedding)
    df["category"] = labels
    df["category"] = df["category"].astype(str)
    return df

# Semantic Search functions
@st.cache_data
def load_company_tagged():
    return pd.read_csv("data/company_tagged.csv", sep=";")


@st.cache_data
def load_doc2vec():
    return Doc2Vec.load("models/d2v.model")


def find_similar_doc(_model, sentence: str, company_df):
    test_data = word_tokenize(sentence.lower())
    v1 = _model.infer_vector(test_data)

    sims = _model.dv.most_similar([v1])
    best_match = [x[0] for x in sims]

    return best_match

#Topic Modeling
@st.cache_data
def load_corpus():
    with open("data/pipe7.json", 'r') as f:
        data = json.load(f)
    return data


@st.cache_data
def LDA(corpus, nb_topics):
    dictionary = corpora.Dictionary(corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus]

    # LSA model
    lda = LdaModel(doc_term_matrix, num_topics=nb_topics, id2word = dictionary)

    vis = pyLDAvis.gensim.prepare(lda, doc_term_matrix, dictionary)

    html_str = pyLDAvis.prepared_data_to_html(vis)

    n_words = 10

    topic_words = pd.DataFrame({})

    for i, topic in enumerate(lda.get_topics()):
        top_feature_ids = topic.argsort()[-n_words:][::-1]
        feature_values = topic[top_feature_ids]
        words = [dictionary[id] for id in top_feature_ids]
        topic_df = pd.DataFrame({'value': feature_values, 'word': words, 'topic': i})
        topic_words = pd.concat([topic_words, topic_df], ignore_index=True)

    # LSA model
    return lda, html_str, topic_words


@st.cache_data
def get_freq(desc_token):
    freq = {}
    for desc in desc_token:
        for item in desc:
            if (item in freq):
                freq[item] += 1
            else:
                freq[item] = 1
    return freq


@st.cache_data
def get_wordcloud(freq):
    return WordCloud().fit_words(freq)


# we create a function that takes the number of words we want in the n-grams as an argument, doing the same thing than before
@st.cache_data
def get_ngrams(n, company_df, pipe7):
    company_id = 0
    company_ngram_freq_dict = dict()
    categories = list(company_df["category"].unique())
    for company in pipe7:
        freq_dict = defaultdict(int)

        for token in ngrams(company, n): # Count token frequency in each company description
            freq_dict[token] += 1
        company_ngram_freq_dict[company_df["company_name"][company_id]] = freq_dict # Add company name as key to dict and the frequency dictionnary as value

        company_id += 1

    dict_cat = {}
    for category in categories:
        # print(category + " :")
        # we merge all the dictionaries of the companies in the same category
        merged_dict = defaultdict(int)
        for company in company_df[company_df["category"] == category]["company_name"]:
            for token in company_ngram_freq_dict[company]:
                merged_dict[token] += company_ngram_freq_dict[company][token]
        # we print the 5 most frequent bigrams in the category
        # print(sorted(merged_dict, key=merged_dict.get, reverse=True)[:5])
        dict_cat[category] = sorted(merged_dict, key=merged_dict.get, reverse=True)[:5]

    return dict_cat


@st.cache_data
def get_UMAP_d2v(_model, df):
    tags = df["tag"]
    labels = df["company_name"]
    category = df["category"]
    vectors = [_model.dv[tag] for tag in tags]
    umap_3d = UMAP(n_components=3, init='random', random_state=0)
    proj_3d = umap_3d.fit_transform(vectors)


    result_df_umap = pd.DataFrame(proj_3d, columns=["x", "y", "z"])
    result_df_umap["word"] = labels
    result_df_umap["cat"] = category

    return result_df_umap

# Clustering
def hdbscan_cluster_2(df):
    clusterable_embedding = list(df[["x", "y", "z"]].values)
    labels = HDBSCAN(min_samples=20,min_cluster_size=50,).fit_predict(clusterable_embedding)
    df["category"] = labels
    df["category"] = df["category"].astype(str)
    return df

# For Whatever reason need to install punkt for chatbot
@st.cache_data
def punkt():
    nltk.download('punkt')


@st.cache_data
def supervised_sklearn(vectors_df):
    df_shuffled = vectors_df.sample(frac = 1)

    y = df_shuffled["category"]
    X = df_shuffled.drop(columns=["category"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

    model_dict = {}

    model_dict.update(SVM_train(X_train, X_test, y_train, y_test))
    model_dict.update(GaussianNB_train(X_train, X_test, y_train, y_test))
    model_dict.update(SGDClassifier_train(X_train, X_test, y_train, y_test))
    model_dict.update(KNeighborsClassifier_train(X_train, X_test, y_train, y_test))
    model_dict.update(DecisionTreeClassifier_train(X_train, X_test, y_train, y_test))
    model_dict.update(RandomForestClassifier_train(X_train, X_test, y_train, y_test))
    model_dict.update(GradientBoostingClassifier_train(X_train, X_test, y_train, y_test))
    model_dict.update(LGBMClassifier_train(X_train, X_test, y_train, y_test))

    return model_dict


@st.cache_data
def SVM_train(X_train, X_test, y_train, y_test):
    svm = SVC(kernel = 'linear', random_state = 0)
    svm.fit(X_train, y_train)
    #Prediction sur le Test set
    y_pred = svm.predict(X_test)

    # st.markdown("SVM")
    a=classification_report(y_test, y_pred, output_dict=True)
    df_a = pd.DataFrame(a).T
    return {"SVM": df_a}


@st.cache_data
def GaussianNB_train(X_train, X_test, y_train, y_test):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    a=classification_report(y_test, y_pred, output_dict=True)
    df_a = pd.DataFrame(a).T
    return {"GaussianNB": df_a}


@st.cache_data
def SGDClassifier_train(X_train, X_test, y_train, y_test):
    clf = SGDClassifier(loss="log_loss", penalty="l2")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    a=classification_report(y_test, y_pred, output_dict=True)
    df_a = pd.DataFrame(a).T
    return {"SGDClassifier": df_a}


@st.cache_data
def KNeighborsClassifier_train(X_train, X_test, y_train, y_test):
    knn_clf=KNeighborsClassifier(n_neighbors=150)
    knn_clf.fit(X_train,y_train)
    y_pred=knn_clf.predict(X_test)
    a=classification_report(y_test, y_pred, output_dict=True)

    df_a = pd.DataFrame(a).T
    return {"KNeighborsClassifier": df_a}


@st.cache_data
def DecisionTreeClassifier_train(X_train, X_test, y_train, y_test):
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    a=classification_report(y_test, y_pred, output_dict=True)
    df_a = pd.DataFrame(a).T
    return {"DecisionTreeClassifier": df_a}


@st.cache_data
def RandomForestClassifier_train(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(max_depth=13, n_estimators=393)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    a=classification_report(y_test, y_pred, output_dict=True)
    df_a = pd.DataFrame(a).T
    return {"RandomForestClassifier": df_a}


@st.cache_data
def GradientBoostingClassifier_train(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,  max_depth=1, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    a=classification_report(y_test, y_pred, output_dict=True)
    df_a = pd.DataFrame(a).T
    return {"GradientBoostingClassifier": df_a}


@st.cache_data
def LGBMClassifier_train(X_train, X_test, y_train, y_test):
    clf = LGBMClassifier()
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)

    a=classification_report(y_test, y_pred, output_dict=True)
    df_a = pd.DataFrame(a).T
    return {"LGBMClassifier": df_a}


def tsnescatterplot(_model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, _model.wv.__getitem__([word]), axis=0)

    # gets list of most similar words
    close_words = _model.wv.most_similar([word])

    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = _model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = _model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=19).fit_transform(arrays)

    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})

    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)
    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))

    return fig