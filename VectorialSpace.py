from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 

def tokenizer_custom(doc):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    sentences = sent_tokenize(doc)
    words = [word.lower() for sentence in sentences for word in word_tokenize(sentence) 
             if word.lower() not in stop_words and word.isalnum()]  

    words_pos = pos_tag(words) 
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in words_pos]

    return " ".join(lemmatized_words) 

def cerca_documenti_simili(query, tf_idf, model_tfidf):
    
    query_tokenized = tokenizer_custom(query)
    query_tfidf = tf_idf.transform([query_tokenized]) 
    cosine_similarities = cosine_similarity(query_tfidf, model_tfidf)
    
    similarity_scores = cosine_similarities.flatten()
    sorted_indices = similarity_scores.argsort()[::-1]
    
    print(f"Risultati della ricerca per la query: '{query}':")
    for i in range(5):
        print(f"Documento {sorted_indices[i]} - Similarità: {similarity_scores[sorted_indices[i]]}")

categories = ['sci.space', 'rec.autos', 'talk.politics.mideast']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

documents = newsgroups.data[:100] 
documents_tokenized = [tokenizer_custom(doc) for doc in documents] 

#creazione del modello TF-IDF
tf_idf = TfidfVectorizer()
model_tfidf = tf_idf.fit_transform(documents_tokenized) 
feature_names = tf_idf.get_feature_names_out()

#matrice TF-IDF dei termini-documenti
df_tfidf = pd.DataFrame(model_tfidf.toarray(), columns=feature_names)

#calcolo della cos_sim fra i doc
cos_sim=cosine_similarity(model_tfidf)
query = "space technology and exploration"
cerca_documenti_simili(query, tf_idf, model_tfidf)

#c Clustering con KMeans
# Usa l'Elbow Method per determinare il numero ottimale di cluster
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(model_tfidf)
    inertia.append(kmeans.inertia_)

# Plot dell'Elbow Method
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Metodo Elbow per il numero ottimale di cluster")
plt.xlabel("Numero di cluster")
plt.ylabel("Inerzia")
plt.show()

# Dopo aver osservato il grafico, scegli il numero ottimale di cluster
optimal_k = 3  

# Applicazione del KMeans con il numero ottimale di cluster
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(model_tfidf)

# Visualizzazione della distribuzione dei documenti nei cluster
plt.hist(kmeans.labels_, bins=optimal_k)
plt.xlabel('Cluster')
plt.ylabel('Numero di Documenti')
plt.title('Distribuzione dei Documenti nei Cluster')
plt.show()

# Mostrare alcuni documenti da ogni cluster
for cluster_num in range(optimal_k):
    print(f"\nCluster {cluster_num}:")
    cluster_indices = np.where(kmeans.labels_ == cluster_num)[0]
    for idx in cluster_indices[:3]:  # Mostra solo i primi 3 documenti per cluster
        print(f"Documento {idx}: {documents[idx][:300]}...")  # Mostriamo solo i primi 300 caratteri per brevità

