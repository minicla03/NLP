'''
Raccomandatore di articoli simili
Descrizione: Utilizzando LSI (Latent Semantic Indexing), 
puoi creare un sistema che raccomanda articoli simili a un dato articolo.
Passaggi:
1-->Raccogli un dataset di articoli o descrizioni di prodotti (ad esempio, articoli di notizie o recensioni di libri).
2-->Usa LSI per ridurre la dimensionalità dei testi e ottenere una rappresentazione semantica latente degli articoli.
3-->Calcola la similarità tra gli articoli usando una misura come la cosine similarity.

Quando un utente consulta un articolo, puoi restituire gli articoli più simili basati sulla loro rappresentazione vettoriale.
'''

from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from gensim import corpora
from gensim import models
from gensim import similarities
import matplotlib.pyplot as plt
import numpy as np

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

def doc_tokeniker(docs):
    docs_tokenized=[]
    stop_word=set(stopwords.words("english"))
    sentence_tokenized=PunktSentenceTokenizer()
    lemmatizer=WordNetLemmatizer()

    for doc in docs:
        sentences=sentence_tokenized.tokenize(doc)
        words = [word.lower() for sentence in sentences for word in word_tokenize(sentence) if word.isalnum() and word not in stop_word]
        words_pos = pos_tag(words) 
        doc_lemm=[lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word,pos in words_pos]
        docs_tokenized.append(doc_lemm)
    return docs_tokenized


newsgroups = fetch_20newsgroups(subset='all')
docs=newsgroups.data[:20]
doc_tokenized=doc_tokeniker(docs)

dictionary=corpora.Dictionary(doc_tokenized)
bow_corpus = [dictionary.doc2bow(doc) for doc in doc_tokenized]

# Creazione del modello TF-IDF
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# Crea il modello LSI
lsi_model = models.LsiModel(corpus_tfidf, num_topics=5, id2word=dictionary)

topics = lsi_model.print_topics(num_topics=5, num_words=5)
sim = similarities.MatrixSimilarity(lsi_model[corpus_tfidf])

#clustering con k-mean
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # Ottieni la rappresentazione LSI del corpus, che è una lista di tuple
    lsi_corpus = lsi_model[corpus_tfidf]
    # Convertiamo la rappresentazione LSI in una matrice densa 2D
    dense_lsi_corpus = np.array([[tfidf for _, tfidf in doc] for doc in lsi_corpus])
    kmeans.fit(dense_lsi_corpus)
    inertia.append(kmeans.inertia_)

# Traccia l'inertia in funzione di k
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Inertia vs. Number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

ko=int(input("Insert the number of cluster: "))

kmeans = KMeans(n_clusters=ko, random_state=42)
kmeans.fit(dense_lsi_corpus)

# Etichette dei cluster per ogni documento
labels = kmeans.labels_
print(labels)

query=input("Insert the topic of the documen you want to read: ")

query_tokenized=doc_tokeniker([query])[0]
query_bow=dictionary.doc2bow(query_tokenized)
query_lsi=lsi_model[query_bow]

query_doc_sim = sim[query_lsi]
recommended_articles_index=query_doc_sim.argsort()[:-5]
for i in recommended_articles_index:
    print(f"Documento {i} (similarità {query_doc_sim[i]})")
print()
print()
print("K-MEANS")

# Converte la query LSI in una matrice densa
dense_query_lsi = np.array([[tfidf for _, tfidf in query_lsi]])
# Predici a quale cluster appartiene la query
cluster_label = kmeans.predict(dense_query_lsi)[0]
# Trova gli indici dei documenti che appartengono a quel cluster
cluster_indices = np.where(labels == cluster_label)[0]
# Restituisci i documenti appartenenti a quel cluster
recommended_docs = [newsgroups.data[i] for i in cluster_indices]

# Visualizza i documenti consigliati
print("\nDocumenti consigliati:")
for i, doc in enumerate(recommended_docs):
    print(f"Documento {i+1}: {doc[:300]}...")

