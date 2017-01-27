from __future__ import print_function
from collections import Counter
from nltk.corpus import stopwords
import re
import codecs
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import scipy
from scipy.cluster.hierarchy import ward, dendrogram,linkage,fcluster,cophenet,distance
import matplotlib.pyplot as plt
from nltk.tag import pos_tag
import random
import numpy as np


def tokenize_and_stem(raw_text):
    tokens=re.findall('[a-zA-Z]+',raw_text.lower())
    allwords_tokenize=set(tokens) - stopword
    return [stemmer.stem(t) for t in allwords_tokenize if len(t)>2]

def get_data():
    return ebooks
    #totalvocab_stemmed,totalvocab_tokenized,terms

def get_filename():
    return doc_name

#creating a set of stopwords provided by nltk
stopword=set(stopwords.words("english"))
raw_path=['C:\Users\Aishwarya Sadasivan\Dataset-1']
for path in raw_path:
    for root,dir,files in os.walk(path):
        print ("Files in: " + root[:])

totalvocab_tokenized=[]
totalvocab_stemmed=[]
#instatiating the class SnowballStemmer for stemming and getting root words
stemmer=SnowballStemmer("english")
ebook=""
ebooks=[]
doc_name=[]
#tokenization,removal of stopwords,stemming
for filename in files:
    with codecs.open(root[:]+'\\'+filename, "r",encoding='utf-8', errors='ignore') as file_name:
        text=file_name.read()

        ebook=ebook+"\n"+text
        ebooks.append(text)

        doc_name.append(filename)

        tokens=re.findall('[a-zA-Z]+',text.lower().strip())
        allwords_tokenize=set(tokens) - stopword

        totalvocab_stemmed.extend([stemmer.stem(t) for t in allwords_tokenize])
        totalvocab_tokenized.extend(allwords_tokenize)


    file_name.close()

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print (vocab_frame.head())

vocab_frequency=Counter(totalvocab_tokenized)
#print vocab_frequency.most_common(10)

tfidf_vectorizer = TfidfVectorizer(max_df=0.7,max_features=200,
                                 min_df=0.2, stop_words='english',
                                 tokenizer=tokenize_and_stem,ngram_range=(1,3),dtype='double')
tfidf_matrix= tfidf_vectorizer.fit_transform(ebooks) #fit the vectorizer to synopses
terms = tfidf_vectorizer.get_feature_names()

#print (terms)
#cosine distance
doc_sim =1-cosine_similarity(tfidf_matrix)
print (doc_sim)



# clustering using hierarchical clustering

linkage_matrix = linkage(doc_sim,method='centroid')
#assignments = fcluster(linkage_matrix,1,criterion='distance')


#assignments = fcluster(,4,'distance')

print(linkage_matrix)
c, coph_dists = cophenet(linkage_matrix, distance.pdist(doc_sim))
print (c)
assignments =fcluster(linkage_matrix, 4, 'maxclust')

cluster_doc = pd.DataFrame({'doc':doc_name , 'cluster':assignments})
print(cluster_doc)

cluster_doc.to_csv('doc_cluster.csv',sep='\t')

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="left", labels=doc_name);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('cosine_cluster_doc1.png', dpi=200) #save figure as ward_clusters

#word clustering
#word to word similarity
word_vector=tfidf_matrix.transpose()
word_vector=word_vector.A


#word_vector=(tfidf_matrix*tfidf_matrix.T).A
word_sim=1-cosine_similarity(word_vector)
print (word_sim)

linkage_matrix1 = linkage(word_sim ,method='average')
c, coph_dists = cophenet(linkage_matrix1, distance.pdist(word_sim))
print (c)
print (coph_dists)

#assignments = fcluster(linkage_matrix1,4,'distance')
word_assignments=fcluster(linkage_matrix1, 10, 'maxclust')
print(word_assignments)
cluster_word = pd.DataFrame({'word':terms , 'cluster':word_assignments})
print (cluster_word)
cluster_word.to_csv('word_cluster.csv', sep='\t')

fig, ax = plt.subplots(figsize=(15, 20)) # set sizeassignments = fcluster(linkage(distanceMatrix, method='complete'),4,'distance')

ax = dendrogram(linkage_matrix1, orientation="left",labels=terms,show_contracted=True);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('cosine_cluster_word1.png', dpi=200) #save figure as ward_clusters
