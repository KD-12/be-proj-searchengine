 # raw text-> removal of stop words, tokenize and stem using nltk Snowball
# created tf-idf matrix using scikit. Feature names are extracted and used as labels in dendrogram
# created similarity matrix using cosine similarity
# Hierarchical clustering for documents and words
# created linkage matrix (using scipy cluster hierarchy) used to plot dendrogram and to create flat clusters using threshold values extracted from the get_cluster_classes function to convert the clusters to csv format.
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
from collections import defaultdict
def get_cluster_classes(den, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_classes ={}
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l

    return cluster_classes

def tokenize(raw_text):
    tokens=re.findall('[a-zA-Z]+',raw_text.lower())
    return set(tokens)-stopword

def tokenize_and_stem(raw_text):
    tokens=re.findall('[a-zA-Z]+',raw_text.lower())
    allwords_tokenize=set(tokens) - stopword
    return [stemmer.stem(t) for t in allwords_tokenize if len(t)>2]

def get_files(raw_path):
    files_extracted=[]
    root_extracted=[]
    for path in raw_path:
        for root,dir,files in os.walk(path):
            print ("Files in: " + root[:])
            root_extracted.append(root)
            files_extracted.extend(files)
    return files_extracted,root_extracted

def linkage_matrix_rep(sim_matrix):
    methods=['average','single','complete','weighted']
    c_final=0.0
    method_final=''
    final_linkage=linkage(sim_matrix)
    for method in methods:
        linkage_matrix = linkage(sim_matrix,method=method)
        c, coph_dists = cophenet(linkage_matrix, distance.pdist(sim_matrix))
        if c>c_final:
            c_final=c
            final_linkage=linkage_matrix
            method_final=method
            cd_final=coph_dists
    return c_final,method_final,final_linkage,cd_final




raw_path=['C:\Users\Aishwarya Sadasivan\Dataset-1']
files,roots=get_files(raw_path)
#creating a set of stopwords provided by nltk
stopword=set(stopwords.words("english"))
totalvocab_tokenized=[]
totalvocab_stemmed=[]
#instatiating the class SnowballStemmer for stemming and getting root words
stemmer=SnowballStemmer("english")
ebook=""
ebooks=[]
doc_name=[]
#tokenization,removal of stopwords,stemming
for root in roots:
    for filename in files:
        with codecs.open(root[:]+'\\'+filename, "r",encoding='utf-8', errors='ignore') as file_name:
            text=file_name.read()

            ebook=ebook+"\n"+text
            ebooks.append(text)

            doc_name.append(filename)
            allwords_tokenize=tokenize(text)

            totalvocab_stemmed.extend([stemmer.stem(t) for t in allwords_tokenize])
            totalvocab_tokenized.extend(allwords_tokenize)


        file_name.close()

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print (vocab_frame.head())

vocab_frequency=Counter(totalvocab_tokenized)
#print vocab_frequency.most_common(10)

tfidf_vectorizer = TfidfVectorizer(max_df=0.75,max_features=200,
                                 min_df=0.3, stop_words='english',
                                 tokenizer=tokenize_and_stem,ngram_range=(1,3),dtype='double')
tfidf_matrix= tfidf_vectorizer.fit_transform(ebooks) #fit the vectorizer to synopses
terms = tfidf_vectorizer.get_feature_names()

#print (terms)
#cosine distance
doc_sim =1-cosine_similarity(tfidf_matrix)
print (doc_sim)



# clustering using hierarchical clustering

doc_cophen,doc_method,doc_linkage_matrix,doc_cd = linkage_matrix_rep(doc_sim)
#assignments = fcluster(linkage_matrix,1,criterion='distance')


#assignments = fcluster(,4,'distance')

print(doc_linkage_matrix)
print (doc_cophen)
print(doc_method)
print(len(doc_cd))
fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(doc_linkage_matrix, orientation="left", labels=doc_name);

doc_classes=get_cluster_classes(ax)
thresh_doc=len(doc_classes)

assignments =fcluster(doc_linkage_matrix, thresh_doc, 'maxclust')
print (assignments)
cluster_doc = pd.DataFrame({'doc':doc_name , 'cluster':assignments})
print(cluster_doc)

cluster_doc.to_csv('doc_cluster.csv',sep='\t')

#print(ax)


plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('cosine_cluster_doc_test.png', dpi=200) #save figure as ward_clusters

#word clustering
#word to word similarity
word_vector=tfidf_matrix.transpose()
word_vector=word_vector.A


#word_vector=(tfidf_matrix*tfidf_matrix.T).A
word_sim=1-cosine_similarity(word_vector)
print (word_sim)

word_cophen,word_method,word_linkage_matrix,word_cd = linkage_matrix_rep(word_sim)
print (word_cophen)
print (word_method)
fig, ax = plt.subplots(figsize=(15, 20)) # set sizeassignments = fcluster(linkage(distanceMatrix, method='complete'),4,'distance')
final_terms=[]
for term in terms:
    final_terms.append(vocab_frame.ix[term].values.tolist()[0][0])


ax = dendrogram(word_linkage_matrix, orientation="left",labels=final_terms,show_contracted=True);

#finding the number of clusters
word_classes=get_cluster_classes(ax)
print(word_classes)
thresh_word=len(word_classes)

#assignments = fcluster(linkage_matrix1,4,'distance')

word_assignments=fcluster(word_linkage_matrix,thresh_word, 'maxclust')
print(word_assignments)
cluster_word = pd.DataFrame({'word':terms , 'cluster':word_assignments})
print (cluster_word)
cluster_word.to_csv('word_cluster.csv', sep='\t')



plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('cosine_cluster_word_test.png', dpi=200) #save figure as ward_clusters
