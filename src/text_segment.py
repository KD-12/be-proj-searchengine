#pwords->returns the product of the probabilities of the words in the sentence.
#Entered sentence is split and checked for a valid sentence using pwords
#if the query is not a valid sentence it is joined and sent to segment function
#segment function->will segment the query into most proable sequence of words.
#segment function uses pwords to calculate the proabilities.

import re
import extract_books as s2
from collections import Counter


def tokens(text):
    "Returns the words contained in the sentence"
    return re.findall('[a-z]+',text.lower())
def pdist(counter):
    "A list of the probability of each word in the doc"
    N=sum(counter.values())
    return lambda x:counter[x]*1.0/N*1.0

def pword(w):
    return prob_dict(w)
def pwords(words):
    return product([pword(w) for w in words])
def splits(words,start,L=20):
    return [(words[:i],words[i:]) for i in range(start,min(len(words),L)+1)]

def segment(word):
    "Returns the most probable segment"
    if not word:
        return []
    else:
        candidate=([first]+ segment(rest)
                    for (first,rest) in splits(word,1))
        return max(candidate,key=pwords)

def product(p):
    result=1
    for x in p:
        result*=x
    return result
def display(sentence):
    text=''
    if pwords(sentence.split(' '))>0:
        return sentence
    else:
        sentence=''.join(sentence.split(' '))
        for word in sentence.split(' '):
                txt=' '.join(segment(word))
                text=text+txt+' '

        return text


text=s2.author
count=Counter(re.findall('[a-z]+',text.lower()))
prob_dict=pdist(count)
print display('henryjames')
