import re
import extract_books as s2
from collections import Counter

def tokens(text):
    "Returns the words contained in the sentence"
    return re.findall('[a-z]+',text.lower())
def pdist(counter):
    "A list of the probability of each word in the doc"
    N=sum(counter.values())
    return lambda x: counter[x]*1.0/N*1.0

def Pword(w):
    return P(w)
def Pwords(words):
    return product([Pword(w) for w in words])
def splits(words,start,L=20):
    return [(words[:i],words[i:]) for i in range(start,min(len(words),L)+1)]

def memo(f):
    cache={}
    def fmemo(*args):
        if args not in cache:
            cache[args]=f(*args)
        return cache[args]
    fmemo.cache=cache
    return fmemo

def segment(word):
    "Returns the most probable segment"
    if not word:
        return []
    else:
        candidate=([first]+ segment(rest)
                    for (first,rest) in splits(word,1))
        return max(candidate,key=Pwords)

def product(p):
    result=1
    for x in p:
        result*=x
    return result
def display(sentence):
    text=''
    for word in sentence.split(' '):
        txt=' '.join(segment(word))
        text=text+txt+' '
    return text

text=s2.string
count=Counter(re.findall('[a-z]+',text.lower()))
P=pdist(count)
print display('chronicles')
