#Bag of words concept
import re
from itertools import permutations as perm
import string
from collections import Counter
import text_segment as ts
import extract_books as s2
def pdist(counter):
    "A list of the probability of each word in the doc"
    N=sum(counter.values())
    return lambda x: counter[x]*1.0/N*1.0

def Pword(w):
    return p(w)

def Pwords(words):
    "Probability of words, assuming each word is independent of others."
    return product(p1w(w) for w in words)


def splits(words,start=0,L=20):
    return [(words[:i],words[i:]) for i in range(start,min(len(words),L)+1)]

def memo(f):
    cache={}
    def fmemo(*args):
        if args not in cache:
            cache[args]=f(*args)
        return cache[args]
    fmemo.cache=cache
    return fmemo

@memo
def segment(word):
    "Returns the most probable segment"
    if not word:
        return []
    else:
        candidate=([first]+ segment(rest)
                    for (first,rest) in splits(word,1))
        return max(candidate,key=Pwords)

def Pwords2(words, prev='<S>'):
    "The probability of a sequence of words, using bigram data, given prev word."
    return product(cPword(w, (prev if (i == 0) else words[i-1]) )
                   for (i, w) in enumerate(words))



#conditional probability
def cPword(word, prev):
    "Conditional probability of word, given previous word."
    bigram = prev + ' ' + word
    if p2w(bigram) > 0 and p1w(prev) > 0:
        return p2w(bigram) / p1w(prev)
    else: # Average the back-off value and zero.
        return p1w(word) / 2


def load_counts(filename):
    bigramw=[]
    trigramw=[]
    keyb={}
    c=Counter()
    for line in filename.split('\n'):
        word=ts.tokens(line)
        bigramw=zip(word,word[1:])
        bigramlist.extend(bigramw)
        trigramw=zip(word,word[1:],word[2:])
        trigramlist.extend(trigramw)
        for (a,b) in zip(word,word[1:]):
            c[a+' '+b]=c[a+' '+b]+1
    return c

def product(p):
    result=1
    for x in p:
        result*=x
    return result
#for gving result of bigram model
def display(sentence):
    #sent='this is good'
    temp=list(perm(sentence.split()))
    print temp
    tp=0.0
    ts=''
    sent=[]
    for word in temp:
        s=' '.join(word)
        sent.append(s)
    for snt in sent:
        if tp < Pwords2(snt.split()):
            tp=Pwords2(snt.split())
            ts=snt


    return ts
bigramlist=[]
trigramlist=[]

text=s2.string
count=ts.count
P=pdist(count)
counts_1=load_counts(text)
p2w=ts.pdist(counts_1)
p1w=ts.pdist(count)

word='jane'
best_probability=0.0
best_word=''
for (prev_word,next_word) in bigramlist:
    p=Pwords2(ts.tokens(prev_word+' '+next_word))
    if re.match(word,prev_word) and p>best_probability:
        best_probability=p
        best_word=prev_word+' '+next_word
print best_word

print bigramlist
#print Pwords2('henry james'.split())
print display('frenchwoman')
