#Entered query is split and words within the sentence undergo permutations to from various sentences.
#pwords2->bigram model(probability of a two words occuring together)
#cPword->calculates conditional probability.
#load_counts->dictionary of all the bigrams of the sentence.
#product->gives the probability of the entire sentence by multiplying the conditional probabilities of the bigram obtained from load_counts.
#pwords2 uses cPword and the product function to calculate the probability of two word occuring together given that the previous word has occured.

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

def pword(w):
    return p(w)

def pwords(words):
    "Probability of words, assuming each word is independent of others."
    return product(p1w(w) for w in words)


def splits(words,start=0,L=20):
    return [(words[:i],words[i:]) for i in range(start,min(len(words),L)+1)]


def pwords2(words, prev='<S>'):
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
    temp_prob=0.0
    temp_sent=''
    sent=[]
    for word in temp:
        s=' '.join(word)
        sent.append(s)
    print sent
    for snt in sent:
        if temp_prob < pwords2(snt.split(' ')):
            temp_prob=pwords2(snt.split(' '))
            temp_sent=snt
    return temp_sent
bigramlist=[]
trigramlist=[]

text=s2.author
count=ts.count
prob_dict=pdist(count)
counts_1=load_counts(text)
p2w=ts.pdist(counts_1)
p1w=ts.pdist(count)

word='jane'
best_probability=0.0
best_word=''
for (prev_word,next_word) in bigramlist:
    p=pwords2(ts.tokens(prev_word+' '+next_word))
    if re.match(word,prev_word) and p>best_probability:
        best_probability=p
        best_word=prev_word+' '+next_word
#print best_word

#print pwords2('henry james'.split())
print display('robert of letters')
