import extract_books as s2
import re
import string
from collections import Counter
import language_model as lm
alphabet='abcdefghijklmnopqrstuvwxyz'
p=[]
delete=[]
replace=[]
transpose=[]
inserts=[]
def known(words):
    "The word already exists in the dictionary"
    return {w for w in words if w in counts}

def edit0(word):
    "The word is zero edits away from the word ie word itself"
    return {word}
def edit1(word):
    pairs=splits(word)
    transpose=[]
    inserts=[]
    replace=[]
    delete=[]
    for(a,b) in pairs:
        if len(b)>0:
            delete.append(a+b[1:])
        if len(b)>1:
            transpose.append(a+b[1]+b[0]+b[2:])
        for c in alphabet:
            #print a+c+b[1:]
            replace.append(a+c+b[1:])
            inserts.append(a+c+b)
    #delete=[a+b[1:]              for (a,b) in pairs if b]
    #transpose=[a+b[1]+b[0]+b[2:] for (a,b) in pairs if len(b)>1]
    #replace=[a+c+b[1:]           for (a,b) in pairs for c in alphabet if b]
    return set(delete+ inserts+ transpose +replace)
def edit2(word):
    "The word is 2 edits away from the original"
    return {e2 for e1 in edit1(word) for e2 in edit1(e1)}

def splits(word):
    return [(word[:i],word[i:])
      for i in range(len(word)+1)]
def tokens(text):
    "Return all the words in the text file"
    return re.findall('[a-z]+',text.lower())
def correct(word):
    candidate=(known(edit0(word)) or
               known(edit1(word)) or
               known(edit2(word)) or
               [word])
    return max(candidate, key=counts.get)
def display(words):
    p= map(correct,tokens(words))
    return p

counts=Counter(tokens(s2.string))
#print map(correct,tokens(words))
print counts.most_common(5)
print display('cndada')
