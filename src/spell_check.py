# Entered query will get tokenised and the tokens will undergo insertion,transposition,deletion,replacement through functions edit1 and edit2.
#edit0-> The word requires no alteration and the same word is returned.
#edit1-> The word requires one insertion/deletion/transposition/replacement and the altered word is returned.
#edit2-> The word is 2 edits away from the original and hence edit1 is performed twice and the altered word is returned.
#counts->stores the frequency of tokens.
#The values returned from edit0,edit1 and edit2 are compared with counts to obtain the maximum frequency amoung them.
#The word with maximum freuency is displayed.
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

counts=Counter(tokens(s2.author))
#print map(correct,tokens(words))
print counts.most_common(5)
print display('hnryy jmes')
