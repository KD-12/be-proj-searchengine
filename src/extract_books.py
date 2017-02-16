# extract names of books and corresponding authors.
#author variable holds all the information regarding book titles and author
import re
import string
import os
import codecs

def extract_book_author(files):
    info=[]
    books=[]
    string=''
    for filename in files:
        with codecs.open(root[:]+'\\'+filename, "r",encoding='utf-8', errors='ignore') as file_name:

            text=file_name.readline()


            info.append(text.lower())
            book_info=''

            non_useful=['the','project',"gutenberg's",'ebooks','etexts','etext','ebook','gutenberg','this','presented','file','s']
            result=[word  for word in re.findall('[a-z]+',text.lower()) if word not in non_useful]

            book_info=' '.join(result)
            book_info=re.sub("of","",book_info,count=1).strip()
            books.append(book_info)

    #print books
    #print book_info
    string='\n'.join(books)

    return string



raw_path=['C:\Users\Aishwarya Sadasivan\Dataset-1']
for path in raw_path:
    for root,dir,files in os.walk(path):
        print ("Files in: " + root[:])
author= extract_book_author(files)
