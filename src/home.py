import spell_check as sc
import text_segment as ts
import language_model as lm
from flask import Flask
from flask import request
from flask import render_template
import string
import re
import sample_2 as s2
final=''

app=Flask(__name__)
@app.route('/',methods=["GET","POST"])
def home():
    return '<html><body><form action ="query.html" method="post"><input type="text" name="query"><input type=submit value="Search"></form></body></html>'

@app.route('/query.html',methods=["GET","POST"])
def query():
    answer1=' '.join(i for i in sc.display(request.form['query'].lower()))
    #answer1=ts.display(request.form['query'])
    #for w in ts.display(request.form['query']):
        #answer1=answer1+w+" "
    answer2=ts.display(answer1)
    #answer2=' '.join(i for i in sc.display(answer1))
    final=lm.display(answer2)

    return 'Did you mean: '+ final+'\n'+'<html><body><form action="result.html"><input type=submit value="Result"></form></body></html>'
