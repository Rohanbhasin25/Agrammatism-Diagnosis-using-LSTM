from nltk.corpus import stopwords
import csv
import nltk
from nltk.corpus import brown
from nltk.corpus import webtext
from nltk.corpus import reuters
from nltk import word_tokenize,pos_tag

Sbrown = brown.sents()
Sweb = webtext.sents()
SReut = reuters.sents()

from csv import reader
# open file in read mode

bi_sents = []

with open('/Users/rohanbhasin/Downloads/sents - Sheet1.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        bi_sents.append(row)

print(bi_sents)            

a_file = open("/Users/rohanbhasin/Downloads/Stopwords hinglish.txt")
file_contents = a_file.read()
stops = file_contents.splitlines()

form = Sbrown[1:2000] + Sweb[1:2000] + SReut[1:2000]

data_path = '/Users/rohanbhasin/Downloads/fra-eng/fra.txt'

input_texts = []

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[1000 : len(lines)-1]:
    li = line.split('\t')
    first = li[0]
    print(first)

print(input_texts)

dfields = ["Type","Sentence"]

fields = ["Non-Grammar","Full"]

def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele + " " 
    
    # return string  
    return str1 

def stripSent(x):
    stop_words = set(stopwords.words('english'))
    d = listToString(x)
    text = word_tokenize(d)
    new_sentence =[]
    for w in text:
        if w not in stop_words: new_sentence.append(w)
    sent = " ".join(new_sentence)

    if sent != "":
        return sent

def Sent(x):
    d = listToString(x)
    text = word_tokenize(d)
    new_sentence =[]
    for w in text:
        new_sentence.append(w)
    sent = " ".join(new_sentence)
    return sent


def poss(sent):
    text = word_tokenize(sent)
    j = nltk.pos_tag(text)

    s = ""

    for i in j:
        s += i[1] + " "

    return s    
    

non_gram = []
gram = []
ones = []
zeroes = []
non_gram_pos = []
gram_pos = []

for line in bi_sents:
    non_gram.append(stripSent(line))
    ones.append("Aggramatic")
    zeroes.append("Non-Aggramatic")
    gram.append(Sent(line))
    non_gram_pos.append(poss(stripSent(line)))
    gram_pos.append(poss(Sent(line)))

    

with open('forma.csv', 'w') as f:
    writer = csv.writer(f)
    

    writer.writerows(zip(ones, non_gram))
    writer.writerows(zip(zeroes, gram))    
    
