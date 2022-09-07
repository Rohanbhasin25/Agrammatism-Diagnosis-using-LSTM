
from nltk.corpus import stopwords
import csv
from nltk.corpus import brown
from nltk.corpus import webtext
from nltk.corpus import reuters
from nltk import word_tokenize,pos_tag

sents = brown.sents() + webtext.sents() +reuters.sents()
form = sents[1:len(sents)]

print(len(webtext.sents()))


'''
empt = []

fields = ["Non-Grammar","Full"]

for i in range(1,len(form)):
	sens = " ".join(form[i])
	ind = [sens]
	empt.append(ind)

def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1


def stripSent(x):
	stop_words = set(stopwords.words('english'))

	text = word_tokenize(x)

	#print(nltk.pos_tag(text))

	new_sentence =[]

	for w in text:
		if w not in stop_words: new_sentence.append(w)

	sent = " ".join(new_sentence)
	return sent


filename = "dnew_records.csv"

with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
      
    # writing the fields
    csvwriter.writerow(fields)

    for i in empt[1:1000]:   
        sentrow = [listToString((stripSent(i))),i]
        csvwriter.writerows(sentrow)
      




'''
