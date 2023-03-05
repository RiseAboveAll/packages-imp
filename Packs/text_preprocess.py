import pandas as pd
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from nltk.sentiment.util import *
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
from nltk import pos_tag
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def clean_data(line):
    if type(line) == str:
         # lower text
        text = line.lower()
        text= text.replace("\n"," ")# remove new lines
        text= text.replace('’',"'")
        text = re.sub(r"n't",' not',text)
        text=re.sub("thanks"," ",text)
        text=re.sub("thank"," ",text)
        text=re.sub("thanx"," ",text)
        text=re.sub("ty"," ",text)
        text = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'," ",text)
        text=re.sub(r'http\S+', '', text)
        # tokenize text and remove puncutation
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        # remove words that contain numbers
        text = [word for word in text if not any(c.isdigit() for c in word)]
        # remove stop words
#         stop = stopwords.words('english')
#         text = [x for x in text if x not in stop]
        # remove empty tokens
        text = [t for t in text if len(t) > 0]
        # pos tag text
        pos_tags = pos_tag(text)
        # lemmatize text
        text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        # remove words with only one letter
        text = [t for t in text if len(t) > 1]
        # join all
        text = " ".join(text)
    else:
        text = line
    return text
def remove_stop_words(line):
    stop_words = list(stopwords.words('english'))
#     stop_words=stop_words+["thanks","thank","thnx","ty"," thanks","thanks."]
    #print(stop_words)
    imp_words = ['against', 'out', 'over', 'most', 'no', 'nor', 'not', 'very',"nope"]
    stop_words=[i for i in stop_words if i not in imp_words]

#     all_stop = []
#     for i in stop_words:
#         if i not in imp_words:
#             all_stop.append(i)

    wt = str(line).split()
    text = [word for word in wt if word not in stop_words]
    text = ' '.join(text)

    return text
def lemmatize_word(line):
    try:
        ps = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(line)
        text =[lemmatizer.lemmatize(token) for token in tokens]
        #text =[ps.stem(word) for word in text] #stemming
        text = ' '.join(text)
    except:
        text = line
    return text