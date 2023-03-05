import pandas as pd
import numpy as np

import re
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
def lemmatize_word(text): 
    from nltk.tokenize import word_tokenize 
    from nltk.stem import WordNetLemmatizer
    lemmatizer=WordNetLemmatizer()
    word_tokens = word_tokenize(text) 
    # provide context i.e. part-of-speech 
    lemmas = [lemmatizer.lemmatize(word,pos='v') for word in word_tokens] 
    return lemmas
def clean_text(text,stop_words=None):
    CLEANR=re.compile('<.*?>')
    text= text.lower().strip()     # Lowering and removing extra spaces
    text= text.replace("\n"," ")# remove new lines
    text= text.replace("weve","we have")
    text= text.replace("'m"," am")# expansion am
    text= text.replace("'ll"," will")# expansion will
    text= text.replace("'d"," would")# expansion would
    #text= text.replace("n't"," not")# expansion not
    text= text.replace("'nt"," not")# expansion not
    text= text.replace("'ve"," have")# expansion have
    text= text.replace("'s"," is")# expansion is
    text= text.replace("'re"," are")# expansion are
    text= text.replace("+"," ")
    text= re.sub(CLEANR,'',text)
    text= re.sub(r"cant",'can not',text)
    text= re.sub(r"dosent","does not",text)
    text= re.sub(r"doesn't","does not",text)
    text= re.sub(r"don't","do not",text)
    text= re.sub(r"doesnt","does not",text)
    text= re.sub(r"can't",'can not',text)
    text= re.sub(r"cannot",'can not',text)
    text= text.replace("\s{1,}(nt)"," not")
    text= re.sub(r"didnot","did not",text)
    text= re.sub(r"didnt","did not",text)
    text= re.sub(r"dont","do not",text)
    text= re.sub(r"wasnt","was not",text)
    text= re.sub(r"reslkve","resolved",text)
    text= re.sub(r"chnage","change",text)
    text= re.sub(r"im","i am",text)
    text= re.sub(r"i've","i have",text)
    text= re.sub(r"won't","will not",text) 
    text= re.sub(r"wont","will not",text)    
    text= re.sub(r"\n"," ",text)    
    #text= re.sub('\S+@\S+','',text) #removing emailadderess
    text= re.sub(r'https?:\/\/.*[\r\n]*', ' ', text) # removing all urls
    text= re.sub(r"\[.*?\]", " ",text)# removing data in square brackets
    text= re.sub("[^a-z-+-@\s]+"," ",text)
#     text= re.sub("thank you|thank you soo much|thank you so much|thanks a lot|thank you very much","thanks",text)
    text= re.sub('\s{2}',' ',text)  # removing extra spaces
    text= lemmatize_word(text)
    
    if stop_words is None:
        stop_words= ["a", "the" , "to" , "be" ,"is" , "an"]
    text = [word for word in text if word not in stop_words]

    return text