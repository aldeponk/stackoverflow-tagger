import json


import pickle
import nltk

#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('stopwords')

import nltk.data
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier


"""
-----------------------------------------------------------------
Desc:   Remove \n 
Input:  string
Output: string without useless characters   

Traitements appliqués : 
- remove linefeed
- 
-----------------------------------------------------------------
"""
def removeLineFeed(s):
    final_s = s.replace('\n',' ')
    return final_s

"""
-----------------------------------------------------------------
Desc:   Remove useless characters 
Input:  Dataset
Output: dataset with body feature without useless characters   

Traitements appliqués : 
- remove 
- create new column with extracted text
- 
-----------------------------------------------------------------
"""
def removeUselessChars(s):
    pattern = re.compile('[^A-Za-z +]')
    final_s = re.sub(pattern, ' ', s)
    return final_s

"""
-----------------------------------------------------------------
Desc:   toLowerCase 
Input:  Dataset
Output: dataset with body feature to lower case   

Traitements appliqués : 
- remove 
- create new column with extracted text
- 
-----------------------------------------------------------------
"""
def toLowerCase(s):
    final_s = s.lower()
    return final_s

"""
-----------------------------------------------------------------
Desc:   unitary stopwords removal function 
Input:  text
Output: text without stopwords   

Traitements appliqués : 
- stop words removal 
-----------------------------------------------------------------
"""
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)

    filtered_text = ' '.join(w for w in word_tokens if not w in stop_words)
   
    return filtered_text

"""
---------------------------------------------------------------------------------------------------
lemmatizer function
---------------------------------------------------------------------------------------------------
"""
def myLemmatizer(text):
    cleanedText = []
    text = removeLineFeed(text)
    text = removeUselessChars(text)
    text = toLowerCase(text)
    text = remove_stopwords(text)
    
    # split string into words (tokens)
    tokens = nltk.tokenize.word_tokenize(text)
    # keep strings with only alphabets
    tokens = [t for t in tokens if t.isalpha()]
    # put words into base form
    tokens = [WordNetLemmatizer().lemmatize(t, pos='v') for t in tokens] 
    # remove short words, they're probably not useful
    tokens = [t for t in tokens if len(t) > 2]
    
    return tokens

"""
---------------------------------------------------------------------------------------------------
main supervised prediction implementation function
---------------------------------------------------------------------------------------------------
"""

def supervised_predict(post, vectorizer, model):
    pattern = re.compile('[^A-Za-z +]')
    intermediate = re.sub(pattern, ' ', post)
    intermediate = post.lower()

    stop_words = set(stopwords.words('english')) 
    #print(text)
    word_tokens = word_tokenize(intermediate) 
    filtered_text = ' '.join(w for w in word_tokens if not w in stop_words)

    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stemmer = EnglishStemmer()
    #stemmed = myTokenizer(filtered_text)
    lemmatized = myLemmatizer(filtered_text)
    x_input = vectorizer.transform(lemmatized).astype('float64')
    tags = model.predict(x_input)
    return tags


"""
---------------------------------------------------------------------------------------------------
supervised prediction function
---------------------------------------------------------------------------------------------------
"""
def supervised_prediction(post):
    
    model = pickle.load(open('models/bagClassifier.obj', 'rb'))
    vectorizer = pickle.load(open('models/vectorizer.obj', 'rb'))
    mlb = pickle.load(open('models/mlb.obj', 'rb'))
    tags = supervised_predict(post, vectorizer, model)
    all_labels = mlb.inverse_transform(tags)
    output_tags = ""
    for label in all_labels:
        s = ''.join(label)
        if len(s) != 0:
            output_tags += s + ' '
    
    return output_tags



"""
---------------------------------------------------------------------------------------------------
main unsupervised prediction implementation function
---------------------------------------------------------------------------------------------------
"""

def unsupervised_predict(post, lda_tags_df_scaled, dictionary, lda):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    stemmer = EnglishStemmer()
    stemmed = ' '.join(stemmer.stem(WordNetLemmatizer().lemmatize(w, pos='v')) for w in w_tokenizer.tokenize(post))

    pattern = re.compile('[^A-Za-z +]')
    normalized = re.sub(pattern, ' ', stemmed)

    result = []
    for token in gensim.utils.simple_preprocess(normalized):
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(token)
    this_dictionary = []
    this_dictionary.append(result)
    
    other_corpus = [dictionary.doc2bow(text) for text in this_dictionary]
    unseen_doc = other_corpus[0]
    vector = lda[unseen_doc]

    topic = vector[0][0][0]
    perc = vector[0][0][1]
    tags = lda_tags_df_scaled[int(topic)]
    tags_output = tags.sort_values(ascending=False).head(5)    
    
    return tags_output

"""
---------------------------------------------------------------------------------------------------
unsupervised prediction function
---------------------------------------------------------------------------------------------------
"""
def unsupervised_prediction(post):
    
    lda_tags_df_scaled = pickle.load(open('models/lda_tags_df_scaled.obj', 'rb'))
    lda = gensim.models.LdaMulticore.load('models/lda_model_tfidf_optimized_2')    
    dictionary = pickle.load(open('models/dictionary.obj', 'rb'))   
    
        
    tags_output = unsupervised_predict(post, lda_tags_df_scaled, dictionary, lda)
    output_tags = ""
    for index in tags_output.index:
        output_tags += index + ' '
        
    return output_tags
