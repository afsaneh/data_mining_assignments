import readingData as data
import pandas as pd
import re
import nltk
#from nltk.stem import PorterStemmer
#from nltk.corpus import stopwords
#from nltk.tokenize import wordpunct_tokenize, sent_tokenize,word_tokenize

### Pre-processing
data.train = data.train.dropna()

# data.train = data.train[:10]


def change_to_uppercase(x):
    return x.upper()

def replace_special_char(x):
    return x.replace('??', '\'')

def removeChar(x):
    return re.sub('[^A-Za-z\']+', ' ', x)
#    return re.sub('[^A-Za-z0-9\']+', ' ', x)

# Change all the letters to uppercase
data.train['title'] = data.train['title'].str.lower()
data.train['Original content'] = data.train['Original content'].str.lower()

# Replace undefined character with '
data.train['title'] = data.train['title'].apply(replace_special_char)
data.train['Original content'] = data.train['Original content'].apply(replace_special_char)

# remove punctuations and nonalphabetical characters
data.train['title'] = data.train['title'].apply(removeChar)
data.train['Original content'] = data.train['Original content'].apply(removeChar)


# Replace words with their lemma
lemma = nltk.wordnet.WordNetLemmatizer()


data.train['title'] = data.train['title'].apply(word_tokenize).apply(lambda x : [lemma.lemmatize(y) for y in x])
data.train['title'] = data.train['title'].apply(lambda x : " ".join(x)) # make a sentence again

data.train['Original content'] = data.train['Original content'].apply(word_tokenize).apply(lambda x : [lemma.lemmatize(y) for y in x])
data.train['Original content'] = data.train['Original content'].apply(lambda x : " ".join(x)) # make a sentence again
#print(data.train['title'])

data.train.to_csv('cleanfile.csv', encoding='utf-8')


sw = stopwords.words('english')
     

from sklearn.feature_extraction.text import CountVectorizer
data.train['title_without_sw'] = data.train['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))
data.train['title_without_sw'] = data.train['Original content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))


countvec = CountVectorizer()
train_title_matrix = countvec.fit_transform(data.train.title_without_sw)

# train_content_matrix = countvec.fit_transform(data.train.title_without_sw)

#print(countvec.fit_transform(data.train.title))
#df = pd.DataFrame(data.train['id'], columns)
#matrix = countvec.fit_transform(data.train.title_without_sw).toarray()
matrix = pd.DataFrame(countvec.fit_transform(data.train.title_without_sw).toarray(), columns=countvec.get_feature_names())
#matrix.to_csv('matrix.csv', encoding='utf-8')




#document_term = data.train.copy()
#print(type(document_term))
#print(type(matrix))
#print(pd.concat([document_term, matrix], axis=1))
#document_term.concat(matrix)



#df = pd.DataFrame({'title': data.train.title_without_sw['title'],                  
#                   \countvec.get_feature_names():countvec.fit_transform(data.train.title_without_sw).toarray()})


    
    
    
    #document_term.to_csv('termDocument.csv', encoding='utf-8')