import readingData as data
import pandas as pd
import re

### Pre-processing
data.train = data.train.dropna()

test = data.train[:10]


test['title'] = test['title'].str.upper()
test['Original content'] = test['Original content'].str.upper()


def change_to_uppercase(x):
    return x.upper()

def replace_special_char(x):
    return x.replace('??', '\'')

# test['title'].apply(replace_special_char)
# test['title'].apply(change_to_uppercase)
print("********************************************* ", test['title'].is_copy)

# print (test['title'])
# print('----------------')

def removeChar(x):
    return re.sub('[^A-Za-z0-9\']+', ' ', x)

test['title'] = test['title'].apply(removeChar)
print ((test['title']))



#data.train.to_csv('cleanfile.csv', encoding='utf-8')

# def dftoList(df, name):
#     lst = []
#     lst.append(str(df[name][1]))
#     print(str(df[name][1]))
#     return lst


# dftoList(data.train, 'title')


# from nltk.corpus import stopwords
# from nltk.tokenize import wordpunct_tokenize

# sw = stopwords.words('english')

# def tok_cln(strin):
#         '''
#         tokenizes string and removes stopwords
#         '''
#         return set(nltk.wordpunct_tokenize(strin)).difference(sw)

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# #docs = ['why hello there', 'omg hello pony', 'she went there? omg']
# #docs = dftoList(data.train, 'title')
# vec = CountVectorizer()
# #vec = TfidfVectorizer()
# print(countvec.fit_transform(data.train.title))

# # df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
# # print(df)