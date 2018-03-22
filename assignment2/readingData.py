import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('2018-ENGR5775-ASSIGNMENT2.csv', encoding = "ISO-8859-1", dtype={'pageviews': float, 'Original content': str,
'title':str, 'id': float})
#print(len(df))


# Split the dataset in 70 to 30 partitions (70% training and 30% test)
train, test = train_test_split(df, test_size=0.3)
#print(type(train), len(train), len(test))
#print(train[:10])


