import readingData as data
import pandas as pd

### Pre-processing
data.train = data.train.dropna()
print(len(data.train))

data.train['title'] = data.train['title'].str.upper()
data.train['Original content'] = data.train['Original content'].str.upper()

data.train.to_csv('cleanfile.csv', encoding='utf-8')

