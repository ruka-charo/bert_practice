'''BERTとLDAの比較'''
%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/bert_practice
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import MeCab

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk


#　LDAでテキストを二値分類する
#%% データの準備
df_train = pd.read_table('train.tsv').drop('Unnamed: 2', axis=1)
df_train.head()
df_train.shape

df_test = pd.read_table('test.tsv').drop('Unnamed: 2', axis=1)
df_test.head()
df_test.shape

#%% 形態素解析(名詞、固有名詞のみ抽出)
m = MeCab.Tagger("mecabrc")
str_train = df_train['テキスト'].values
df_train['テキスト']
meisi_train = []
k_meisi_train = []
for text in str_train:
    res_train = m.parseToNode(text)
    while res_train:
        arr = res_train.feature.split(",")
        if(arr[1] == "固有名詞"):
            k_meisi_train.append(arr[6])
        elif (arr[0] == "名詞"):
            meisi_train.append(arr[6])
        res_train = res_train.next

meisi_train = [n for n in meisi_train if n != '*']
k_meisi_train = [n for n in k_meisi_train if n != '*']
