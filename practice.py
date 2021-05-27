'''BERTとLDAの比較'''
%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/bert_practice
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import MeCab
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import LatentDirichletAllocation as LDA

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# 不要文字の削除関数
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text


#　LDAでテキストを二値分類する
#%% データの準備
df_train = pd.read_table('train.tsv').drop('Unnamed: 2', axis=1)
df_train.head()
df_train.shape
df_train['テキスト'] = df_train['テキスト'].apply(preprocessor)

df_test = pd.read_table('test.tsv').drop('Unnamed: 2', axis=1)
df_test.head()
df_test.shape

#%% 形態素解析(名詞、固有名詞のみ抽出)
df = pd.DataFrame()
str_train = df_train['テキスト'].values
class_num = df_train['クラス'].values
m = MeCab.Tagger("-Ochasen")

for idx, num, text in zip(range(len(str_train)+1), class_num, str_train):
    nouns = [line.split()[0] for line in m.parse(text).splitlines()
        if "名詞" in line.split()[-1]]
    nouns = [n for n in nouns if n != '*']
    data_str = " ".join(map(str, nouns))

    df = df.append([[idx, num, data_str]], ignore_index=False)

df.columns = ['id', 'class', 'text']
df = df.reset_index().drop('index', axis=1)
df = df.sample(frac=1, random_state=1)
df.head()

#%% 文書全体の単語の特徴
vocabulary = " ".join(df.text).split(" ")
print("全単語数", len(vocabulary))
print("ユニークな単語数", len(set(vocabulary)))
# 単語をカウントしてくれる
counter = Counter(vocabulary)
# 出現頻度で並べる
counter.most_common(n=100)

#%% 文書の単語のカウント表
# dfを最大10%まで。　出現が多い順の5000個を使う
cv = CountVectorizer()
X = cv.fit_transform(df["text"].values)
pd.DataFrame(X.toarray(), columns=cv.get_feature_names())

#%% LDAで分解する
lda = LDA(
    # トピックは2つと仮定
    n_components=2, random_state=42,
    # batchは全データで一気に学習
    learning_method='batch')
X_topics = lda.fit_transform(X)

#%% X_topicsをDataframeに
topic_columns = ["トピック{}".format(i+1) for i in range(2)]
df_doc = pd.DataFrame(X_topics, columns=topic_columns)
df_doc

#%% 正解率の算出
index_0 = df_doc[df_doc['トピック1'] < df_doc['トピック2']].index
index_1 = df_doc[df_doc['トピック1'] > df_doc['トピック2']].index

df_0 = df.iloc[index_0]
df_1 = df.iloc[index_1]

df_0_correct = df_0[df_0['class'] == 0]
df_1_correct = df_1[df_1['class'] == 1]

print('クラス0の正解率=', len(df_0_correct) / len(df_0))
print('クラス1の正解率=', len(df_1_correct) / len(df_1))

#%% 不正解の確認
df_0_incorrect = df_0[df_0['class'] == 1]
df_1_incorrect = df_1[df_1['class'] == 0]
df_0_incorrect
df_1_incorrect
