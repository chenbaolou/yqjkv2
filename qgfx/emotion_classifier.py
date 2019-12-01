import jieba
import pandas as pd

#数据准备
df_pos = pd.read_excel("./data/content_1.xlsx", encoding='utf-8')
df_pos = df_pos.dropna()

df_neg = pd.read_excel("./data/content_0.xlsx", encoding='utf-8')
df_neg = df_neg.dropna()

df_neu = pd.read_excel("./data/content_5.xlsx", encoding='utf-8')
df_neu = df_neu.dropna()

pos = df_pos.content.values.tolist()[:461]
neg = df_neg.content.values.tolist()[:461]
neu = df_neu.content.values.tolist()[:461]

#停用词
stopwords = pd.read_csv("data/stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values

#去停用词
def preprocess_text(content_line, sentences, category):
    for line in content_line:
        try:
            segs = jieba.lcut(line)
            segs = filter(lambda x: len(x) > 1, segs)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append((" ".join(segs), category))
        except Exception:
            print(line)
            continue

#生成训练数据
sentences = []

preprocess_text(pos, sentences, 'pos')
preprocess_text(neg, sentences, 'neg')
preprocess_text(neu, sentences, 'neu')


#生成训练集
import random

random.shuffle(sentences)

# for sentences in sentences[:]:
#      print(sentences[0], sentences[1])

# 定义文本抽取词袋模型特征
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(
    analyzer='word',
    max_features=4000,
)

# 把语料数据切分
from sklearn.model_selection import train_test_split
x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1256)

# print(x_train[10])
# print(y_train[10])

# 把训练数据转换为词袋模型
vec.fit(x_train)
# 算法建模和模型训练
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)
# 计算 AUC 值
print(classifier.score(vec.transform(x_test), y_test))

#交叉验证
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score
import numpy as np

def stratifiedkfold_cv(x, y, clf_class, shuffle=True, n_folds=5, **kwargs):
    stratifiedk_fold = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y[:]
    for train_index, test_index in stratifiedk_fold:
        X_train, X_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred



NB = MultinomialNB
#完成一个文本分类器
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class TextClassifier():

    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,4), max_features=20000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)


text_classifier = TextClassifier()
text_classifier.fit(x_train, y_train)
export_text = '太差劲了'
print(text_classifier.predict(export_text))
print(text_classifier.score(x_test, y_test))









