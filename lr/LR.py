import pandas as pd 
import numpy as np 
from sklearn import preprocessing,decomposition,model_selection,metrics,pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from nltk import word_tokenize
import jieba
import re
from  sklearn.metrics import confusion_matrix,precision_score,recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def rp_stop(stopwords, str):
    str_new=""
    for w in str.split(" "):
        if w not in stopwords and w not in ["\n"]:
            w = re.sub("[\da-zA-Z]+","",w)
            str_new += w.strip()+" "
    return str_new.strip()

#多分类的混淆矩阵
def multiclass_matrix(x, y_act, y_predict):
    print("精准率为：", precision_score(y_act,y_predict,average="weighted"))
    print("召回率为：", recall_score(y_act,y_predict,average="weighted"))
    cfm = confusion_matrix(y_act,y_predict)
    print("多分类混淆矩阵：", cfm)
    sns.heatmap(cfm, linewidths=0.1, annot=False, fmt=".2f", cmap="rainbow", robust=False, center=10)

#根据预测结果，生成唯一预测值
def get_pred_rs(predictions):
    y_pred = [-1]*len(predictions)
    for i,dist in enumerate(predictions):
        dist = dist.tolist()
        if len(set(dist)) != 1:
            topic_index = dist.index(max(dist))
            y_pred[i]= topic_index
    return y_pred

def number_normalizer(tokens):
    #将所有数字标记映射为一个占位符（placeholder）
    #对于许多实际应用场景来说，以数字开头的tokens不是很有用
    #但这样的tokens存在也有一定相关性，通过将所有数字都表示成同一个富豪，可以达到降维的目的
    return("#NUMBER" if token[0].isdigit() else token for token in tokens)

class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer,self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))

def load_train_data(stopwords):
    realpath = os.path.dirname(os.path.realpath(__file__))
    data1 = pd.read_excel(os.path.join(realpath, "sample_have_nb_type_1009_new.xls"), sheet_name=0)

    # 样本量少于10的类别，标记为“其他”（样本数量较少，此项需要调整）
    aa = data1["type"].value_counts().reset_index()
    to_del=aa[aa["type"]<10]["index"]

    #直接删掉<10类别
    data = data1[~data1["type"].isin(to_del)]

    jieba.load_userdict(os.path.join(realpath, "自定义词典.txt"))

    data['文本分词'] = data["theme_new"].apply(lambda i:jieba.cut(str(i),False))
    data['文本分词'] = [" ".join(i) for i in data["文本分词"]]
    data['文本分词'] = data["文本分词"].apply(lambda i:rp_stop(stopwords, str(i)))

    data.head(5)

    return data

def load_stop_words():
    realpath = os.path.dirname(os.path.realpath(__file__))
    stopwords = [line.strip() for line in open(os.path.join(realpath, "停用词.txt"), "r", encoding="utf-8").readlines()]
    return stopwords

def train_and_model(data, stopwords):
    
    #接下来用scikit-learn中的LabelEncoder将文本标签（text label）-本次为type，转化为数字（integer）
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(data.type.values)
    y_labels = lbl_enc.classes_

    xtrain, xvalid, ytrain, yvalid = train_test_split(data['文本分词'].values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)

    ctv = CountVectorizer(min_df=3,max_df=0.5,ngram_range=(1,2),stop_words = stopwords)

    #使用count Vecorizer来fit训练集和测试集（半监督学习）
    ctv.fit(list(xtrain)+list(xvalid))
    xtrain_ctv = ctv.transform(xtrain)
    #xvalid_ctv = ctv.transform(xvalid)

    clf = LogisticRegression(C=1.0, solver="lbfgs", multi_class="multinomial")
    realpath = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(realpath, 'lr.model')
    if os.path.exists(model_path) == False:
        #利用提取的word count特征来fit一个简单的logistic regression
        clf.fit(xtrain_ctv, ytrain)
        joblib.dump(clf, model_path)
    else:
        print("load model")
        clf = joblib.load(model_path)
    #predictions = clf.predict_proba(xvalid_ctv)

    #y_predict = get_pred_rs(predictions)
    #multiclass_matrix(xvalid_ctv, yvalid, y_predict)

    return clf, ctv, y_labels


def classify(content):
    print(content)
    stopwords = load_stop_words()
    data = load_train_data(stopwords)

    clf, ctv, y_labels = train_and_model(data, stopwords)
    test = []
    test_ctv = ''
    if content == None:
        test = pd.read_excel("test_no_type_1009.xlsx",sheet_name=0)
        #最终采用LR模型，进行结果预测
        test["文本分词"] = test["theme_new"].apply(lambda i:jieba.cut(str(i),False))
        test["文本分词"] = [" ".join(i) for i in test["文本分词"]]
        test["文本分词"] = test["文本分词"].apply(lambda i:rp_stop(stopwords, str(i)))
        test.head()
        test_ctv = ctv.transform(test.文本分词.values)

        test_prd = clf.predict_proba(test_ctv)

        rs = get_pred_rs(test_prd)
        print(rs)
        test["label"]=rs
        test["预测结果"] = test["label"].apply(lambda i:y_labels[i])
        test.head()

        test.to_csv(r"LR预测type结果_20191009.csv",mode="w",header=True,encoding="utf-8")

    else:
        _content = jieba.cut(str(content),False)
        _content = " ".join(_content)
        _content = rp_stop(stopwords, _content)
        test_ctv = ctv.transform([_content])

        test_prd = clf.predict_proba(test_ctv)

        rs = get_pred_rs(test_prd)
        if rs[0] < len(y_labels):
            return y_labels[rs[0]]
        else:
            return ""
    
#xxx = classify("阅读器战况如图_我是刷还是不刷，估计2万拿不下礼品，好纠结刷还是不刷")
#print(xxx)