import pandas as pd 
import numpy as np 
from sklearn import preprocessing,decomposition,model_selection,metrics,pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk import word_tokenize
import jieba
import re
from  sklearn.metrics import confusion_matrix,precision_score,recall_score
import matplotlib.pyplot as plt
import seaborn as sns
# 作用待定 
# hea  
data1 = pd.read_excel("sample_have_nb_type_1009_new.xls",sheet_name=0)
test = pd.read_excel("test_no_type_1009.xlsx",sheet_name=0)
# 样本量少于10的类别，标记为“其他”（样本数量较少，此项需要调整）
aa = data1["type"].value_counts().reset_index()
to_del=aa[aa["type"]<10]["index"]
to_del

#直接删掉<10类别
data = data1[~data1["type"].isin(to_del)]

jieba.load_userdict("自定义词典.txt")
stopwords = [line.strip() for line in open("停用词.txt","r",encoding="utf-8").readlines()]
# jieba.enable_parallel() #并行分词开启
def rp_stop(str):
    str_new=""
    for w in str.split(" "):
        if w not in stopwords and w not in ["\n"]:
            w = re.sub("[\da-zA-Z]+","",w)
            str_new += w.strip()+" "
    return str_new.strip()

data['文本分词'] = data["theme_new"].apply(lambda i:jieba.cut(str(i),False))
data['文本分词'] = [" ".join(i) for i in data["文本分词"]]
data['文本分词'] = data["文本分词"].apply(lambda i:rp_stop(str(i)))

data.head(5)

def multiclass_logloss(actual,predicted,eps=1e-15):
    #对数损失度量（logarithmic Loss Metric）的多分类版本
    #param actual:包含actual target classes 的数组
    #param predicted:分类预测结果矩阵，每个类别都有一个概率
    #convert 'actual' to a binary array if it's not  already:
        if len(actual.shape) == 1:
            actual2 = np.zeros((actual.shape[0],predicted.shape[1]))
            for i,val in enumerate(actual):
                actual2[i,val] = 1
            actual = actual2
        clip = np.clip(predicted,eps,1-eps)
        rows = actual.shape[0]
        vsota = np.sum(actual * np.log(clip))
        return -1.0/rows*vsota

#接下来用scikit-learn中的LabelEncoder将文本标签（text label）-本次为type，转化为数字（integer）
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.type.values)
y_labels = lbl_enc.classes_
y_labels



#根据预测结果，生成唯一预测值
def get_pred_rs(predictions):
    y_pred = [-1]*len(predictions)
    for i,dist in enumerate(predictions):
        dist = dist.tolist()
        if len(set(dist)) != 1:
            topic_index = dist.index(max(dist))
            max_topic_prob = dist[topic_index]
            y_pred[i]= topic_index
    return y_pred



#多分类的混淆矩阵
def multiclass_matrix(x,y_act,y_predict):
    print("精准率为：",precision_score(y_act,y_predict,average="weighted"))
    print("召回率为：",recall_score(y_act,y_predict,average="weighted"))
    cfm = confusion_matrix(y_act,y_predict)
    print("多分类混淆矩阵：",cfm)
    sns.heatmap(cfm,linewidths=0.1,annot=False,fmt=".2f",cmap="rainbow",robust=False,center=10)

xtrain,xvalid,ytrain,yvalid = train_test_split(data['文本分词'].values,y,stratify=y,random_state=42,test_size=0.1,shuffle=True)

print(xtrain.shape)
print(xvalid.shape)

def number_normalizer(tokens):
    #将所有数字标记映射为一个占位符（placeholder）
    #对于许多实际应用场景来说，以数字开头的tokens不是很有用
    #但这样的tokens存在也有一定相关性，通过将所有数字都表示成同一个富豪，可以达到降维的目的
    return("#NUMBER" if token[0].isdigit() else token for token in tokens)

class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer,self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))

ctv = CountVectorizer(min_df=3,max_df=0.5,ngram_range=(1,2),stop_words = stopwords)

#最终采用LR模型，进行结果预测
test["文本分词"] = test["theme_new"].apply(lambda i:jieba.cut(str(i),False))
test["文本分词"] = [" ".join(i) for i in test["文本分词"]]
test["文本分词"] = test["文本分词"].apply(lambda i:rp_stop(str(i)))

test.head()

#使用count Vecorizer来fit训练集和测试集（半监督学习）
ctv.fit(list(xtrain)+list(xvalid))
xtrain_ctv = ctv.transform(xtrain)
xvalid_ctv = ctv.transform(xvalid)
    
#利用提取的word count特征来fit一个简单的logistic regression
clf = LogisticRegression(C=1.0,solver="lbfgs",multi_class="multinomial")
clf.fit(xtrain_ctv,ytrain)
predictions = clf.predict_proba(xvalid_ctv)

print("logloss: %0.3f " %multiclass_logloss(yvalid,predictions))

y_predict = get_pred_rs(predictions)
multiclass_matrix(xvalid_ctv, yvalid, y_predict)


#最终采用LR模型，进行结果预测
test["文本分词"] = test["theme_new"].apply(lambda i:jieba.cut(str(i), False))
test["文本分词"] = [" ".join(i) for i in test["文本分词"]]
test["文本分词"] = test["文本分词"].apply(lambda i:rp_stop(str(i)))

test_ctv = ctv.transform(test.文本分词.values)
test_prd = clf.predict_proba(test_ctv)

rs = get_pred_rs(test_prd)
test["label"]=rs
test["预测结果"] = test["label"].apply(lambda i:y_labels[i])

test.head()

test.to_csv(r"LR预测type结果_20191009.csv",mode="w",header=True,encoding="utf-8")

#输出训练集结果
rs_valid = pd.DataFrame()
#rs_valid["theme_new"]=data["theme_new"]
rs_valid["分词结果"] = xvalid
rs_valid["label"] = y_predict
rs_valid["act_label"] = yvalid
rs_valid["预测结果"] = rs_valid["label"].apply(lambda i:y_labels[i])
rs_valid["人工分类"] = rs_valid["act_label"].apply(lambda i:y_labels[i])

rs_valid.head()


#输出
rs_valid.to_csv(r"训练集预测type结果_20191009.csv",mode="w",header=True,encoding="utf-8")



