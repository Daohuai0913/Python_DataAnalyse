# coding: utf-8


# In[1]:

import pandas
titanic = pandas.read_csv("titanic_train.csv")  # 数据源可以搜索也可以加微信：nemoon
titanic.head(5)
# print (titanic.describe())  # 查看数据基本统计参数
# print(titanic.info())  # 查看数据基本类型和大小


# In[2]:

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
# print(titanic.describe()) # 用中位数来处理缺失值


# In[3]:

# print(titanic["Sex"].unique()) # 当年没有第三类人，否则会打印出NAN

# 将性别0,1化，男人0，女人1；在用pandas作统计或者后续的数据分析时，文本型数据要预处理。
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1


# In[4]:

print(titanic["Embarked"].unique())  # 登船港口有未知的，说明当年偷渡已经是常态，套票哪里都有。
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2


# In[22]:

# Import 线性回归类
from sklearn.linear_model import LinearRegression
# 交叉验证走起
from sklearn.cross_validation import KFold
# 自选特征量，船票本身和获救关系不大所以就没有入选
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# 实例化一个分类器
alg = LinearRegression()
# 生成一个交叉验证实例，titanic.shape[0]：数据集行数；n_splits：表示划分几等份；random_state：随机种子数
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    # 训练集
    train_predictors = (titanic[predictors].iloc[train,:])
    # 目标集（标签）
    train_target = titanic["Survived"].iloc[train]
    # 开始训练走起
    alg.fit(train_predictors, train_target)
    #  测试集
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    #  记录测试结果
    predictions.append(test_predictions)
# print(predictions)


# In[4]:

import numpy as np
# 测试结果是3个独立的矩阵（三份测试数据），接下来进行合并
predictions = np.concatenate(predictions, axis=0)
# 预测存货概率大于0.5生，小于等于0.5死（也是一瞬间）
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
# print(accuracy) # 精度


# In[5]:

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
alg = LogisticRegression(random_state=1)
# 直接计算交叉验证的结果，结果略有差异，下方法对三个分组的精度进行了平均
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# print(scores.mean())


# #### 预测

# In[6]:

titanic_test = pandas.read_csv("test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2


# In[7]:

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier  # 随机森林
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# random_state：随机数种子；子模型数:n_estimators;min_samples_split: 内部节点再划分所需最小样本数；min_samples_leaf:叶子节点最少样本数
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
# 平均预测结果
print(scores.mean())


# In[8]:

# 调整参数
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
print(scores.mean())


# In[9]:

# 加入家庭成员数作为特征
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
# titanic["NameLength"].head()


# In[10]:

import re
# 获取名字的title
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
# 获取title的词频
titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))  # 打印词频
# 将主要title数字化
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
# 验证假定
print(pandas.value_counts(titles))
# 增加一个title列
titanic["Title"] = titles


# In[11]:

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "NameLength"]

# 选择K个最好的特征，返回选择特征后的数据
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# 获取每个特征的p-values, 然后将其转化为得分
scores = -np.log10(selector.pvalues_)

# 选择四个最佳的特征
# predictors = ["Pclass", "Sex", "Fare", "Title"]
# alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

# Bokeh



# In[12]:
# 看看哪个特征获救的几率最大?
from bokeh.io import output_notebook, show
output_notebook()
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FactorRange


# In[13]:

source = ColumnDataSource({'predictors':predictors,'scores':scores})
source


# In[14]:

p = figure(title='泰坦尼克号乘客特征与存活率关系', y_range=FactorRange(factors=predictors), x_range=(0, 100), tools='save')
p.grid.grid_line_color = None
p.hbar(left=0, right='scores', y='predictors',height=0.5 ,color='seagreen', legend= None, source=source)
show(p)


# In[15]:

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# 迭代决策树
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title",]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms:
        # 训练集
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # 测试集
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # 测试准确率
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy,len(predictions))