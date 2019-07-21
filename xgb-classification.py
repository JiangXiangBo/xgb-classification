import pandas as pd
import random
import pickle
from sklearn.model_selection import ShuffleSplit, cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings

data_df = pd.read_csv('data.csv',parse_dates=["time"])
print(data_df.shape)
# 查看数据信息，有无缺失值
# data_df.info()
data_df.head()

"""
数据共三十九万多行，28个特征值 data.csv文件是带有所有特征字段的数据集； 
failure.csv文件是风力发电机叶片故障时间段，时间段包括2个字段：开始时间startTime、结束时间endTime；
normal.csv文件是风力发电机叶片正常时间段，时间段包括2个字段：开始时间startTime、结束时间endTime。
"""

# 1 数据处理，获取正常时间段
normalTime_df = pd.read_csv('normal.csv', parse_dates=["startTime", "endTime"])
print(normalTime_df.shape)
# normalTime_df.info()
normalTime_df.head()
# (26,2)

# 数据处理，获取故障时间段
failureTime_df = pd.read_csv('failure.csv', parse_dates=["startTime", "endTime"])
print(normalTime_df.shape)
# normalTime_df.info()
normalTime_df.head()
# (26,2)


# 2-取出预测目标值为正常的样本-根据时间段
normal_list = []
for index in normalTime_df.index:
    startTime = normalTime_df.loc[index].startTime
    endTime = normalTime_df.loc[index].endTime
    # 从样本中取出在这一段正常时间内所有的样本数据
    part_df = data_df[data_df.time.between(startTime, endTime)]
    print(part_df.shape)
    normal_list.append(part_df)

# 将正常样本合并为一个新的DataFrame
normal_df = pd.concat(normal_list).reset_index(drop=True)
print(normal_df.shape)
# (350255, 28)

# 取出预测目标值为故障的样本-根据时间段
failure_list = []
for index in failureTime_df.index:
    startTime = failureTime_df.loc[index].startTime
    endTime = failureTime_df.loc[index].endTime
    # 从样本中取出在这一段故障时间内所有的样本数据
    part_df = data_df[data_df.time.between(startTime, endTime)]
    print(part_df.shape)
    failure_list.append(part_df)

# 将故障样本合并为一个新的DataFrame
failure_df = pd.concat(failure_list).reset_index(drop=True)
print(failure_df.shape)
# (23892, 28)

# 统计正常，故障，无效样本占比
stat_df = pd.DataFrame(
    { 'number' : [normal_df.shape[0], failure_df.shape[0], data_df.shape[0]-normal_df.shape[0]-failure_df.shape[0]]},
      index = ['normal', 'failure', 'invalid'])
stat_df['ratio'] = stat_df['number'] / stat_df['number'].sum()
print(stat_df)
"""
正常-0.88/异常-0.05/无效-0.06
"""


# 3-下采样：
# 因为预测目标值为正常的样本远远多于预测目标值为故障的样本，所以对预测目标值为正常的样本做下采样。
# 下采样指减少样本或者减少特征，具体方法是选取一部分正常样本，数量为故障样本的2倍。
normalpart_df = normal_df.loc[random.sample(list(normal_df.index), k = failure_df.shape[0]*2)]
print(normalpart_df.shape)
# (47784, 28)

# 形成特征矩阵和预测目标值-把抽取出来的部分正常样本和故障样本和为一个样本
import numpy as np
feature_df = pd.concat([normalpart_df, failure_df]).reset_index(drop=True)
x= feature_df.drop('time', axis=1).values   # 丢掉时间列（无用）
print(x.shape)
y = np.append(np.ones(len(normalpart_df)), np.zeros(len(failure_df)))
print(y.shape)
# (71676, 27)
# (71676,)

# 保存数据
with open('x.pickle', 'wb') as file:
    pickle.dump(x, file)
with open('y.pickle', 'wb') as file:
    pickle.dump(y, file)


# 4 模型训练
# 4.1、加载处理好的数据
with open('x.pickle', 'rb') as file:
    x = pickle.load(file)

with open('y.pickle', 'rb') as file:
    y = pickle.load(file)

# 4.2xgboost模型
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
xgb_model = XGBClassifier(nthread=8)
xgb_model.fit(x_train, y_train)
xgb_model.score(x_test, y_test).round(4)


# 5.保存模型
with open("xgb_model.pickle", "wb") as file:
    pickle.dump(xgb_model, file)

# 6.模型加载
with open("xgb_model.pickle", "rb") as file:
    xgb_model = pickle.load(file)

# 6.2、测试数据准备
test_df = pd.read_csv("data_test.csv", index_col=0)
print(test_df.shape)
# (190494, 29)

# 预测目标值是clf字段，查看clf字段的统计计数情况
# 数字0代表故障的样本，1代表正常，2代表无效样本
test_df["clf"].value_counts()

y = test_df["clf"].values
x = test_df.drop(["time", "clf"], axis=1).values
# 去除无效样本
exam_x= x[y<2]
exam_y = y[y<2]
print(exam_x.shape)
print(exam_y.shape)
# (179567, 27)
# (179567,)

# 6.3 绘制precision、recall、f1-score、support报告表
def eval_model(y_true, y_pred, labels):
    # 计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    # 计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': ['总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [999]
    res = pd.concat([res1, res2])
    return res[['Label', 'Precision', 'Recall', 'F1', 'Support']]

predict_exam_y = xgb_model.predict(exam_x)
labels = ['故障', '正常']
eval_model(exam_y, predict_exam_y, labels)
# Precision-0.92
# Recall---0.93
