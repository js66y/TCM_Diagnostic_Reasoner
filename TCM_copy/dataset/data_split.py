import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:\\Project\\JST\\TCM\\dataset\\clear.csv', encoding="gbk")

# print("年龄最小值:", df['年龄'].min())
# print("年龄最大值:", df['年龄'].max())
# train_df = df.iloc[:100000, 2:]
# test_df = df.iloc[100000:, 2:]

# train_df.to_csv('dataset/train.csv', index=False)
# test_df.to_csv('dataset/test.csv', index=False)

# print(f"训练集大小: {len(train_df)} 条记录\n", train_df)
# print(f"测试集大小: {len(test_df)} 条记录\n", test_df)



def index2onehot(index, num_classes):
    method_hot = [0] * num_classes
    for i in index:
        method_hot[i] = 1
    return method_hot



num_herb = 510

def process_row(row):
    herb_i = sorted(list(eval(row['药名与单帖重量']).keys()))
    herb = np.array(index2onehot(herb_i, num_herb))
    measures = np.zeros_like(herb)
    for idx, val in eval(row['药名与单帖重量']).items():
        measures[idx] = val
    zz = eval(row['治则治法'])
    if len(zz) == 1:
        zz = [zz[0], zz[0], zz[0]]
    elif len(zz) == 2:
        zz = [zz[0], zz[1], zz[0]]
    iddata = [
        row['性别'],
        int(row['初步诊断']),
        int(row['中医诊断']),
        int(row['证型'])
    ] + zz
    return [int(row['年龄'])], iddata, herb_i + [-1] * (num_herb - len(herb_i)), list(herb), list(measures)
results = df.apply(process_row, axis=1)
results = list(results)
train_results, test_results = train_test_split(results, train_size=100000, random_state=42)

import pickle
with open('train.pkl', 'wb') as f:
    pickle.dump(train_results, f)
with open('test.pkl', 'wb') as f:
    pickle.dump(test_results, f)
