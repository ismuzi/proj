import pandas as pd
import numpy as np
import math
import time
from sklearn.neighbors import NearestNeighbors
import heapq
import warnings

warnings.filterwarnings('ignore')

# 获取数据集
def getAllData(fileName):
    csv_data = pd.read_csv(fileName, header=None)
    dataLables = csv_data.iloc[:, -1].values
    dataLists = csv_data.iloc[:, 0:-1].values
    return dataLists, dataLables
def tck():
    k = 5
    if dataSet == 'D3' or dataSet == 'vowels':
        k = 7
    if dataSet == 'D4' or dataSet == 'iris':
        k = 4
    if dataSet == 'wdbc':
        k = 6
    return k


def tcc():
    c = 2
    if dataSet == 'Ionosphere-1':
        c = 5
    if dataSet == 'iris' or dataSet == 'vowels':
        c = 3
    return c

def eudis(a, b):
    return np.sqrt(sum((a - b) ** 2))

# 对数组降序排列，并返回对应索引值
def orderListDes(list):
    resultListIndex = np.argsort(-list)
    return resultListIndex

def outilercount(indexList, dataLable):
    count = 0
    for i in indexList:
        if dataLable[i] == 0:
            count += 1
    return count

def fdpc_of(dataSet):
    start = time.perf_counter()
    data, dataLable = getAllData('D:/研究生/数据集/dataset/dataset2/' + dataSet + '.csv')
    k = tck()
    c = tcc()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(data)
    distances, indices = nbrs.kneighbors(data)
    distance = np.mat(distances)
    density = (1 / distance.sum(axis=1)) / (k - 1)
    dis = np.full(len(data), 999.9999)
    maxArray = []
    for i in range(len(data)):
        maxD = 0
        for j in range(k):
            if j != 0 and density[indices[i][j]] > density[i] and distances[i][j] < dis[i]:
                maxD = 1
                dis[i] = min(dis[i], distances[i][j])
        if maxD == 0:
            maxArray.append(i)

    for i in maxArray:
        for j in maxArray:
            if i != j:
                dis[i] = min(dis[i], eudis(data[i], data[j]))
    xy = np.full(len(data), 0.0)
    for i in range(len(data)):
        xy[i] = density[i] * dis[i]
    maxxy = heapq.nlargest(c, range(len(xy)), xy.take)
    cindex = np.full(len(data), -1)
    for i in range(len(data)):
        if i not in maxxy:
            mind = 999.9
            for j in maxxy:
                if eudis(data[i], data[j]) < mind:
                    mind = eudis(data[i], data[j])
                    cindex[i] = j
    summ = np.full(c, 0)
    cou = np.full(c, 0)
    for i in range(len(data)):
        for j in range(c):
            if cindex[i] == maxxy[j]:
                summ[j] = summ[j] + eudis(data[i], maxxy[j])
                cou[j] = cou[j] + 1
    for i in range(c):
        summ[i] = summ[i] / cou[i]
    d = np.full(len(data), 0.001)
    for i in range(len(data)):
        if i not in maxxy:
            for j in range(c):
                if cindex[i] == maxxy[j]:
                    d[i] = dis[i] / summ[j]
    DPC1 = np.full(len(data), 0.001)
    for i in range(len(data)):
        DPC1[i] = d[i] / density[i]
    indexList = orderListDes(DPC1)
    Detectoutlercount = 0.0
    count = -1
    outs = outilercount(indexList, dataLable)
    end = time.perf_counter()  # 结束时刻

    for i in indexList:
        count = count + 1
        if dataLable[i] == 0:
            if count < outs:
                Detectoutlercount = Detectoutlercount + 1
    pr = Detectoutlercount / outs
    print(
        "算法FDPC-OF在数据集" + dataSet + "中的实验结果-----------------------------------------------------------------------------------")
    print("数据集中真实的离群点有：" + str(outs))
    print("前" + str(outs) + "个最大的里离群分数中是离群点的有" + str(Detectoutlercount))
    print('执行所用时间 %s' % str(end - start))
    print("精确率为：" + str(pr))
    Detectoutlercount = 0.0
    count = -1
    ott = math.floor(outs * 1.05)
    for i in indexList:
        count = count + 1
        if dataLable[i] == 0:
            if count < ott:
                Detectoutlercount = Detectoutlercount + 1
    re = Detectoutlercount / ott
    print("F1为：" + str((2 * re * pr) / (re + pr)))

if __name__ == '__main__':
    # dataSet = 'D1'
    # dataSet = 'D2'
    # dataSet = 'D3'
    # dataSet = 'D4'
    # dataSet = 'Ionosphere-1'
    # dataSet = 'iris'
    # dataSet = 'wdbc'
    dataSet = 'vowels'
    fdpc_of(dataSet)

