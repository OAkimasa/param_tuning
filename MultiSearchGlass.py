import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
import csv

from multiprocessing import Pool
#import tensorflow as tf

from numpy.core.defchararray import array
from numpy.core.fromnumeric import shape

from GlassData import GlassData
from setting.MacroGlassMatrix import returnFocus
from setting.Out_GlassList_4core import trainParam
from setting.Out_GlassList_4core2 import trainParam2



# ガラスデータ参照、変数のマトリックスを作成
def searchParam_GlassData():
    print('----------generating matrix----------')
    GlassList = GlassData()

    ParamsMatrix = np.array(list(
            itertools.permutations(GlassList, 5)
            ))
    return ParamsMatrix

# 無限遠の場合から 1 mへ変更して計算するときの変数のマトリックスを作成
def searchParam_OutGlass():
    print('----------generating matrix----------')
    param = trainParam2()

    paramMatrix = []
    for i in param:
        paramMatrix.append(i[1])
    return paramMatrix


# 無限遠、推定される組み合わせを返す関数
def infMakeFocusList(args):
    results = []
    argsList = [args]
    for i in argsList:
        focus = returnFocus(i)
        dfocus = focus[2]
        if -0.01<=dfocus<=0.01 and 12.9<=focus[0]<=13.7 and 12.9<=focus[1]<=13.7:
            result = (focus, i)
            results.append(result)
    return results

# 1 m、推定される組み合わせを返す関数
def nearMakeFocusList(args):
    results = []
    argsList = [args]
    for i in argsList:
        focus = returnFocus(i)
        dfocus = focus[2]
        if -0.01<=dfocus<=0.01 and 13.3<=focus[0]<=13.7 and 13.3<=focus[1]<=13.7:
            result = (focus, i)
            results.append(result)
    return results

# 推定される組み合わせを返す関数、制限無し
def makeFocusList(args):
    results = []
    argsList = [args]
    for i in argsList:
        focus = returnFocus(i)
        dfocus = focus[2]
        result = (focus, i)
        results.append(result)
    return results

# csvファイルの作成
def out_csvFile(args):
    print('---------------writing---------------')
    results = []
    for i in args:
        if i != []:
            results.append(i)

    file = open('Out_GlassList_after_inf12345.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerows(results)
    file.close()



if __name__ == "__main__":
    print('\n----------------START----------------\n')
    start = time.time()

    ParamsMatrix = searchParam_GlassData()  # 総当たりか既存のリストか
    print('-------------calculating-------------')
    p = Pool(processes=4)
    results = p.map(func=infMakeFocusList, iterable=ParamsMatrix)
    out_csvFile(results)
    #print('result =', results)

    print('time =', time.time()-start)
    print('\n-----------------END-----------------\n')
