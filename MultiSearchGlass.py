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
from setting.MacroGlassMatrix_Lens12 import returnFocus_Lens12
from setting.MacroGlassMatrix_Lens45 import returnFocus_Lens45
#from setting.MacroGlassMatrix_Lens12re import returnFocus_Lens12re
#from setting.MacroGlassMatrix_Lens45re import returnFocus_Lens45re

from setting.Out_GlassList_4core import trainParam
from setting.Out_GlassList_4core2 import trainParam2



# ガラスデータ参照、変数のマトリックスを作製
# レンズ12345について作製
def searchParam_GlassData():
    print('----------generating matrix----------')
    GlassList = GlassData()

    ParamsMatrix = np.array(list(
            itertools.permutations(GlassList, 5)
            ))
    return ParamsMatrix

# レンズ12、レンズ45について作製
def searchParam_GlassData_Lens12and45():
    print('----------generating matrix----------')
    GlassList = GlassData()

    ParamsMatrix = np.array(list(
            itertools.permutations(GlassList, 2)
            ))
    return ParamsMatrix

'''
# 無限遠の場合から 1 mへ変更して計算するときの変数のマトリックスを作成
def searchParam_OutGlass():
    print('----------generating matrix----------')
    param = trainParam2()

    paramMatrix = []
    for i in param:
        paramMatrix.append(i[1])
    return paramMatrix
'''


# 無限遠、推定される組み合わせを返す関数
# レンズ12345、探索
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

# レンズ12、探索
def infMakeFocusList_Lens12(args):
    results = []
    argsList = [args]
    for i in argsList:
        focus = returnFocus_Lens12(i)
        dfocus = focus[2]
        if -0.6<=dfocus<=0.6 and 4.7<=focus[0]<=5.9 and 4.7<=focus[1]<=5.9:
            result = (focus, i)
            results.append(result)
    return results

# レンズ45、探索
def infMakeFocusList_Lens45(args):
    results = []
    argsList = [args]
    for i in argsList:
        focus = returnFocus_Lens45(i)
        dfocus = focus[2]
        if -0.4<=dfocus<=0.4 and 7.6<=focus[0]<=8.4 and 7.6<=focus[1]<=8.4:
            result = (focus, i)
            results.append(result)
    return results


'''
# レンズ12re、探索
def infMakeFocusList_Lens12re(args):
    results = []
    argsList = [args]
    for i in argsList:
        focus = returnFocus_Lens12re(i)
        dfocus = focus[2]
        if -1.1<=dfocus<=1.1 and 5.4<=focus[0]<=7.6 and 5.4<=focus[1]<=7.6:
            result = (focus, i)
            results.append(result)
    return results

# レンズ45re、探索
def infMakeFocusList_Lens45re(args):
    results = []
    argsList = [args]
    for i in argsList:
        focus = returnFocus_Lens45re(i)
        dfocus = focus[2]
        if -0.6<=dfocus<=0.6 and 7.1<=focus[0]<=8.3 and 7.1<=focus[1]<=8.3:
            result = (focus, i)
            results.append(result)
    return results
'''


'''
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
'''


# csvファイルの作成
def out_csvFile(args):
    print('---------------writing---------------')
    results = []
    for i in args:
        if i != []:
            results.append(i)

    file = open('Out_GlassList_after_inf12.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerows(results)
    file.close()



if __name__ == "__main__":
    print('\n----------------START----------------\n')
    start = time.time()

    ParamsMatrix = searchParam_GlassData_Lens12and45()  # 総当たりか既存のリストか
    print('-------------calculating-------------')
    p = Pool(processes=4)
    results = p.map(func=infMakeFocusList_Lens12, iterable=ParamsMatrix)
    out_csvFile(results)
    #print('result =', results)

    print('time =', time.time()-start)
    print('\n-----------------END-----------------\n')
