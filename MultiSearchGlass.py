import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
import csv

from multiprocessing import Pool

from numpy.core.defchararray import array
from numpy.core.fromnumeric import shape

from GlassData import GlassData
from setting.MacroGlassMatrix import returnFocus


# ガラスデータ参照、変数のマトリックスを作成
def searchParam_GlassData():
    print('----------generating matrix----------')
    GlassList = GlassData()

    ParamsMatrix = np.array(list(
            itertools.permutations(GlassList, 5)
            ))
    return ParamsMatrix

# 推定される組み合わせを返す関数
def makeFocusList(args):
    results = []
    argsList = [args]
    for i in argsList:
        focus = returnFocus(i)
        dfocus = focus[2]
        if -0.03<=dfocus<=0.03 and 13.0<=focus[0]<=14.0 and 13.0<=focus[1]<=14.0:
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

    file = open('Out_GlassList.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerows(results)
    file.close()


if __name__ == "__main__":
    print('\n----------------START----------------\n')
    start = time.time()

    ParamsMatrix = searchParam_GlassData()
    print('-------------calculataing------------')
    p = Pool()
    results = p.map(func=makeFocusList, iterable=ParamsMatrix)
    out_csvFile(results)
    #print('result =', results)

    print('time =', time.time()-start)
    print('\n-----------------END-----------------\n')
