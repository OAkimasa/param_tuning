import numpy as np
import matplotlib.pyplot as plt
import time
from Tessar import pointsTessar
from ZoomLens import pointsZoomLens
from Macro135f4 import MacroLens
#import pulp


def calcNorm(Nlens1=1.43, Nlens2=1.43, Nlens3=1.43, Nlens4=1.70,
            NBlueRay1=1.01, NBlueRay2=1.01, NBlueRay3=1.01, NBlueRay4=1.01):
    Params = np.array([Nlens1, Nlens2, Nlens3, Nlens4,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4])  # ここから変化させて最適化する
    #Params = [1.5, 1.5, 1.45, 1.8, 1.04, 1.04, 1.1048, 1.01]  # 自力で見つけた設定値 norm = 0.04309

    pointsRed = pointsTessar(*Params)[0]
    pointsBlue = pointsTessar(*Params)[1]
    diff = np.array(pointsRed) - np.array(pointsBlue)

    '''
    print('\n----------------RED----------------\n')
    for i in pointsRed:
        print(i)
    print('\n----------------BLUE----------------\n')
    for i in pointsBlue:
        print(i)

    print('\n----------------DIFF----------------\n')
    for i in diff:
        print(i)
    '''
    resultNorm = np.linalg.norm(diff, ord=2)
    print('norm =', resultNorm, '  :   params =', Params)
    return resultNorm, Params

def calcNorm_ZoomLens(Nlens1=1.8, Nlens2=1.8, Nlens3=1.8, Nlens4=1.5,
            Nlens5=1.6, Nlens6=1.7, Nlens7=1.6, Nlens8=1.5,
            Nlens9=1.5, Nlens10=1.6, Nlens11=1.8, Nlens12=1.6,
            Nlens13=1.6, Nlens14=1.5, Nlens15=1.6,
            NBlueRay1=1.00, NBlueRay2=1.00, NBlueRay3=1.00, NBlueRay4=1.00,
            NBlueRay5=1.00, NBlueRay6=1.00, NBlueRay7=1.00, NBlueRay8=1.00,
            NBlueRay9=1.00, NBlueRay10=1.00, NBlueRay11=1.00, NBlueRay12=1.00,
            NBlueRay13=1.00, NBlueRay14=1.00, NBlueRay15=1.00):
    Params = np.array([Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15])  # ここから変化させて最適化する

    points = pointsZoomLens(*Params)
    pointsRed = points[0]
    pointsBlue = points[1]
    diff = np.array(pointsRed) - np.array(pointsBlue)

    resultNorm = np.linalg.norm(np.nan_to_num(
        diff, copy=False), ord=2)
    #print('norm =', resultNorm, '  :   params =', Params)
    return resultNorm, Params

def calcNorm_MacroLens(Nlens1=1.8, Nlens2=1.7, Nlens3=1.56, Nlens4=1.56, Nlens5=1.8,
            NBlueRay1=1.010, NBlueRay2=1.010, NBlueRay3=1.010, NBlueRay4=1.013, NBlueRay5=1.008):
    Params = np.array([Nlens1, Nlens2, Nlens3, Nlens4, Nlens5,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, NBlueRay5])  # ここから変化させて最適化する

    points = MacroLens(*Params)
    pointsRed = points[0]
    pointsBlue = points[1]
    diff = np.array(pointsRed) - np.array(pointsBlue)

    resultNorm = np.linalg.norm(np.nan_to_num(
        diff, copy=False), ord=2)
    #print('norm =', resultNorm, '  :   params =', Params)
    return resultNorm, Params

'''
# pulpは線形計画法なので、屈折の処理などができなかった
def pulpSearch():
    problem = pulp.LpProblem()

    # 変数の定義
    Nlens1 = pulp.LpVariable('Nlens1', lowBound = 1.4)
    Nlens2 = pulp.LpVariable('Nlens2', lowBound = 1.4)
    Nlens3 = pulp.LpVariable('Nlens3', lowBound = 1.4)
    Nlens4 = pulp.LpVariable('Nlens4', lowBound = 1.4)
    NBlueRay1 = pulp.LpVariable('NBlueRay1', lowBound = 1.008)
    NBlueRay2 = pulp.LpVariable('NBlueRay2', lowBound = 1.008)
    NBlueRay3 = pulp.LpVariable('NBlueRay3', lowBound = 1.008)
    NBlueRay4 = pulp.LpVariable('NBlueRay4', lowBound = 1.008)

    # 目的関数の設定
    problem += calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)

    # 制約条件の追加
    problem += Nlens1 <= 1.55
    problem += Nlens2 <= 1.55
    problem += Nlens3 <= 1.55
    problem += Nlens4 <= 1.81
    problem += NBlueRay1 <= 1.08
    problem += NBlueRay2 <= 1.08
    problem += NBlueRay3 <= 1.1
    problem += NBlueRay4 <= 1.15

    print(problem)

    # 実行
    status = problem.solve()
    print('status', pulp.LpStatus[status])

    # 結果の表示
    print('norm =', problem.objective.value())
'''


def searchParam_Tessar(Nlens1=1.43, Nlens2=1.43, Nlens3=1.43, Nlens4=1.45,
            NBlueRay1=1.008, NBlueRay2=1.008, NBlueRay3=1.008, NBlueRay4=1.006,
            dNl=0.01, dNB=0.001):
    minNorm12 = [10, [0]]
    for i in range(10):
        print('Lens12 :', i+1, '/10')
        calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)[0]
        Nlens1 += dNl
        Nlens2 += dNl

        for j in range(10):
            norm12 = calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                    NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)
            NBlueRay1 += dNB
            NBlueRay2 += dNB

            #print(minNorm12[0])
            if norm12[0] <= minNorm12[0]:
                print('!')
                minNorm12 = norm12
    Nlens1 = minNorm12[1][0]
    Nlens2 = minNorm12[1][1]
    NBlueRay1 = minNorm12[1][4]
    NBlueRay2 = minNorm12[1][5]

    minNorm3 = [10, [0]]
    for i in range(10):
        print('Lens3 :', i+1, '/10')
        calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)[0]
        Nlens3 += dNl

        for j in range(10):
            norm3 = calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                    NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)
            NBlueRay3 += dNB

            #print(minNorm3[0])
            if norm3[0] <= minNorm3[0]:
                print('!')
                minNorm3 = norm3
    Nlens3 = minNorm3[1][2]
    NBlueRay3 = minNorm3[1][6]

    minNorm4 = [10, [0]]
    for i in range(10):
        print('Lens4 :', i+1, '/10')
        calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)[0]
        Nlens4 += dNl

        for j in range(10):
            norm4 = calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                    NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)
            NBlueRay4 += dNB/10

            #print(minNorm4[0])
            if norm4[0] <= minNorm4[0]:
                print('!')
                minNorm4 = norm4
    Nlens4 = minNorm4[1][3]
    NBlueRay4 = minNorm4[1][7]

    print('12:best =', 'norm =', minNorm12[0], 'params =', *minNorm12[1])
    print('3:best =', 'norm =', minNorm3[0], 'params =', *minNorm3[1])
    print('4:best =', 'norm =', minNorm4[0], 'params =', *minNorm4[1])


def searchParam_Tessar_Layer(Nlens1=1.4, Nlens2=1.4, Nlens3=1.45, Nlens4=1.7,
            NBlueRay1=1.002, NBlueRay2=1.002, NBlueRay3=1.010, NBlueRay4=1.002,
            dNl=0.0001, dNB=0.00001):
    minNorm_toNext = [30, []]
    def Lens1Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm1=minNorm_toNext):
        count = 0
        for i in range(10):
            if 10 <= count:
                print('\n----Lens1 STOP----\n')
                break
            print('Lens1 :', i+1, '/10')
            calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                    NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)[0]
            Nlens1 += dNl
            for j in range(10):
                norm1 = calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                        NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)
                NBlueRay1 += dNB
                #print(minNorm1[0])
                if norm1[0] <= minNorm1[0]:
                    print('!')
                    minNorm1 = norm1
                    count = 0
                else:
                    count += 1
        Nlens1 = minNorm1[1][0]
        NBlueRay1 = minNorm1[1][4]
        return minNorm1, Nlens1, NBlueRay1

    def Lens2Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm2=minNorm_toNext):
        count = 0
        for i in range(10):
            if 10 <= count:
                print('\n----Lens2 STOP----\n')
                break
            print('Lens2 :', i+1, '/10')
            calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                    NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)[0]
            Nlens2 += dNl
            for j in range(10):
                norm2 = calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                        NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)
                NBlueRay2 += dNB
                #print(minNorm2[0])
                if norm2[0] <= minNorm2[0]:
                    print('!')
                    minNorm2 = norm2
                    count = 0
                else:
                    count += 1
        Nlens2 = minNorm2[1][1]
        NBlueRay2 = minNorm2[1][5]
        return minNorm2, Nlens2, NBlueRay2

    def Lens3Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm3=minNorm_toNext):
        count = 0
        for i in range(10):
            if 10 <= count:
                print('\n----Lens3 STOP----\n')
                break
            print('Lens3 :', i+1, '/10')
            calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                    NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)[0]
            Nlens3 += dNl
            for j in range(10):
                norm3 = calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                        NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)
                NBlueRay3 += dNB
                #print(minNorm3[0])
                if norm3[0] <= minNorm3[0]:
                    print('!')
                    minNorm3 = norm3
                    count = 0
                else:
                    count += 1
        Nlens3 = minNorm3[1][2]
        NBlueRay3 = minNorm3[1][6]
        return minNorm3, Nlens3, NBlueRay3

    def Lens4Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm4=minNorm_toNext):
        count = 0
        for i in range(10):
            if 10 <= count:
                print('\n----Lens4 STOP----\n')
                break
            print('Lens4 :', i+1, '/10')
            calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                    NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)[0]
            Nlens4 += dNl
            for j in range(10):
                norm4 = calcNorm(Nlens1, Nlens2, Nlens3, Nlens4,
                        NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4)
                NBlueRay4 += dNB/10
                #print(minNorm4[0])
                if norm4[0] <= minNorm4[0]:
                    print('!')
                    minNorm4 = norm4
                    count = 0
                else:
                    count += 1
        Nlens4 = minNorm4[1][3]
        NBlueRay4 = minNorm4[1][7]
        return minNorm4, Nlens4, NBlueRay4


    resultLayer3 = Lens3Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm_toNext)
    minNorm3 = resultLayer3[0]
    Nlens3 = resultLayer3[1]
    NBlueRay3 = resultLayer3[2]
    minNorm_toNext = minNorm3

    resultLayer2 = Lens2Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm_toNext)
    minNorm2 = resultLayer2[0]
    Nlens2 = resultLayer2[1]
    NBlueRay2 = resultLayer2[2]
    minNorm_toNext = minNorm2

    resultLayer4 = Lens4Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm_toNext)
    minNorm4 = resultLayer4[0]
    Nlens4 = resultLayer4[1]
    NBlueRay4 = resultLayer4[2]
    minNorm_toNext = minNorm4

    resultLayer1 = Lens1Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm_toNext)
    minNorm1 = resultLayer1[0]
    Nlens1 = resultLayer1[1]
    NBlueRay1 = resultLayer1[2]
    minNorm_toNext = minNorm1


    resultLayer1 = Lens1Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm_toNext)
    minNorm1 = resultLayer1[0]
    Nlens1 = resultLayer1[1]
    NBlueRay1 = resultLayer1[2]
    minNorm_toNext = minNorm1

    resultLayer2 = Lens2Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm_toNext)
    minNorm2 = resultLayer2[0]
    Nlens2 = resultLayer2[1]
    NBlueRay2 = resultLayer2[2]
    minNorm_toNext = minNorm2

    resultLayer3 = Lens3Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm_toNext)
    minNorm3 = resultLayer3[0]
    Nlens3 = resultLayer3[1]
    NBlueRay3 = resultLayer3[2]
    minNorm_toNext = minNorm3

    resultLayer4 = Lens4Layer(Nlens1, Nlens2, Nlens3, Nlens4,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4, minNorm_toNext)
    minNorm4 = resultLayer4[0]
    Nlens4 = resultLayer4[1]
    NBlueRay4 = resultLayer4[2]
    minNorm_toNext = minNorm4


    print('1:best =', 'norm =', minNorm1[0], 'params =', *minNorm1[1])
    print('2:best =', 'norm =', minNorm2[0], 'params =', *minNorm2[1])
    print('3:best =', 'norm =', minNorm3[0], 'params =', *minNorm3[1])
    print('4:best =', 'norm =', minNorm4[0], 'params =', *minNorm4[1])


def searchParam_ZoomLens_Layer(Nlens1=1.5, Nlens2=1.5, Nlens3=1.5, Nlens4=1.5,
            Nlens5=1.5, Nlens6=1.5, Nlens7=1.5, Nlens8=1.5,
            Nlens9=1.5, Nlens10=1.5, Nlens11=1.5, Nlens12=1.5,
            Nlens13=1.5, Nlens14=1.5, Nlens15=1.5,
            NBlueRay1=1.008, NBlueRay2=1.008, NBlueRay3=1.008, NBlueRay4=1.008,
            NBlueRay5=1.008, NBlueRay6=1.008, NBlueRay7=1.008, NBlueRay8=1.008,
            NBlueRay9=1.008, NBlueRay10=1.008, NBlueRay11=1.008, NBlueRay12=1.008,
            NBlueRay13=1.008, NBlueRay14=1.008, NBlueRay15=1.008,
            dNl=0.01, dNB=0.0001):
    minNorm_toNext = [30, []]
    def Lens1Layer(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15, minNorm1=minNorm_toNext):
        count = 0
        for i in range(10):
            if 10 <= count:
                print('\n----Lens1 STOP----\n')
                break
            print('Lens1 :', i+1, '/10')
            calcNorm_ZoomLens(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15)[0]
            Nlens1 += dNl
            for j in range(10):
                norm1 = calcNorm_ZoomLens(Nlens1, Nlens2, Nlens3, Nlens4,
                Nlens5, Nlens6, Nlens7, Nlens8,
                Nlens9, Nlens10, Nlens11, Nlens12,
                Nlens13, Nlens14, Nlens15,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
                NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
                NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
                NBlueRay13, NBlueRay14, NBlueRay15)
                NBlueRay1 += dNB
                #print(minNorm1[0])
                if norm1[0] <= minNorm1[0]:
                    print('!')
                    minNorm1 = norm1
                    count = 0
                else:
                    count += 1
        Nlens1 = minNorm1[1][0]
        NBlueRay1 = minNorm1[1][15]
        return minNorm1, Nlens1, NBlueRay1

    def Lens2Layer(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15, minNorm2=minNorm_toNext):
        count = 0
        for i in range(10):
            if 10 <= count:
                print('\n----Lens2 STOP----\n')
                break
            print('Lens2 :', i+1, '/10')
            calcNorm_ZoomLens(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15)[0]
            Nlens2 += dNl
            for j in range(10):
                norm2 = calcNorm_ZoomLens(Nlens1, Nlens2, Nlens3, Nlens4,
                Nlens5, Nlens6, Nlens7, Nlens8,
                Nlens9, Nlens10, Nlens11, Nlens12,
                Nlens13, Nlens14, Nlens15,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
                NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
                NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
                NBlueRay13, NBlueRay14, NBlueRay15)
                NBlueRay2 += dNB

                if norm2[0] <= minNorm2[0]:
                    print('!')
                    minNorm2 = norm2
                    count = 0
                else:
                    count += 1
        Nlens2 = minNorm2[1][2]
        NBlueRay2 = minNorm2[1][16]
        return minNorm2, Nlens2, NBlueRay2

    def Lens3Layer(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15, minNorm3=minNorm_toNext):
        count = 0
        for i in range(10):
            if 10 <= count:
                print('\n----Lens3 STOP----\n')
                break
            print('Lens3 :', i+1, '/10')
            calcNorm_ZoomLens(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15)[0]
            Nlens3 += dNl
            for j in range(10):
                norm3 = calcNorm_ZoomLens(Nlens1, Nlens2, Nlens3, Nlens4,
                Nlens5, Nlens6, Nlens7, Nlens8,
                Nlens9, Nlens10, Nlens11, Nlens12,
                Nlens13, Nlens14, Nlens15,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
                NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
                NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
                NBlueRay13, NBlueRay14, NBlueRay15)
                NBlueRay3 += dNB

                if norm3[0] <= minNorm3[0]:
                    print('!')
                    minNorm3 = norm3
                    count = 0
                else:
                    count += 1
        Nlens3 = minNorm3[1][3]
        NBlueRay3 = minNorm3[1][17]
        return minNorm3, Nlens3, NBlueRay3

    def Lens4Layer(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15, minNorm4=minNorm_toNext):
        count = 0
        for i in range(10):
            if 10 <= count:
                print('\n----Lens4 STOP----\n')
                break
            print('Lens4 :', i+1, '/10')
            calcNorm_ZoomLens(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15)[0]
            Nlens4 += dNl
            for j in range(10):
                norm4 = calcNorm_ZoomLens(Nlens1, Nlens2, Nlens3, Nlens4,
                Nlens5, Nlens6, Nlens7, Nlens8,
                Nlens9, Nlens10, Nlens11, Nlens12,
                Nlens13, Nlens14, Nlens15,
                NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
                NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
                NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
                NBlueRay13, NBlueRay14, NBlueRay15)
                NBlueRay4 += dNB

                if norm4[0] <= minNorm4[0]:
                    print('!')
                    minNorm4 = norm4
                    count = 0
                else:
                    count += 1
        Nlens4 = minNorm4[1][4]
        NBlueRay4 = minNorm4[1][18]
        return minNorm4, Nlens4, NBlueRay4

    resultLayer1 = Lens1Layer(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15, minNorm_toNext)
    minNorm1 = resultLayer1[0]
    Nlens1 = resultLayer1[1]
    NBlueRay1 = resultLayer1[2]
    minNorm_toNext = minNorm1

    resultLayer2 = Lens2Layer(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15, minNorm_toNext)
    minNorm2 = resultLayer2[0]
    Nlens2 = resultLayer2[1]
    NBlueRay2 = resultLayer2[2]
    minNorm_toNext = minNorm2

    resultLayer3 = Lens3Layer(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15, minNorm_toNext)
    minNorm3 = resultLayer3[0]
    Nlens3 = resultLayer3[1]
    NBlueRay3 = resultLayer3[2]
    minNorm_toNext = minNorm3

    resultLayer4 = Lens4Layer(Nlens1, Nlens2, Nlens3, Nlens4,
            Nlens5, Nlens6, Nlens7, Nlens8,
            Nlens9, Nlens10, Nlens11, Nlens12,
            Nlens13, Nlens14, Nlens15,
            NBlueRay1, NBlueRay2, NBlueRay3, NBlueRay4,
            NBlueRay5, NBlueRay6, NBlueRay7, NBlueRay8,
            NBlueRay9, NBlueRay10, NBlueRay11, NBlueRay12,
            NBlueRay13, NBlueRay14, NBlueRay15, minNorm_toNext)
    minNorm4 = resultLayer4[0]
    Nlens4 = resultLayer4[1]
    NBlueRay4 = resultLayer4[2]
    minNorm_toNext = minNorm4

    print('best =', 'norm =', minNorm_toNext[0], 'params =', *minNorm_toNext[1])


def searchParam_ZoomLens_Layer_v2():
    LayerOrder = [2, 1, 6, 5, 11,
                10, 15, 14, 3, 4,
                7, 8, 9, 12, 13]
    Nlensargs = [1.600, 1.701, 1.70, 1.50, 1.50,
                1.607, 1.60, 1.50, 1.50, 1.60,
                1.70, 1.50, 1.50, 1.70, 1.60]
    NBlueRayargs = [1.015, 1.012, 1.016, 1.013, 1.004,
                1.015, 1.005, 1.006, 1.004, 1.006,
                1.007, 1.004, 1.006, 1.008, 1.007]
    Nargs = Nlensargs + NBlueRayargs
    minNorm_toNext = [30, []]
    for k in LayerOrder:
        def LensLayer(Nlensargs, NBlueRayargs, minNorm=minNorm_toNext, dNl=0.001, dNB=0.0001):
            count = 0
            for i in range(10):
                if 10 <= count:
                    print('\n----Lens', k, ' STOP----\n')
                    break
                print('Lens', k, ' :', i+1, '/10')
                calcNorm_ZoomLens(*Nargs)[0]
                Nargs[k-1] += dNl
                for j in range(10):
                    norm = calcNorm_ZoomLens(*Nargs)
                    Nargs[k+14] += dNB
                    #print(minNorm[0])
                    if norm[0] <= minNorm[0]:
                        #print('!')
                        minNorm = norm
                        count = 0
                    else:
                        count += 1
            Nlens = minNorm[1][k-1]
            NBlueRay = minNorm[1][k+14]
            return minNorm, Nlens, NBlueRay

        resultLayer = LensLayer(Nlensargs, NBlueRayargs, minNorm_toNext)
        minNorm = resultLayer[0]
        Nlensargs = resultLayer[1]
        NBlueRayargs = resultLayer[2]
        minNorm_toNext = minNorm

        print('best =', 'norm =', minNorm_toNext[0], 'params =', *minNorm_toNext[1])


def searchParam_MacroLens_Layer():
    LayerOrder = [3, 2, 4, 1, 5]
    Nlensargs = [1.8, 1.7, 1.56, 1.56, 1.8]
    NBlueRayargs = [1.010, 1.010, 1.010, 1.013, 1.008]
    Nargs = Nlensargs + NBlueRayargs
    minNorm_toNext = [30, []]
    for k in LayerOrder:
        def LensLayer(Nlensargs, NBlueRayargs, minNorm=minNorm_toNext, dNl=0.001, dNB=0.0001):
            count = 0
            for i in range(10):
                if 10 <= count:
                    print('\n----Lens', k, ' STOP----\n')
                    break
                print('Lens', k, ' :', i+1, '/10')
                calcNorm_MacroLens(*Nargs)[0]
                Nargs[k-1] += dNl
                for j in range(10):
                    norm = calcNorm_MacroLens(*Nargs)
                    Nargs[k+4] += dNB
                    #print(minNorm[0])
                    if norm[0] <= minNorm[0]:
                        #print('!')
                        minNorm = norm
                        count = 0
                    else:
                        count += 1
            Nlens = minNorm[1][k-1]
            NBlueRay = minNorm[1][k+4]
            return minNorm, Nlens, NBlueRay

        resultLayer = LensLayer(Nlensargs, NBlueRayargs, minNorm_toNext)
        minNorm = resultLayer[0]
        Nlensargs = resultLayer[1]
        NBlueRayargs = resultLayer[2]
        minNorm_toNext = minNorm

        print('best =', 'norm =', minNorm_toNext[0], 'params =', *minNorm_toNext[1])


if __name__ == "__main__":
    print('\n----------------START----------------\n')
    start = time.time()

    # searchParam_Tessar_Layer or searchParam_ZoomLens_Layer
    searchParam_MacroLens_Layer()

    print('time =', time.time()-start)
