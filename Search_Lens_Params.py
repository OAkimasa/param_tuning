import numpy as np
import matplotlib.pyplot as plt
import time
from Tessar import pointsTessar

import pulp


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


def searchParam(Nlens1=1.43, Nlens2=1.43, Nlens3=1.43, Nlens4=1.45,
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


if __name__ == "__main__":
    print('\n----------------START----------------\n')
    start = time.time()

    searchParam()

    print('time =', time.time()-start)