import numpy as np
import time
from Tessar import pointsTessar

start = time.time()

firstParams = [1.5, 1.5, 1.5, 1.8, 1.01, 1.01, 1.01, 1.01]  # ここから変化させて最適化する
#firstParams = [1.5, 1.5, 1.45, 1.8, 1.04, 1.04, 1.1048, 1.01]  # 自力で見つけた設定値

pointsRed = pointsTessar(*firstParams)[0]
pointsBlue = pointsTessar(*firstParams)[1]

'''
print('\n----------------RED----------------\n')
for i in pointsRed:
    print(i)
print('\n----------------BLUE----------------\n')
for i in pointsBlue:
    print(i)
'''

diff = np.array(pointsRed) - np.array(pointsBlue)
print('\n----------------DIFF----------------\n')
for i in diff:
    print(i)
print('norm=', np.linalg.norm(diff, ord=2))

print('time=', time.time()-start)
