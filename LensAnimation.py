# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
import time


start = time.time()

LX = 6
LY = 6
LZ = 6
geneNum = 500
Nair = 1  # 空気の屈折率

rayStartV = [1000, 0, 0]
centerX = -10  # 入射光の始点の中心座標
centerY = 0  # 入射光の始点の中心座標
centerZ = 0  # 入射光の始点の中心座標
rayDensity = 0.1  # 入射光の密度
focusX = 0  # 焦点付近の描画範囲を平行移動

# アニメーション
for j in range(40):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    if 0<=j<=19:
        Unit1X = 0 - j/5
    elif 20<=j<=39:
        Unit1X = -4

    if 0<=j<=19:
        Unit2X = -0 - (j) / 10
        print(Unit2X)
    elif 20<=j<=39:
        Unit2X = -2 + ((j-20) / 10)

    Unit3X = 0
    Unit4X = 1.92 + Unit3X + 0

    Rx11 = 16.67  # レンズ１の倍率1
    Rx12 = 7.06  # レンズ１の倍率2
    Rx21 = Rx12  # レンズ２の倍率１
    Rx22 = 44.44  # レンズ２の倍率２
    Rx31 = 7.78  # レンズ３の倍率１
    Rx32 = 12.78  # レンズ３の倍率２
    Rx41 = 14.44  # レンズ４の倍率1
    Rx42 = 1.44  # レンズ４の倍率2
    Rx51 = 3.33  # レンズ5の倍率1
    Rx52 = 3.33  # レンズ5の倍率2
    Rx61 = Rx52  # レンズ6の倍率１
    Rx62 = 6.94  # レンズ6の倍率２
    Rx71 = 3.61  # レンズ7の倍率１
    Rx72 = 2.50  # レンズ7の倍率２
    Rx81 = 2.22  # レンズ8の倍率1
    Rx82 = 8.33  # レンズ8の倍率2
    Rx91 = 4.17  # レンズ9の倍率1
    Rx92 = 4.17  # レンズ9の倍率2
    Rx101 = 3.06  # レンズ１0の倍率1
    Rx102 = 3.06  # レンズ１0の倍率2
    Rx111 = Rx102  # レンズ１1の倍率1
    Rx112 = 5.00  # レンズ１1の倍率2
    Rx121 = 3.72  # レンズ1２の倍率１
    Rx122 = 3.72  # レンズ1２の倍率２
    Rx131 = 9.44  # レンズ1３の倍率１
    Rx132 = 3.33  # レンズ1３の倍率２
    Rx141 = 8.33  # レンズ14の倍率１
    Rx142 = 2.50  # レンズ14の倍率２
    Rx151 = Rx142  # レンズ15の倍率１
    Rx152 = 5.00  # レンズ15の倍率２

    lens1V = np.array([-0-6+Unit1X, 0, 0])  # レンズ１の位置ベクトル
    lens2V = np.array([0.22-6+Unit1X, 0, 0])  # レンズ２の位置ベクトル
    lens3V = np.array([1.16-6+Unit1X, 0, 0])  # レンズ３の位置ベクトル

    lens4V = np.array([0-3.7+Unit2X, 0, 0])  # レンズ４の位置ベクトル
    lens5V = np.array([0.88-3.7+Unit2X, 0, 0])  # レンズ5の位置ベクトル
    lens6V = np.array([1.05-3.7+Unit2X, 0, 0])  # レンズ6の位置ベクトル
    lens7V = np.array([1.33-3.7+Unit2X, 0, 0])  # レンズ7の位置ベクトル
    lens8V = np.array([2.05-3.7+Unit2X, 0, 0])  # レンズ8の位置ベクトル

    lens9V = np.array([0+Unit3X, 0, 0])  # レンズ9の位置ベクトル
    lens10V = np.array([0.39+Unit3X, 0, 0])  # レンズ１0の位置ベクトル
    lens11V = np.array([1.06+Unit3X, 0, 0])  # レンズ１1の位置ベクトル

    lens12V = np.array([0+Unit4X, 0, 0])  # レンズ1２の位置ベクトル
    lens13V = np.array([0.56+Unit4X, 0, 0])  # レンズ1３の位置ベクトル
    lens14V = np.array([1+Unit4X, 0, 0])  # レンズ14の位置ベクトル
    lens15V = np.array([1.17+Unit4X, 0, 0])  # レンズ15の位置ベクトル

    screenV = np.array([6, 0, 0])  # スクリーンの位置ベクトル

    Lens1Param = [16.67, 7.06, 0.22, [3.69, 3.44], [-1, -1], lens1V]
    Lens2Param = [7.06, 44.44, 0.94, [3.44, 3.44], [-1, -1], lens2V]
    Lens3Param = [7.78, 12.78, 0.72, [3.06, 3.06], [-1, -1], lens3V]
    Lens4Param = [14.44, 1.44, 0.22, [1.75, 1.19], [-1, -1], lens4V]
    Lens5Param = [3.33, 3.33, 0.17, [1.19, 1.19], [1, -1], lens5V]
    Lens6Param = [3.33, 6.94, 0.28, [1.08, 1.08], [-1, -1], lens6V]
    Lens7Param = [3.61, 2.50, 0.67, [1.14, 1.14], [-1, 1], lens7V]
    Lens8Param = [2.22, 8.33, 0.17, [0.94, 0.94], [1, 1], lens8V]
    Lens9Param = [4.17, 4.17, 0.39, [1.08, 1.08], [-1, 1], lens9V]
    Lens10Param = [3.06, 3.06, 0.67, [1.11, 1.11], [-1, 1], lens10V]
    Lens11Param = [3.06, 5.00, 0.22, [1.11, 1.11], [1, -1], lens11V]
    Lens12Param = [3.72, 3.72, 0.56, [1.00, 1.00], [-1, 1], lens12V]
    Lens13Param = [9.44, 3.33, 0.22, [0.97, 0.97], [-1, -1], lens13V]
    Lens14Param = [8.33, 2.50, 0.17, [1.17, 1.17], [1, -1], lens14V]
    Lens15Param = [2.50, 5.00, 0.56, [1.08, 1.08], [-1, 1], lens15V]

    Params = [Lens1Param, Lens2Param, Lens3Param, Lens4Param,
            Lens5Param, Lens6Param, Lens7Param, Lens8Param,
            Lens9Param, Lens10Param, Lens11Param, Lens12Param,
            Lens13Param, Lens14Param, Lens15Param]


    class VectorFunctions:
        # 受け取ったx,y,z座標から(x,y,z)の組を作る関数
        def makePoints(self, point0, point1, point2, shape0, shape1):
                result = [None]*(len(point0)+len(point1)+len(point2))
                result[::3] = point0
                result[1::3] = point1
                result[2::3] = point2
                result = np.array(result)
                result = result.reshape(shape0, shape1)
                return result


        # レイトレーシング、光線ベクトルとレンズ１の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens1L(self, startV, directionV):
                startV = startV - lens1V
                A = (directionV[0]**2/Rx11**2)+(
                        directionV[1]**2/Rx11**2)+(
                        directionV[2]**2/Rx11**2)
                #print(A)
                B = ((startV[0] - 16.67)*directionV[0]/Rx11**2)+(
                        startV[1]*directionV[1]/Rx11**2)+(
                        startV[2]*directionV[2]/Rx11**2)
                #print(B)
                C = -1+((startV[0] - 16.67)**2/Rx11**2)+(
                        startV[1]**2/Rx11**2)+(
                        startV[2]**2/Rx11**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens1R(self, startV, directionV):
                startV = startV - lens1V
                A = (directionV[0]**2/Rx12**2)+(
                        directionV[1]**2/Rx12**2)+(
                        directionV[2]**2/Rx12**2)
                #print(A)
                B = ((startV[0] - 7.06 - 0.22)*directionV[0]/Rx12**2)+(
                        startV[1]*directionV[1]/Rx12**2)+(
                        startV[2]*directionV[2]/Rx12**2)
                #print(B)
                C = -1+((startV[0] - 7.06 - 0.22)**2/Rx12**2)+(
                        startV[1]**2/Rx12**2)+(
                        startV[2]**2/Rx12**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        # レンズ１表面の法線を求める関数
        def decideNormalV_Lens1L(self, pointV):
                pointV = pointV - lens1V
                nornalVx = (2/Rx11**2)*(pointV[0] - 16.67)
                nornalVy = (2/Rx11**2)*pointV[1]
                nornalVz = (2/Rx11**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens1R(self, pointV):
                pointV = pointV - lens1V
                nornalVx = -(2/Rx12**2)*(pointV[0] - 7.06 - 0.22)
                nornalVy = -(2/Rx12**2)*pointV[1]
                nornalVz = -(2/Rx12**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ２の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens2L(self, startV, directionV):
                startV = startV - lens2V
                A = (directionV[0]**2/Rx21**2)+(
                        directionV[1]**2/Rx21**2)+(
                        directionV[2]**2/Rx21**2)
                #print(A)
                B = ((startV[0] - 7.06)*directionV[0]/Rx21**2)+(
                        startV[1]*directionV[1]/Rx21**2)+(
                        startV[2]*directionV[2]/Rx21**2)
                #print(B)
                C = -1+((startV[0] - 7.06)**2/Rx21**2)+(
                        startV[1]**2/Rx21**2)+(
                        startV[2]**2/Rx21**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens2R(self, startV, directionV):
                startV = startV - lens2V
                A = (directionV[0]**2/Rx22**2)+(
                        directionV[1]**2/Rx22**2)+(
                        directionV[2]**2/Rx22**2)
                #print(A)
                B = ((startV[0] - 44.44 - 0.94)*directionV[0]/Rx22**2)+(
                        startV[1]*directionV[1]/Rx22**2)+(
                        startV[2]*directionV[2]/Rx22**2)
                #print(B)
                C = -1+((startV[0] - 44.44 - 0.94)**2/Rx22**2)+(
                        startV[1]**2/Rx22**2)+(
                        startV[2]**2/Rx22**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        # レンズ２表面の法線を求める関数
        def decideNormalV_Lens2L(self, pointV):
                pointV = pointV - lens2V
                nornalVx = (2/Rx21**2)*(pointV[0] - 7.06)
                nornalVy = (2/Rx21**2)*pointV[1]
                nornalVz = (2/Rx21**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens2R(self, pointV):
                pointV = pointV - lens2V
                nornalVx = -(2/Rx22**2)*(pointV[0] - 44.44 - 0.94)
                nornalVy = -(2/Rx22**2)*pointV[1]
                nornalVz = -(2/Rx22**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ３の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens3L(self, startV, directionV):
                startV = startV - lens3V
                A = (directionV[0]**2/Rx31**2)+(
                        directionV[1]**2/Rx31**2)+(
                        directionV[2]**2/Rx31**2)
                #print(A)
                B = ((startV[0] - 7.78)*directionV[0]/Rx31**2)+(
                        startV[1]*directionV[1]/Rx31**2)+(
                        startV[2]*directionV[2]/Rx31**2)
                #print(B)
                C = -1+((startV[0] - 7.78)**2/Rx31**2)+(
                        startV[1]**2/Rx31**2)+(
                        startV[2]**2/Rx31**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens3R(self, startV, directionV):
                startV = startV - lens3V
                A = (directionV[0]**2/Rx32**2)+(
                        directionV[1]**2/Rx32**2)+(
                        directionV[2]**2/Rx32**2)
                #print(A)
                B = ((startV[0] - 12.78 - 0.72)*directionV[0]/Rx32**2)+(
                        startV[1]*directionV[1]/Rx32**2)+(
                        startV[2]*directionV[2]/Rx32**2)
                #print(B)
                C = -1+((startV[0] - 12.78 - 0.72)**2/Rx32**2)+(
                        startV[1]**2/Rx32**2)+(
                        startV[2]**2/Rx32**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        # レンズ３表面の法線を求める関数
        def decideNormalV_Lens3L(self, pointV):
                pointV = pointV - lens3V
                nornalVx = (2/Rx31**2)*(pointV[0] - 7.78)
                nornalVy = (2/Rx31**2)*pointV[1]
                nornalVz = (2/Rx31**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens3R(self, pointV):
                pointV = pointV - lens3V
                nornalVx = -(2/Rx32**2)*(pointV[0] - 12.78 - 0.72)
                nornalVy = -(2/Rx32**2)*pointV[1]
                nornalVz = -(2/Rx32**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ４の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens4L(self, startV, directionV):
                startV = startV - lens4V
                A = (directionV[0]**2/Rx41**2)+(
                        directionV[1]**2/Rx41**2)+(
                        directionV[2]**2/Rx41**2)
                #print(A)
                B = ((startV[0] - 14.44)*directionV[0]/Rx41**2)+(
                        startV[1]*directionV[1]/Rx41**2)+(
                        startV[2]*directionV[2]/Rx41**2)
                #print(B)
                C = -1+((startV[0] - 14.44)**2/Rx41**2)+(
                        startV[1]**2/Rx41**2)+(
                        startV[2]**2/Rx41**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens4R(self, startV, directionV):
                startV = startV - lens4V
                A = (directionV[0]**2/Rx42**2)+(
                        directionV[1]**2/Rx42**2)+(
                        directionV[2]**2/Rx42**2)
                #print(A)
                B = ((startV[0] - 1.44 - 0.22)*directionV[0]/Rx42**2)+(
                        startV[1]*directionV[1]/Rx42**2)+(
                        startV[2]*directionV[2]/Rx42**2)
                #print(B)
                C = -1+((startV[0] - 1.44 - 0.22)**2/Rx42**2)+(
                        startV[1]**2/Rx42**2)+(
                        startV[2]**2/Rx42**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        # レンズ４表面の法線を求める関数
        def decideNormalV_Lens4L(self, pointV):
                pointV = pointV - lens4V
                nornalVx = (2/Rx41**2)*(pointV[0] - 14.44)
                nornalVy = (2/Rx41**2)*pointV[1]
                nornalVz = (2/Rx41**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens4R(self, pointV):
                pointV = pointV - lens4V
                nornalVx = -(2/Rx42**2)*(pointV[0] - 1.44 - 0.22)
                nornalVy = -(2/Rx42**2)*pointV[1]
                nornalVz = -(2/Rx42**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ5の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens5L(self, startV, directionV):
                startV = startV - lens5V
                A = (directionV[0]**2/Rx51**2)+(
                        directionV[1]**2/Rx51**2)+(
                        directionV[2]**2/Rx51**2)
                #print(A)
                B = ((startV[0] + 3.33)*directionV[0]/Rx51**2)+(
                        startV[1]*directionV[1]/Rx51**2)+(
                        startV[2]*directionV[2]/Rx51**2)
                #print(B)
                C = -1+((startV[0] + 3.33)**2/Rx51**2)+(
                        startV[1]**2/Rx51**2)+(
                        startV[2]**2/Rx51**2)
                #print(C)
                T = (-B+np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens5R(self, startV, directionV):
                startV = startV - lens5V
                A = (directionV[0]**2/Rx52**2)+(
                        directionV[1]**2/Rx52**2)+(
                        directionV[2]**2/Rx52**2)
                #print(A)
                B = ((startV[0] - 3.33 - 0.17)*directionV[0]/Rx52**2)+(
                        startV[1]*directionV[1]/Rx52**2)+(
                        startV[2]*directionV[2]/Rx52**2)
                #print(B)
                C = -1+((startV[0] - 3.33 - 0.17)**2/Rx52**2)+(
                        startV[1]**2/Rx52**2)+(
                        startV[2]**2/Rx52**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        # レンズ5表面の法線を求める関数
        def decideNormalV_Lens5L(self, pointV):
                pointV = pointV - lens5V
                nornalVx = -(2/Rx51**2)*(pointV[0] + 3.33)
                nornalVy = -(2/Rx51**2)*pointV[1]
                nornalVz = -(2/Rx51**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens5R(self, pointV):
                pointV = pointV - lens5V
                nornalVx = -(2/Rx52**2)*(pointV[0] - 3.33 - 0.17)
                nornalVy = -(2/Rx52**2)*pointV[1]
                nornalVz = -(2/Rx52**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ6の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens6L(self, startV, directionV):
                startV = startV - lens6V
                A = (directionV[0]**2/Rx61**2)+(
                        directionV[1]**2/Rx61**2)+(
                        directionV[2]**2/Rx61**2)
                #print(A)
                B = ((startV[0] - 3.33)*directionV[0]/Rx61**2)+(
                        startV[1]*directionV[1]/Rx61**2)+(
                        startV[2]*directionV[2]/Rx61**2)
                #print(B)
                C = -1+((startV[0] - 3.33)**2/Rx61**2)+(
                        startV[1]**2/Rx61**2)+(
                        startV[2]**2/Rx61**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens6R(self, startV, directionV):
                startV = startV - lens6V
                A = (directionV[0]**2/Rx62**2)+(
                        directionV[1]**2/Rx62**2)+(
                        directionV[2]**2/Rx62**2)
                #print(A)
                B = ((startV[0] - 6.94 - 0.28)*directionV[0]/Rx62**2)+(
                        startV[1]*directionV[1]/Rx62**2)+(
                        startV[2]*directionV[2]/Rx62**2)
                #print(B)
                C = -1+((startV[0] - 6.94 - 0.28)**2/Rx62**2)+(
                        startV[1]**2/Rx62**2)+(
                        startV[2]**2/Rx62**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        # レンズ6表面の法線を求める関数
        def decideNormalV_Lens6L(self, pointV):
                pointV = pointV - lens6V
                nornalVx = (2/Rx61**2)*(pointV[0] - 3.33)
                nornalVy = (2/Rx61**2)*pointV[1]
                nornalVz = (2/Rx61**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens6R(self, pointV):
                pointV = pointV - lens6V
                nornalVx = -(2/Rx62**2)*(pointV[0] - 6.94 - 0.28)
                nornalVy = -(2/Rx62**2)*pointV[1]
                nornalVz = -(2/Rx62**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ7の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens7L(self, startV, directionV):
                startV = startV - lens7V
                A = (directionV[0]**2/Rx71**2)+(
                        directionV[1]**2/Rx71**2)+(
                        directionV[2]**2/Rx71**2)
                #print(A)
                B = ((startV[0] - 3.61)*directionV[0]/Rx71**2)+(
                        startV[1]*directionV[1]/Rx71**2)+(
                        startV[2]*directionV[2]/Rx71**2)
                #print(B)
                C = -1+((startV[0] - 3.61)**2/Rx71**2)+(
                        startV[1]**2/Rx71**2)+(
                        startV[2]**2/Rx71**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens7R(self, startV, directionV):
                startV = startV - lens7V
                A = (directionV[0]**2/Rx72**2)+(
                        directionV[1]**2/Rx72**2)+(
                        directionV[2]**2/Rx72**2)
                #print(A)
                B = ((startV[0] + 2.50 - 0.67)*directionV[0]/Rx72**2)+(
                        startV[1]*directionV[1]/Rx72**2)+(
                        startV[2]*directionV[2]/Rx72**2)
                #print(B)
                C = -1+((startV[0] + 2.50 - 0.67)**2/Rx72**2)+(
                        startV[1]**2/Rx72**2)+(
                        startV[2]**2/Rx72**2)
                #print(C)
                T = (-B+np.sqrt(B**2-A*C))/A
                return T

        # レンズ7表面の法線を求める関数
        def decideNormalV_Lens7L(self, pointV):
                pointV = pointV - lens7V
                nornalVx = (2/Rx71**2)*(pointV[0] - 3.61)
                nornalVy = (2/Rx71**2)*pointV[1]
                nornalVz = (2/Rx71**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens7R(self, pointV):
                pointV = pointV - lens7V
                nornalVx = (2/Rx72**2)*(pointV[0] + 2.50 - 0.67)
                nornalVy = (2/Rx72**2)*pointV[1]
                nornalVz = (2/Rx72**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ8の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens8L(self, startV, directionV):
                startV = startV - lens8V
                A = (directionV[0]**2/Rx81**2)+(
                        directionV[1]**2/Rx81**2)+(
                        directionV[2]**2/Rx81**2)
                #print(A)
                B = ((startV[0] + 2.22)*directionV[0]/Rx81**2)+(
                        startV[1]*directionV[1]/Rx81**2)+(
                        startV[2]*directionV[2]/Rx81**2)
                #print(B)
                C = -1+((startV[0] + 2.22)**2/Rx81**2)+(
                        startV[1]**2/Rx81**2)+(
                        startV[2]**2/Rx81**2)
                #print(C)
                T = (-B+np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens8R(self, startV, directionV):
                startV = startV - lens8V
                A = (directionV[0]**2/Rx82**2)+(
                        directionV[1]**2/Rx82**2)+(
                        directionV[2]**2/Rx82**2)
                #print(A)
                B = ((startV[0] + 8.33 - 0.17)*directionV[0]/Rx82**2)+(
                        startV[1]*directionV[1]/Rx82**2)+(
                        startV[2]*directionV[2]/Rx82**2)
                #print(B)
                C = -1+((startV[0] + 8.33 - 0.17)**2/Rx82**2)+(
                        startV[1]**2/Rx82**2)+(
                        startV[2]**2/Rx82**2)
                #print(C)
                T = (-B+np.sqrt(B**2-A*C))/A
                return T

        # レンズ8表面の法線を求める関数
        def decideNormalV_Lens8L(self, pointV):
                pointV = pointV - lens8V
                nornalVx = -(2/Rx81**2)*(pointV[0] + 2.22)
                nornalVy = -(2/Rx81**2)*pointV[1]
                nornalVz = -(2/Rx81**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens8R(self, pointV):
                pointV = pointV - lens8V
                nornalVx = (2/Rx82**2)*(pointV[0] + 8.33 - 0.17)
                nornalVy = (2/Rx82**2)*pointV[1]
                nornalVz = (2/Rx82**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ9の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens9L(self, startV, directionV):
                startV = startV - lens9V
                A = (directionV[0]**2/Rx91**2)+(
                        directionV[1]**2/Rx91**2)+(
                        directionV[2]**2/Rx91**2)
                #print(A)
                B = ((startV[0] - 4.17)*directionV[0]/Rx91**2)+(
                        startV[1]*directionV[1]/Rx91**2)+(
                        startV[2]*directionV[2]/Rx91**2)
                #print(B)
                C = -1+((startV[0] - 4.17)**2/Rx91**2)+(
                        startV[1]**2/Rx91**2)+(
                        startV[2]**2/Rx91**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens9R(self, startV, directionV):
                startV = startV - lens9V
                A = (directionV[0]**2/Rx92**2)+(
                        directionV[1]**2/Rx92**2)+(
                        directionV[2]**2/Rx92**2)
                #print(A)
                B = ((startV[0] + 4.17 - 0.39)*directionV[0]/Rx92**2)+(
                        startV[1]*directionV[1]/Rx92**2)+(
                        startV[2]*directionV[2]/Rx92**2)
                #print(B)
                C = -1+((startV[0] + 4.17 - 0.39)**2/Rx92**2)+(
                        startV[1]**2/Rx92**2)+(
                        startV[2]**2/Rx92**2)
                #print(C)
                T = (-B+np.sqrt(B**2-A*C))/A
                return T

        # レンズ9表面の法線を求める関数
        def decideNormalV_Lens9L(self, pointV):
                pointV = pointV - lens9V
                nornalVx = (2/Rx91**2)*(pointV[0] - 4.17)
                nornalVy = (2/Rx91**2)*pointV[1]
                nornalVz = (2/Rx91**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens9R(self, pointV):
                pointV = pointV - lens9V
                nornalVx = (2/Rx92**2)*(pointV[0] + 4.17 - 0.39)
                nornalVy = (2/Rx92**2)*pointV[1]
                nornalVz = (2/Rx92**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ10の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens10L(self, startV, directionV):
                startV = startV - lens10V
                A = (directionV[0]**2/Rx101**2)+(
                        directionV[1]**2/Rx101**2)+(
                        directionV[2]**2/Rx101**2)
                #print(A)
                B = ((startV[0] - 3.06)*directionV[0]/Rx101**2)+(
                        startV[1]*directionV[1]/Rx101**2)+(
                        startV[2]*directionV[2]/Rx101**2)
                #print(B)
                C = -1+((startV[0] - 3.06)**2/Rx101**2)+(
                        startV[1]**2/Rx101**2)+(
                        startV[2]**2/Rx101**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens10R(self, startV, directionV):
                startV = startV - lens10V
                A = (directionV[0]**2/Rx102**2)+(
                        directionV[1]**2/Rx102**2)+(
                        directionV[2]**2/Rx102**2)
                #print(A)
                B = ((startV[0] + 3.06 - 0.67)*directionV[0]/Rx102**2)+(
                        startV[1]*directionV[1]/Rx102**2)+(
                        startV[2]*directionV[2]/Rx102**2)
                #print(B)
                C = -1+((startV[0] + 3.06 - 0.67)**2/Rx102**2)+(
                        startV[1]**2/Rx102**2)+(
                        startV[2]**2/Rx102**2)
                #print(C)
                T = (-B+np.sqrt(B**2-A*C))/A
                return T

        # レンズ10表面の法線を求める関数
        def decideNormalV_Lens10L(self, pointV):
                pointV = pointV - lens10V
                nornalVx = (2/Rx101**2)*(pointV[0] - 3.06)
                nornalVy = (2/Rx101**2)*pointV[1]
                nornalVz = (2/Rx101**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens10R(self, pointV):
                pointV = pointV - lens10V
                nornalVx = (2/Rx102**2)*(pointV[0] + 3.06 - 0.67)
                nornalVy = (2/Rx102**2)*pointV[1]
                nornalVz = (2/Rx102**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ11の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens11L(self, startV, directionV):
                startV = startV - lens11V
                A = (directionV[0]**2/Rx111**2)+(
                        directionV[1]**2/Rx111**2)+(
                        directionV[2]**2/Rx111**2)
                #print(A)
                B = ((startV[0] + 3.06)*directionV[0]/Rx111**2)+(
                        startV[1]*directionV[1]/Rx111**2)+(
                        startV[2]*directionV[2]/Rx111**2)
                #print(B)
                C = -1+((startV[0] + 3.06)**2/Rx111**2)+(
                        startV[1]**2/Rx111**2)+(
                        startV[2]**2/Rx111**2)
                #print(C)
                T = (-B+np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens11R(self, startV, directionV):
                startV = startV - lens11V
                A = (directionV[0]**2/Rx112**2)+(
                        directionV[1]**2/Rx112**2)+(
                        directionV[2]**2/Rx112**2)
                #print(A)
                B = ((startV[0] - 5.00 - 0.22)*directionV[0]/Rx112**2)+(
                        startV[1]*directionV[1]/Rx112**2)+(
                        startV[2]*directionV[2]/Rx112**2)
                #print(B)
                C = -1+((startV[0] - 5.00 - 0.22)**2/Rx112**2)+(
                        startV[1]**2/Rx112**2)+(
                        startV[2]**2/Rx112**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        # レンズ11表面の法線を求める関数
        def decideNormalV_Lens11L(self, pointV):
                pointV = pointV - lens11V
                nornalVx = -(2/Rx111**2)*(pointV[0] + 3.06)
                nornalVy = -(2/Rx111**2)*pointV[1]
                nornalVz = -(2/Rx111**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens11R(self, pointV):
                pointV = pointV - lens11V
                nornalVx = -(2/Rx112**2)*(pointV[0] - 5.00 - 0.22)
                nornalVy = -(2/Rx112**2)*pointV[1]
                nornalVz = -(2/Rx112**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ12の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens12L(self, startV, directionV):
                startV = startV - lens12V
                A = (directionV[0]**2/Rx121**2)+(
                        directionV[1]**2/Rx121**2)+(
                        directionV[2]**2/Rx121**2)
                #print(A)
                B = ((startV[0] - 3.72)*directionV[0]/Rx121**2)+(
                        startV[1]*directionV[1]/Rx121**2)+(
                        startV[2]*directionV[2]/Rx121**2)
                #print(B)
                C = -1+((startV[0] - 3.72)**2/Rx121**2)+(
                        startV[1]**2/Rx121**2)+(
                        startV[2]**2/Rx121**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens12R(self, startV, directionV):
                startV = startV - lens12V
                A = (directionV[0]**2/Rx122**2)+(
                        directionV[1]**2/Rx122**2)+(
                        directionV[2]**2/Rx122**2)
                #print(A)
                B = ((startV[0] + 3.72 - 0.56)*directionV[0]/Rx122**2)+(
                        startV[1]*directionV[1]/Rx122**2)+(
                        startV[2]*directionV[2]/Rx122**2)
                #print(B)
                C = -1+((startV[0] + 3.72 - 0.56)**2/Rx122**2)+(
                        startV[1]**2/Rx122**2)+(
                        startV[2]**2/Rx122**2)
                #print(C)
                T = (-B+np.sqrt(B**2-A*C))/A
                return T

        # レンズ12表面の法線を求める関数
        def decideNormalV_Lens12L(self, pointV):
                pointV = pointV - lens12V
                nornalVx = (2/Rx121**2)*(pointV[0] - 3.72)
                nornalVy = (2/Rx121**2)*pointV[1]
                nornalVz = (2/Rx121**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens12R(self, pointV):
                pointV = pointV - lens12V
                nornalVx = (2/Rx122**2)*(pointV[0] + 3.72 - 0.56)
                nornalVy = (2/Rx122**2)*pointV[1]
                nornalVz = (2/Rx122**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ13の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens13L(self, startV, directionV):
                startV = startV - lens13V
                A = (directionV[0]**2/Rx131**2)+(
                        directionV[1]**2/Rx131**2)+(
                        directionV[2]**2/Rx131**2)
                #print(A)
                B = ((startV[0] - 9.44)*directionV[0]/Rx131**2)+(
                        startV[1]*directionV[1]/Rx131**2)+(
                        startV[2]*directionV[2]/Rx131**2)
                #print(B)
                C = -1+((startV[0] - 9.44)**2/Rx131**2)+(
                        startV[1]**2/Rx131**2)+(
                        startV[2]**2/Rx131**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens13R(self, startV, directionV):
                startV = startV - lens13V
                A = (directionV[0]**2/Rx132**2)+(
                        directionV[1]**2/Rx132**2)+(
                        directionV[2]**2/Rx132**2)
                #print(A)
                B = ((startV[0] - 3.33 - 0.22)*directionV[0]/Rx132**2)+(
                        startV[1]*directionV[1]/Rx132**2)+(
                        startV[2]*directionV[2]/Rx132**2)
                #print(B)
                C = -1+((startV[0] - 3.33 - 0.22)**2/Rx132**2)+(
                        startV[1]**2/Rx132**2)+(
                        startV[2]**2/Rx132**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        # レンズ13表面の法線を求める関数
        def decideNormalV_Lens13L(self, pointV):
                pointV = pointV - lens13V
                nornalVx = (2/Rx131**2)*(pointV[0] - 9.44)
                nornalVy = (2/Rx131**2)*pointV[1]
                nornalVz = (2/Rx131**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens13R(self, pointV):
                pointV = pointV - lens13V
                nornalVx = -(2/Rx132**2)*(pointV[0] - 3.33 - 0.22)
                nornalVy = -(2/Rx132**2)*pointV[1]
                nornalVz = -(2/Rx132**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ14の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens14L(self, startV, directionV):
                startV = startV - lens14V
                A = (directionV[0]**2/Rx141**2)+(
                        directionV[1]**2/Rx141**2)+(
                        directionV[2]**2/Rx141**2)
                #print(A)
                B = ((startV[0] + 8.33)*directionV[0]/Rx141**2)+(
                        startV[1]*directionV[1]/Rx141**2)+(
                        startV[2]*directionV[2]/Rx141**2)
                #print(B)
                C = -1+((startV[0] + 8.33)**2/Rx141**2)+(
                        startV[1]**2/Rx141**2)+(
                        startV[2]**2/Rx141**2)
                #print(C)
                T = (-B+np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens14R(self, startV, directionV):
                startV = startV - lens14V
                A = (directionV[0]**2/Rx142**2)+(
                        directionV[1]**2/Rx142**2)+(
                        directionV[2]**2/Rx142**2)
                #print(A)
                B = ((startV[0] - 2.50 - 0.17)*directionV[0]/Rx142**2)+(
                        startV[1]*directionV[1]/Rx142**2)+(
                        startV[2]*directionV[2]/Rx142**2)
                #print(B)
                C = -1+((startV[0] - 2.50 - 0.17)**2/Rx142**2)+(
                        startV[1]**2/Rx142**2)+(
                        startV[2]**2/Rx142**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        # レンズ14表面の法線を求める関数
        def decideNormalV_Lens14L(self, pointV):
                pointV = pointV - lens14V
                nornalVx = -(2/Rx141**2)*(pointV[0] + 8.33)
                nornalVy = -(2/Rx141**2)*pointV[1]
                nornalVz = -(2/Rx141**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens14R(self, pointV):
                pointV = pointV - lens14V
                nornalVx = -(2/Rx142**2)*(pointV[0] - 2.50 - 0.17)
                nornalVy = -(2/Rx142**2)*pointV[1]
                nornalVz = -(2/Rx142**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # レイトレーシング、光線ベクトルとレンズ15の交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Lens15L(self, startV, directionV):
                startV = startV - lens15V
                A = (directionV[0]**2/Rx151**2)+(
                        directionV[1]**2/Rx151**2)+(
                        directionV[2]**2/Rx151**2)
                #print(A)
                B = ((startV[0] - 2.50)*directionV[0]/Rx151**2)+(
                        startV[1]*directionV[1]/Rx151**2)+(
                        startV[2]*directionV[2]/Rx151**2)
                #print(B)
                C = -1+((startV[0] - 2.50)**2/Rx151**2)+(
                        startV[1]**2/Rx151**2)+(
                        startV[2]**2/Rx151**2)
                #print(C)
                T = (-B-np.sqrt(B**2-A*C))/A
                return T

        def rayTraceDecideT_Lens15R(self, startV, directionV):
                startV = startV - lens15V
                A = (directionV[0]**2/Rx152**2)+(
                        directionV[1]**2/Rx152**2)+(
                        directionV[2]**2/Rx152**2)
                #print(A)
                B = ((startV[0] + 5.00 - 0.56)*directionV[0]/Rx152**2)+(
                        startV[1]*directionV[1]/Rx152**2)+(
                        startV[2]*directionV[2]/Rx152**2)
                #print(B)
                C = -1+((startV[0] + 5.00 - 0.56)**2/Rx152**2)+(
                        startV[1]**2/Rx152**2)+(
                        startV[2]**2/Rx152**2)
                #print(C)
                T = (-B+np.sqrt(B**2-A*C))/A
                return T

        # レンズ15表面の法線を求める関数
        def decideNormalV_Lens15L(self, pointV):
                pointV = pointV - lens15V
                nornalVx = (2/Rx151**2)*(pointV[0] - 2.50)
                nornalVy = (2/Rx151**2)*pointV[1]
                nornalVz = (2/Rx151**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV

        def decideNormalV_Lens15R(self, pointV):
                pointV = pointV - lens15V
                nornalVx = (2/Rx152**2)*(pointV[0] + 5.00 - 0.56)
                nornalVy = (2/Rx152**2)*pointV[1]
                nornalVz = (2/Rx152**2)*pointV[2]
                normalV = np.array([nornalVx, nornalVy, nornalVz])
                return normalV


        # スクリーンとの交点を持つときの係数Ｔを求める関数
        def rayTraceDecideT_Screen(self, startV, directionV):
                T = (screenV[0]-startV[0])/directionV[0]
                return T


        # スネルの公式から屈折光の方向ベクトルを求める関数
        def decideRefractionVL(self, rayV, normalV, Nair, Nn):
                # 正規化
                rayV = rayV/np.linalg.norm(rayV)
                normalV = normalV/np.linalg.norm(normalV)
                # 係数A
                A = Nair/Nn
                # 入射角
                cos_t_in = -np.dot(rayV,normalV)
                #量子化誤差対策
                if cos_t_in<-1.:
                cos_t_in = -1.
                elif cos_t_in>1.:
                cos_t_in = 1.
                # スネルの法則
                sin_t_in = np.sqrt(1.0 - cos_t_in**2)
                sin_t_out = sin_t_in*A
                if sin_t_out>1.0:
                #全反射する場合
                return np.zeros(3)
                cos_t_out = np.sqrt(1 - sin_t_out**2)
                # 係数B
                B = -cos_t_out + A*cos_t_in
                # 出射光線の方向ベクトル
                outRayV = A*rayV + B*normalV
                # 正規化
                outRayV = outRayV/np.linalg.norm(outRayV)
                return outRayV

        def decideRefractionVR(self, rayV, normalV, Nair, Nn):
                # 正規化
                rayV = rayV/np.linalg.norm(rayV)
                normalV = normalV/np.linalg.norm(normalV)
                # 係数A
                A = Nair/Nn
                # 入射角
                cos_t_in = np.dot(rayV,normalV)
                #量子化誤差対策
                if cos_t_in<-1.:
                cos_t_in = -1.
                elif cos_t_in>1.:
                cos_t_in = 1.
                # スネルの法則
                sin_t_in = np.sqrt(1.0 - cos_t_in**2)
                sin_t_out = sin_t_in*A
                if sin_t_out>1.0:
                #全反射する場合
                return np.zeros(3)
                cos_t_out = np.sqrt(1 - sin_t_out**2)
                # 係数B
                B = -cos_t_out + A*cos_t_in
                # 出射光線の方向ベクトル
                outRayV = A*rayV + B*normalV
                # 正規化
                outRayV = outRayV/np.linalg.norm(outRayV)
                return outRayV

        # ２点の位置ベクトルから直線を引く関数
        def plotLineRed(self, startPointV, endPointV):
                startX = startPointV[0]
                startY = startPointV[1]
                startZ = startPointV[2]
                endX = endPointV[0]
                endY = endPointV[1]
                endZ = endPointV[2]
                ax.plot([startX,endX],[startY,endY],[startZ,endZ],
                'o-',ms='2',linewidth=0.5,color='r')

        def plotLinePurple(self, startPointV, endPointV):
                startX = startPointV[0]
                startY = startPointV[1]
                startZ = startPointV[2]
                endX = endPointV[0]
                endY = endPointV[1]
                endZ = endPointV[2]
                ax.plot([startX,endX],[startY,endY],[startZ,endZ],
                'o-',ms='2',linewidth=0.5,color='purple')

        def plotLineBlue(self, startPointV, endPointV):
                startX = startPointV[0]
                startY = startPointV[1]
                startZ = startPointV[2]
                endX = endPointV[0]
                endY = endPointV[1]
                endZ = endPointV[2]
                ax.plot([startX,endX],[startY,endY],[startZ,endZ],
                'o-',ms='2',linewidth=0.5,color='blue')


    # ズームレンズのスクリーン上に映った点を返す関数
    def colorZoomLens(Nlens1=1.74, Nlens2=1.8, Nlens3=1.7, Nlens4=1.5,
                Nlens5=1.5, Nlens6=1.6, Nlens7=1.6, Nlens8=1.5,
                Nlens9=1.5, Nlens10=1.6, Nlens11=1.7, Nlens12=1.5,
                Nlens13=1.5, Nlens14=1.5, Nlens15=1.5, EXNlens45=1, EXNlens1314=1,
                NBlueRay1=1.054, NBlueRay2=1.106, NBlueRay3=1.004, NBlueRay4=1.004,
                NBlueRay5=1.004, NBlueRay6=1.008, NBlueRay7=1.004, NBlueRay8=1.006,
                NBlueRay9=1.004, NBlueRay10=1.004, NBlueRay11=1.006, NBlueRay12=1.004,
                NBlueRay13=1.006, NBlueRay14=1.006, NBlueRay15=1.004):
        # スクリーン描画
        Ys, Zs = np.meshgrid(
                np.arange(-3, 3.5, 0.5),
                np.arange(-3, 3.5, 0.5))
        Xs = 0*Ys + 0*Zs + screenV[0]
        ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, color='k')

        limitTheta = 2*np.pi  # theta生成数
        limitPhi = np.pi  # phi生成数
        theta = np.linspace(0, limitTheta, geneNum)
        phi = np.linspace(0, limitPhi, geneNum)

        # レンズ描画
        def plotLens(Rxi1, Rxi2, LensD, RLimit, LensType, lensiV):
                Ys = np.outer(np.sin(theta), np.sin(phi))
                Zs = np.outer(np.ones(np.size(theta)), np.cos(phi))

                Ysi1 = RLimit[0] * Ys
                Zsi1 = RLimit[0] * Zs
                Xsi1 = LensType[0] * (
                        Rxi1**2-Ysi1**2-Zsi1**2)**(1/2) - LensType[0]*Rxi1 + lensiV[0]
                ax.plot_wireframe(Xsi1, Ysi1, Zsi1, linewidth=0.1)

                Ysi2 = RLimit[1] * Ys
                Zsi2 = RLimit[1] * Zs
                Xsi2 = LensType[1] * (
                        Rxi2**2-Ysi2**2-Zsi2**2)**(1/2) + LensType[1]*(
                        -Rxi2 + LensType[1]*LensD) + lensiV[0]
                ax.plot_wireframe(Xsi2, Ysi2, Zsi2, linewidth=0.1)

        for i in Params:
                plotLens(*i)
        VF = VectorFunctions()  # インスタンス化

        LastRedPoints = []
        LastBluePoints = []

        # 始点を生成する
        width = 3
        space = 0.5
        size = len(np.arange(-width+centerY, 1+width+centerY, space))**2
        pointsY, pointsZ = np.meshgrid(
            np.arange(-width+centerY, 1+width+centerY, space),
            np.arange(-width+centerZ, 1+width+centerZ, space))
        pointsX = np.array([centerX]*size) + lens1V[0]
        pointsY = pointsY.reshape(size)*rayDensity + lens1V[1]
        pointsZ = pointsZ.reshape(size)*rayDensity + lens1V[2]
        raySPoint0 = VF.makePoints(pointsX, pointsY, pointsZ, size, 3)

        for i in raySPoint0:
            raySPoint0 = i
            #directionVector0 = np.array([1, 0, 0])  # 入射光の方向ベクトルを設定
            directionVector0 = [100, 0, 0] + raySPoint0
            T = VF.rayTraceDecideT_Lens1L(raySPoint0, directionVector0)  # 交点のための係数
            rayEPoint0 = raySPoint0 + T*directionVector0  # 入射光の終点
            VF.plotLinePurple(raySPoint0, rayEPoint0)  # 入射光描画

            # 赤色光
            refractSPoint0 = rayEPoint0  # 入射光の終点を引き継ぐ。以下レンズ１についての計算
            normalV_Lens1L = VF.decideNormalV_Lens1L(refractSPoint0)  # レンズの法線を求める
            # 屈折光の方向ベクトルを求める
            refractionV_Lens1L = VF.decideRefractionVL(directionVector0, normalV_Lens1L, Nair, Nlens1)
            # 係数Tを求めて、屈折光の終点も求める
            T = VF.rayTraceDecideT_Lens1R(refractSPoint0, refractionV_Lens1L)
            refractEPoint0 = refractSPoint0 + T*refractionV_Lens1L
            VF.plotLineRed(refractSPoint0,refractEPoint0)  # 屈折光の描画
            raySPoint1 = refractEPoint0  # 屈折光の終点を引き継ぐ
            normalV1 = VF.decideNormalV_Lens1R(raySPoint1)  # レンズの法線を求める
            # 屈折光の方向ベクトルを求める
            refractionV_Lens1R = VF.decideRefractionVR(refractionV_Lens1L, normalV1, Nlens2, Nlens1)
            T = 0  # レンズの接着を考えた
            rayEPoint1 = raySPoint1 + T*refractionV_Lens1R  # 空気中の屈折光の終点
            VF.plotLineRed(raySPoint1,rayEPoint1)  # 空気中の屈折光の描画

            refractSPoint_Lens2L = rayEPoint1  # 以下、レンズ２についての計算
            normalV_Lens2L = VF.decideNormalV_Lens2L(refractSPoint_Lens2L)  # レンズの法線を求める
            # 屈折光の方向ベクトルを求める
            refractionV_Lens2L = VF.decideRefractionVL(refractionV_Lens1R, normalV_Lens2L, Nlens1, Nlens2)
            # 係数Tを求めて、屈折光の終点も求める
            T = VF.rayTraceDecideT_Lens2R(refractSPoint_Lens2L, refractionV_Lens2L)
            refractEPoint_Lens2L = refractSPoint_Lens2L + T*refractionV_Lens2L
            VF.plotLineRed(refractSPoint_Lens2L,refractEPoint_Lens2L)  # 屈折光の描画
            raySPoint_Lens2R = refractEPoint_Lens2L
            normalV_Lens2R = VF.decideNormalV_Lens2R(raySPoint_Lens2R)
            refractionV_Lens2R = VF.decideRefractionVR(refractionV_Lens2L, normalV_Lens2R, Nair, Nlens2)
            T = VF.rayTraceDecideT_Lens3L(raySPoint_Lens2R, refractionV_Lens2R)
            rayEPoint_Lens3L = raySPoint_Lens2R + T*refractionV_Lens2R
            VF.plotLineRed(raySPoint_Lens2R, rayEPoint_Lens3L)

            refractSPoint_Lens3L = rayEPoint_Lens3L  # 以下、レンズ３についての計算
            normalV_Lens3L = VF.decideNormalV_Lens3L(refractSPoint_Lens3L)  # レンズの法線を求める
            # 屈折光の方向ベクトルを求める
            refractionV_Lens3L = VF.decideRefractionVL(refractionV_Lens2R, normalV_Lens3L, Nair, Nlens3)
            # 係数Tを求めて、屈折光の終点も求める
            T = VF.rayTraceDecideT_Lens3R(refractSPoint_Lens3L, refractionV_Lens3L)
            refractEPoint_Lens3L = refractSPoint_Lens3L + T*refractionV_Lens3L
            VF.plotLineRed(refractSPoint_Lens3L,refractEPoint_Lens3L)  # 屈折光の描画
            raySPoint_Lens3R = refractEPoint_Lens3L
            normalV_Lens3R = VF.decideNormalV_Lens3R(raySPoint_Lens3R)
            refractionV_Lens3R = VF.decideRefractionVR(refractionV_Lens3L, normalV_Lens3R, Nair, Nlens3)
            T = VF.rayTraceDecideT_Lens4L(raySPoint_Lens3R, refractionV_Lens3R)
            rayEPoint_Lens4L = raySPoint_Lens3R + T*refractionV_Lens3R
            VF.plotLineRed(raySPoint_Lens3R, rayEPoint_Lens4L)

            refractSPoint_Lens4L = rayEPoint_Lens4L  # 以下、レンズ４についての計算
            normalV_Lens4L = VF.decideNormalV_Lens4L(refractSPoint_Lens4L)
            refractionV_Lens4L = VF.decideRefractionVL(refractionV_Lens3R, normalV_Lens4L, Nair, Nlens4)
            T = VF.rayTraceDecideT_Lens4R(refractSPoint_Lens4L, refractionV_Lens4L)
            refractEPoint_Lens4L = refractSPoint_Lens4L + T*refractionV_Lens4L
            VF.plotLineRed(refractSPoint_Lens4L,refractEPoint_Lens4L)  # 屈折光の描画
            raySPoint_Lens4R = refractEPoint_Lens4L
            normalV_Lens4R = VF.decideNormalV_Lens4R(raySPoint_Lens4R)
            refractionV_Lens4R = VF.decideRefractionVR(refractionV_Lens4L, normalV_Lens4R, EXNlens45, Nlens4)
            T = VF.rayTraceDecideT_Lens5L(raySPoint_Lens4R, refractionV_Lens4R)
            rayEPoint_Lens5L = raySPoint_Lens4R + T*refractionV_Lens4R
            VF.plotLineRed(raySPoint_Lens4R, rayEPoint_Lens5L)

            refractSPoint_Lens5L = rayEPoint_Lens5L  # 以下、レンズ5についての計算
            normalV_Lens5L = VF.decideNormalV_Lens5L(refractSPoint_Lens5L)
            refractionV_Lens5L = VF.decideRefractionVL(refractionV_Lens4R, normalV_Lens5L, EXNlens45, Nlens5)
            T = VF.rayTraceDecideT_Lens5R(refractSPoint_Lens5L, refractionV_Lens5L)
            refractEPoint_Lens5L = refractSPoint_Lens5L + T*refractionV_Lens5L
            VF.plotLineRed(refractSPoint_Lens5L,refractEPoint_Lens5L)  # 屈折光の描画
            raySPoint_Lens5R = refractEPoint_Lens5L
            normalV_Lens5R = VF.decideNormalV_Lens5R(raySPoint_Lens5R)
            refractionV_Lens5R = VF.decideRefractionVR(refractionV_Lens5L, normalV_Lens5R, Nlens6, Nlens5)
            T = 0
            rayEPoint_Lens6L = raySPoint_Lens5R + T*refractionV_Lens5R
            VF.plotLineRed(raySPoint_Lens5R, rayEPoint_Lens6L)

            refractSPoint_Lens6L = rayEPoint_Lens6L  # 以下、レンズ6についての計算
            normalV_Lens6L = VF.decideNormalV_Lens6L(refractSPoint_Lens6L)
            refractionV_Lens6L = VF.decideRefractionVL(refractionV_Lens5R, normalV_Lens6L, Nlens5, Nlens6)
            T = VF.rayTraceDecideT_Lens6R(refractSPoint_Lens6L, refractionV_Lens6L)
            refractEPoint_Lens6L = refractSPoint_Lens6L + T*refractionV_Lens6L
            VF.plotLineRed(refractSPoint_Lens6L,refractEPoint_Lens6L)  # 屈折光の描画
            raySPoint_Lens6R = refractEPoint_Lens6L
            normalV_Lens6R = VF.decideNormalV_Lens6R(raySPoint_Lens6R)
            refractionV_Lens6R = VF.decideRefractionVR(refractionV_Lens6L, normalV_Lens6R, Nair, Nlens6)
            T = VF.rayTraceDecideT_Lens7L(raySPoint_Lens6R, refractionV_Lens6R)
            rayEPoint_Lens7L = raySPoint_Lens6R + T*refractionV_Lens6R
            VF.plotLineRed(raySPoint_Lens6R, rayEPoint_Lens7L)

            refractSPoint_Lens7L = rayEPoint_Lens7L  # 以下、レンズ7についての計算
            normalV_Lens7L = VF.decideNormalV_Lens7L(refractSPoint_Lens7L)
            refractionV_Lens7L = VF.decideRefractionVL(refractionV_Lens6R, normalV_Lens7L, Nair, Nlens7)
            T = VF.rayTraceDecideT_Lens7R(refractSPoint_Lens7L, refractionV_Lens7L)
            refractEPoint_Lens7L = refractSPoint_Lens7L + T*refractionV_Lens7L
            VF.plotLineRed(refractSPoint_Lens7L,refractEPoint_Lens7L)  # 屈折光の描画
            raySPoint_Lens7R = refractEPoint_Lens7L
            normalV_Lens7R = VF.decideNormalV_Lens7R(raySPoint_Lens7R)
            refractionV_Lens7R = VF.decideRefractionVR(refractionV_Lens7L, normalV_Lens7R, Nair, Nlens7)
            T = VF.rayTraceDecideT_Lens8L(raySPoint_Lens7R, refractionV_Lens7R)
            rayEPoint_Lens8L = raySPoint_Lens7R + T*refractionV_Lens7R
            VF.plotLineRed(raySPoint_Lens7R, rayEPoint_Lens8L)

            refractSPoint_Lens8L = rayEPoint_Lens8L  # 以下、レンズ8についての計算
            normalV_Lens8L = VF.decideNormalV_Lens8L(refractSPoint_Lens8L)
            refractionV_Lens8L = VF.decideRefractionVL(refractionV_Lens7R, normalV_Lens8L, Nair, Nlens8)
            T = VF.rayTraceDecideT_Lens8R(refractSPoint_Lens8L, refractionV_Lens8L)
            refractEPoint_Lens8L = refractSPoint_Lens8L + T*refractionV_Lens8L
            VF.plotLineRed(refractSPoint_Lens8L,refractEPoint_Lens8L)  # 屈折光の描画
            raySPoint_Lens8R = refractEPoint_Lens8L
            normalV_Lens8R = VF.decideNormalV_Lens8R(raySPoint_Lens8R)
            refractionV_Lens8R = VF.decideRefractionVR(refractionV_Lens8L, normalV_Lens8R, Nair, Nlens8)
            T = VF.rayTraceDecideT_Lens9L(raySPoint_Lens8R, refractionV_Lens8R)
            rayEPoint_Lens9L = raySPoint_Lens8R + T*refractionV_Lens8R
            VF.plotLineRed(raySPoint_Lens8R, rayEPoint_Lens9L)

            # 絞り
            rayEPoint_Lens9L_TEST = raySPoint_Lens8R + 1.5*refractionV_Lens8R
            TestR = rayEPoint_Lens9L_TEST[1]**2 + rayEPoint_Lens9L_TEST[2]**2
            if TestR >= 1.4:
                refractionV_Lens8R = [0,0,0]

            refractSPoint_Lens9L = rayEPoint_Lens9L  # 以下、レンズ9についての計算
            normalV_Lens9L = VF.decideNormalV_Lens9L(refractSPoint_Lens9L)
            refractionV_Lens9L = VF.decideRefractionVL(refractionV_Lens8R, normalV_Lens9L, Nair, Nlens9)
            T = VF.rayTraceDecideT_Lens9R(refractSPoint_Lens9L, refractionV_Lens9L)
            refractEPoint_Lens9L = refractSPoint_Lens9L + T*refractionV_Lens9L
            VF.plotLineRed(refractSPoint_Lens9L,refractEPoint_Lens9L)  # 屈折光の描画
            raySPoint_Lens9R = refractEPoint_Lens9L
            normalV_Lens9R = VF.decideNormalV_Lens9R(raySPoint_Lens9R)
            refractionV_Lens9R = VF.decideRefractionVR(refractionV_Lens9L, normalV_Lens9R, Nair, Nlens9)
            T = VF.rayTraceDecideT_Lens10L(raySPoint_Lens9R, refractionV_Lens9R)
            rayEPoint_Lens10L = raySPoint_Lens9R + T*refractionV_Lens9R
            VF.plotLineRed(raySPoint_Lens9R, rayEPoint_Lens10L)

            refractSPoint_Lens10L = rayEPoint_Lens10L  # 以下、レンズ10についての計算
            normalV_Lens10L = VF.decideNormalV_Lens10L(refractSPoint_Lens10L)
            refractionV_Lens10L = VF.decideRefractionVL(refractionV_Lens9R, normalV_Lens10L, Nair, Nlens10)
            T = VF.rayTraceDecideT_Lens10R(refractSPoint_Lens10L, refractionV_Lens10L)
            refractEPoint_Lens10L = refractSPoint_Lens10L + T*refractionV_Lens10L
            VF.plotLineRed(refractSPoint_Lens10L,refractEPoint_Lens10L)  # 屈折光の描画
            raySPoint_Lens10R = refractEPoint_Lens10L
            normalV_Lens10R = VF.decideNormalV_Lens10R(raySPoint_Lens10R)
            refractionV_Lens10R = VF.decideRefractionVR(refractionV_Lens10L, normalV_Lens10R, Nlens11, Nlens10)
            T = 0  # レンズの接着を考えた
            rayEPoint_Lens11L = raySPoint_Lens10R + T*refractionV_Lens10R
            VF.plotLineRed(raySPoint_Lens10R, rayEPoint_Lens11L)

            refractSPoint_Lens11L = rayEPoint_Lens11L  # 以下、レンズ11についての計算
            normalV_Lens11L = VF.decideNormalV_Lens11L(refractSPoint_Lens11L)
            refractionV_Lens11L = VF.decideRefractionVL(refractionV_Lens10R, normalV_Lens11L, Nlens10, Nlens11)
            T = VF.rayTraceDecideT_Lens11R(refractSPoint_Lens11L, refractionV_Lens11L)
            refractEPoint_Lens11L = refractSPoint_Lens11L + T*refractionV_Lens11L
            VF.plotLineRed(refractSPoint_Lens11L,refractEPoint_Lens11L)  # 屈折光の描画
            raySPoint_Lens11R = refractEPoint_Lens11L
            normalV_Lens11R = VF.decideNormalV_Lens11R(raySPoint_Lens11R)
            refractionV_Lens11R = VF.decideRefractionVR(refractionV_Lens11L, normalV_Lens11R, Nair, Nlens11)
            T = VF.rayTraceDecideT_Lens12L(raySPoint_Lens11R, refractionV_Lens11R)
            rayEPoint_Lens12L = raySPoint_Lens11R + T*refractionV_Lens11R
            VF.plotLineRed(raySPoint_Lens11R, rayEPoint_Lens12L)

            refractSPoint_Lens12L = rayEPoint_Lens12L  # 以下、レンズ12についての計算
            normalV_Lens12L = VF.decideNormalV_Lens12L(refractSPoint_Lens12L)
            refractionV_Lens12L = VF.decideRefractionVL(refractionV_Lens11R, normalV_Lens12L, Nair, Nlens12)
            T = VF.rayTraceDecideT_Lens12R(refractSPoint_Lens12L, refractionV_Lens12L)
            refractEPoint_Lens12L = refractSPoint_Lens12L + T*refractionV_Lens12L
            VF.plotLineRed(refractSPoint_Lens12L,refractEPoint_Lens12L)  # 屈折光の描画
            raySPoint_Lens12R = refractEPoint_Lens12L
            normalV_Lens12R = VF.decideNormalV_Lens12R(raySPoint_Lens12R)
            refractionV_Lens12R = VF.decideRefractionVR(refractionV_Lens12L, normalV_Lens12R, Nair, Nlens12)
            T = VF.rayTraceDecideT_Lens13L(raySPoint_Lens12R, refractionV_Lens12R)
            rayEPoint_Lens13L = raySPoint_Lens12R + T*refractionV_Lens12R
            VF.plotLineRed(raySPoint_Lens12R, rayEPoint_Lens13L)

            refractSPoint_Lens13L = rayEPoint_Lens13L  # 以下、レンズ13についての計算
            normalV_Lens13L = VF.decideNormalV_Lens13L(refractSPoint_Lens13L)
            refractionV_Lens13L = VF.decideRefractionVL(refractionV_Lens12R, normalV_Lens13L, Nair, Nlens13)
            T = VF.rayTraceDecideT_Lens13R(refractSPoint_Lens13L, refractionV_Lens13L)
            refractEPoint_Lens13L = refractSPoint_Lens13L + T*refractionV_Lens13L
            VF.plotLineRed(refractSPoint_Lens13L,refractEPoint_Lens13L)  # 屈折光の描画
            raySPoint_Lens13R = refractEPoint_Lens13L
            normalV_Lens13R = VF.decideNormalV_Lens13R(raySPoint_Lens13R)
            refractionV_Lens13R = VF.decideRefractionVR(refractionV_Lens13L, normalV_Lens13R, EXNlens1314, Nlens13)
            T = VF.rayTraceDecideT_Lens14L(raySPoint_Lens13R, refractionV_Lens13R)
            rayEPoint_Lens14L = raySPoint_Lens13R + T*refractionV_Lens13R
            VF.plotLineRed(raySPoint_Lens13R, rayEPoint_Lens14L)

            refractSPoint_Lens14L = rayEPoint_Lens14L  # 以下、レンズ14についての計算
            normalV_Lens14L = VF.decideNormalV_Lens14L(refractSPoint_Lens14L)
            refractionV_Lens14L = VF.decideRefractionVL(refractionV_Lens13R, normalV_Lens14L, EXNlens1314, Nlens14)
            T = VF.rayTraceDecideT_Lens14R(refractSPoint_Lens14L, refractionV_Lens14L)
            refractEPoint_Lens14L = refractSPoint_Lens14L + T*refractionV_Lens14L
            VF.plotLineRed(refractSPoint_Lens14L,refractEPoint_Lens14L)  # 屈折光の描画
            raySPoint_Lens14R = refractEPoint_Lens14L
            normalV_Lens14R = VF.decideNormalV_Lens14R(raySPoint_Lens14R)
            refractionV_Lens14R = VF.decideRefractionVR(refractionV_Lens14L, normalV_Lens14R, Nlens15, Nlens14)
            T = 0  # レンズの接着を考えた
            rayEPoint_Lens15L = raySPoint_Lens14R + T*refractionV_Lens14R
            VF.plotLineRed(raySPoint_Lens14R, rayEPoint_Lens15L)

            refractSPoint_Lens15L = rayEPoint_Lens15L  # 以下、レンズ15についての計算
            normalV_Lens15L = VF.decideNormalV_Lens15L(refractSPoint_Lens15L)
            refractionV_Lens15L = VF.decideRefractionVL(refractionV_Lens14R, normalV_Lens15L, Nlens14, Nlens15)
            T = VF.rayTraceDecideT_Lens15R(refractSPoint_Lens15L, refractionV_Lens15L)
            refractEPoint_Lens15L = refractSPoint_Lens15L + T*refractionV_Lens15L
            VF.plotLineRed(refractSPoint_Lens15L,refractEPoint_Lens15L)  # 屈折光の描画
            raySPoint_Lens15R = refractEPoint_Lens15L
            normalV_Lens15R = VF.decideNormalV_Lens15R(raySPoint_Lens15R)
            refractionV_Lens15R = VF.decideRefractionVR(refractionV_Lens15L, normalV_Lens15R, Nair, Nlens15)
            T = VF.rayTraceDecideT_Screen(raySPoint_Lens15R, refractionV_Lens15R)
            rayEPoint_Last = raySPoint_Lens15R + T*refractionV_Lens15R
            VF.plotLineRed(raySPoint_Lens15R, rayEPoint_Last)

            LastRedPoints.append(rayEPoint_Last)

            # 青色光
            refractSPoint0 = rayEPoint0  # 入射光の終点を引き継ぐ。以下レンズ１についての計算
            normalV_Lens1L = VF.decideNormalV_Lens1L(refractSPoint0)  # レンズの法線を求める
            # 屈折光の方向ベクトルを求める
            refractionV_Lens1L = VF.decideRefractionVL(directionVector0, normalV_Lens1L, Nair, Nlens1*NBlueRay1)
            # 係数Tを求めて、屈折光の終点も求める
            T = VF.rayTraceDecideT_Lens1R(refractSPoint0, refractionV_Lens1L)
            refractEPoint0 = refractSPoint0 + T*refractionV_Lens1L
            VF.plotLineBlue(refractSPoint0,refractEPoint0)  # 屈折光の描画
            raySPoint1 = refractEPoint0  # 屈折光の終点を引き継ぐ
            normalV1 = VF.decideNormalV_Lens1R(raySPoint1)  # レンズの法線を求める
            # 屈折光の方向ベクトルを求める
            refractionV_Lens1R = VF.decideRefractionVR(refractionV_Lens1L, normalV1, Nlens2*NBlueRay2, Nlens1*NBlueRay1)
            T = 0  # レンズの接着を考えた
            rayEPoint1 = raySPoint1 + T*refractionV_Lens1R  # 空気中の屈折光の終点
            VF.plotLineBlue(raySPoint1,rayEPoint1)  # 空気中の屈折光の描画

            refractSPoint_Lens2L = rayEPoint1  # 以下、レンズ２についての計算
            normalV_Lens2L = VF.decideNormalV_Lens2L(refractSPoint_Lens2L)  # レンズの法線を求める
            # 屈折光の方向ベクトルを求める
            refractionV_Lens2L = VF.decideRefractionVL(refractionV_Lens1R, normalV_Lens2L, Nlens1*NBlueRay1, Nlens2*NBlueRay2)
            # 係数Tを求めて、屈折光の終点も求める
            T = VF.rayTraceDecideT_Lens2R(refractSPoint_Lens2L, refractionV_Lens2L)
            refractEPoint_Lens2L = refractSPoint_Lens2L + T*refractionV_Lens2L
            VF.plotLineBlue(refractSPoint_Lens2L,refractEPoint_Lens2L)  # 屈折光の描画
            raySPoint_Lens2R = refractEPoint_Lens2L
            normalV_Lens2R = VF.decideNormalV_Lens2R(raySPoint_Lens2R)
            refractionV_Lens2R = VF.decideRefractionVR(refractionV_Lens2L, normalV_Lens2R, Nair, Nlens2*NBlueRay2)
            T = VF.rayTraceDecideT_Lens3L(raySPoint_Lens2R, refractionV_Lens2R)
            rayEPoint_Lens3L = raySPoint_Lens2R + T*refractionV_Lens2R
            VF.plotLineBlue(raySPoint_Lens2R, rayEPoint_Lens3L)

            refractSPoint_Lens3L = rayEPoint_Lens3L  # 以下、レンズ３についての計算
            normalV_Lens3L = VF.decideNormalV_Lens3L(refractSPoint_Lens3L)  # レンズの法線を求める
            # 屈折光の方向ベクトルを求める
            refractionV_Lens3L = VF.decideRefractionVL(refractionV_Lens2R, normalV_Lens3L, Nair, Nlens3*NBlueRay3)
            # 係数Tを求めて、屈折光の終点も求める
            T = VF.rayTraceDecideT_Lens3R(refractSPoint_Lens3L, refractionV_Lens3L)
            refractEPoint_Lens3L = refractSPoint_Lens3L + T*refractionV_Lens3L
            VF.plotLineBlue(refractSPoint_Lens3L,refractEPoint_Lens3L)  # 屈折光の描画
            raySPoint_Lens3R = refractEPoint_Lens3L
            normalV_Lens3R = VF.decideNormalV_Lens3R(raySPoint_Lens3R)
            refractionV_Lens3R = VF.decideRefractionVR(refractionV_Lens3L, normalV_Lens3R, Nair, Nlens3*NBlueRay3)
            T = VF.rayTraceDecideT_Lens4L(raySPoint_Lens3R, refractionV_Lens3R)
            rayEPoint_Lens4L = raySPoint_Lens3R + T*refractionV_Lens3R
            VF.plotLineBlue(raySPoint_Lens3R, rayEPoint_Lens4L)

            refractSPoint_Lens4L = rayEPoint_Lens4L  # 以下、レンズ４についての計算
            normalV_Lens4L = VF.decideNormalV_Lens4L(refractSPoint_Lens4L)
            refractionV_Lens4L = VF.decideRefractionVL(refractionV_Lens3R, normalV_Lens4L, Nair, Nlens4*NBlueRay4)
            T = VF.rayTraceDecideT_Lens4R(refractSPoint_Lens4L, refractionV_Lens4L)
            refractEPoint_Lens4L = refractSPoint_Lens4L + T*refractionV_Lens4L
            VF.plotLineBlue(refractSPoint_Lens4L,refractEPoint_Lens4L)  # 屈折光の描画
            raySPoint_Lens4R = refractEPoint_Lens4L
            normalV_Lens4R = VF.decideNormalV_Lens4R(raySPoint_Lens4R)
            refractionV_Lens4R = VF.decideRefractionVR(refractionV_Lens4L, normalV_Lens4R, EXNlens45*1.008, Nlens4*NBlueRay4)
            T = VF.rayTraceDecideT_Lens5L(raySPoint_Lens4R, refractionV_Lens4R)
            rayEPoint_Lens5L = raySPoint_Lens4R + T*refractionV_Lens4R
            VF.plotLineBlue(raySPoint_Lens4R, rayEPoint_Lens5L)

            refractSPoint_Lens5L = rayEPoint_Lens5L  # 以下、レンズ5についての計算
            normalV_Lens5L = VF.decideNormalV_Lens5L(refractSPoint_Lens5L)
            refractionV_Lens5L = VF.decideRefractionVL(refractionV_Lens4R, normalV_Lens5L, EXNlens45*1.008, Nlens5*NBlueRay5)
            T = VF.rayTraceDecideT_Lens5R(refractSPoint_Lens5L, refractionV_Lens5L)
            refractEPoint_Lens5L = refractSPoint_Lens5L + T*refractionV_Lens5L
            VF.plotLineBlue(refractSPoint_Lens5L,refractEPoint_Lens5L)  # 屈折光の描画
            raySPoint_Lens5R = refractEPoint_Lens5L
            normalV_Lens5R = VF.decideNormalV_Lens5R(raySPoint_Lens5R)
            refractionV_Lens5R = VF.decideRefractionVR(refractionV_Lens5L, normalV_Lens5R, Nlens6*NBlueRay6, Nlens5*NBlueRay5)
            T = 0
            rayEPoint_Lens6L = raySPoint_Lens5R + T*refractionV_Lens5R
            VF.plotLineBlue(raySPoint_Lens5R, rayEPoint_Lens6L)

            refractSPoint_Lens6L = rayEPoint_Lens6L  # 以下、レンズ6についての計算
            normalV_Lens6L = VF.decideNormalV_Lens6L(refractSPoint_Lens6L)
            refractionV_Lens6L = VF.decideRefractionVL(refractionV_Lens5R, normalV_Lens6L, Nlens5*NBlueRay5, Nlens6*NBlueRay6)
            T = VF.rayTraceDecideT_Lens6R(refractSPoint_Lens6L, refractionV_Lens6L)
            refractEPoint_Lens6L = refractSPoint_Lens6L + T*refractionV_Lens6L
            VF.plotLineBlue(refractSPoint_Lens6L,refractEPoint_Lens6L)  # 屈折光の描画
            raySPoint_Lens6R = refractEPoint_Lens6L
            normalV_Lens6R = VF.decideNormalV_Lens6R(raySPoint_Lens6R)
            refractionV_Lens6R = VF.decideRefractionVR(refractionV_Lens6L, normalV_Lens6R, Nair, Nlens6*NBlueRay6)
            T = VF.rayTraceDecideT_Lens7L(raySPoint_Lens6R, refractionV_Lens6R)
            rayEPoint_Lens7L = raySPoint_Lens6R + T*refractionV_Lens6R
            VF.plotLineBlue(raySPoint_Lens6R, rayEPoint_Lens7L)

            refractSPoint_Lens7L = rayEPoint_Lens7L  # 以下、レンズ7についての計算
            normalV_Lens7L = VF.decideNormalV_Lens7L(refractSPoint_Lens7L)
            refractionV_Lens7L = VF.decideRefractionVL(refractionV_Lens6R, normalV_Lens7L, Nair, Nlens7*NBlueRay7)
            T = VF.rayTraceDecideT_Lens7R(refractSPoint_Lens7L, refractionV_Lens7L)
            refractEPoint_Lens7L = refractSPoint_Lens7L + T*refractionV_Lens7L
            VF.plotLineBlue(refractSPoint_Lens7L,refractEPoint_Lens7L)  # 屈折光の描画
            raySPoint_Lens7R = refractEPoint_Lens7L
            normalV_Lens7R = VF.decideNormalV_Lens7R(raySPoint_Lens7R)
            refractionV_Lens7R = VF.decideRefractionVR(refractionV_Lens7L, normalV_Lens7R, Nair, Nlens7*NBlueRay7)
            T = VF.rayTraceDecideT_Lens8L(raySPoint_Lens7R, refractionV_Lens7R)
            rayEPoint_Lens8L = raySPoint_Lens7R + T*refractionV_Lens7R
            VF.plotLineBlue(raySPoint_Lens7R, rayEPoint_Lens8L)

            refractSPoint_Lens8L = rayEPoint_Lens8L  # 以下、レンズ8についての計算
            normalV_Lens8L = VF.decideNormalV_Lens8L(refractSPoint_Lens8L)
            refractionV_Lens8L = VF.decideRefractionVL(refractionV_Lens7R, normalV_Lens8L, Nair, Nlens8*NBlueRay8)
            T = VF.rayTraceDecideT_Lens8R(refractSPoint_Lens8L, refractionV_Lens8L)
            refractEPoint_Lens8L = refractSPoint_Lens8L + T*refractionV_Lens8L
            VF.plotLineBlue(refractSPoint_Lens8L,refractEPoint_Lens8L)  # 屈折光の描画
            raySPoint_Lens8R = refractEPoint_Lens8L
            normalV_Lens8R = VF.decideNormalV_Lens8R(raySPoint_Lens8R)
            refractionV_Lens8R = VF.decideRefractionVR(refractionV_Lens8L, normalV_Lens8R, Nair, Nlens8*NBlueRay8)
            T = VF.rayTraceDecideT_Lens9L(raySPoint_Lens8R, refractionV_Lens8R)
            rayEPoint_Lens9L = raySPoint_Lens8R + T*refractionV_Lens8R
            VF.plotLineBlue(raySPoint_Lens8R, rayEPoint_Lens9L)

            # 絞り
            rayEPoint_Lens9L_TEST = raySPoint_Lens8R + 1.5*refractionV_Lens8R
            TestR = rayEPoint_Lens9L_TEST[1]**2 + rayEPoint_Lens9L_TEST[2]**2
            if TestR >= 1.4:
                refractionV_Lens8R = [0,0,0]

            refractSPoint_Lens9L = rayEPoint_Lens9L  # 以下、レンズ9についての計算
            normalV_Lens9L = VF.decideNormalV_Lens9L(refractSPoint_Lens9L)
            refractionV_Lens9L = VF.decideRefractionVL(refractionV_Lens8R, normalV_Lens9L, Nair, Nlens9*NBlueRay9)
            T = VF.rayTraceDecideT_Lens9R(refractSPoint_Lens9L, refractionV_Lens9L)
            refractEPoint_Lens9L = refractSPoint_Lens9L + T*refractionV_Lens9L
            VF.plotLineBlue(refractSPoint_Lens9L,refractEPoint_Lens9L)  # 屈折光の描画
            raySPoint_Lens9R = refractEPoint_Lens9L
            normalV_Lens9R = VF.decideNormalV_Lens9R(raySPoint_Lens9R)
            refractionV_Lens9R = VF.decideRefractionVR(refractionV_Lens9L, normalV_Lens9R, Nair, Nlens9*NBlueRay9)
            T = VF.rayTraceDecideT_Lens10L(raySPoint_Lens9R, refractionV_Lens9R)
            rayEPoint_Lens10L = raySPoint_Lens9R + T*refractionV_Lens9R
            VF.plotLineBlue(raySPoint_Lens9R, rayEPoint_Lens10L)

            refractSPoint_Lens10L = rayEPoint_Lens10L  # 以下、レンズ10についての計算
            normalV_Lens10L = VF.decideNormalV_Lens10L(refractSPoint_Lens10L)
            refractionV_Lens10L = VF.decideRefractionVL(refractionV_Lens9R, normalV_Lens10L, Nair, Nlens10*NBlueRay10)
            T = VF.rayTraceDecideT_Lens10R(refractSPoint_Lens10L, refractionV_Lens10L)
            refractEPoint_Lens10L = refractSPoint_Lens10L + T*refractionV_Lens10L
            VF.plotLineBlue(refractSPoint_Lens10L,refractEPoint_Lens10L)  # 屈折光の描画
            raySPoint_Lens10R = refractEPoint_Lens10L
            normalV_Lens10R = VF.decideNormalV_Lens10R(raySPoint_Lens10R)
            refractionV_Lens10R = VF.decideRefractionVR(refractionV_Lens10L, normalV_Lens10R, Nlens11*NBlueRay11, Nlens10*NBlueRay10)
            T = 0  # レンズの接着を考えた
            rayEPoint_Lens11L = raySPoint_Lens10R + T*refractionV_Lens10R
            VF.plotLineBlue(raySPoint_Lens10R, rayEPoint_Lens11L)

            refractSPoint_Lens11L = rayEPoint_Lens11L  # 以下、レンズ11についての計算
            normalV_Lens11L = VF.decideNormalV_Lens11L(refractSPoint_Lens11L)
            refractionV_Lens11L = VF.decideRefractionVL(refractionV_Lens10R, normalV_Lens11L, Nlens10*NBlueRay10, Nlens11*NBlueRay11)
            T = VF.rayTraceDecideT_Lens11R(refractSPoint_Lens11L, refractionV_Lens11L)
            refractEPoint_Lens11L = refractSPoint_Lens11L + T*refractionV_Lens11L
            VF.plotLineBlue(refractSPoint_Lens11L,refractEPoint_Lens11L)  # 屈折光の描画
            raySPoint_Lens11R = refractEPoint_Lens11L
            normalV_Lens11R = VF.decideNormalV_Lens11R(raySPoint_Lens11R)
            refractionV_Lens11R = VF.decideRefractionVR(refractionV_Lens11L, normalV_Lens11R, Nair, Nlens11*NBlueRay11)
            T = VF.rayTraceDecideT_Lens12L(raySPoint_Lens11R, refractionV_Lens11R)
            rayEPoint_Lens12L = raySPoint_Lens11R + T*refractionV_Lens11R
            VF.plotLineBlue(raySPoint_Lens11R, rayEPoint_Lens12L)

            refractSPoint_Lens12L = rayEPoint_Lens12L  # 以下、レンズ12についての計算
            normalV_Lens12L = VF.decideNormalV_Lens12L(refractSPoint_Lens12L)
            refractionV_Lens12L = VF.decideRefractionVL(refractionV_Lens11R, normalV_Lens12L, Nair, Nlens12*NBlueRay12)
            T = VF.rayTraceDecideT_Lens12R(refractSPoint_Lens12L, refractionV_Lens12L)
            refractEPoint_Lens12L = refractSPoint_Lens12L + T*refractionV_Lens12L
            VF.plotLineBlue(refractSPoint_Lens12L,refractEPoint_Lens12L)  # 屈折光の描画
            raySPoint_Lens12R = refractEPoint_Lens12L
            normalV_Lens12R = VF.decideNormalV_Lens12R(raySPoint_Lens12R)
            refractionV_Lens12R = VF.decideRefractionVR(refractionV_Lens12L, normalV_Lens12R, Nair, Nlens12*NBlueRay12)
            T = VF.rayTraceDecideT_Lens13L(raySPoint_Lens12R, refractionV_Lens12R)
            rayEPoint_Lens13L = raySPoint_Lens12R + T*refractionV_Lens12R
            VF.plotLineBlue(raySPoint_Lens12R, rayEPoint_Lens13L)

            refractSPoint_Lens13L = rayEPoint_Lens13L  # 以下、レンズ13についての計算
            normalV_Lens13L = VF.decideNormalV_Lens13L(refractSPoint_Lens13L)
            refractionV_Lens13L = VF.decideRefractionVL(refractionV_Lens12R, normalV_Lens13L, Nair, Nlens13*NBlueRay13)
            T = VF.rayTraceDecideT_Lens13R(refractSPoint_Lens13L, refractionV_Lens13L)
            refractEPoint_Lens13L = refractSPoint_Lens13L + T*refractionV_Lens13L
            VF.plotLineBlue(refractSPoint_Lens13L,refractEPoint_Lens13L)  # 屈折光の描画
            raySPoint_Lens13R = refractEPoint_Lens13L
            normalV_Lens13R = VF.decideNormalV_Lens13R(raySPoint_Lens13R)
            refractionV_Lens13R = VF.decideRefractionVR(refractionV_Lens13L, normalV_Lens13R, EXNlens1314*1.002, Nlens13*NBlueRay13)
            T = VF.rayTraceDecideT_Lens14L(raySPoint_Lens13R, refractionV_Lens13R)
            rayEPoint_Lens14L = raySPoint_Lens13R + T*refractionV_Lens13R
            VF.plotLineBlue(raySPoint_Lens13R, rayEPoint_Lens14L)

            refractSPoint_Lens14L = rayEPoint_Lens14L  # 以下、レンズ14についての計算
            normalV_Lens14L = VF.decideNormalV_Lens14L(refractSPoint_Lens14L)
            refractionV_Lens14L = VF.decideRefractionVL(refractionV_Lens13R, normalV_Lens14L, EXNlens1314*1.002, Nlens14*NBlueRay14)
            T = VF.rayTraceDecideT_Lens14R(refractSPoint_Lens14L, refractionV_Lens14L)
            refractEPoint_Lens14L = refractSPoint_Lens14L + T*refractionV_Lens14L
            VF.plotLineBlue(refractSPoint_Lens14L,refractEPoint_Lens14L)  # 屈折光の描画
            raySPoint_Lens14R = refractEPoint_Lens14L
            normalV_Lens14R = VF.decideNormalV_Lens14R(raySPoint_Lens14R)
            refractionV_Lens14R = VF.decideRefractionVR(refractionV_Lens14L, normalV_Lens14R, Nlens15*NBlueRay15, Nlens14*NBlueRay14)
            T = 0  # レンズの接着を考えた
            rayEPoint_Lens15L = raySPoint_Lens14R + T*refractionV_Lens14R
            VF.plotLineBlue(raySPoint_Lens14R, rayEPoint_Lens15L)

            refractSPoint_Lens15L = rayEPoint_Lens15L  # 以下、レンズ15についての計算
            normalV_Lens15L = VF.decideNormalV_Lens15L(refractSPoint_Lens15L)
            refractionV_Lens15L = VF.decideRefractionVL(refractionV_Lens14R, normalV_Lens15L, Nlens14*NBlueRay14, Nlens15*NBlueRay15)
            T = VF.rayTraceDecideT_Lens15R(refractSPoint_Lens15L, refractionV_Lens15L)
            refractEPoint_Lens15L = refractSPoint_Lens15L + T*refractionV_Lens15L
            VF.plotLineBlue(refractSPoint_Lens15L,refractEPoint_Lens15L)  # 屈折光の描画
            raySPoint_Lens15R = refractEPoint_Lens15L
            normalV_Lens15R = VF.decideNormalV_Lens15R(raySPoint_Lens15R)
            refractionV_Lens15R = VF.decideRefractionVR(refractionV_Lens15L, normalV_Lens15R, Nair, Nlens15*NBlueRay15)
            T = VF.rayTraceDecideT_Screen(raySPoint_Lens15R, refractionV_Lens15R)
            rayEPoint_Last = raySPoint_Lens15R + T*refractionV_Lens15R
            VF.plotLineBlue(raySPoint_Lens15R, rayEPoint_Last)

            LastBluePoints.append(rayEPoint_Last)


    ax.set_xlim(-LX+focusX, LX+focusX)
    ax.set_ylim(-LY, LY)
    ax.set_zlim(-LZ, LZ)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=5, azim=-110)



    colorZoomLens()

    print('time =', time.time()-start)
    #plt.show()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig('ZLens%d.png' % j)
