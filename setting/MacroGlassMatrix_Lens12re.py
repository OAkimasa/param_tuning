# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import time

#start = time.time()

LX = 5
LY = 5
LZ = 5
geneNum = 500
Nair = 1  # 空気の屈折率

rayStartV = np.array([100000*100, 0*100, 0.0*100])  # m から cm へ変換
centerX = 0  # 入射光表示の中心座標
centerY = 0  # 入射光表示の中心座標
centerZ = 0  # 入射光表示の中心座標
rayDensity = 0.25  # 入射光の密度
focusX = -2  # 焦点付近の描画範囲を平行移動

screenV = np.array([16, 0, 0])  # スクリーンの位置ベクトル
UnitX = -0


lens1V = np.array([-4.456+UnitX, 0, 0])  # レンズ１の位置ベクトル
lens2V = np.array([-3.7+UnitX, 0, 0])  # レンズ２の位置ベクトル
lens3V = np.array([-2.57+UnitX, 0, 0])  # レンズ３の位置ベクトル
lens4V = np.array([-0.78+UnitX, 0, 0])  # レンズ４の位置ベクトル
lens5V = np.array([-0.58+UnitX, 0, 0])  # レンズ5の位置ベクトル

Lens1Param = [4.78, 37.2/4, 0.8-0.044, [1.791, 1.791], [-1, 1], lens1V]
Lens2Param = [37.2/4, 26.25, 0.27, [1.788, 1.788], [1, -1], lens2V]
Lens3Param = [11.27, 4.3, 0.514, [1.26, 1.26], [1, -1], lens3V]
Lens4Param = [1000000, 26.32/4, 0.2, [1.395, 1.395], [1, -1], lens4V]
Lens5Param = [26.32/4, 9.305, 0.58, [1.395, 1.395], [-1, 1], lens5V]

# レンズ曲率半径
Rx11 = Lens1Param[0]
Rx12 = Lens1Param[1]
Rx21 = Lens2Param[0]
Rx22 = Lens2Param[1]
Rx31 = Lens3Param[0]
Rx32 = Lens3Param[1]
Rx41 = Lens4Param[0]
Rx42 = Lens4Param[1]
Rx51 = Lens5Param[0]
Rx52 = Lens5Param[1]

# レンズ厚さ
Lensd1 = Lens1Param[2]
Lensd2 = Lens2Param[2]
Lensd3 = Lens3Param[2]
Lensd4 = Lens4Param[2]
Lensd5 = Lens5Param[2]

Params = [Lens1Param, Lens2Param, Lens3Param, Lens4Param, Lens5Param]


# 逆伝播用、係数 Tを求める過程で符号逆転
class VectorFunctions_reverse:
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
        B = ((startV[0] - Rx11)*directionV[0]/Rx11**2)+(
                startV[1]*directionV[1]/Rx11**2)+(
                startV[2]*directionV[2]/Rx11**2)
        #print(B)
        C = -1+((startV[0] - Rx11)**2/Rx11**2)+(
                startV[1]**2/Rx11**2)+(
                startV[2]**2/Rx11**2)
        #print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    def rayTraceDecideT_Lens1R(self, startV, directionV):
        startV = startV - lens1V
        A = (directionV[0]**2/Rx12**2)+(
                directionV[1]**2/Rx12**2)+(
                directionV[2]**2/Rx12**2)
        #print(A)
        B = ((startV[0] + Rx12 - Lensd1)*directionV[0]/Rx12**2)+(
                startV[1]*directionV[1]/Rx12**2)+(
                startV[2]*directionV[2]/Rx12**2)
        #print(B)
        C = -1+((startV[0] + Rx12 - Lensd1)**2/Rx12**2)+(
                startV[1]**2/Rx12**2)+(
                startV[2]**2/Rx12**2)
        #print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    # レンズ１表面の法線を求める関数
    def decideNormalV_Lens1L(self, pointV):
        pointV = pointV - lens1V
        nornalVx = (2/Rx11**2)*(pointV[0] - Rx11)
        nornalVy = (2/Rx11**2)*pointV[1]
        nornalVz = (2/Rx11**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens1R(self, pointV):
        pointV = pointV - lens1V
        nornalVx = (2/Rx12**2)*(pointV[0] + Rx12 - Lensd1)
        nornalVy = (2/Rx12**2)*pointV[1]
        nornalVz = (2/Rx12**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV


    # レイトレーシング、光線ベクトルとレンズ２の交点を持つときの係数Ｔを求める関数
    def rayTraceDecideT_Lens2L(self, startV, directionV):
        startV = startV - lens2V
        A = (directionV[0]**2/Rx21**2)+(
                directionV[1]**2/Rx21**2)+(
                directionV[2]**2/Rx21**2)
        #print(A)
        B = ((startV[0] + Rx21)*directionV[0]/Rx21**2)+(
                startV[1]*directionV[1]/Rx21**2)+(
                startV[2]*directionV[2]/Rx21**2)
        #print(B)
        C = -1+((startV[0] + Rx21)**2/Rx21**2)+(
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
        B = ((startV[0] - Rx22 - Lensd2)*directionV[0]/Rx22**2)+(
                startV[1]*directionV[1]/Rx22**2)+(
                startV[2]*directionV[2]/Rx22**2)
        #print(B)
        C = -1+((startV[0] - Rx22 - Lensd2)**2/Rx22**2)+(
                startV[1]**2/Rx22**2)+(
                startV[2]**2/Rx22**2)
        #print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    # レンズ２表面の法線を求める関数
    def decideNormalV_Lens2L(self, pointV):
        pointV = pointV - lens2V
        nornalVx = -(2/Rx21**2)*(pointV[0] + Rx21)
        nornalVy = -(2/Rx21**2)*pointV[1]
        nornalVz = -(2/Rx21**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens2R(self, pointV):
        pointV = pointV - lens2V
        nornalVx = -(2/Rx22**2)*(pointV[0] - Rx22 - Lensd2)
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
        B = ((startV[0] + Rx31)*directionV[0]/Rx31**2)+(
                startV[1]*directionV[1]/Rx31**2)+(
                startV[2]*directionV[2]/Rx31**2)
        #print(B)
        C = -1+((startV[0] + Rx31)**2/Rx31**2)+(
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
        B = ((startV[0] - Rx32 - Lensd3)*directionV[0]/Rx32**2)+(
                startV[1]*directionV[1]/Rx32**2)+(
                startV[2]*directionV[2]/Rx32**2)
        #print(B)
        C = -1+((startV[0] - Rx32 - Lensd3)**2/Rx32**2)+(
                startV[1]**2/Rx32**2)+(
                startV[2]**2/Rx32**2)
        #print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    # レンズ３表面の法線を求める関数
    def decideNormalV_Lens3L(self, pointV):
        pointV = pointV - lens3V
        nornalVx = -(2/Rx31**2)*(pointV[0] + Rx31)
        nornalVy = -(2/Rx31**2)*pointV[1]
        nornalVz = -(2/Rx31**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens3R(self, pointV):
        pointV = pointV - lens3V
        nornalVx = -(2/Rx32**2)*(pointV[0] - Rx32 - Lensd3)
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
        B = ((startV[0] + Rx41)*directionV[0]/Rx41**2)+(
                startV[1]*directionV[1]/Rx41**2)+(
                startV[2]*directionV[2]/Rx41**2)
        #print(B)
        C = -1+((startV[0] + Rx41)**2/Rx41**2)+(
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
        B = ((startV[0] - Rx42 - Lensd4)*directionV[0]/Rx42**2)+(
                startV[1]*directionV[1]/Rx42**2)+(
                startV[2]*directionV[2]/Rx42**2)
        #print(B)
        C = -1+((startV[0] - Rx42 - Lensd4)**2/Rx42**2)+(
                startV[1]**2/Rx42**2)+(
                startV[2]**2/Rx42**2)
        #print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    # レンズ４表面の法線を求める関数
    def decideNormalV_Lens4L(self, pointV):
        pointV = pointV - lens4V
        nornalVx = -(2/Rx41**2)*(pointV[0] + Rx41)
        nornalVy = -(2/Rx41**2)*pointV[1]
        nornalVz = -(2/Rx41**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens4R(self, pointV):
        pointV = pointV - lens4V
        nornalVx = -(2/Rx42**2)*(pointV[0] - Rx42 - Lensd4)
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
        B = ((startV[0] - Rx51)*directionV[0]/Rx51**2)+(
                startV[1]*directionV[1]/Rx51**2)+(
                startV[2]*directionV[2]/Rx51**2)
        #print(B)
        C = -1+((startV[0] - Rx51)**2/Rx51**2)+(
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
        B = ((startV[0] + Rx52 - Lensd5)*directionV[0]/Rx52**2)+(
                startV[1]*directionV[1]/Rx52**2)+(
                startV[2]*directionV[2]/Rx52**2)
        #print(B)
        C = -1+((startV[0] + Rx52 - Lensd5)**2/Rx52**2)+(
                startV[1]**2/Rx52**2)+(
                startV[2]**2/Rx52**2)
        #print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    # レンズ5表面の法線を求める関数
    def decideNormalV_Lens5L(self, pointV):
        pointV = pointV - lens5V
        nornalVx = (2/Rx51**2)*(pointV[0] - Rx51)
        nornalVy = (2/Rx51**2)*pointV[1]
        nornalVz = (2/Rx51**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens5R(self, pointV):
        pointV = pointV - lens5V
        nornalVx = (2/Rx52**2)*(pointV[0] + Rx52 - Lensd5)
        nornalVy = (2/Rx52**2)*pointV[1]
        nornalVz = (2/Rx52**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV


    # スクリーンとの交点を持つときの係数Ｔを求める関数
    def rayTraceDecideT_Screen(self, startV, directionV):
        T = (-10-startV[0])/directionV[0]
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

    def plotLineBlack(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX,endX],[startY,endY],[startZ,endZ],
            'o-',ms='2',linewidth=0.5,color='black')


# 焦点距離を返す
def returnFocus_Lens12re(paramList):
    Nlens1 = paramList[0][0]
    NBlueRay1 = paramList[0][1]
    Nlens2 = paramList[1][0]
    NBlueRay2 = paramList[1][1]

    def T_FocusGraph(startV, directionV):
        T = -startV[2]/directionV[2]
        return T

    def T_FocalLength(startV, directionV):
        T = -startV[2]/directionV[2]
        return T

    VF = VectorFunctions_reverse()  # インスタンス化

    #LastRedPoints = []
    #LastBluePoints = []

    # 始点を生成する
    pointsZ = np.arange(0.001, 0.1, 0.1) + 0  # 近軸光線
    #pointsZ = np.arange(1, 1.1, 0.2) + 0            ##########仮
    pointsX = np.array([centerX]*len(pointsZ)) + 5
    pointsY = np.array([0]*len(pointsZ)) + 0
    raySPoint0 = VF.makePoints(pointsX, pointsY, pointsZ, len(pointsZ), 3)

    for i in raySPoint0:
        raySPoint0 = i
        directionVector0 = np.array([-1, 0, 0])  # 入射光の方向ベクトルを設定
        T = VF.rayTraceDecideT_Lens2R(raySPoint0, directionVector0)
        rayEPoint0 = raySPoint0 + T*directionVector0
        #VF.plotLinePurple(raySPoint0, rayEPoint0)  # 入射光描画

        # 赤色光
        refractionV_Lens3L = directionVector0

        refractSPoint_Lens2R = rayEPoint0  # 以下、レンズ２についての計算
        normalV_Lens2R = VF.decideNormalV_Lens2R(refractSPoint_Lens2R)
        refractionV_Lens2R = VF.decideRefractionVL(refractionV_Lens3L, normalV_Lens2R, Nair, Nlens2)
        T = VF.rayTraceDecideT_Lens2L(refractSPoint_Lens2R, refractionV_Lens2R)
        refractEPoint_Lens2R = refractSPoint_Lens2R + T*refractionV_Lens2R
        #VF.plotLineRed(refractSPoint_Lens2R,refractEPoint_Lens2R)  # 屈折光の描画
        raySPoint_Lens2L = refractEPoint_Lens2R
        normalV_Lens2L = VF.decideNormalV_Lens2L(raySPoint_Lens2L)
        refractionV_Lens2L = VF.decideRefractionVR(refractionV_Lens2R, normalV_Lens2L, Nlens1, Nlens2)
        T = 0  # レンズの接着を考えた
        rayEPoint_Lens1R = raySPoint_Lens2L + T*refractionV_Lens2L
        #VF.plotLineRed(raySPoint_Lens2L, rayEPoint_Lens1R)

        refractSPoint_Lens1R = rayEPoint_Lens1R  # 以下、レンズ1についての計算
        normalV_Lens1R = VF.decideNormalV_Lens1R(refractSPoint_Lens1R)
        refractionV_Lens1R = VF.decideRefractionVL(refractionV_Lens2L, normalV_Lens1R, Nlens2, Nlens1)
        T = VF.rayTraceDecideT_Lens1L(refractSPoint_Lens1R, refractionV_Lens1R)
        refractEPoint_Lens1R = refractSPoint_Lens1R + T*refractionV_Lens1R
        #VF.plotLineRed(refractSPoint_Lens1R,refractEPoint_Lens1R)  # 屈折光の描画
        raySPoint_Lens1L = refractEPoint_Lens1R
        normalV_Lens1L = VF.decideNormalV_Lens1L(raySPoint_Lens1L)
        refractionV_Lens1L = VF.decideRefractionVR(refractionV_Lens1R, normalV_Lens1L, Nair, Nlens1)
        T = VF.rayTraceDecideT_Screen(raySPoint_Lens1L, refractionV_Lens1L)
        rayEPoint_Last = raySPoint_Lens1L + T*refractionV_Lens1L
        #VF.plotLineRed(raySPoint_Lens1L, rayEPoint_Last)


        T = T_FocusGraph(raySPoint_Lens1L, refractionV_Lens1L)
        focusPoint = raySPoint_Lens1L + T*refractionV_Lens1L
        RedFocusPoints=focusPoint[0]

        T = T_FocalLength(refractSPoint_Lens2R, refractionV_Lens1L)
        principalPoint = focusPoint - T*refractionV_Lens1L
        RedPrincipalPoints=principalPoint[0]
        #print(RedFocusPoints)
        #print(RedPrincipalPoints)
        #VF.plotLineBlack(focusPoint+[0,1,0], principalPoint+[0,1,0])
        #VF.plotLineBlack(raySPoint0+[0,1,0], principalPoint+[0,1,0])
        #VF.plotLineBlack(focusPoint+[0,1,0], rayEPoint_Last+[0,1,0])

        # 青色光

        refractionV_Lens3L = directionVector0

        refractSPoint_Lens2R = rayEPoint0  # 以下、レンズ２についての計算
        normalV_Lens2R = VF.decideNormalV_Lens2R(refractSPoint_Lens2R)
        refractionV_Lens2R = VF.decideRefractionVL(refractionV_Lens3L, normalV_Lens2R, Nair, Nlens2*NBlueRay2)
        T = VF.rayTraceDecideT_Lens2L(refractSPoint_Lens2R, refractionV_Lens2R)
        refractEPoint_Lens2R = refractSPoint_Lens2R + T*refractionV_Lens2R
        #VF.plotLineBlue(refractSPoint_Lens2R,refractEPoint_Lens2R)  # 屈折光の描画
        raySPoint_Lens2L = refractEPoint_Lens2R
        normalV_Lens2L = VF.decideNormalV_Lens2L(raySPoint_Lens2L)
        refractionV_Lens2L = VF.decideRefractionVR(refractionV_Lens2R, normalV_Lens2L, Nlens1*NBlueRay1, Nlens2*NBlueRay2)
        T = 0  # レンズの接着を考えた
        rayEPoint_Lens1R = raySPoint_Lens2L + T*refractionV_Lens2L
        #VF.plotLineBlue(raySPoint_Lens2L, rayEPoint_Lens1R)

        refractSPoint_Lens1R = rayEPoint_Lens1R  # 以下、レンズ1についての計算
        normalV_Lens1R = VF.decideNormalV_Lens1R(refractSPoint_Lens1R)
        refractionV_Lens1R = VF.decideRefractionVL(refractionV_Lens2L, normalV_Lens1R, Nlens2*NBlueRay2, Nlens1*NBlueRay1)
        T = VF.rayTraceDecideT_Lens1L(refractSPoint_Lens1R, refractionV_Lens1R)
        refractEPoint_Lens1R = refractSPoint_Lens1R + T*refractionV_Lens1R
        #VF.plotLineBlue(refractSPoint_Lens1R,refractEPoint_Lens1R)  # 屈折光の描画
        raySPoint_Lens1L = refractEPoint_Lens1R
        normalV_Lens1L = VF.decideNormalV_Lens1L(raySPoint_Lens1L)
        refractionV_Lens1L = VF.decideRefractionVR(refractionV_Lens1R, normalV_Lens1L, Nair, Nlens1*NBlueRay1)
        T = VF.rayTraceDecideT_Screen(raySPoint_Lens1L, refractionV_Lens1L)
        rayEPoint_Last = raySPoint_Lens1L + T*refractionV_Lens1L
        #VF.plotLineBlue(raySPoint_Lens1L, rayEPoint_Last)


        T = T_FocusGraph(raySPoint_Lens1L, refractionV_Lens1L)
        focusPoint = raySPoint_Lens1L + T*refractionV_Lens1L
        BlueFocusPoints=focusPoint[0]

        T = T_FocalLength(refractSPoint_Lens2R, refractionV_Lens1L)
        principalPoint = focusPoint - T*refractionV_Lens1L
        BluePrincipalPoints=principalPoint[0]
        #VF.plotLineBlack(focusPoint+[0,1,0], principalPoint+[0,1,0])
        #VF.plotLineBlack(raySPoint0+[0,1,0], principalPoint+[0,1,0])
        #VF.plotLineBlack(focusPoint+[0,1,0], rayEPoint_Last+[0,1,0])


    #ax.plot(RedFocusPoints, pointsY, pointsZ, color='r')
    #ax.plot(BlueFocusPoints, pointsY, pointsZ, color='b')

    #return RedFocusPoints
    #return BlueFocusPoints
    #return RedPrincipalPoints
    #return BluePrincipalPoints
    #print('time =', time.time()-start)

    return -RedFocusPoints+RedPrincipalPoints, -BlueFocusPoints+BluePrincipalPoints, -RedFocusPoints+RedPrincipalPoints+BlueFocusPoints-BluePrincipalPoints

'''
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


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
limitTheta = 2*np.pi  # theta生成数
limitPhi = np.pi  # phi生成数
theta = np.linspace(0, limitTheta, geneNum)
phi = np.linspace(0, limitPhi, geneNum)
for i in Params:
    plotLens(*i)
returnFocus_Lens12re([[1.61506, 1.01054], [1.54457, 1.00776]])
ax.set_xlim(-LX+focusX, LX+focusX)
ax.set_ylim(-LY, LY)
ax.set_zlim(-LZ, LZ)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=0, azim=-90)
plt.show()
'''