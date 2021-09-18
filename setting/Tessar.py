# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import time


LX = 4
LY = 4
LZ = 4
geneNum = 500
Nair = 1  # 空気の屈折率

centerX = -8  # 入射光の始点の中心座標
centerY = 0  # 入射光の始点の中心座標
centerZ = 0  # 入射光の始点の中心座標
rayDensity = 0.25  # 入射光の密度
focusX = 4  # 焦点付近の描画範囲を平行移動

Rx1 = 0.5  # レンズ１の倍率
Ry1 = 3  # レンズ１の倍率
Rz1 = 3  # レンズ１の倍率
lens1V = np.array([-1.8, 0, 0])  # レンズ１の位置ベクトル
Rx21 = 0.2  # レンズ２の倍率１
Rx22 = 0.3  # レンズ２の倍率２
Ry21 = 3  # レンズ２の倍率１
Ry22 = 2.5  # レンズ２の倍率２
Rz21 = 3  # レンズ２の倍率１
Rz22 = 2.5  # レンズ２の倍率２
lens2V = np.array([-1.1, 0, 0])  # レンズ２の位置ベクトル
Rx31 = 0.01  # レンズ３の倍率１
Rx32 = 0.8  # レンズ３の倍率２
Ry31 = 2.5  # レンズ３の倍率１
Ry32 = 3  # レンズ３の倍率２
Rz31 = 2.5  # レンズ３の倍率１
Rz32 = 3  # レンズ３の倍率２
lens3V = np.array([1.5, 0, 0])  # レンズ３の位置ベクトル
Rx4 = Rx32  # レンズ４の倍率
Ry4 = 3  # レンズ４の倍率
Rz4 = 3  # レンズ４の倍率
lens4V = np.array([1.5, 0, 0])  # レンズ４の位置ベクトル

screenV = np.array([8, 0, 0])  # スクリーンの位置ベクトル

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

    '''
    # レイトレーシング、光線ベクトルとレンズ0の交点を持つときの係数Ｔを求める関数
    def rayTraceDecideT_Lens0L(self, startV, directionV):
        startV = startV - lens0V
        A = (directionV[0]**2/Rx0**2)+(
                directionV[1]**2/Ry0**2)+(
                directionV[2]**2/Rz0**2)
        #print(A)
        B = (startV[0]*directionV[0]/Rx0**2)+(
                startV[1]*directionV[1]/Ry0**2)+(
                startV[2]*directionV[2]/Rz0**2)
        #print(B)
        C = -1+(startV[0]**2/Rx0**2)+(
                startV[1]**2/Ry0**2)+(
                startV[2]**2/Rz0**2)
        #print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    def rayTraceDecideT_Lens0R(self, startV, directionV):
        startV = startV - lens0V
        A = (directionV[0]**2/Rx0**2)+(
                directionV[1]**2/Ry0**2)+(
                directionV[2]**2/Rz0**2)
        #print(A)
        B = (startV[0]*directionV[0]/Rx0**2)+(
                startV[1]*directionV[1]/Ry0**2)+(
                startV[2]*directionV[2]/Rz0**2)
        #print(B)
        C = -1+(startV[0]**2/Rx0**2)+(
                startV[1]**2/Ry0**2)+(
                startV[2]**2/Rz0**2)
        #print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    # レンズ0表面の法線を求める関数
    def decideNormalV_Lens0(self, pointV):
        pointV = pointV - lens0V
        nornalVx = (2/Rx0**2)*pointV[0]
        nornalVy = (2/Ry0**2)*pointV[1]
        nornalVz = (2/Rz0**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV
    '''

    # レイトレーシング、光線ベクトルとレンズ１の交点を持つときの係数Ｔを求める関数
    def rayTraceDecideT_Lens1L(self, startV, directionV):
        startV = startV - lens1V
        A = (directionV[0]**2/Rx1**2)+(
                directionV[1]**2/Ry1**2)+(
                directionV[2]**2/Rz1**2)
        #print(A)
        B = (startV[0]*directionV[0]/Rx1**2)+(
                startV[1]*directionV[1]/Ry1**2)+(
                startV[2]*directionV[2]/Rz1**2)
        #print(B)
        C = -1+(startV[0]**2/Rx1**2)+(
                startV[1]**2/Ry1**2)+(
                startV[2]**2/Rz1**2)
        #print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    def rayTraceDecideT_Lens1R(self, startV, directionV):
        T = (lens1V[0] - startV[0])/directionV[0]
        return T

    # レンズ１表面の法線を求める関数
    def decideNormalV_Lens1L(self, pointV):
        pointV = pointV - lens1V
        nornalVx = (2/Rx1**2)*pointV[0]
        nornalVy = (2/Ry1**2)*pointV[1]
        nornalVz = (2/Rz1**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens1R(self, pointV):
        pointV = pointV - lens1V
        nornalVx = 1
        nornalVy = 0
        nornalVz = 0
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV


    # レイトレーシング、光線ベクトルとレンズ２の交点を持つときの係数Ｔを求める関数
    def rayTraceDecideT_Lens2L(self, startV, directionV):
        startV = startV - lens2V
        A = (directionV[0]**2/Rx21**2)+(
                directionV[1]**2/Ry21**2)+(
                directionV[2]**2/Rz21**2)
        #print(A)
        B = (startV[0]*directionV[0]/Rx21**2)+(
                startV[1]*directionV[1]/Ry21**2)+(
                startV[2]*directionV[2]/Rz21**2)
        #print(B)
        C = -1+(startV[0]**2/Rx21**2)+(
                startV[1]**2/Ry21**2)+(
                startV[2]**2/Rz21**2)
        #print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    def rayTraceDecideT_Lens2R(self, startV, directionV):
        startV = startV - lens2V
        A = (directionV[0]**2/Rx22**2)+(
                directionV[1]**2/Ry22**2)+(
                directionV[2]**2/Rz22**2)
        #print(A)
        B = ((startV[0] - 0.8)*directionV[0]/Rx22**2)+(
                startV[1]*directionV[1]/Ry22**2)+(
                startV[2]*directionV[2]/Rz22**2)
        #print(B)
        C = -1+((startV[0] - 0.8)**2/Rx22**2)+(
                startV[1]**2/Ry22**2)+(
                startV[2]**2/Rz22**2)
        #print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    # レンズ２表面の法線を求める関数
    def decideNormalV_Lens2L(self, pointV):
        pointV = pointV - lens2V
        nornalVx = -(2/Rx21**2)*pointV[0]
        nornalVy = -(2/Ry21**2)*pointV[1]
        nornalVz = -(2/Rz21**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens2R(self, pointV):
        pointV = pointV - lens2V
        nornalVx = -(2/Rx22**2)*(pointV[0] - 0.8)
        nornalVy = -(2/Ry22**2)*pointV[1]
        nornalVz = -(2/Rz22**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV


    # レイトレーシング、光線ベクトルとレンズ３の交点を持つときの係数Ｔを求める関数
    def rayTraceDecideT_Lens3L(self, startV, directionV):
        startV = startV - lens3V
        A = (directionV[0]**2/Rx31**2)+(
                directionV[1]**2/Ry31**2)+(
                directionV[2]**2/Rz31**2)
        #print(A)
        B = ((startV[0] + 1.1)*directionV[0]/Rx31**2)+(
                startV[1]*directionV[1]/Ry31**2)+(
                startV[2]*directionV[2]/Rz31**2)
        #print(B)
        C = -1+((startV[0] + 1.1)**2/Rx31**2)+(
                startV[1]**2/Ry31**2)+(
                startV[2]**2/Rz31**2)
        #print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    def rayTraceDecideT_Lens3R(self, startV, directionV):
        startV = startV - lens3V
        A = (directionV[0]**2/Rx32**2)+(
                directionV[1]**2/Ry32**2)+(
                directionV[2]**2/Rz32**2)
        #print(A)
        B = (startV[0]*directionV[0]/Rx32**2)+(
                startV[1]*directionV[1]/Ry32**2)+(
                startV[2]*directionV[2]/Rz32**2)
        #print(B)
        C = -1+(startV[0]**2/Rx32**2)+(
                startV[1]**2/Ry32**2)+(
                startV[2]**2/Rz32**2)
        #print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    # レンズ３表面の法線を求める関数
    def decideNormalV_Lens3L(self, pointV):
        pointV = pointV - lens3V
        nornalVx = -(2/Rx31**2)*(pointV[0] + 1.6)
        nornalVy = -(2/Ry31**2)*pointV[1]
        nornalVz = -(2/Rz31**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens3R(self, pointV):
        pointV = pointV - lens3V
        nornalVx = -(2/Rx32**2)*pointV[0]
        nornalVy = -(2/Ry32**2)*pointV[1]
        nornalVz = -(2/Rz32**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV


    # レイトレーシング、光線ベクトルとレンズ４の交点を持つときの係数Ｔを求める関数
    def rayTraceDecideT_Lens4L(self, startV, directionV):
        startV = startV - lens4V
        A = (directionV[0]**2/Rx4**2)+(
                directionV[1]**2/Ry4**2)+(
                directionV[2]**2/Rz4**2)
        #print(A)
        B = (startV[0]*directionV[0]/Rx4**2)+(
                startV[1]*directionV[1]/Ry4**2)+(
                startV[2]*directionV[2]/Rz4**2)
        #print(B)
        C = -1+(startV[0]**2/Rx4**2)+(
                startV[1]**2/Ry4**2)+(
                startV[2]**2/Rz4**2)
        #print(C)
        T = (-B-np.sqrt(B**2-A*C))/A
        return T

    def rayTraceDecideT_Lens4R(self, startV, directionV):
        startV = startV - lens4V
        A = (directionV[0]**2/Rx4**2)+(
                directionV[1]**2/Ry4**2)+(
                directionV[2]**2/Rz4**2)
        #print(A)
        B = (startV[0]*directionV[0]/Rx4**2)+(
                startV[1]*directionV[1]/Ry4**2)+(
                startV[2]*directionV[2]/Rz4**2)
        #print(B)
        C = -1+(startV[0]**2/Rx4**2)+(
                startV[1]**2/Ry4**2)+(
                startV[2]**2/Rz4**2)
        #print(C)
        T = (-B+np.sqrt(B**2-A*C))/A
        return T

    # レンズ４表面の法線を求める関数
    def decideNormalV_Lens4L(self, pointV):
        pointV = pointV - lens4V
        nornalVx = (2/Rx4**2)*pointV[0]
        nornalVy = (2/Ry4**2)*pointV[1]
        nornalVz = (2/Rz4**2)*pointV[2]
        normalV = np.array([nornalVx, nornalVy, nornalVz])
        return normalV

    def decideNormalV_Lens4R(self, pointV):
        pointV = pointV - lens4V
        nornalVx = (2/Rx4**2)*pointV[0]
        nornalVy = (2/Ry4**2)*pointV[1]
        nornalVz = (2/Rz4**2)*pointV[2]
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
    '''
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
    '''


# トリプレット（テッサー）のスクリーン上に映った点を返す関数
def pointsTessar(Nlens1=1.44, Nlens2=1.44, Nlens3=1.44, Nlens4=1.46,
            NBlueRay1=1.016, NBlueRay2=1.016, NBlueRay3=1.013, NBlueRay4=1.006):
    '''
    def plotLensTriplet():
        # １枚目の凸レンズを再現する
        limitTheta = 2*np.pi  # theta生成数
        limitPhi = np.pi  # phi生成数
        theta = np.linspace(0, limitTheta, geneNum)
        phi = np.linspace(0, limitPhi, geneNum)
        Xs = Rx1 * np.outer(np.cos(theta), np.sin(phi))
        Xs = np.where(0<Xs, 0, Xs) + lens1V[0]
        Ys = Ry1 * np.outer(np.sin(theta), np.sin(phi)) + lens1V[1]
        Zs = Rz1 * np.outer(np.ones(np.size(theta)), np.cos(phi)) + lens1V[2]
        ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2)

        # ２枚目の凹レンズを再現する
        limitTheta = 2*np.pi  # theta生成数
        limitPhi = np.pi  # phi生成数
        theta = np.linspace(0, limitTheta, geneNum)
        phi = np.linspace(0, limitPhi, geneNum)
        Xs = Rx21 * np.outer(np.cos(theta), np.sin(phi))
        Ys = Ry21 * np.outer(np.sin(theta), np.sin(phi))
        Zs = Rz21 * np.outer(np.ones(np.size(theta)), np.cos(phi))
        Xs1 = Rx21 * np.outer(np.cos(theta), np.sin(phi))
        Xs2 = Rx22 * np.outer(np.cos(theta), np.sin(phi)) + 0.8
        Ys1 = Ry21 * np.outer(np.sin(theta), np.sin(phi))
        Ys2 = Ry22 * np.outer(np.sin(theta), np.sin(phi))
        Zs1 = Rz21 * np.outer(np.ones(np.size(theta)), np.cos(phi))
        Zs2 = Rz22 * np.outer(np.ones(np.size(theta)), np.cos(phi))
        Ys = np.where(Xs<0, Ys2, Ys1) + lens2V[1]
        Zs = np.where(Xs<0, Zs2, Zs1) + lens2V[2]
        Xs = np.where(Xs<0, Xs2, Xs1) + lens2V[0]
        ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2)

        # ３枚目の凹レンズを再現する
        limitTheta = 2*np.pi  # theta生成数
        limitPhi = np.pi  # phi生成数
        theta = np.linspace(0, limitTheta, geneNum)
        phi = np.linspace(0, limitPhi, geneNum)
        Xs = Rx31 * np.outer(np.cos(theta), np.sin(phi))
        Ys = Ry31 * np.outer(np.sin(theta), np.sin(phi))
        Zs = Rz31 * np.outer(np.ones(np.size(theta)), np.cos(phi))
        Xs1 = Rx31 * np.outer(np.cos(theta), np.sin(phi))
        Xs2 = Rx32 * np.outer(np.cos(theta), np.sin(phi)) + 1.1
        Ys1 = Ry31 * np.outer(np.sin(theta), np.sin(phi))
        Ys2 = Ry32 * np.outer(np.sin(theta), np.sin(phi))
        Zs1 = Rz31 * np.outer(np.ones(np.size(theta)), np.cos(phi))
        Zs2 = Rz32 * np.outer(np.ones(np.size(theta)), np.cos(phi))
        Ys = np.where(Xs<0, Ys2, Ys1) + lens3V[1]
        Zs = np.where(Xs<0, Zs2, Zs1) + lens3V[2]
        Xs = np.where(Xs<0, Xs2, Xs1) + lens3V[0] - 1.1
        ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2)

        # ４枚目のレンズを再現する
        limitTheta = 2*np.pi  # theta生成数
        limitPhi = np.pi  # phi生成数
        theta = np.linspace(0, limitTheta, geneNum)
        phi = np.linspace(0, limitPhi, geneNum)
        Xs = Rx4 * np.outer(np.cos(theta), np.sin(phi)) + lens4V[0]
        Ys = Ry4 * np.outer(np.sin(theta), np.sin(phi)) + lens4V[1]
        Zs = Rz4 * np.outer(np.ones(np.size(theta)), np.cos(phi)) + lens4V[2]
        ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2)

        # スクリーン
        Ys, Zs = np.meshgrid(
            np.arange(-3, 3.5, 0.5),
            np.arange(-3, 3.5, 0.5))
        Xs = 0*Ys + 0*Zs + screenV[0]
        ax.plot_wireframe(Xs, Ys, Zs, linewidth=0.2, color='k')
    '''

    #plotLensTriplet()
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
        directionVector0 = np.array([1, 0, 0])  # 入射光の方向ベクトルを設定
        T = VF.rayTraceDecideT_Lens1L(raySPoint0, directionVector0)  # 交点のための係数
        rayEPoint0 = raySPoint0 + T*directionVector0  # 入射光の終点
        #VF.plotLinePurple(raySPoint0, rayEPoint0)  # 入射光描画

        # 赤色光
        refractSPoint0 = rayEPoint0  # 入射光の終点を引き継ぐ。以下レンズ１についての計算
        normalV_Lens1L = VF.decideNormalV_Lens1L(refractSPoint0)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1L = VF.decideRefractionVL(directionVector0, normalV_Lens1L, Nair, Nlens1)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens1R(refractSPoint0, refractionV_Lens1L)
        refractEPoint0 = refractSPoint0 + T*refractionV_Lens1L
        #VF.plotLineRed(refractSPoint0,refractEPoint0)  # 屈折光の描画
        raySPoint1 = refractEPoint0  # 屈折光の終点を引き継ぐ
        normalV1 = VF.decideNormalV_Lens1R(raySPoint1)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1R = VF.decideRefractionVR(refractionV_Lens1L, normalV1, Nair, Nlens1)
        T = VF.rayTraceDecideT_Lens2L(raySPoint1, refractionV_Lens1R)
        rayEPoint1 = raySPoint1 + T*refractionV_Lens1R  # 空気中の屈折光の終点
        #VF.plotLineRed(raySPoint1,rayEPoint1)  # 空気中の屈折光の描画

        refractSPoint_Lens2L = rayEPoint1  # 以下、レンズ２についての計算
        normalV_Lens2L = VF.decideNormalV_Lens2L(refractSPoint_Lens2L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens2L = VF.decideRefractionVL(refractionV_Lens1R, normalV_Lens2L, Nair, Nlens2)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens2R(refractSPoint_Lens2L, refractionV_Lens2L)
        refractEPoint_Lens2L = refractSPoint_Lens2L + T*refractionV_Lens2L
        #VF.plotLineRed(refractSPoint_Lens2L,refractEPoint_Lens2L)  # 屈折光の描画
        raySPoint_Lens2R = refractEPoint_Lens2L
        normalV_Lens2R = VF.decideNormalV_Lens2R(raySPoint_Lens2R)
        refractionV_Lens2R = VF.decideRefractionVR(refractionV_Lens2L, normalV_Lens2R, Nair, Nlens2)
        T = VF.rayTraceDecideT_Lens3L(raySPoint_Lens2R, refractionV_Lens2R)
        rayEPoint_Lens3L = raySPoint_Lens2R + T*refractionV_Lens2R
        #VF.plotLineRed(raySPoint_Lens2R, rayEPoint_Lens3L)

        refractSPoint_Lens3L = rayEPoint_Lens3L  # 以下、レンズ３についての計算
        normalV_Lens3L = VF.decideNormalV_Lens3L(refractSPoint_Lens3L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens3L = VF.decideRefractionVL(refractionV_Lens2R, normalV_Lens3L, Nair, Nlens3)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens3R(refractSPoint_Lens3L, refractionV_Lens3L)
        refractEPoint_Lens3L = refractSPoint_Lens3L + T*refractionV_Lens3L
        #VF.plotLineRed(refractSPoint_Lens3L,refractEPoint_Lens3L)  # 屈折光の描画
        raySPoint_Lens3R = refractEPoint_Lens3L
        normalV_Lens3R = VF.decideNormalV_Lens3R(raySPoint_Lens3R)
        refractionV_Lens3R = VF.decideRefractionVR(refractionV_Lens3L, normalV_Lens3R, Nlens4, Nlens3)
        T = 0  # レンズ３とレンズ４の接着を考えた
        rayEPoint_Lens4L = raySPoint_Lens3R + T*refractionV_Lens3R
        #VF.plotLineRed(raySPoint_Lens3R, rayEPoint_Lens4L)

        refractSPoint_Lens4L = rayEPoint_Lens4L  # 以下、レンズ４についての計算
        normalV_Lens4L = VF.decideNormalV_Lens4L(refractSPoint_Lens4L)
        refractionV_Lens4L = VF.decideRefractionVL(refractionV_Lens3R, normalV_Lens4L, Nlens3, Nlens4)
        T = VF.rayTraceDecideT_Lens4R(refractSPoint_Lens4L, refractionV_Lens4L)
        refractEPoint_Lens4L = refractSPoint_Lens4L + T*refractionV_Lens4L
        #VF.plotLineRed(refractSPoint_Lens4L,refractEPoint_Lens4L)  # 屈折光の描画
        raySPoint_Lens4R = refractEPoint_Lens4L
        normalV_Lens4R = VF.decideNormalV_Lens4R(raySPoint_Lens4R)
        refractionV_Lens4R = VF.decideRefractionVR(refractionV_Lens4L, normalV_Lens4R, Nair, Nlens4)
        T = VF.rayTraceDecideT_Screen(raySPoint_Lens4R, refractionV_Lens4R)  # スクリーン
        rayEPoint_Last = raySPoint_Lens4R + T*refractionV_Lens4R
        #VF.plotLineRed(raySPoint_Lens4R, rayEPoint_Last)

        LastRedPoints.append(rayEPoint_Last)

        # 青色光
        refractSPoint0 = rayEPoint0  # 入射光の終点を引き継ぐ。以下レンズ１についての計算
        normalV_Lens1L = VF.decideNormalV_Lens1L(refractSPoint0)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1L = VF.decideRefractionVL(directionVector0, normalV_Lens1L, Nair, Nlens1*NBlueRay1)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens1R(refractSPoint0, refractionV_Lens1L)
        refractEPoint0 = refractSPoint0 + T*refractionV_Lens1L
        #VF.plotLineBlue(refractSPoint0,refractEPoint0)  # 屈折光の描画
        raySPoint1 = refractEPoint0  # 屈折光の終点を引き継ぐ
        normalV1 = VF.decideNormalV_Lens1R(raySPoint1)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens1R = VF.decideRefractionVR(refractionV_Lens1L, normalV1, Nair, Nlens1*NBlueRay1)
        T = VF.rayTraceDecideT_Lens2L(raySPoint1, refractionV_Lens1R)
        rayEPoint1 = raySPoint1 + T*refractionV_Lens1R  # 空気中の屈折光の終点
        #VF.plotLineBlue(raySPoint1,rayEPoint1)  # 空気中の屈折光の描画

        refractSPoint_Lens2L = rayEPoint1  # 以下、レンズ２についての計算
        normalV_Lens2L = VF.decideNormalV_Lens2L(refractSPoint_Lens2L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens2L = VF.decideRefractionVL(refractionV_Lens1R, normalV_Lens2L, Nair, Nlens2*NBlueRay2)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens2R(refractSPoint_Lens2L, refractionV_Lens2L)
        refractEPoint_Lens2L = refractSPoint_Lens2L + T*refractionV_Lens2L
        #VF.plotLineBlue(refractSPoint_Lens2L,refractEPoint_Lens2L)  # 屈折光の描画
        raySPoint_Lens2R = refractEPoint_Lens2L
        normalV_Lens2R = VF.decideNormalV_Lens2R(raySPoint_Lens2R)
        refractionV_Lens2R = VF.decideRefractionVR(refractionV_Lens2L, normalV_Lens2R, Nair, Nlens2*NBlueRay2)
        T = VF.rayTraceDecideT_Lens3L(raySPoint_Lens2R, refractionV_Lens2R)
        rayEPoint_Lens3L = raySPoint_Lens2R + T*refractionV_Lens2R
        #VF.plotLineBlue(raySPoint_Lens2R, rayEPoint_Lens3L)

        refractSPoint_Lens3L = rayEPoint_Lens3L  # 以下、レンズ３についての計算
        normalV_Lens3L = VF.decideNormalV_Lens3L(refractSPoint_Lens3L)  # レンズの法線を求める
        # 屈折光の方向ベクトルを求める
        refractionV_Lens3L = VF.decideRefractionVL(refractionV_Lens2R, normalV_Lens3L, Nair, Nlens3*NBlueRay3)
        # 係数Tを求めて、屈折光の終点も求める
        T = VF.rayTraceDecideT_Lens3R(refractSPoint_Lens3L, refractionV_Lens3L)
        refractEPoint_Lens3L = refractSPoint_Lens3L + T*refractionV_Lens3L
        #VF.plotLineBlue(refractSPoint_Lens3L,refractEPoint_Lens3L)  # 屈折光の描画
        raySPoint_Lens3R = refractEPoint_Lens3L
        normalV_Lens3R = VF.decideNormalV_Lens3R(raySPoint_Lens3R)
        refractionV_Lens3R = VF.decideRefractionVR(refractionV_Lens3L, normalV_Lens3R, Nlens4*NBlueRay4, Nlens3*NBlueRay3)
        T = 0  # レンズ３とレンズ４の接着を考えた
        rayEPoint_Lens4L = raySPoint_Lens3R + T*refractionV_Lens3R
        #VF.plotLineBlue(raySPoint_Lens3R, rayEPoint_Lens4L)

        refractSPoint_Lens4L = rayEPoint_Lens4L  # 以下、レンズ４についての計算
        normalV_Lens4L = VF.decideNormalV_Lens4L(refractSPoint_Lens4L)
        refractionV_Lens4L = VF.decideRefractionVL(refractionV_Lens3R, normalV_Lens4L, Nlens3*NBlueRay3, Nlens4*NBlueRay4)
        T = VF.rayTraceDecideT_Lens4R(refractSPoint_Lens4L, refractionV_Lens4L)
        refractEPoint_Lens4L = refractSPoint_Lens4L + T*refractionV_Lens4L
        #VF.plotLineBlue(refractSPoint_Lens4L,refractEPoint_Lens4L)  # 屈折光の描画
        raySPoint_Lens4R = refractEPoint_Lens4L
        normalV_Lens4R = VF.decideNormalV_Lens4R(raySPoint_Lens4R)
        refractionV_Lens4R = VF.decideRefractionVR(refractionV_Lens4L, normalV_Lens4R, Nair, Nlens4*NBlueRay4)
        T = VF.rayTraceDecideT_Screen(raySPoint_Lens4R, refractionV_Lens4R)  # スクリーン
        rayEPoint_Last = raySPoint_Lens4R + T*refractionV_Lens4R
        #VF.plotLineBlue(raySPoint_Lens4R, rayEPoint_Last)

        LastBluePoints.append(rayEPoint_Last)
    return LastRedPoints, LastBluePoints
