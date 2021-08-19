import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import time

from Tessar import pointsTessar
from ZoomLens_Rendering import pointsZoomLens


def getScreenPoint(rayStartV):
    pointsZoomLens(rayStartV)


if __name__ == "__main__":
    print('\n----------------START----------------\n')
    start = time.time()

    # searchParam_Tessar_Layer or searchParam_ZoomLens_Layer
    getScreenPoint()

    print('time =', time.time()-start)
