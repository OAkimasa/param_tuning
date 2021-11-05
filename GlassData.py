import numpy as np
from numpy.lib.function_base import append

def GlassData():
    # [nC, nF/nC]  GlassName
    GlassList = [
        [1.48534, np.round(1.49225/1.48534, 5)],  # FK5HTi
        [1.45446, np.round(1.45948/1.45446, 5)],  # N-FK58
        [1.54965, np.round(1.55835/1.54965, 5)],  # N-PSK3
        [1.61503, np.round(1.62478/1.61503, 5)],  # N-PSK53A
        [1.51432, np.round(1.52238/1.51432, 5)],  # SCHOTT N-BK 7
        [1.49552, np.round(1.50296/1.49552, 5)],  # N-BK10
        [1.50854, np.round(1.51700/1.50854, 5)],  # K7
        [1.49867, np.round(1.50756/1.49867, 5)],  # K10
        [1.51982, np.round(1.52860/1.51982, 5)],  # N-K5
        [1.50592, np.round(1.51423/1.50592, 5)],  # N-ZK7
        [1.56949, np.round(1.57943/1.56949, 5)],  # N-BAK1
        [1.53721, np.round(1.54625/1.53721, 5)],  # N-BAK2
        [1.56575, np.round(1.57591/1.56575, 5)],  # N-BAK4
        [1.60157, np.round(1.61542/1.60157, 5)],  # N-BAF4
        [1.66578, np.round(1.68000/1.66578, 5)],  # N-BAF10
        [1.64792, np.round(1.66243/1.64792, 5)],  # N-BAF51
        [1.60473, np.round(1.61779/1.60473, 5)],  # N-BAF52
        [1.57631, np.round(1.58707/1.57631, 5)],  # N-BALF4
        [1.54430, np.round(1.55451/1.54430, 5)],  # N-BALF5
        [1.60414, np.round(1.61486/1.60414, 5)],  # N-SK2
        [1.60954, np.round(1.61999/1.60954, 5)],  # N-SK4
        [1.58619, np.round(1.59581/1.58619, 5)],  # N-SK5
        [1.56101, np.round(1.57028/1.56101, 5)],  # N-SK11
        [1.60008, np.round(1.61003/1.60008, 5)],  # N-SK14
        [1.61727, np.round(1.62756/1.61727, 5)],  # N-SK16
        [1.52040, np.round(1.53056/1.52040, 5)],  # N-KF9
        [1.54457, np.round(1.55655/1.54457, 5)],  # LLF1
        [1.57723, np.round(1.59146/1.57723, 5)],  # LF5
        [1.61503, np.round(1.63208/1.61503, 5)],  # F2
        [1.59875, np.round(1.61461/1.59875, 5)],  # F5
        [1.61506, np.round(1.63208/1.61506, 5)],  # N-F2
        [1.84255, np.round(1.86898/1.84255, 5)],  # N-LASF9
        [1.79436, np.round(1.81726/1.79436, 5)],  # N-LASF45
        [1.79608, np.round(1.82783/1.79608, 5)],  # N-SF6
        [1.83650, np.round(1.87210/1.83650, 5)],  # N-SF57
        [1.79609, np.round(1.82775/1.79609, 5)]  # SF6
    ]
    return GlassList