'''
此代码包含IDW插值方法、球面距离和平面距离计算
编写：柴熠垲
'''

import numpy as np
from math import radians, cos, sin, asin, sqrt

def distance_lonlat(lon1, lat1, lon2, lat2):
    """
    计算两点间球面距离

    Args:
        lon1 (float): 点1的经度
        lat1 (float): 点1的纬度
        lon2 (float): 点2的经度
        lat2 (float): 点2的纬度

    returns:
        d (float): 两点间球面距离
    """
    R = 6372.8 # 设置地球半径的常数，单位为千米
    # Haversine Formula
    dLon = radians(lon2 - lon1)
    dLat = radians(lat2 - lat1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
    c = 2 * asin(sqrt(a))
    d = R * c
    return d

def IDW(x, y, z, xi, yi):
    """
    计算每个点的插值结果

    Args:
        x (array): 已知点的x坐标
        y (array): 已知点的y坐标
        z (array): 已知点的值
        xi (array): 目标点的x坐标
        yi (array): 目标点的y坐标

    returns:
        lstxyzi (list): 目标点的x、y坐标和插值结果
    """
    lstxyzi = [] # 用于存储目标点的x、y坐标和插值结果
    for p in range(len(xi)): # 遍历目标点的x坐标
        for q in range(len(yi)): # 遍历目标点的y坐标
            lstdist = [] # 用于存储该目标点到所有已知点的距离
            for s in range(len(x)): # 遍历已知点
                d = distance_lonlat(x[s], y[s], xi[p], yi[q]) # 计算该目标点到所有已知点的球面距离
                lstdist.append(d)
            # 计算插值
            '''
            np.power(基数，指数)
            可调整参数: 括号里指数可以输入1或者2
            '''
            w = list((1 / np.power(lstdist, 2))) # 权重。
            sumw = np.sum(w) # 权重总和
            sumwzi = np.sum(np.array(w) * np.array(z)) # 加权和
            u = sumwzi / sumw # 加权平均
            xyzi = [xi[p], yi[q], u] # 组合目标点的x、y坐标和插值结果为列表
            lstxyzi.append(xyzi)
    return lstxyzi