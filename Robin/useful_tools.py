import numpy as np
from math import *
import matplotlib.pyplot as plt
def isPointinPolygon(point, rangelist):  #[[0,0],[1,1],[0,1],[0,0]] [1,0.8]
    # 判断是否在外包矩形内，如果不在，直接返回false
    lnglist = []
    latlist = []
    for i in range(len(rangelist)-1):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)
    if (point[0] > maxlng or point[0] < minlng or
        point[1] > maxlat or point[1] < minlat):
        return False
    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        # 点与多边形顶点重合
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            return False
        # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
            # 求线段与射线交点 再和lat比较
            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0])/(point2[1] - point1[1])
            # 点在多边形边上
            if (point12lng == point[0]):
                return False
            if (point12lng < point[0]):
                count +=1
        point1 = point2
    if count%2 == 0:
        return False
    else:
        return True

def polyshow(pol):
    x = np.random.rand(1000,2)
    i= 0
    while i < 1000:
        p = np.random.rand(2)
        p[0] = p[0]*np.max(pol)
        p[1] = p[1]*np.max(pol)
        if isPointinPolygon(p, pol ):
            x[i,:] = p
            i= i+1
    plt.scatter(x[:,0],x[:,1])
    
def Genpoint(Batchsize,pol):
    lnglist = []
    latlist = []
    for i in range(len(pol)-1):
        lnglist.append(pol[i][0])
        latlist.append(pol[i][1])
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)
    x = np.random.rand(Batchsize,2)
    i= 0
    while i < Batchsize:
        p = np.random.rand(2)
        p[0] = p[0]*maxlng
        p[1] = p[1]*maxlat
        if isPointinPolygon(p,pol):
            x[i,:] = p
            i= i+1
    return x

def Distance_square(p,a,b):
    AB = b-a
    AP = p-a
    AB_AP = AB[0] * AP[0] + AB[1] * AP[1]
    AB2 = AB[0] * AB[0] + AB[1] * AB[1]
    t = AB_AP/AB2
    if t>1:
        D = b
    elif t>0:
        D = a + AB*t
    else:
        D = a
    return np.sum((p-D)**2)

def Distance_pol(p,pol):
    Distance_min = 36
    for i in range(np.shape(pol)[0]-1):
        Distance_min =  min(Distance_min,Distance_square(p,pol[i],pol[i+1]))
    return Distance_min

def GenBC(pol):
    x = pol
    for i in range(np.shape(pol)[0]-1):
        R = np.sqrt(np.sum((pol[i+1] - pol[i])**2))
        theta = asin(pol[1][1]/R)
        r = R * np.random.rand(int(R*100))
        x_i = np.random.rand(int(R*100),2)
        x_i[:,0] = r*cos(theta)
        x_i[:,1] = r*sin(theta)
        x = np.vstack((x,x_i))
    return x