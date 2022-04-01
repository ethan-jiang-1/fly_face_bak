import numpy as np
import os

def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def get_vectorDistance(vector1,vector2):
    diff = np.subtract(vector1,vector2)
    distance = np.sum(np.square(diff))
    return distance

def maxminnorm(array):
    nmax = np.max(array)
    nmin = np.min(array)
    t = []
    for i in range(len(array)):
        t.append(array[i]-nmin/(nmax-nmin))    
    return np.array(t)

def get_uni_vector(A):
    dists = np.linalg.norm(A) 
    out = np.where(np.isclose(dists,0), 0, A/dists) 
    return out

if __name__ == "__main__":
    dir_this = os.path.dirname(__file__)
    name1 = r'{}/npy/cancan_facepaint.npy'.format(dir_this)
    name2 = r'.{}/npy/me_facepaint.npy'.format(dir_this)
    n1 = np.load(name1,allow_pickle=True)
    n2 = np.load(name2,allow_pickle=True)
    n1 = get_uni_vector(n1)
    n2 = get_uni_vector(n2)
    cos = get_cos_similar(n1,n2)
    print(name1.split('\\')[-1]+" "+name2.split('\\')[-1]+" cos:"+str(cos))
    distance = get_vectorDistance(n1,n2)
    print(name1.split('\\')[-1]+" "+name2.split('\\')[-1]+" distance:"+str(distance))
