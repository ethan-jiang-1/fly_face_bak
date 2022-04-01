import numpy as np
import pandas as pd
import os

def name_filter(fname):
    last_name = fname.split("_")[-1][:-4]
    return last_name

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
    files_path = "./npy"
    file_list = os.listdir(files_path)
    
    # filter name 
    new_file_list = []
    for file in file_list:
        if 'beard'  in file:
            new_file_list.append(file)
    
    cos_matrix = []
    c_list = []
    
    for file in new_file_list:         
            
        c_list = []
        
        for file2 in new_file_list:            
            
            n1 = np.load("./npy/"+file,allow_pickle=True)
            n2 = np.load("./npy/"+file2,allow_pickle=True)
            cos = get_cos_similar(n1,n2)            
            # distance = get_vectorDistance(n1,n2)            
            c_list.append(cos)
            
        cos_matrix.append(c_list)
        
    df = pd.DataFrame(cos_matrix, columns = new_file_list, index = new_file_list)
    df.to_excel('./output/beard_matrix.xlsx',index=True)
     