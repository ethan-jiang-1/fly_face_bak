def RodriguesRotate(lst_v, lst_u, theta:float):
    try:
        # RodriguesRotate(v, u:np.ndarray, theta:float)->np.ndarray:
        '''向量v绕向量u旋转角度θ,得到新的向量P_new
        罗德里格斯旋转公式:v' = vcosθ + (u×v)sinθ + (u·v)u(1-cosθ) 
        
        args:
            v:向量,维度为(3,)
            u:作为旋转轴的向量,维度为(3,)
            theta:旋转角度θ,此处默认为角度值
        returns:
            v_new:旋转后得到的向量,维度为(3,)
        '''
        import numpy as np 

        v = np.array(lst_v)
        u = np.array(lst_u)

        u = u/np.linalg.norm(u) # 计算单位向量
        sin_theta = np.sin(theta*np.pi/180)
        cos_theta = np.cos(theta*np.pi/180)
        v_new = v*cos_theta + np.cross(u,v)*sin_theta + np.dot(u,v)*u*(1-cos_theta)
        return list(v_new)
    except Exception as ex:
        print("Exception occured (RodriguesRotate): {}".format(ex))
    return None
