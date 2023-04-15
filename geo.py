import numpy as np

def skew(v : np.ndarray):
    #v - 3-vector
    #v_skew - 3x3 skew symmetric matrix
    v_skew = np.zeros((3,3), dtype = type(v[0]))

    v_skew[0,1] = -v[2]
    v_skew[0,2] = v[1]
    v_skew[1,0] = v[2]
    v_skew[1,2] = -v[0]
    v_skew[2,0] = -v[1]
    v_skew[2,1] = v[0]
    return v_skew

def SO3_to_storage(R):
    #R - 3x3 rotation matrix
    #q - 4-vector, x, y, z, w
    q = np.zeros(4,dtype = type(R[0,0]))

    q[3] = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    q[0] = (R[2,1] - R[1,2]) / (4 * q[3])
    q[1] = (R[0,2] - R[2,0]) / (4 * q[3])
    q[2] = (R[1,0] - R[0,1]) / (4 * q[3])
    return q


def SO3_Log(R):
    #R - 3x3 rotation matrix
    th = np.arccos(np.trace(R)-1)/2
    u = th * skew(R-R.T) / (2 * np.sin(th))
    return u, th

def SO3_Exp(theta):
    th = np.linalg.norm(theta)
    u = theta / th
    sk = skew(u)
    sk2 = sk @ sk
    return np.eye(3) + \
            np.multiply(np.sin(th),sk) + \
            np.multiply((1 - np.cos(th)),sk2)

def SO3_V(theta):
    th = np.linalg.norm(theta)
    sk = skew(theta)
    sk2 = sk @ sk
    return np.eye(3) + \
            np.multiply((1 - np.cos(th)) / th**2, sk) +\
            np.multiply((th - np.sin(th)) / th**3, sk2)
    

def SE3_Exp(chi : np.ndarray):
    #chi - tangent vector
    #X - 4x4 matrix
    X = np.zeros((4,4),dtype = type(chi[0]))

    theta = chi[:3]
    rho = chi[3:]

    X[:3,:3] = SO3_Exp(theta)
    X[:3,3] = SO3_V(theta) @ rho
    X[3,3] = 1
    return X

def SE3_to_storage(X):
    #X - 4x4 matrix
    #q - 4-vector
    #t - 3-vector
    t = X[:3,3]
    q = SO3_to_storage(X[:3,:3])
    return np.hstack((q, t))