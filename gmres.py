import numpy as np
import numpy.linalg as npla

import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from householder import lcd, householder

def gmres(A: sp.matrix, b: sp.matrix, x_0: sp.matrix=None):
    m, n = A.shape
    if x_0 is None: 
        x_0 = np.zeros((n, 1))
    x_0 = sps.csc_matrix(x_0)
    b = sps.csc_matrix(b)
    r = b - A @ x_0
    H = sps.lil_matrix((1, 0))
    Q = r / spsla.norm(r)
    Q.reshape((n, 1))
    k = 0
    zero = sps.coo_matrix([0], (1, 1))
    while True:
        k += 1
        print(H)
        h = sps.lil_matrix((k, 1))
        j = k
        q_j = Q[:,j-1]
        z = A @ q_j
        for i in range(j):
            q_i = Q[:, i]
            h[i, 0] = (q_i.T @ A @ q_j)[0,0]
            # print(h)
            # print(q_i)
            z -= h[i, 0] * q_i 
        # h = sps.csc_matrix([h], shape=(len(h), 1))
        h_j_plus_1 = spsla.norm(z)
        if h_j_plus_1 == 0:
            print('BREAK')
            return 0 
        q_j_plus_1 = z / h_j_plus_1
        q_j_plus_1 = q_j_plus_1.reshape((n, 1))
        # print(q_j_plus_1.shape)
        Q = sps.hstack([Q, q_j_plus_1], format='csc')
        if k > 1:
            H = sps.vstack([H, zero])
        H = sps.hstack([H, h])
        print(H)



if __name__ == "__main__":
    A = lcd(0.1, 0.1, 10)
    b = A @ np.ones((A.shape[1], 1))
    gmres(A, b, None)