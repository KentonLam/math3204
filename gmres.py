import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import time

from householder import lcd, householder

sp.set_printoptions(edgeitems=30, linewidth=100000)

def gmres(A: sp.matrix, b: sp.matrix, x_0: sp.matrix=None):
    start = time.time() 

    m, n = A.shape
    if x_0 is None: 
        x_0 = sp.zeros((n, 1))
    r = b - A @ x_0
    r_0_norm = spla.norm(r)
    b_norm = spla.norm(b)
    H = sp.ndarray((1, 0))
    Q = r / spla.norm(r)
    Q.reshape((n, 1))
    k = 0

    while True:
        k += 1
        # print(H.shape)
        h = sp.ndarray((k, 1))
        j = k
        q_j = Q[:,j-1]
        z = A @ q_j
        for i in range(j):
            q_i = Q[:, i]
            h[i, 0] = (q_i.T @ z).item()
            # print(h)
            # print(q_i)
            z -= h[i, 0] * q_i 
        # h = sps.csc_matrix([h], shape=(len(h), 1))
        h_last = spla.norm(z)
        q_new = z / h_last
        # q_new.reshape((n, 1))
        # print(q_new.shape)
        
        H = sp.block([
            [H, h],
            [sp.zeros((1, H.shape[1])), h_last]
        ])

        e_1 = sp.zeros((k+1, 1)) # k+1 because it is multiplied by U_{k+1,k}^T
        e_1[0,0] = 1

        U, R = householder(H, True)
        y_j = spla.solve(R, r_0_norm * U.T @ e_1)
        # print(y_j)
        x_j = x_0 + Q @ y_j
        # print(x_j)
        # print(H)
        Q = sp.hstack([Q, q_new])
        r_j_norm = r_0_norm * sp.sqrt(1 - spla.norm(U.T @ e_1)**2)
        print(f'relative residual {k}:', r_j_norm / b_norm)

        if k > 1000:
            print('terminating from iteration limit')
            break 
        if r_j_norm / b_norm <= 10**-6:
            print('terminating from residual norm')
            break
    
    print('iterations:', k)
    print('time:', time.time() - start)
    print('x')
    print(x_j.flatten().round(3).tolist())


if __name__ == "__main__":
    A = lcd(0.1, 0.1, 10).todense()
    b = A @ sp.ones((A.shape[1], 1))
    gmres(A, b, None)