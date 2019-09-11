# import numpy as np
# from numpy import linalg as npla
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import time
from typing import Tuple

# sp.set_printoptions(edgeitems=30, linewidth=100000)

def sign(x):
    if x == 0: return 0
    return 1 if x > 0 else -1

def lcd(beta, gamma, N):
    eee = sp.ones(N)
    ee = sp.ones(N-1)
    a = 4
    b = -1 - gamma
    c = -1 - beta 
    d = -1 + beta 
    e = -1 + gamma
    t1 = sps.diags([ c*ee, a*eee, d*ee ], [-1, 0, 1])
    t2 = sps.diags([ b*ee, sp.zeros(N), e*ee ], [-1, 0, 1])
    A = sps.kron(sps.eye(N), t1) + sps.kron(t2, sps.eye(N))
    return A

def householder(A, reduced=False) -> Tuple[sp.matrix, sp.matrix]:
    '''
    Given a matrix A, computes its QR factorisation using Householder
    reflections.

    Returns (Q, R) such that A = QR, Q is orthogonal and R is triangular.
    '''
    m, n = A.shape

    A_full = sp.ndarray(A.shape)
    A_sub = A.copy()

    Q_full = sp.identity(m)
    # iterate over smaller dimension of A
    for i in range(min(A.shape)):
        # leftmost vector of A submatrix
        v = A_sub[:, 0] 
        # vector with 1 in the first position.
        e_i = sp.zeros(v.shape[0])
        e_i[0] = 1 
        
        # compute householder vector for P
        u = v + sign(v.item(0)) * spla.norm(v) * e_i 
        # normalise
        u = u / spla.norm(u) 
        # compute submatrix _P
        _P = sp.identity(v.shape[0]) - 2 * sp.outer(u, u) 

        # embed this submatrix _P into the full size P
        P = spla.block_diag(sp.identity(i), _P)

        # compute next iteration of Q
        Q_full = P @ Q_full

        # compute next iteration of R
        A_sub = _P @ A_sub

        # copy first rows/cols to A_full
        A_full[i,i:] = A_sub[0,:]
        A_full[i:,i] = A_sub[:,0]

        # iterate into submatrix
        A_sub = A_sub[1:,1:]
    

    # Q_full is currently the inverse because it is applied to A.
    # thus, Q = Q_full^T.
    Q_full = Q_full.T
    if reduced:
        Q_full = Q_full[:, :n]
        A_full = A_full[:n, :]

    # A = QR
    # note that A has been reduced to R by applying the P's.
    return (Q_full, A_full)
    

if __name__ == "__main__":
    # example adapted from: http://www.math.niu.edu/~ammar/m434/hhexamp.pdf
    A = sp.array([
        [3, 1, 0],
        [1, 4, 2],
        [0, 2, 1],
        [10, 2, 3]
    ])

    # L = lcd(0.1, 0.1, 30).todense()
    
    # A = L
    Q, R = householder(A)
    print(R.round(4))
    print(Q @ R)
    print(Q @ Q.T)
    print(Q.T @ Q)
    print('QR - A:', spla.norm(Q @ R - A))
    print('QQ^T - I:', spla.norm(Q @ Q.transpose() - sp.identity(Q.shape[0])))
    print('Q^TQ - I:', spla.norm(Q.transpose() @ Q - sp.identity(Q.shape[0])))
    