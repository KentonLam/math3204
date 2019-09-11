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
    # Q_full = sps.identity(A.shape[0], format='csr')
    start = time.time()
    m, n = A.shape

    A_full = sp.ndarray(A.shape)
    A_sub = A.copy()

    Q_full = sp.identity(m)

    for i in range(min(A.shape)): # iterate over smaller dimension of A
        # _A = A[i:,i:] # submatrix of A left to triangularise
        # print(_A)
        if i % 100 == 0:
            # print(i, time.time() - start)
            start = time.time()
        # print(i)
        v = A_sub[:, 0] # leftmost vector of A matrix
        e_i = sp.zeros(v.shape[0])
        e_i[0] = 1 # vector with 1 in the first position.
        
        u = v + sign(v.item(0)) * spla.norm(v) * e_i # compute u vector for P
        u = u / spla.norm(u) # normalise
        # print(u.shape)
        _P = sp.identity(v.shape[0]) - 2 * sp.outer(u, u) # compute sub _P

        # _Q = sps.csr_matrix(Q_full[i:, i:])
        
        # embed this submatrix _P into the full size P
        P = spla.block_diag(sp.identity(i), _P)

        Q_full = P @ Q_full
        # print(Q_full.toarray())
        # print()

        # Q_full[i:,i:] = _P @ Q_full[i:,i:] # accumulated orthogonal Q
        # A[i:,i:] = _P @ A[i:,i:] # applying to A
        A_sub = _P @ A_sub
        # A = P @ A
        A_full[i,i:] = A_sub[0,:]
        A_full[i:,i] = A_sub[:,0]

        A_sub = A_sub[1:,1:] # iterate into submatrix.
    
    Q_full = Q_full.T

    if reduced:
        Q_full = Q_full[:, :n]
        A_full = A_full[:n, :]
    A_full = sp.triu(A_full)

    # A = QR
    # Q_full is currently the inverse because it is applied to A.
    # thus, Q = Q_full^T.
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
    # A = sp.array([
    #     [1, 2, 3, 4],
    #     [5, 6, 7, 8]
    # ])
    # A = sps.csr_matrix(A)

    L = lcd(0.1, 0.1, 30).todense()
    
    # A = L
    Q, R = householder(A)
    print(Q @ R)
    print(Q @ Q.T)
    print(Q.T @ Q)
    print('QR - A:', spla.norm(Q @ R - A))
    print('QQ^T - I:', spla.norm(Q @ Q.transpose() - sp.identity(Q.shape[0])))
    print('Q^TQ - I:', spla.norm(Q.transpose() @ Q - sp.identity(Q.shape[0])))
    