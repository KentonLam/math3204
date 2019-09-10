# import numpy as np
# from numpy import linalg as npla
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import time
from typing import Tuple

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

def householder(A) -> Tuple[sp.matrix, sp.matrix]:
    '''
    Given a matrix A, computes its QR factorisation using Householder
    reflections.

    Returns (Q, R) such that A = QR, Q is orthogonal and R is triangular.
    '''
    Q_full = sps.identity(A.shape[0])
    start = time.time()
    for i in range(min(A.shape)): # iterate over smaller dimension of A
        _A = A[i:,i:] # submatrix of A left to triangularise
        # print(_A)
        if i % 100 == 0:
            print(i, time.time() - start)
            start = time.time()
        # print(i)
        v = (_A[:,0]) # leftmost vector of A matrix
        e_i = sp.zeros((v.shape[0], 1))
        e_i[0] = 1 # vector with 1 in the first position.
        e_i = sps.csr_matrix(e_i)
        
        v_max = v.min() if -v.min() > v.max() else v.max()
        u = v + sign(v_max) * spsla.norm(v) * e_i # compute u vector for P
        u = u / spsla.norm(u) # normalise
        _P = sps.identity(u.shape[0], format='csr') - 2 * (u @ u.T) # compute sub _P

        # embed this submatrix _P into the full size P
        P = sps.block_diag((sps.identity(i, format='dia'), _P))

        Q_full = P @ Q_full # accumulated orthogonal Q
        A = P @ A # applying to A
    
    # A = QR
    # Q_full is currently the inverse because it is applied to A, to R.
    # thus, Q = Q_full^T.
    # note that A has been reduced to R by applying the P's.
    return (Q_full.transpose(), A)
    

if __name__ == "__main__":
    # example adapted from: http://www.math.niu.edu/~ammar/m434/hhexamp.pdf
    A = sp.array([
        [3, 1, 0],
        [1, 4, 2],
        [0, 2, 1],
        [10, 2, 3]
    ])
    A = sp.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

    L = lcd(0.1, 0.1, 100)
    L = sps.csr_matrix(L)
    
    # A = sps.csr_matrix(A)
    A = L
    Q, R = householder(A)
    print('QR - A:', spsla.norm(Q @ R - A))
    print('QQ^T - I:', spsla.norm(Q @ Q.transpose() - sps.identity(Q.shape[0])))
    print('Q^TQ - I:', spsla.norm(Q.transpose() @ Q - sps.identity(Q.shape[0])))
    