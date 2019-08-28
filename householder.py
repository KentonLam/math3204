import numpy
from numpy import linalg as LA

def sign(x):
    if x == 0: return 0
    return 1 if x > 0 else -1

def householder(A):
    '''
    Given a matrix A, computes its QR factorisation using Householder
    reflections.

    Returns (Q, R) such that A = QR, Q is orthogonal and R is triangular.
    '''
    Q_full = numpy.identity(A.shape[0])
    for i in range(min(A.shape)): # iterate over smaller dimension of A
        _A = A[i:,i:] # submatrix of A left to triangularise
        print(_A)
        v = _A[:,0] # leftmost vector of A matrix
        e_i = numpy.zeros(len(v))
        e_i[0] = 1 # vector with 1 in the first position.
        
        v_max = v.min() if -v.min() > v.max() else v.max()
        u = v + sign(v_max) * LA.norm(v) * e_i # compute u vector for P
        u = u / LA.norm(u) # normalise
        _P = numpy.identity(len(u)) - 2 * numpy.outer(u, u) # compute sub _P
        P = numpy.identity(A.shape[0])
        P[i:, i:] = _P # embed this submatrix _P into the full size P
        print(_P)
        print(_P @ _A)
        Q_full = P @ Q_full # accumulated orthogonal Q
        A = P @ A # applying to A
        print()
    print('result:')
    print(Q_full)
    print(A.round(4))
    
    # A = QR
    # Q_full is currently the inverse because it is applied to A, to R.
    # this, Q = Q_full^T.
    # note that A has been reduced to R by applying the P's.
    return (Q_full.transpose(), A)
    

if __name__ == "__main__":
    # example adapted from: http://www.math.niu.edu/~ammar/m434/hhexamp.pdf
    A = numpy.array([
        [3, 1, 0],
        [1, 4, 2],
        [0, 2, 1],
        [10, 2, 3]
    ])
    Q, R = householder(A)
    
    print()
    print('original A:')
    print(A)
    print('recomputing A from QR:')
    print((Q @ R))
    print('QQ^T')
    print(Q @ Q.transpose())