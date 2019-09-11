import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from householder import lcd

def cg(A, b, x_0, P=None):
    if P is None:
        P = sp.identity(A.shape[0])
    r_0 = b - A @ x_0 

    r_prev = r = r_0
    x = x_0
    p = r_0

    k = 0
    while True:
        k += 1
        Ap = A @ p
        alpha = ((r.T @ r) / (p.T @ Ap)).item()
        x = x + alpha*p # x_k initially stores x_{k-1}

        r_prev = r
        r = r - alpha * Ap
        beta = ((r.T @ r) / (r_prev.T @ r_prev)).item()
        p = r + beta * p 
        print(k, spla.norm(r))

        if spla.norm(r) <= 10**-12:
            print('terminating from residual')
            break 
    
    print(x)
    print(A @ x)


if __name__ == "__main__":
    N = 50
    A = lcd(0, 0, N).todense()
    b = sp.ones((N**2, 1))
    x_0 = sp.zeros((N**2, 1))
    cg(A, b, x_0)