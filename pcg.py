import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from householder import lcd

def pcg(A, b, x_0, P=None):
    if P is None:
        P = sp.identity(A.shape[0])
    P_inv = spla.inv(P)
    r_0 = b - A @ x_0
    A = P_inv @ A

    r_prev = r = r_0
    x = x_0
    p = r_0

    k = 0
    while True:
        k += 1
        Ap = A @ p
        alpha = ((r.T @ P @ r) / (p.T @ P @ Ap)).item()
        x = x + alpha*p # x_k initially stores x_{k-1}

        r_prev = r
        r = r - alpha * Ap
        # r = P_inv @ r
        beta = ((r.T @ P @ r) / (r_prev.T @ P @ r_prev)).item()
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
    pcg(A, b, x_0, sp.diag(sp.diag(A)))