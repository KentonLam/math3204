import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from householder import lcd

def pcg(A, b, x_0, P=None):
    if P is None:
        P = sp.identity(A.shape[0])
    # P_inv = spla.inv(P)
    print('starting')
    r_0 = b - A @ x_0

    r_prev = r = r_0
    r_prod = (r.T @ P.solve(r))
    x = x_0
    p = r_0

    k = 0
    while True:
        k += 1
        Ap = A @ p
        alpha = (r_prod / (p.T @ Ap)).item()
        x = x + alpha*p # x_k initially stores x_{k-1}

        r_prev = r
        r_prev_prod = r_prod
        r = r - alpha * Ap
        r_prod = (r.T @ P.solve(r))
        beta = (r_prod / r_prev_prod).item()
        p = P.solve(r) + beta * p 
        print(k, spla.norm(r))

        if spla.norm(r) <= 10**-12:
            print('terminating from residual')
            break 
    
    print(x)
    print(A @ x)
    return x


if __name__ == "__main__":
    P = None
    N = 50
    A = lcd(0, 0, N)
    b = sp.ones((N**2, 1))
    x_0 = sp.zeros((N**2, 1))
    # pcg(A, b, x_0, sp.diag(sp.diag(A)))
    print() 
    # P = spsla.spilu(A).solve(b)
    P = spsla.spilu(A)
    # P = spla.inv(P)
    # print(P)
    sol1 = pcg(A.todense(), b, x_0, P)


    sol2, info = spsla.cg(lcd(0, 0, N), b, x_0)
    sol2 = sol2.reshape((sol2.shape[0], 1))

    print(sol1 - sol2)