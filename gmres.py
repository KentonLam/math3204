import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from householder import lcd, householder

sp.set_printoptions(edgeitems=30, linewidth=100000)

def gmres(A: sp.matrix, b: sp.matrix, x_0: sp.matrix=None):
    m, n = A.shape
    if x_0 is None: 
        x_0 = sp.zeros((n, 1))
    r = b - A @ x_0
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
        h_j_plus_1 = spla.norm(z)
        if sp.isclose(h_j_plus_1, 0):
            print('BREAK')
            break
        q_j_plus_1 = z / h_j_plus_1
        q_j_plus_1.reshape((n, 1))
        # print(q_j_plus_1.shape)
        Q_prev = Q
        Q = sp.hstack([Q, q_j_plus_1])
        H = sp.block([
            [H, h],
            [sp.zeros((1, H.shape[1])), h_j_plus_1]
        ])
        # print(H)

    # print((Q.T @ A @ Q).round(4))
    # print((A @ Q_prev - Q @ H))

if __name__ == "__main__":
    A = lcd(0.1, 0.1, 4).todense()
    b = A @ sp.ones((A.shape[1], 1))
    gmres(A, b, None)