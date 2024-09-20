import numpy as np
from scipy.integrate import solve_ivp


def riccati_eq(t, P_flat, A, B, Q, R, dim=2):
    P = P_flat.reshape((dim,dim))
    R_inv = np.linalg.inv(R)
    dPdt = -(A.T@P + P@A - P@B@R_inv@B.T@P + Q)
    return dPdt.flatten()



def solve_riccati(A, B, Q, R, Q_f, T, dt, dim=2):
    steps = int(T/dt)
    P_T = Q_f
    t_span = [T, 0]
    P_T_flat = P_T.flatten()
    sol = solve_ivp(riccati_eq, t_span, P_T_flat, args=(A, B, Q, R, dim), method='RK45', dense_output=True)
    t = np.linspace(T, 0, steps)
    G = sol.sol(t)
    G_reversed = G[:,::-1]
    G_reversed = G_reversed.reshape((dim,dim,steps))
    return G_reversed