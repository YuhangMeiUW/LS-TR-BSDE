import numpy as np
import matplotlib.pyplot as plt
from solve_riccati import solve_riccati
from LSV_highdim import LeastSquareValue_highdim
from TRCo_highdim import TimeReversalCostate_highdim
from LSCo_highdim import LeastSquareCostate_highdim
from TRV_highdim import TimeReversalValue_highdim

# Parameters
exp_num = 15 # experiment number
dt = 0.02 # time step
tf = 4 # time horizon
noise_level = 1 # noise level
kf = 100 # number of iterations
dim_list = [2, 4, 6, 8, 10, 20] # dimension
N =1000 # sample number


# Noise
def noise(dt, N, dim=2):
    """
    Generate noise for the system.
    Args:
        dt (float): Time step size.
        N (int): Number of samples.
        dim (int): Dimension of the noise.
    Returns:
        np.ndarray: Noise samples of shape (N, dim).
    """
    return np.random.multivariate_normal(np.zeros(dim), dt*np.eye(dim), N)

LSV_MSE = np.zeros((exp_num, len(dim_list)))
TRCo_MSE = np.zeros((exp_num, len(dim_list)))
LSCo_MSE = np.zeros((exp_num, len(dim_list)))
TRV_MSE = np.zeros((exp_num, len(dim_list)))
# Start experiment
for exp in range(exp_num):

    for dim_index in range(len(dim_list)):
        dim = dim_list[dim_index]
        print('experiment: {}, dimension: {}'.format(exp, dim))
        # System matrices
        T = np.zeros((int(dim/2), int(dim/2)))
        np.fill_diagonal(T, 2) # Diagonal
        np.fill_diagonal(T[:-1, 1:], -1)  # First super-diagonal
        np.fill_diagonal(T[1:, :-1], -1)  # First sub-diagonal
        A = np.block([[np.zeros((int(dim/2), int(dim/2))), np.eye(int(dim/2))], [-T, -np.eye(int(dim/2))]])
        B = np.block([[np.zeros((int(dim/2), int(dim/2)))], [np.eye(int(dim/2))]])
        Noise_sigma = np.eye(dim) * noise_level
        Q = np.eye(dim)
        R = np.eye(int(dim/2))
        Q_f = np.eye(dim) 
        D = Noise_sigma @ Noise_sigma.T
        steps = int(tf/dt)

        # Generate data
        m_0 = np.zeros(dim)
        m_0[0] = 1
        sigma_0 = np.eye(dim)
        X_0 = np.random.multivariate_normal(m_0, sigma_0, N)# initial state
        W_f = np.zeros((steps+1, N, dim))# forward noise
        W_b = np.zeros((steps+1, N, dim))# backward noise
        for noise_step in range(steps+1):
            W_f[noise_step, :, :] = noise(dt, N, dim)
            W_b[noise_step, :, :] = noise(dt, N, dim)
    
        # experiment results
        try:
            G_LSV = LeastSquareValue_highdim(A, B, Noise_sigma, Q, R, Q_f, D, tf, dt, X_0, W_f, W_b, kf, N, dim)
        except:
            G_LSV = None
        try:
            G_TRCo = TimeReversalCostate_highdim(A, B, Noise_sigma, Q, R, Q_f, D, tf, dt, X_0, W_f, W_b, kf, N, dim)
        except:
            G_TRCo = None
        try:
            G_LSCo = LeastSquareCostate_highdim(A, B, Noise_sigma, Q, R, Q_f, D, tf, dt, X_0, W_f, W_b, kf, N, dim)
        except:
            G_LSCo = None
        try:
            G_TRV = TimeReversalValue_highdim(A, B, Noise_sigma, Q, R, Q_f, D, tf, dt, X_0, W_f, W_b, kf, N, dim)
        except:
            G_TRV = None

        # solve Riccati equation
        G_ref = solve_riccati(A, B, Q, R, Q_f, tf, dt, dim).transpose(2,0,1)

        # MSE
        if G_LSV is None:
            LSV_MSE[exp, dim_index] = np.nan
        else:
            LSV_MSE[exp, dim_index] = np.mean((G_LSV - G_ref)**2)
        if G_TRCo is None:
            TRCo_MSE[exp, dim_index] = np.nan
        else:
            TRCo_MSE[exp, dim_index] = np.mean((G_TRCo - G_ref)**2)
        if G_LSCo is None:
            LSCo_MSE[exp, dim_index] = np.nan
        else:
            LSCo_MSE[exp, dim_index] = np.mean((G_LSCo - G_ref)**2)
        if G_TRV is None:
            TRV_MSE[exp, dim_index] = np.nan
        else:
            TRV_MSE[exp, dim_index] = np.mean((G_TRV - G_ref)**2)




### Data Saving ###
np.save('data/LSV_MSE_exp15_N1000.npy', LSV_MSE)
np.save('data/TRCo_MSE_exp15_N1000.npy', TRCo_MSE)
np.save('data/LSCo_MSE_exp15_N1000.npy', LSCo_MSE)
np.save('data/TRV_MSE_exp15_N1000.npy', TRV_MSE)


