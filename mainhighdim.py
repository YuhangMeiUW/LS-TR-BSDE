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
kf = 200 # number of iterations
dim_list = [2, 4, 6, 8] # dimension
N =1000 # sample number


# Noise
def noise(dt, N, dim=2):
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
        N_sigma = np.eye(dim) * noise_level
        Q = np.eye(dim)
        R = np.eye(int(dim/2))
        Q_f = np.eye(dim) 
        D = N_sigma @ N_sigma.T
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
            G_LSV = LeastSquareValue_highdim(A, B, N_sigma, Q, R, Q_f, D, tf, dt, X_0, W_f, W_b, kf, N, dim)
        except:
            G_LSV = None
        try:
            G_TRCo = TimeReversalCostate_highdim(A, B, N_sigma, Q, R, Q_f, D, tf, dt, X_0, W_f, W_b, kf, N, dim)
        except:
            G_TRCo = None
        try:
            G_LSCo = LeastSquareCostate_highdim(A, B, N_sigma, Q, R, Q_f, D, tf, dt, X_0, W_f, W_b, kf, N, dim)
        except:
            G_LSCo = None
        try:
            G_TRV = TimeReversalValue_highdim(A, B, N_sigma, Q, R, Q_f, D, tf, dt, X_0, W_f, W_b, kf, N, dim)
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

        # plt.figure()
        # plt.plot(G_TRV[:,0,0])
        # plt.plot(G_ref[:,0,0], linestyle='--')
        # plt.plot(G_TRV[:,0,1])
        # plt.plot(G_ref[:,0,1], linestyle='--')
        # plt.plot(G_TRV[:,1,0])
        # plt.plot(G_ref[:,1,0], linestyle='--')
        # plt.plot(G_TRV[:,1,1])
        # plt.plot(G_ref[:,1,1], linestyle='--')
        # for i in range(dim):
        #     for j in range(dim):
        #         plt.plot(G_ref[:,i,j], linestyle='--', color='C0')
        #         plt.plot(G_TRV[:,i,j], color='C0')
        # plt.show()
    
print('LSV_MSE:', LSV_MSE)
print('TRCo_MSE:', TRCo_MSE)
print('LSCo_MSE:', LSCo_MSE)
print('TRV_MSE:', TRV_MSE)
# plot
mean_LSV_MSE = np.mean(LSV_MSE, axis=0)
mean_TRCo_MSE = np.mean(TRCo_MSE, axis=0)
mean_LSCo_MSE = np.mean(LSCo_MSE, axis=0)
mean_TRV_MSE = np.mean(TRV_MSE, axis=0)
std_LSV_MSE = np.std(LSV_MSE, axis=0)
std_TRCo_MSE = np.std(TRCo_MSE, axis=0)
std_LSCo_MSE = np.std(LSCo_MSE, axis=0)
std_TRV_MSE = np.std(TRV_MSE, axis=0)
plt.figure()
plt.fill_between(dim_list, mean_LSV_MSE - std_LSV_MSE, mean_LSV_MSE + std_LSV_MSE, color='C0', alpha=0.3)
plt.fill_between(dim_list, mean_TRCo_MSE - std_TRCo_MSE, mean_TRCo_MSE + std_TRCo_MSE, color='C1', alpha=0.3)
plt.fill_between(dim_list, mean_LSCo_MSE - std_LSCo_MSE, mean_LSCo_MSE + std_LSCo_MSE, color='C2', alpha=0.3)
plt.fill_between(dim_list, mean_TRV_MSE - std_TRV_MSE, mean_TRV_MSE + std_TRV_MSE, color='C3', alpha=0.3)
plt.plot(dim_list, mean_LSV_MSE, label='Least Square Value', color='C0')
plt.plot(dim_list, mean_TRCo_MSE, label='Time Reversal Costate', color='C1')
plt.plot(dim_list, mean_LSCo_MSE, label='Least Square Costate', color='C2')
plt.plot(dim_list, mean_TRV_MSE, label='Time Reversal Value', color='C3')
plt.xlabel('dimension')
plt.ylabel('MSE')
plt.yscale('log')
plt.title('T={}, dt={}, sample={}'.format(tf, dt, N))
plt.legend()
plt.show()


