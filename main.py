import numpy as np
import matplotlib.pyplot as plt
from solve_riccati import solve_riccati
from LSV import LeastSquareValue
from TRCo import TimeReversalCostate
from LSCo import LeastSquareCostate
from TRV import TimeReversalValue

# Parameters
exp_num = 1 # experiment number
sample_list = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]  
# N = 1000 # sample number
m_0 = [1, 0] # initial mean
sigma_0 = np.eye(2) # initial covariance
kf = 200 # number of iterations
dt_list = [0.004, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4] # time step
tf_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # time horizon
noise_level_list = [0.1, 0.5, 1, 2, 3, 4, 5] # noise level
stability_list = [-1, -0.1, 0, 0.1, 1, 2] # stability


# Noise
def noise(dt, N):
    return np.random.multivariate_normal(np.zeros(2), dt*np.eye(2), N)


LSV_MSE = np.zeros((exp_num, len(sample_list)))
TRCo_MSE = np.zeros((exp_num, len(sample_list)))
LSCo_MSE = np.zeros((exp_num, len(sample_list)))
TRV_MSE = np.zeros((exp_num, len(sample_list)))
# Start experiment
for exp in range(exp_num):
    
    print('experiment: {}'.format(exp))
    # Experiment for different T, fixed (dt, noise level, and stability)
    dt = 0.4
    noise_level = 1
    stability = -0.1
    T = 4
    # N = 10000

    for sample_index in range(len(sample_list)):
        print('sample number: {}'.format(sample_list[sample_index]))
        N = sample_list[sample_index]

        # System matrices
        A = np.array([[0, 1], [-1, stability]])
        B = np.array([[0], [1]])
        N_sigma = np.eye(2) * noise_level
        Q = np.eye(2)
        R = np.eye(1)
        Q_f = np.eye(2) 
        D = N_sigma @ N_sigma.T
        steps = int(T/dt)

        # Generate data
        X_0 = np.random.multivariate_normal(m_0, sigma_0, N)# initial state
        W_f = np.zeros((steps+1, N, 2))# forward noise
        W_b = np.zeros((steps+1, N, 2))# backward noise
        for noise_step in range(steps+1):
            W_f[noise_step, :, :] = noise(dt, N)
            W_b[noise_step, :, :] = noise(dt, N)
    
        # experiment results
        try:
            G_LSV = LeastSquareValue(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)
        except:
            G_LSV = None
        try:
            G_TRCo = TimeReversalCostate(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)
        except:
            G_TRCo = None
        try:
            G_LSCo = LeastSquareCostate(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)
        except:
            G_LSCo = None
        try:
            G_TRV = TimeReversalValue(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)
        except:
            G_TRV = None

        # solve Riccati equation
        G_ref = solve_riccati(A, B, Q, R, Q_f, T, dt, N).transpose(2,0,1)

        # MSE
        if G_LSV is None:
            LSV_MSE[exp, sample_index] = np.nan
        else:
            LSV_MSE[exp, sample_index] = np.mean((G_LSV - G_ref)**2)
        if G_TRCo is None:
            TRCo_MSE[exp, sample_index] = np.nan
        else:
            TRCo_MSE[exp, sample_index] = np.mean((G_TRCo - G_ref)**2)
        if G_LSCo is None:
            LSCo_MSE[exp, sample_index] = np.nan
        else:
            LSCo_MSE[exp, sample_index] = np.mean((G_LSCo - G_ref)**2)
        if G_TRV is None:
            TRV_MSE[exp, sample_index] = np.nan
        else:
            TRV_MSE[exp, sample_index] = np.mean((G_TRV - G_ref)**2)

        # plt.figure()
        # plt.plot(G_TRV[:,0,0])
        # plt.plot(G_ref[:,0,0], linestyle='--')
        # plt.plot(G_TRV[:,0,1])
        # plt.plot(G_ref[:,0,1], linestyle='--')
        # plt.plot(G_TRV[:,1,0])
        # plt.plot(G_ref[:,1,0], linestyle='--')
        # plt.plot(G_TRV[:,1,1])
        # plt.plot(G_ref[:,1,1], linestyle='--')
        # plt.show()
    
print('LSV_MSE: {}'.format(LSV_MSE))
print('TRCo_MSE: {}'.format(TRCo_MSE))
print('LSCo_MSE: {}'.format(LSCo_MSE))
print('TRV_MSE: {}'.format(TRV_MSE))
mean_LSV_MSE = np.mean(LSV_MSE, axis=0)
mean_TRCo_MSE = np.mean(TRCo_MSE, axis=0)
mean_LSCo_MSE = np.mean(LSCo_MSE, axis=0)
mean_TRV_MSE = np.mean(TRV_MSE, axis=0)
std_LSV_MSE = np.std(LSV_MSE, axis=0)
std_TRCo_MSE = np.std(TRCo_MSE, axis=0)
std_LSCo_MSE = np.std(LSCo_MSE, axis=0)
std_TRV_MSE = np.std(TRV_MSE, axis=0)
plt.figure()
plt.fill_between(sample_list, mean_LSV_MSE - std_LSV_MSE, mean_LSV_MSE + std_LSV_MSE, color='C0', alpha=0.3)
plt.fill_between(sample_list, mean_TRCo_MSE - std_TRCo_MSE, mean_TRCo_MSE + std_TRCo_MSE, color='C1', alpha=0.3)
plt.fill_between(sample_list, mean_LSCo_MSE - std_LSCo_MSE, mean_LSCo_MSE + std_LSCo_MSE, color='C2', alpha=0.3)
plt.fill_between(sample_list, mean_TRV_MSE - std_TRV_MSE, mean_TRV_MSE + std_TRV_MSE, color='C3', alpha=0.3)
plt.plot(sample_list, mean_LSV_MSE, label='Least Square Value', color='C0')
plt.plot(sample_list, mean_TRCo_MSE, label='Time Reversal Costate', color='C1')
plt.plot(sample_list, mean_LSCo_MSE, label='Least Square Costate', color='C2')
plt.plot(sample_list, mean_TRV_MSE, label='Time Reversal Value', color='C3')
plt.xlabel('sample number')
plt.ylabel('MSE')
plt.yscale('log')
plt.xscale('log')
plt.title('T={}, dt={}'.format(T, dt))
plt.legend()
plt.show()
