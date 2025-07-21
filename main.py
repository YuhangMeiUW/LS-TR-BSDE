import numpy as np
import matplotlib.pyplot as plt
from solve_riccati import solve_riccati
from LSV import LeastSquareValue
from TRCo import TimeReversalCostate
from LSCo import LeastSquareCostate
from TRV import TimeReversalValue
import time

# Parameters
exp_num = 1 # experiment number
# sample_list = [10, 50, 100, 500, 1000, 2000, 4000]  ### For testing sensitivity to sample number ###
sample_list = [2000]
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
LSV_time = []
TRCo_time = []
LSCo_time = []
TRV_time = []

dt = 0.02 # time step size
noise_level = 1
stability = -0.1 # (2,2) entry of A matrix
T = 4 # time horizon
# N = 1000

# System matrices
A = np.array([[0, 1], [-1, stability]])
B = np.array([[0], [1]])
Noise_sigma = np.eye(2) * noise_level
Q = np.eye(2) # state cost
R = np.eye(1) # control cost
Q_f = np.eye(2) # terminal state cost
D = Noise_sigma @ Noise_sigma.T

# Start experiment
for exp in range(exp_num):
    
    print('experiment: {}'.format(exp))
    

    for sample_index in range(len(sample_list)):

        ### iterate over sample number ###
        print('N: {}'.format(sample_list[sample_index]))
        N = sample_list[sample_index]
        # dt = dt_list[dt_index]

        
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
            LSV_start = time.time()
            G_LSV, LSV_J = LeastSquareValue(A, B, Noise_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)
            LSV_time.append(time.time() - LSV_start)
        except:
            print('LSV failed')
            G_LSV = None
        try:
            TRCo_start = time.time()
            G_TRCo, TRCo_J = TimeReversalCostate(A, B, Noise_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)
            TRCo_time.append(time.time() - TRCo_start)
        except:
            print('TRCo failed')
            G_TRCo = None
        try:
            LSCo_start = time.time()
            G_LSCo, LSCo_J = LeastSquareCostate(A, B, Noise_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)
            LSCo_time.append(time.time() - LSCo_start)
        except:
            print('LSCo failed')
            G_LSCo = None
        try:
            TRV_start = time.time()
            G_TRV, TRV_J = TimeReversalValue(A, B, Noise_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)
            TRV_time.append(time.time() - TRV_start)
        except:
            print('TRV failed')
            G_TRV = None

        ### Solve Riccati equation ###
        G_ref = solve_riccati(A, B, Q, R, Q_f, T, dt).transpose(2,0,1)



        ### Calculate Mean Square Error ###
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



### Save results ###
            
np.save('data/LSV_MSE_T4_dt002_N2000_exp15.npy', LSV_MSE)
np.save('data/TRCo_MSE_T4_dt002_N2000_exp15.npy', TRCo_MSE)
np.save('data/LSCo_MSE_T4_dt002_N2000_exp15.npy', LSCo_MSE)
np.save('data/TRV_MSE_T4_dt002_N2000_exp15.npy', TRV_MSE)
np.save('data/G_ref_T4_dt002_N2000_exp15.npy', G_ref)
np.save('data/G_LSV_exp15.npy', G_LSV)
np.save('data/G_TRCo_exp15.npy', G_TRCo)
np.save('data/G_LSCo_exp15.npy', G_LSCo)
np.save('data/G_TRV_exp15.npy', G_TRV)
np.save('data/LSV_J.npy', LSV_J)
np.save('data/TRCo_J.npy', TRCo_J)
np.save('data/LSCo_J.npy', LSCo_J)
np.save('data/TRV_J.npy', TRV_J)
np.savetxt('data/LSV_time_exp15.txt', LSV_time)
np.savetxt('data/TRCo_time_exp15.txt', TRCo_time)
np.savetxt('data/LSCo_time_exp15.txt', LSCo_time)
np.savetxt('data/TRV_time_exp15.txt', TRV_time)


