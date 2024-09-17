import numpy as np
import matplotlib.pyplot as plt
from solve_riccati import solve_riccati
from LSV import LeastSquareValue
from TRCo import TimeReversalCostate
from LSCo import LeastSquareCostate
from TRV import TimeReversalValue

# Parameters
exp_num = 1 # experiment number
N = 1000 # sample number
m_0 = [1, 0] # initial mean
sigma_0 = np.eye(2) # initial covariance
kf = 200 # number of iterations
dt_list = [0.004, 0.02, 0.1] # time step
tf_list = [1, 2, 3, 4, 5, 6, 7, 8] # time horizon
noise_level_list = [0.1, 1, 2, 3, 4] # noise level
stability_list = [-0.1, 0, 0.1] # stability


# Noise
def noise(dt):
    return np.random.multivariate_normal([0, 0], dt*np.eye(2), N)

# Start experiment
for exp in range(exp_num):
    
    # Experiment for different T, fixed (dt, noise level, and stability)
    T = 4
    noise_level = 1
    dt = 0.02
    LSV_MSE = []
    TRCo_MSE = []
    LSCo_MSE = []
    TRV_MSE = []
    for stability in stability_list:
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
            W_f[noise_step, :, :] = noise(dt)
            W_b[noise_step, :, :] = noise(dt)
    
        # experiment results
        G_LSV = LeastSquareValue(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)
        G_TRCo = TimeReversalCostate(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)
        G_LSCo = LeastSquareCostate(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)
        G_TRV = TimeReversalValue(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N)

        # solve Riccati equation
        G_ref = solve_riccati(A, B, Q, R, Q_f, T, dt).transpose(2,0,1)

        # MSE
        LSV_MSE.append(np.mean((G_LSV - G_ref)**2))
        TRCo_MSE.append(np.mean((G_TRCo - G_ref)**2))
        LSCo_MSE.append(np.mean((G_LSCo - G_ref)**2))
        TRV_MSE.append(np.mean((G_TRV - G_ref)**2))

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

    # plot
    plt.figure()
    plt.plot(stability_list, LSV_MSE, label='Least Square Value', color='C0')
    plt.plot(stability_list, TRCo_MSE, label='Time Reversal Costate', color='C1')
    plt.plot(stability_list, LSCo_MSE, label='Least Square Costate', color='C2')
    plt.plot(stability_list, TRV_MSE, label='Time Reversal Value', color='C3')
    plt.xlabel('Stability parameter')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.title('Experiment: T={}, Noise_level={}, dt={}'.format(T, noise_level, dt))
    plt.legend()
    plt.show()
