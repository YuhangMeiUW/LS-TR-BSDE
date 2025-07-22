import numpy as np
import torch

def TimeReversalCostate(A, B, Noise_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N):
    """
    Perform the time-reversal costate method for a linear quadratic model

    Args:
        A, B: Dynamics matrices (numpy arrays)
        Noise_sigma: Noise matrix
        Q, R: Cost matrices for state and control
        Q_f: Terminal cost matrix
        D: Noise_sigma @ Noise_sigma.T (covariance matrix)
        T: Total time horizon
        dt: Time step size
        X_0: Initial state (N x state_dim)
        W_f, W_b: Forward and backward noise samples (steps x N x state_dim)
        kf: Number of algorithm iterations
        N: Number of particles

    Returns:
        G: Gain matrices for each time step (steps x 2 x 2)
        J: Cost for each iteration (kf x 1)
    """

    steps = int(T/dt)

    # Data storage
    X_f = np.zeros((kf, steps+1, N, 2)) # Forward state trajectory
    X_b = np.zeros((kf, steps+1, N, 2)) # Backward state trajectory
    Y_b = np.zeros((kf, steps+1, N, 2)) # Backward costate trajectory
    U_forward = np.zeros((kf+1, steps+1, N, 1)) # Forward control input
    U_backward = np.zeros((kf+1, steps+1, N, 1)) # Backward control input
    J = np.zeros((kf, 1)) # Cost for each iteration
    

    for k in range(kf):

        x = X_0.copy()
        X_f[k, 0, :, :] = x.copy()
        
        if k == 0:
            # In first iteration use zero control input
            u_f = np.zeros((steps+1, N, 1))  # Forward control input
            u_b = np.zeros((steps+1, N, 1))  # Backward control input

        # Forward pass
        for i in range(steps):
            if k > 0:
                # Feedback control input 
                u_f[i, :, :] = - (np.linalg.inv(R) @ B.T @ G_record[i, :, :] @ x.T).T
                U_forward[k, i, :, :] = u_f[i, :, :]
            dx = (A @ x.T + B @ u_f[i, :, :].T).T * dt + (Noise_sigma @ W_f[i, :, :].T).T
            x = x + dx
            X_f[k, i+1, :, :] = x.copy()
         
        # Follmer drift
        m_k_t = X_f[k, :, :, :].mean(axis=1)
        x_minus_m = X_f[k, :, :, :] - np.repeat(m_k_t[:, np.newaxis], N, axis=1)
        Sigma_k_t = np.einsum('tni,tnj->tij', x_minus_m, x_minus_m) / N
        
        # Backward pass for the state
        m_final = m_k_t[-1, :].copy()
        Sigma_final = Sigma_k_t[-1, :, :].copy()
        x_b = np.random.multivariate_normal(m_final, Sigma_final, N)
        X_b[k, -1, :, :] = x_b.copy()
        for i in range(steps, 0, -1):
            back_noise = W_b[i, :, :].copy()
            mean = m_k_t[i, :].copy()
            mean_repeated = np.repeat(mean[:, np.newaxis], N, axis=1).T
            if k > 0:
                # Feedback control input
                u_b[i, :, :] = - (np.linalg.inv(R) @ B.T @ G_record[i, :, :] @ x_b.T).T
                U_backward[k, i, :, :] = u_b[i, :, :].copy()
            follmer = (D @ np.linalg.pinv(Sigma_k_t[i,:,:]) @ (x_b - mean_repeated).T).T
            dx = (A @ x_b.T + B @ u_b[i, :, :].T).T * dt + (Noise_sigma @ back_noise.T).T + follmer * dt
            x_b = x_b - dx
            X_b[k, i-1, :, :] = x_b.copy()

        # Backward pass for the costate
        y_b = (Q_f @ X_b[k, -1, :, :].T).T
        Y_b[k, -1, :, :] = y_b.copy()
        G_record = np.zeros((steps+1, 2, 2))
        for i in range(steps, 0, -1):
            x_b = X_b[k, i, :, :].copy()
            # G = (y_b.T @ x_b @ np.linalg.pinv(x_b.T @ x_b))
            G = (y_b.T @ x_b @ torch.pinverse(torch.from_numpy(x_b.T @ x_b)).numpy())
            G_record[i, :, :] = G.copy()
            back_noise = W_b[i, :, :].copy()
            mean = m_k_t[i, :].copy()
            mean_repeated = np.repeat(mean[:, np.newaxis], N, axis=1).T
            minus_dy = (A.T @ y_b.T + Q @ x_b.T).T * dt - (G @ Noise_sigma @ back_noise.T).T - (G @ D @ np.linalg.pinv(Sigma_k_t[i,:,:]) @ (x_b - mean_repeated).T).T*dt
            y_b = y_b + minus_dy
            Y_b[k, i-1, :, :] = y_b.copy()
        # G = (y_b.T @ X_b[k,0,:,:] @ np.linalg.pinv(X_b[k,0,:,:].T @ X_b[k,0,:,:]))
        G = (y_b.T @ X_b[k,0,:,:] @ torch.pinverse(torch.from_numpy(X_b[k,0,:,:].T @ X_b[k,0,:,:])).numpy())
        G_record[0, :, :] = G.copy()


        # Cost calculation 
        J[k] += 0.5 * (X_f[k,:,:,:] @ Q * X_f[k,:,:,:]).mean(axis=1).sum() * dt
        J[k] += 0.5 * (U_forward[k,:,:,:] @ R * U_forward[k,:,:,:]).mean(axis=1).sum() * dt
        J[k] += 0.5 * (X_f[k,-1,:,:] @ Q_f * X_f[k,-1,:,:]).mean(axis=0).sum() 
    
    G = G_record[1:, :, :]
    return G, J