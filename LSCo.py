import numpy as np
import torch

def LeastSquareCostate(A, B, Noise_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N):
    """
    Perform the Least square monte carlo costate method for a linear quadratic model

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
    X_f = np.zeros((kf, steps+1, N,2)) # Forward state trajectory
    Y_sample = np.zeros((kf, steps+1, N, 2)) # Sampled costate trajectory
    Y_corrected = np.zeros((kf, steps+1, N, 2)) # Corrected costate trajectory
    U_forward = np.zeros((kf+1, steps+1, N, 1)) # Forward control input
    Z_backward = np.zeros((kf, steps+1, 2, 2)) # Z term in BSDE
    Alpha_records = np.zeros((kf, steps, 2, 2)) # Riccati matrix
    J = np.zeros((kf, 1)) # Cost for each iteration

    for k in range(kf):

        if k == 0:
            # In first iteration use zero control input
            u_f = np.zeros((steps+1, N, 1))
        x = X_0.copy()
        X_f[k, 0, :, :] = x.copy()

        # Forward pass
        for i in range(steps):
            if k > 0:
                u_f[i, :, :] = - (np.linalg.inv(R) @ B.T @ Alpha_records[k-1, i, :, :].T @ x.T).T
                U_forward[k, i, :, :] = u_f[i, :, :]
            dx = (A @ x.T + B @ u_f[i, :, :].T).T * dt + (Noise_sigma @ W_f[i, :, :].T).T
            x = x + dx
            X_f[k, i+1, :, :] = x.copy()

        # Backward pass
        y_b = (Q_f @ X_f[k, -1, :, :].T).T
        z_b = Q_f @ Noise_sigma
        Y_sample[k, -1, :, :] = y_b.copy()
        Y_corrected[k, -1, :, :] = y_b.copy()
        Z_backward[k, -1, :, :] = z_b.copy()
        for i in range(steps, 0, -1):
            x_b = X_f[k, i, :, :]
            h = (Q @ x_b.T + A.T @ y_b.T).T
            y_s = y_b + h * dt
            phi_mat = X_f[k, i-1, :, :]
            # alpha = y_s.T @ phi_mat @ np.linalg.pinv(phi_mat.T @ phi_mat)
            alpha = y_s.T @ phi_mat @ torch.pinverse(torch.from_numpy(phi_mat.T @ phi_mat)).numpy()
            Alpha_records[k, i-1, :, :] = alpha.copy()
            y_c = phi_mat @ alpha.T
            Y_sample[k, i-1, :, :] = y_s.copy()
            Y_corrected[k, i-1, :, :] = y_c.copy()
            z_b = (Noise_sigma.T @ alpha).T
            Z_backward[k, i-1, :, :] = z_b.copy()
            y_b = y_c.copy()

    
        # Cost calculation
        J[k] += 0.5 * (X_f[k,:,:,:] @ Q * X_f[k,:,:,:]).mean(axis=1).sum() * dt
        J[k] += 0.5 * (U_forward[k,:,:,:] @ R * U_forward[k,:,:,:]).mean(axis=1).sum() * dt
        J[k] += 0.5 * (X_f[k,-1,:,:] @ Q_f * X_f[k,-1,:,:]).mean(axis=0).sum() 
    
    G = Alpha_records[kf-1, :, :, :]
    return G, J