import numpy as np
import torch


def LeastSquareValue(A, B, Noise_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N):
    """
    Perform the Least square monte carlo Value method for a linear quadratic model

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
    Gamma = np.linalg.inv(Noise_sigma) @ B
    # Data storage
    X_f = np.zeros((kf, steps+1, N, 2)) # Forward state trajectory
    Y_sample = np.zeros((kf, steps+1, N, 1)) # Sampled Y in BSDE
    Y_corrected = np.zeros((kf, steps+1, N, 1)) # Corrected Y in BSDE
    U_forward = np.zeros((kf+1, steps+1, N, 1)) # Forward control input
    Z_backward = np.zeros((kf, steps+1, N, 2)) # Z term in BSDE
    Alpha_records = np.zeros((kf, steps, 4, 1)) # basis function coefficients
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
                x_1 = x[:,0]
                x_2 = x[:,1]
                phi_x = np.zeros((N, 2, 4))
                phi_x[:, 0, 0] = x_1 * 2
                phi_x[:, 0, 2] = x_2
                phi_x[:, 1, 1] = x_2 * 2
                phi_x[:, 1, 2] = x_1
                alpha = Alpha_records[k-1, i, :, :]
                v_x = (phi_x @ alpha).squeeze(-1)
                u_f[i, :, :] = - (np.linalg.inv(R) @ B.T @ v_x.T).T
                U_forward[k, i, :, :] = u_f[i, :, :]
            dx = (A @ x.T + B @ u_f[i, :, :].T).T * dt + (Noise_sigma @ W_f[i, :, :].T).T
            x = x + dx
            X_f[k, i+1, :, :] = x.copy()

        # Backward pass
        y_b = 0.5 * (X_f[k, -1, :, :] @ Q_f * X_f[k, -1, :, :]).sum(axis=1, keepdims=True)
        z_b = (Noise_sigma.T @ Q_f @ X_f[k, -1, :, :].T).T
        Y_sample[k, -1, :, :] = y_b.copy()
        Y_corrected[k, -1, :, :] = y_b.copy()
        Z_backward[k, -1, :, :] = z_b.copy()
        for i in range(steps, 0, -1):
            x_b = X_f[k, i, :, :]
            h = 0.5 * (((x_b @ Q) * x_b).sum(axis=1, keepdims=True) - ((z_b @ Gamma @ np.linalg.inv(R) @ Gamma.T) * z_b).sum(axis=1, keepdims=True))
            h_hat = (h - (z_b @ Gamma * u_f[i,:,:]).sum(axis=1, keepdims=True))
            y_s = y_b + h_hat * dt
            x_pre = X_f[k, i-1, :, :]
            x_1 = x_pre[:,0]
            x_2 = x_pre[:,1]
            phi_1 = x_1 ** 2
            phi_2 = x_2 ** 2
            phi_3 = x_1 * x_2
            phi_4 = np.ones_like(phi_1)
            phi_mat = np.concatenate((phi_1[:,np.newaxis], phi_2[:,np.newaxis], phi_3[:,np.newaxis], phi_4[:,np.newaxis]), axis=1)
            alpha = torch.pinverse(torch.from_numpy(phi_mat.T @ phi_mat)).numpy() @ phi_mat.T @ y_s
            # alpha = np.linalg.pinv(phi_mat.T @ phi_mat) @ phi_mat.T @ y_s
            Alpha_records[k, i-1, :, :] = alpha.copy()
            y_c = phi_mat @ alpha
            Y_sample[k, i-1, :, :] = y_s.copy()
            Y_corrected[k, i-1, :, :] = y_c.copy()
            phi_x = np.zeros((N, 2, 4))
            x_1 = x_pre[:,0]
            x_2 = x_pre[:,1]
            phi_x[:, 0, 0] = x_1 * 2
            phi_x[:, 0, 2] = x_2
            phi_x[:, 1, 1] = x_2 * 2
            phi_x[:, 1, 2] = x_1
            v_x = (phi_x @ alpha).squeeze(-1)
            z_b = (Noise_sigma.T @ v_x.T).T
            Z_backward[k, i-1, :, :] = z_b.copy()
            y_b = y_c.copy()
    

        # Cost calculation
        J[k] += 0.5 * (X_f[k,:,:,:] @ Q * X_f[k,:,:,:]).mean(axis=1).sum() * dt
        J[k] += 0.5 * (U_forward[k,:,:,:] @ R * U_forward[k,:,:,:]).mean(axis=1).sum() * dt
        J[k] += 0.5 * (X_f[k,-1,:,:] @ Q_f * X_f[k,-1,:,:]).mean(axis=0).sum() 
    
    # Converge basis function coefficients to gain matrix
    alpha_final = Alpha_records[kf-1,:,:,:]
    G = np.zeros((steps, 2, 2))
    G[:, 0, 0] = alpha_final[:, 0, 0] * 2
    G[:, 0, 1] = alpha_final[:, 2, 0]
    G[:, 1, 0] = alpha_final[:, 2, 0]
    G[:, 1, 1] = alpha_final[:, 1, 0] * 2

   
    return G, J