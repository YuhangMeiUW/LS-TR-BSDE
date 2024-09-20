import numpy as np
import torch


def LeastSquareValue_highdim(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N, dim=2):
    
    steps = int(T/dt)
    Gamma = np.linalg.inv(N_sigma) @ B
    # Data storage
    X_f = np.zeros((kf, steps+1, N, dim))
    Y_sample = np.zeros((kf, steps+1, N, 1))
    Y_corrected = np.zeros((kf, steps+1, N, 1))
    U_forward = np.zeros((kf+1, steps+1, N, int(dim/2)))
    Z_backward = np.zeros((kf, steps+1, N, dim))
    G_record = np.zeros((kf, steps, dim, dim))
    
    for k in range(kf):

        u_f = U_forward[k, :, :, :].copy()
        x = X_0.copy()
        X_f[k, 0, :, :] = x.copy()

        # Forward pass
        for i in range(steps):
            if k > 0:
                u_f[i, :, :] = - (np.linalg.inv(R) @ B.T @ G_record[k-1, i, :, :] @ x.T).T
                U_forward[k, i, :, :] = u_f[i, :, :].copy()
            dX = (A @ x.T + B @ u_f[i, :, :].T).T * dt + (N_sigma @ W_f[i, :, :].T).T
            x = x + dX
            X_f[k, i+1, :, :] = x.copy()

        # Backward pass
        y_b = 0.5 * (X_f[k, -1, :, :] @ Q_f * X_f[k, -1, :, :]).sum(axis=1, keepdims=True)
        z_b = (N_sigma.T @ Q_f @ X_f[k, -1, :, :].T).T
        Y_sample[k, -1, :, :] = y_b.copy()
        Y_corrected[k, -1, :, :] = y_b.copy()
        Z_backward[k, -1, :, :] = z_b.copy()
        for i in range(steps, 0, -1):
            x_b = X_f[k, i, :, :].copy()
            h = 0.5 * (((x_b @ Q) * x_b).sum(axis=1, keepdims=True) - ((z_b @ Gamma @ np.linalg.inv(R) @ Gamma.T) * z_b).sum(axis=1, keepdims=True))
            h_hat = (h - (z_b @ Gamma * u_f[i,:,:]).sum(axis=1, keepdims=True))
            y_s = y_b + h_hat * dt
            x_pre = X_f[k, i-1, :, :].copy()
            kron_x = (x_pre[:, :, None] * x_pre[:, None, :]).reshape(N, dim*dim)
            aug_kron_x = np.concatenate((kron_x, np.ones((N, 1))), axis=1)
            G_b_vec = (np.linalg.pinv(aug_kron_x.T @ aug_kron_x) @ aug_kron_x.T @ y_s)
            G_record[k, i-1, :, :] = G_b_vec[0:dim*dim, 0].reshape(dim, dim) * 2
            y_c = aug_kron_x @ G_b_vec
            Y_sample[k, i-1, :, :] = y_s.copy()
            Y_corrected[k, i-1, :, :] = y_c.copy()
            v_x = (G_record[k, i-1, :, :] @ X_f[k, i-1, :, :].T).T
            z_b = (N_sigma.T @ v_x.T).T
            Z_backward[k, i-1, :, :] = z_b.copy()
            y_b = y_c.copy()
    

    # Cost
    J = np.zeros((kf, 1))
    for k in range(kf):
        x_mean = X_f[k, :, :, :].mean(axis=1)
        u_mean = U_forward[k, :, :, :].mean(axis=1)
        J[k] = 0.5 * ((x_mean @ Q * x_mean).sum() + (u_mean @ R * u_mean).sum()) * dt
        J[k] += 0.5 * ((x_mean[-1,:] @ Q_f * x_mean[-1,:]).sum()) * (1-dt)

    G = G_record[kf-1, :, :, :].copy()
    return G