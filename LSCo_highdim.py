import numpy as np
import torch

def LeastSquareCostate_highdim(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N, dim=2):

    steps = int(T/dt)

    # Data storage
    X_f = np.zeros((kf, steps+1, N, dim))
    Y_sample = np.zeros((kf, steps+1, N, dim))
    Y_corrected = np.zeros((kf, steps+1, N, dim))
    U_forward = np.zeros((kf+1, steps+1, N, int(dim/2)))
    Z_backward = np.zeros((kf, steps+1, dim, dim))
    Alpha_records = np.zeros((kf, steps, dim, dim))

    for k in range(kf):

        u_f = U_forward[k, :, :, :].copy()
        x = X_0.copy()
        X_f[k, 0, :, :] = x.copy()

        # Forward pass
        for i in range(steps):
            if k > 0:
                u_f[i, :, :] = - (np.linalg.inv(R) @ B.T @ Alpha_records[k-1, i, :, :].T @ x.T).T
                U_forward[k, i, :, :] = u_f[i, :, :].copy()
            dX = (A @ x.T + B @ u_f[i, :, :].T).T * dt + (N_sigma @ W_f[i, :, :].T).T
            x = x + dX
            X_f[k, i+1, :, :] = x.copy()

        # Backward pass
        y_b = (Q_f @ X_f[k, -1, :, :].T).T
        z_b = Q_f @ N_sigma
        Y_sample[k, -1, :, :] = y_b.copy()
        Y_corrected[k, -1, :, :] = y_b.copy()
        Z_backward[k, -1, :, :] = z_b.copy()
        for i in range(steps, 0, -1):
            x_b = X_f[k, i, :, :].copy()
            h = (Q @ x_b.T + A.T @ y_b.T).T
            y_s = y_b + h * dt
            phi_mat = X_f[k, i-1, :, :].copy()
            # alpha = y_s.T @ phi_mat @ np.linalg.pinv(phi_mat.T @ phi_mat)
            alpha = y_s.T @ phi_mat @ torch.pinverse(torch.from_numpy(phi_mat.T @ phi_mat)).numpy()
            Alpha_records[k, i-1, :, :] = alpha.copy()
            y_c = (alpha @ phi_mat.T).T
            Y_sample[k, i-1, :, :] = y_s.copy()
            Y_corrected[k, i-1, :, :] = y_c.copy()
            z_b = (N_sigma.T @ alpha).T
            Z_backward[k, i-1, :, :] = z_b.copy()
            y_b = y_c.copy()

    
    # Cost
    J = np.zeros((kf, 1))
    for k in range(kf):
        x_mean = X_f[k, :, :, :].mean(axis=1)
        u_mean = U_forward[k, :, :, :].mean(axis=1)
        J[k] = 0.5 * ((x_mean @ Q * x_mean).sum() + (u_mean @ R * u_mean).sum()) * dt
        J[k] += 0.5 * ((x_mean[-1,:] @ Q_f * x_mean[-1,:]).sum()) * (1-dt)
    
    return Alpha_records[kf-1, :, :, :]