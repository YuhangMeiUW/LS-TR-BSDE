import numpy as np
import torch

def TimeReversalValue_highdim(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N, dim=2):

    steps = int(T/dt)

    # Data storage
    X_f = np.zeros((kf, steps+1, N, dim))
    X_b = np.zeros((kf, steps+1, N, dim))
    Y_b = np.zeros((kf, steps+1, N, 1))
    Z_b = np.zeros((kf, steps+1, N, dim))
    U_forward = np.zeros((kf+1, steps+1, N, int(dim/2)))
    U_backward = np.zeros((kf+1, steps+1, N, int(dim/2)))

    for k in range(kf):
        
        u_f = U_forward[k,:,:,:].copy()
        u_b = U_backward[k,:,:,:].copy()
        x = X_0.copy()
        X_f[k,0,:,:] = x.copy()

        # Forward pass
        for i in range(steps):
            if k > 0:
                u_f[i,:,:] = - (np.linalg.inv(R) @ B.T @ G_record[i,:,:] @ X_f[k,i,:,:].T).T
                U_forward[k,i,:,:] = u_f[i,:,:].copy()
            dX = (A @ x.T + B @ u_f[i,:,:].T).T * dt + (N_sigma @ W_f[i,:,:].T).T
            x = x + dX
            X_f[k,i+1,:,:] = x.copy()


        # Folmer's drift
        m_k_t = X_f[k, :, :, :].mean(axis=1)
        x_minus_m = X_f[k, :, :, :] - np.repeat(m_k_t[:, np.newaxis], N, axis=1)
        Sigma_k_t = np.einsum('tni,tnj->tij', x_minus_m, x_minus_m) / N   


        # Time-reversed state
        m_final = m_k_t[-1, :].copy()
        Sigma_final = Sigma_k_t[-1, :, :].copy()
        x_b_final = np.random.multivariate_normal(m_final, Sigma_final, N)
        X_b[k, -1, :, :] = x_b_final.copy()
        for i in range(steps, 0, -1):
            back_noise = W_b[i, :, :].copy()
            mean = m_k_t[i, :].copy()
            mean_repeated = np.repeat(mean[:, np.newaxis], N, axis=1).T
            if k > 0:
                u_b[i, :, :] = - (np.linalg.inv(R) @ B.T @ G_record[i, :, :] @ x_b_final.T).T
                U_backward[k, i, :, :] = u_b[i, :, :].copy()
            dX = (A @ x_b_final.T + B @ u_b[i, :, :].T).T * dt + (N_sigma @ back_noise.T).T + (D @ np.linalg.inv(Sigma_k_t[i,:,:]) @ (x_b_final - mean_repeated).T).T*dt
            x_b_final = x_b_final - dX
            X_b[k, i-1, :, :] = x_b_final.copy()
    
        # time-reversal y and z
        y_b_final = 0.5 * (X_b[k,-1,:,:] @ Q_f * X_b[k,-1,:,:]).sum(axis=1, keepdims=True)
        z_b_final = (N_sigma.T @ Q_f @ X_b[k,-1,:,:].T).T
        Y_b[k,-1,:,:] = y_b_final.copy()
        Z_b[k,-1,:,:] = z_b_final.copy()
        G_record = np.zeros((steps+1, dim, dim))
        G_record[-1,:,:] = Q_f.copy()
        for i in range(steps,0,-1):
            x_b = X_b[k,i,:,:].copy()
            G = G_record[i,:,:].copy()
            v_x = (G @ x_b.T).T
            back_noise = W_b[i,:,:].copy()
            mean = m_k_t[i,:].copy()
            mean_rep = np.repeat(mean[:,np.newaxis],N,axis=1).T
            h = 0.5 * (x_b @ Q * x_b).sum(axis=1, keepdims=True) - 0.5 * (v_x @ B @ np.linalg.inv(R) @ B.T * v_x).sum(axis=1, keepdims=True)
            h_hat = h - (v_x @ B * u_b[i,:,:]).sum(axis=1, keepdims=True) + np.trace(G.T @ N_sigma @ N_sigma.T) - (v_x * (D @ np.linalg.inv(Sigma_k_t[i,:,:]) @ (x_b - mean_rep).T).T).sum(axis=1, keepdims=True)
            minus_dY = h_hat*dt - (z_b_final * back_noise).sum(axis=1, keepdims=True)
            y_b_final = y_b_final + minus_dY
            Y_b[k,i-1,:,:] = y_b_final.copy() 
            x_pre = X_b[k,i-1,:,:].copy()
            kron_x = (x_pre[:, :, None] * x_pre[:, None, :]).reshape(N, dim*dim)
            aug_kron_x = np.concatenate((kron_x, np.ones((N, 1))), axis=1)
            G_b_vec = (np.linalg.pinv(aug_kron_x.T @ aug_kron_x) @ aug_kron_x.T @ y_b_final)
            G = G_b_vec[0:dim*dim, 0].reshape(dim, dim) * 2
            G_record[i-1, :, :] = G.copy()
            v_x = (G @ x_pre.T).T
            z_b_final = (N_sigma.T @ v_x.T).T
            Z_b[k, i-1, :, :] = z_b_final.copy()
    
    # Cost
    J = np.zeros((kf, 1))
    for k in range(kf):
        x_mean = X_f[k, :, :, :].mean(axis=1)
        u_mean = U_forward[k, :, :, :].mean(axis=1)
        J[k] = 0.5 * ((x_mean @ Q * x_mean).sum() + (u_mean @ R * u_mean).sum()) * dt
        J[k] += 0.5 * ((x_mean[-1,:] @ Q_f * x_mean[-1,:]).sum()) * (1-dt)

    return G_record[1:,:,:]