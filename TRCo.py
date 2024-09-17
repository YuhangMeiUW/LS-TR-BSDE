import numpy as np

def TimeReversalCostate(A, B, N_sigma, Q, R, Q_f, D, T, dt, X_0, W_f, W_b, kf, N):

    steps = int(T/dt)

    # Data storage
    X_f = np.zeros((kf, steps+1, N,2))
    X_b = np.zeros((kf, steps+1, N,2))
    Y_b = np.zeros((kf, steps+1, N, 2))
    U_forward = np.zeros((kf+1, steps+1, N, 1))
    U_backward = np.zeros((kf+1, steps+1, N, 1))
    

    for k in range(kf):

        u_f = U_forward[k, :, :, :].copy()
        u_b = U_backward[k, :, :, :].copy()
        x = X_0.copy()
        X_f[k, 0, :, :] = x.copy()

        # Forward pass
        for i in range(steps):
            if k > 0:
                u_f[i, :, :] = - (np.linalg.inv(R) @ B.T @ G_record[i, :, :] @ x.T).T
                U_forward[k, i, :, :] = u_f[i, :, :].copy()
            dX = (A @ x.T + B @ u_f[i, :, :].T).T * dt + (N_sigma @ W_f[i, :, :].T).T
            x = x + dX
            X_f[k, i+1, :, :] = x.copy()
         
        # Folmer drift
        m_k_t = X_f[k, :, :, :].mean(axis=1)
        x_minus_m = X_f[k, :, :, :] - np.repeat(m_k_t[:, np.newaxis], N, axis=1)
        Sigma_k_t = np.einsum('tni,tnj->tij', x_minus_m, x_minus_m) / N
        
        # Backward pass for the state
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

        # Backward pass for the costate
        y_b_final = (Q_f @ X_b[k, -1, :, :].T).T
        Y_b[k, -1, :, :] = y_b_final.copy()
        G_record = np.zeros((steps+1, 2, 2))
        for i in range(steps, 0, -1):
            x_b = X_b[k, i, :, :].copy()
            G = (y_b_final.T @ x_b @ np.linalg.inv(x_b.T @ x_b))
            G_record[i, :, :] = G.copy()
            back_noise = W_b[i, :, :].copy()
            mean = m_k_t[i, :].copy()
            mean_repeated = np.repeat(mean[:, np.newaxis], N, axis=1).T
            minus_dY = (A.T @ y_b_final.T + Q @ x_b.T).T * dt - (G @ N_sigma @ back_noise.T).T - (G @ D @ np.linalg.inv(Sigma_k_t[i,:,:]) @ (x_b - mean_repeated).T).T*dt
            y_b_final = y_b_final + minus_dY
            Y_b[k, i-1, :, :] = y_b_final.copy()
        G = (y_b_final.T @ X_b[k,0,:,:] @ np.linalg.inv(X_b[k,0,:,:].T @ X_b[k,0,:,:]))
        G_record[0, :, :] = G.copy()


    # Cost
    J = np.zeros((kf, 1))
    for k in range(kf):
        x_mean = X_f[k, :, :, :].mean(axis=1)
        u_mean = U_forward[k, :, :, :].mean(axis=1)
        J[k] = 0.5 * ((x_mean @ Q * x_mean).sum() + (u_mean @ R * u_mean).sum()) * dt
        J[k] += 0.5 * ((x_mean[-1,:] @ Q_f * x_mean[-1,:]).sum()) * (1-dt)
    
    return G_record[1:, :, :]