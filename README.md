# Time-reversal solution of BSDEs in stochastic optimal control: a linear quadratic study
This repository is created by Yuhang Mei and contains the Python source code to reproduce the experiments in our paper [Time-reversal solution of BSDEs in stochastic optimal control: a linear quadratic study] (https://arxiv.org/pdf/2410.04615).

The four algorithms are implemented on a two-dimensional LQ example with the model parameters.

$$
\begin{align*}
&A = \begin{bmatrix}
	0 & 1 \\
	-1 & -0.1
\end{bmatrix},~B = \begin{bmatrix}
0\\
1
\end{bmatrix},~\sigma = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
,R=1,~m_0=\begin{bmatrix}
1\\
0
\end{bmatrix},\quad Q=Q_f=\Sigma_0=\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\end{align*}
$$

where $m_0$ and $\Sigma_0$ are the mean and covariance of the Gaussian initial distribution $p_0$. The time horizon $T=4$. The number of samples $N=2000$ and the time-discretization step-size $\Delta t = 0.02$. All four algorithms start with a zero control law $k(t,x)=0$. Each run of the algorithm results in a $2\times 2$ time-varying matrix $G_t$, which is used to update the control law according to the formula 

$$
\begin{align*}
     k(t,x)=-R^{-1} B^\top G_t x. 
\end{align*}
$$

The new control law is used to run the algorithm again, and this procedure is repeated 200 times to ensure convergence and fair comparison among all algorithms. 



## Setup
* Python/Numpy,Scipy,Matplotlib
* Pytorch

## Running the code and regenerating data and figures.
1. For 2 dimsional example, run the 'main.py' to generate and save the data. We already make the four algorithms as four functions. You can play with different time step size, sample size. Use 'plot_result.ipynb' to plot figures.
2. For high dimensional example, run the 'mainhighdim.py' to generate and save the data. You can play with different dimensions. Use 'plot_result.ipynb' to plot figures.
