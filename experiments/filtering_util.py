"""
Utility functions for solving Filtering Equation.
Original code from Alexander Lobbe and converted into PyTorch.
https://arxiv.org/abs/2201.03283
"""

import time
import torch

def lin_signal_update(x0, M, eta, sigma, delta_t):
    noise_incr = torch.sqrt(delta_t) * torch.randn(1)[0]
    x = x0 + delta_t * M * x0 + delta_t * eta + sigma * noise_incr
    return x


def observation_update(y0, signal, H, gamma, delta_t):
    noise_incr = torch.sqrt(delta_t) * torch.randn(1)[0]
    y = y0 + delta_t * H * signal + delta_t * gamma + noise_incr
    return y

def lin_sig_obs_paths(x0, y0, delta_t, n_steps, M, eta, sigma, H, gamma):

    signal = torch.zeros(n_steps+1)
    observation = torch.zeros(n_steps+1)
    signal[0] = x0
    observation[0] = y0

    for i in range(n_steps):
        signal[i+1] = lin_signal_update(signal[i], M, eta, sigma, delta_t)
        observation[i+1] = observation_update(observation[i], signal[i],
                                            H, gamma, delta_t)

    return signal, observation


def likelihood_fn(y0, y1, delta_t, H, gamma, x):
    """This is eta_n(x) in the paper"""
    z = (y1-y0)/delta_t
    return torch.exp(-0.5*delta_t*(z - H*x-gamma)**2)


def solution_lin(sig_0, obs_path, delta_t, n_steps, M, eta, sigma, H, gamma):
    '''
    sig_0   : float
    obs_path: array
    delta_t : float
    n_steps : int
    M       : float
    eta     : float
    sigma   : float
    H       : float
    gamma   : float

    x_hat   : array
    r       : array
    '''

    times = torch.arange(n_steps+1) * delta_t # array; time stemps

    p = torch.sqrt((M/sigma)**2 + H**2) # float
    q = M*eta/sigma**2 + H*gamma # float

    path_incr = torch.diff(obs_path)
    psi_new = torch.cumsum(torch.sinh(times[1:]*p*sigma)*path_incr, dim=0) \
            / torch.sinh(times[1:]*p*sigma)

    A = eta/sigma**2 + H*psi_new \
        +(q+p**2*sig_0)/(p*sigma*torch.sinh(times[1:]*p*sigma))\
            - q/(p*sigma) * torch.cosh(times[1:]*p*sigma)/torch.sinh(times[1:]*p*sigma) # A: array

    B = p/(2*sigma) * torch.cosh(times[1:]*p*sigma)/torch.sinh(times[1:]*p*sigma) \
        - M/(2*sigma**2) # B: array

    x_hat = A/(2*B)
    r = 1./(2*B)

    # Insert a 0 at the start of the array
    x_hat = torch.concatenate([torch.tensor([0.]), x_hat])
    r     = torch.concatenate([torch.tensor([0.]), r])

    return x_hat, r


def normalisation_constant(y0, y1, delta_t, H, gamma,
                           network, mc_samples=int(1e6)):

    start = time.perf_counter_ns()
    z = (y1-y0)/delta_t
    mean = (z-gamma) / H
    std = 1./(torch.sqrt(delta_t)*H)
    samples = mean + torch.randn(mc_samples)*std
    mc_correction = torch.sqrt(2. * torch.pi / (delta_t * H**2))

    # Perform rejection sampling
    samples = samples[(network.domain[0] <= samples)
                      & (samples <= network.domain[1])]

    mc_vals = mc_correction * network(samples)
    const = torch.mean(mc_vals)
    timing = time.perf_counter_ns() - start

    acc_rate = samples.shape[0] / mc_samples

    return 1./const, acc_rate, timing


class CorrectedDensity:
    """Second step of Splitting up Method: Correction.
    network is the neural network used to solve the PDE in step 1.
    multiple by eta_n
    normalise using precomputed constant.
    """
    def __init__(self, H, gamma, network, y0, y1, delta_t, normalisation_constant):
        self.H = H
        self.gamma = gamma
        self.network = network
        self.y0 = y0
        self.y1 = y1
        self.delta_t = delta_t
        self.normalisation_constant = normalisation_constant

    def likelihood_fn(self, x):
        """This is eta_n(x) in the paper"""
        z = (self.y1 - self.y0) / self.delta_t
        return torch.exp(-0.5 * self.delta_t * (z - self.H*x - self.gamma)**2)

    def __call__(self, x):
        """Second step of Splitting up Method: Correction.
        network is the neural network used to solve the PDE in step 1.
        multiple by eta_n
        normalise using precomputed constant.
        """
        xi = self.likelihood_fn(x)
        p_tilde = self.network(x)
        return self.normalisation_constant * xi * p_tilde