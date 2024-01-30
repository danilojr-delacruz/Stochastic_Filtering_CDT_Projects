import time
import os
import numpy as np
import torch
import lightning

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from neural_network_util import LinearFilterPDESolver, BasicNet,\
                                UniformDomainDataset
from filtering_util import lin_sig_obs_paths, solution_lin, \
                           CorrectedDensity, normalisation_constant


torch.manual_seed(1705415)


def gaussian_pdf(x, mean, std):
    """Probability density function of 1D Gaussian with specified mean and std."""
    # Ensure value is a float
    z = (x - mean) / std
    c = 2. * (std ** 2) * torch.pi
    return torch.exp(-z**2 / 2.) / torch.sqrt(c)

TIMESTAMP = time.strftime("%y-%m-%d_%H-%M-%S")
RESULT_DIR = f"./results/{TIMESTAMP}"
os.makedirs(RESULT_DIR, exist_ok=True)

# TODO: Use CUDA at some point
ACCELERATOR = "cpu"
DEVICE = torch.device(ACCELERATOR)

### Filtering Params:
M       = -1.
eta     = 0.
sigma   = 0.1
H       = 90.
gamma   = 0.
delta_t = 0.01

# These are x_0, y_0 in the paper
sig0, obs0 = 0., 0.

# Initial Distribution X_0 of signal process
INITIAL_MEAN      = 0.0
INITIAL_STD       = 0.01
INITIAL_CONDITION = lambda x: gaussian_pdf(x, torch.tensor(INITIAL_MEAN),
                                              torch.tensor(INITIAL_STD))

# This is number of points in the grid points
RESOLUTION = 1000

# Number of timesteps
NUM_STEPS = 1

### PDE Parameters:
### -------------------
INPUT_DIMENSION  = 1           # Dimension of Signal Process
DOMAIN           = [-.5, .5]   # [a, b] in the paper
COEFFS           = [M , -eta/M, sigma]

### Neural Network Parameters:
### --------------------
MAX_EPOCHS = 300
NEURONS    = [51, 51, 1]

### Simulate filtering paths (Signal & Observation)
### -----------------------------------------------
# Use the same path for all experiments
# TODO: These need to be tensors, what is nicer way to write?
SIGNAL, OBSERVATION = lin_sig_obs_paths(
    *torch.tensor([sig0, obs0, delta_t]), NUM_STEPS,
    *torch.tensor([M, eta, sigma, H, gamma]))
SOL_MEAN, SOL_VAR   = solution_lin(
    torch.tensor(sig0), OBSERVATION, torch.tensor(delta_t), NUM_STEPS,
    *torch.tensor([M, eta, sigma, H, gamma]))


def experiment(use_quasi_mc, use_modified_loss, N1, N2):
    filename = f"qmc={use_quasi_mc}_modloss={use_modified_loss}"
    ### Store posterior density evolution
    ### ---
    # Each row corresponds to a timestep
    # Each column represents a gridpoint (as determined by RESOLUTION)
    posterior_evol = torch.zeros((NUM_STEPS, RESOLUTION))
    net_evol       = torch.zeros((NUM_STEPS, RESOLUTION))
    mcref_evol     = torch.zeros((NUM_STEPS, RESOLUTION))
    # Equivalent to doing torch.linspace(...)[:, None]
    # Dimension is (RESOLUTION, 1)
    x              = torch.linspace(*DOMAIN, RESOLUTION)

    ### Store statistics
    normalisation_constants = torch.zeros(NUM_STEPS)
    NNet_masses             = torch.zeros(NUM_STEPS)
    MCRef_masses            = torch.zeros(NUM_STEPS)
    acc_rates               = torch.zeros(NUM_STEPS)
    mc_times                = torch.zeros(NUM_STEPS)
    loss_function           = {}
    l2_residual             = {}
    train_times             = torch.zeros(NUM_STEPS)

    ### Main Loop
    initial_condition = INITIAL_CONDITION
    for i in range(NUM_STEPS):
        print(f'Timestep {i+1} of {NUM_STEPS}')

        ### 1. Splitting up method step 1: Prediction (solving PDE)
        ##  1.1 Use neural network to solve PDE
        neural_network = BasicNet(NEURONS, INPUT_DIMENSION, DOMAIN)
        pde_solver     = LinearFilwterPDESolver(
                               COEFFS, initial_condition,
                               delta_t, DOMAIN, INPUT_DIMENSION,
                               neural_network,
                               use_modified_loss=use_modified_loss,
                               num_diffusions_per_input=N2,
                               use_mc_reference=True,
                               comparison_resolution=RESOLUTION)
        dataset = UniformDomainDataset(DOMAIN, mc_sample_size=N1,
                                       use_quasi_mc=use_quasi_mc)
        train_loader = DataLoader(dataset)

        ##  1.2 Train the neural network
        logger = TensorBoardLogger("tb_logs",
            name=f"{TIMESTAMP}/{filename}")

        start_time = time.time() # This is in seconds
        pde_solver.train() # Switch to training mode
        # Each version will correspond to a time step
        trainer = lightning.Trainer(accelerator=ACCELERATOR, max_epochs=MAX_EPOCHS,
                                    logger=logger, log_every_n_steps=1)
        trainer.fit(model=pde_solver, train_dataloaders=train_loader)
        train_times[i] = time.time() - start_time

        pde_solver.eval() # Switch to evaluation mode
        # Freeze parameters here to ensure they don't get updated
        for param in pde_solver.parameters():
            param.requires_grad = False


        ### 2. Splitting up method step 2: Correction (normalisation)
        ##  2.1 Get new observations and track signal process
        # Careful! sig is only used as entry to coef_dict.
        # Should not be used anywhere else!
        old_sig, old_obs = SIGNAL[i], OBSERVATION[i]
        sig, obs = SIGNAL[i+1], OBSERVATION[i+1]

        ##  2.2 Compute the normalisation constant for the neural network
        # Use Monte Carlo Estimation
        con, acc_rate, mc_time = normalisation_constant(
            *torch.tensor([old_obs, obs, delta_t, H, gamma]),
            pde_solver, mc_samples=int(1e6))
        normalisation_constants[i] = con
        # Accepting only points within the domain of the neural network.
        acc_rates[i] = acc_rate
        mc_times[i]  = mc_time

        ##  2.3 Incorporate new observation and normalise the prior density
        posterior_density = CorrectedDensity(
            *torch.tensor([H, gamma]), pde_solver,
            *torch.tensor([old_obs, obs, delta_t]),
            normalisation_constant=con)


        ### 3. Statistics for figures
        ##  3.1 Compute the mass of the neural network over the domain
        domain_mesh     = torch.linspace(*DOMAIN, RESOLUTION)
        nn_value        = pde_solver(domain_mesh)
        NNet_masses[i]  = torch.trapezoid(nn_value, domain_mesh).item()

        ## 3.2 Compute the mass of the MC Reference
        mc_value        = pde_solver.mc_reference(domain_mesh)
        MCRef_masses[i] = torch.trapezoid(mc_value, domain_mesh).item()

        ## 3.3 Snapshots
        posterior_evol[i,:] = torch.squeeze(posterior_density(x))
        net_evol[i,:]       = torch.squeeze(pde_solver(x))
        mcref_evol[i,:]     = torch.squeeze(pde_solver.mc_reference(x))

        ## 3.4 Training times
        loss_function.update({f"loss_function_{i+1}": pde_solver.training_losses})
        l2_residual.update({f"l2_residual_{i+1}": pde_solver.l2_residual})


        ### 4. Prepare new iteration
        initial_condition = posterior_density

    # Save things in NumPy Format
    coef_dict = {"n": NUM_STEPS,
        "M": M,
        "eta": eta,
        "sigma": sigma,
        "H": H,
        "gamma": gamma,
        "delta_t" : delta_t,
        "sig" : sig,
        "obs" : obs,
        "initial_cond_mean_std" : [INITIAL_MEAN, INITIAL_STD],
        "neurons": NEURONS,
        "batch_size": N1,
        "input_dimension" : INPUT_DIMENSION,
        "coefs" : np.array(COEFFS),
        "epochs": MAX_EPOCHS,
        "resolution": RESOLUTION
    }

    np.savez(f"{RESULT_DIR}/{filename}",
                time_grid               = np.arange(NUM_STEPS+1)*delta_t,
                domain                  = np.array(DOMAIN),
                signal                  = np.array(SIGNAL),
                observation             = np.array(OBSERVATION),
                sol_mean                = np.array(SOL_MEAN),
                sol_var                 = np.array(SOL_VAR),
                normalisation_constants = np.array(normalisation_constants),
                NNet_masses             = np.array(NNet_masses),
                mcref_masses            = np.array(MCRef_masses),
                posterior_evolution     = np.array(posterior_evol),
                net_evolution           = np.array(net_evol),
                mcref_evol              = np.array(mcref_evol),
                **loss_function,
                **l2_residual,
                train_times             = np.array(train_times),
                acc_rates               = np.array(acc_rates),
                mc_times                = np.array(mc_times),
                **coef_dict,
                timestamp               = np.array([TIMESTAMP])
                )



if __name__ == '__main__':
    for use_quasi_mc in [True, False]:
        for use_modified_loss in [True, False]:
            # Ensure N1*N2 constant for fair comparison
            NUM_SAMPLES = 2 ** 9
            if use_modified_loss:
                N1 = 2 ** 5
            else:
                N1 = 2 ** 9
            N2 = NUM_SAMPLES // N1

            experiment(use_quasi_mc, use_modified_loss, N1, N2)

    from generate_figures import COMMANDS, OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for command in COMMANDS:
        command()
