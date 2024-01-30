"""TODO: What is a better way to run all these scripts?"""

OUTPUT_DIR = "../figures"


def generate_figure_1():
    import torch
    import matplotlib.pyplot as plt
    from neural_network_util import MCReference, AuxiliaryLinearDiffusion, UniformDomainDataset, LinearFilterPDESolver
    from run_experiments import INPUT_DIMENSION, DOMAIN, COEFFS, INITIAL_CONDITION, delta_t, RESOLUTION

    torch.manual_seed(23571113)

    N1 = 2 ** 5
    N2 = 4

    mc_reference = MCReference(INPUT_DIMENSION, DOMAIN, delta_t,
                            COEFFS, INITIAL_CONDITION, 2**15)
    auxiliary_diffusion = AuxiliaryLinearDiffusion(
        COEFFS, num_diffusions_per_input=N2)

    x0 = UniformDomainDataset(DOMAIN, mc_sample_size=N1, dim=INPUT_DIMENSION, use_quasi_mc=True).x
    xT = auxiliary_diffusion.simulate(x0, delta_t)
    pde_solver = LinearFilterPDESolver(COEFFS, INITIAL_CONDITION, delta_t, DOMAIN, INPUT_DIMENSION, None)
    values = pde_solver.feynman_kac_stochastic_representation(xT).squeeze()
    mean = torch.mean(values, dim=0)
    mc_domain = torch.linspace(*DOMAIN, RESOLUTION)
    plt.plot(mc_domain, mc_reference.exact(mc_domain), label="True Solution")

    x = x0.squeeze()
    for i in range(N2):
        plt.scatter(x, values[i], label=f"MC_{i}")

    plt.scatter(x, mean, label="Mean", marker="x")
    plt.legend(loc="upper right")
    plt.xlabel(r"$x$")
    plt.savefig(f"{OUTPUT_DIR}/stochastic_approximations.png")


def generate_figure_2():
    import torch
    import matplotlib.pyplot as plt
    from neural_network_util import AuxiliaryLinearDiffusion, LinearFilterPDESolver
    from run_experiments import INPUT_DIMENSION, DOMAIN, COEFFS, INITIAL_CONDITION, delta_t, RESOLUTION

    torch.manual_seed(1705415)

    BATCH_SIZE = 2 ** 4
    N2 = 2 ** 15

    auxiliary_diffusion = AuxiliaryLinearDiffusion(
        COEFFS, num_diffusions_per_input=N2)

    x0 = torch.tensor([0])
    xT = auxiliary_diffusion.simulate(x0, delta_t)
    pde_solver = LinearFilterPDESolver(COEFFS, INITIAL_CONDITION, delta_t,
                                    DOMAIN, INPUT_DIMENSION, None)
    values = pde_solver.feynman_kac_stochastic_representation(xT).squeeze()
    batched = values.reshape((BATCH_SIZE, -1)).mean(dim=0)
    # Also the mean value. Mean is the same for values and batched
    true_value = values.mean().item()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    _ax = ax[0]
    _ax.hist(batched, density=True, bins=100, alpha=0.9, label="Modified")
    _ax.hist(values, density=True, bins=100, alpha=0.8, label="Original")
    _ax.axvline(true_value, color="black", label="True Value")
    _ax.set_xlabel("Stochastic Approximation")
    _ax.legend()

    # Want blue to be on top as it's smaller
    # Use zorder, the higher the number the closer to the front layer
    # Keep Alpha for consistency
    _ax = ax[1]
    ## Modified
    squared_residuals = (batched - true_value)**2
    mean_squared_residuals = torch.mean(squared_residuals)
    _ax.hist(squared_residuals, density=True, bins=100, label="Modified", zorder=10, alpha=0.9)
    _ax.axvline(mean_squared_residuals, label=f"Modified MSE = {mean_squared_residuals:.2f}",
                color="black", ls="--", zorder=15)
    ## Naive
    squared_residuals = (values - true_value)**2
    mean_squared_residuals = torch.mean(squared_residuals)
    _ax.hist(squared_residuals, density=True, bins=100, label="Original", zorder=0, alpha=0.8)
    _ax.axvline(mean_squared_residuals, label=f"Naive MSE = {mean_squared_residuals:.2f}",
                color="black", zorder=15)

    _ax.set_yscale("log")
    _ax.set_xlabel(r"Squared Residuals ($N_2 = 2^4$)")
    _ax.legend()

    plt.savefig(f"{OUTPUT_DIR}/residual_distributions.png")


def generate_figure_3():
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # Use the most recent experiment
    RESULTS_DIR = "results"
    INPUT_DIR = f"{RESULTS_DIR}/{sorted(os.listdir(RESULTS_DIR))[-1]}"


    LOSS_DATA = {}
    L2_DATA = {}
    for use_quasi_mc in [True, False]:
        for use_modified_loss in [True, False]:
            filename = f"qmc={use_quasi_mc}_modloss={use_modified_loss}"
            # Make the USE_MOD_LOSS align. Because True and False are off by one char
            gap = " " if use_quasi_mc else ""
            output_name = f"USE_QMC={use_quasi_mc},{gap} USE_MOD_LOSS={use_modified_loss}"
            data = np.load(f"{INPUT_DIR}/{filename}.npz")
            LOSS_DATA[output_name] = data["loss_function_1"]
            L2_DATA[output_name] = data["l2_residual_1"]


    # L2 Residual Plot on a log y scale
    for name, ts in L2_DATA.items():
        plt.semilogy(ts, label=name)
    plt.legend(loc=1)
    plt.title("L2 Residual")
    plt.savefig(f"{OUTPUT_DIR}/l2_residual.png", bbox_inches="tight")
    plt.clf()


    # Plot a smoothened loss
    WINDOW_SIZE=30
    for name, ts in LOSS_DATA.items():
        plt.plot(pd.Series(ts).rolling(window=WINDOW_SIZE).mean(), label=name)
    plt.legend(loc=1)
    plt.title("Smoothened Loss")
    plt.savefig(f"{OUTPUT_DIR}/smoothened_loss.png", bbox_inches="tight")
    plt.clf()


COMMANDS = [
    generate_figure_1,
    generate_figure_2,
    generate_figure_3
]