"""
Utility functions for Neural Network.
Original code from Alexander Lobbe and converted into PyTorch.
https://arxiv.org/abs/2201.03283

PDESolver is to solve the Linear Filtering Problem. It has a:
- Details of PDE: E.g. pde_coeffs, initial_condition and time-space domain
- Solver: E.g. BasicNet for the Neural Network
          Auxiliary Diffusion
          Formula for the Stochastic Representation

We also use MC Reference to compute the actual solution by using
a lot of mc samples for each point. (This would be unfeasible in practice)

Assuming everything is 1D otherwise things will break.
"""


import torch
import openturns as ot
from scipy.interpolate import interp1d
import lightning
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR


PLOT_LEVEL = 0
"""
Control what plots are logged
Level 0 - No plots
Level 1 - Plots which occur at most every 100 epochs
Level 2 - Plots which occur at most every 10 epochs
"""


class PDESolver(lightning.LightningModule):
    def __init__(self, pde_coeffs, initial_condition, T, domain, dim,
                 neural_network, auxiliary_diffusion,
                 use_modified_loss=False, num_diffusions_per_input=1,
                 use_mc_reference=False, comparison_resolution=1000):
        """
        Args:
            pde_coeffs (tensor): List containing the pde coefficients.
            initial_condition (function): Function which returns the initial condition
            T (float): Time value we are getting solution for
            domain (tensor): [a, b]
            dim (int): Will be 1 for now
            neural_network (function): Trainable nn.Module
            auxiliary_diffusion (class constructor): Associated Auxiliary Diffusion
            use_modified_loss (bool, optional):
            num_diffusions_per_input (int, optional): How many samples to generate per input
            use_mc_reference (bool, optional): If we should compare against mc_reference
            comparison_resolution: Resolution of domain mesh for comparison
        """
        super().__init__()

        # Details of the PDE
        ## PDE Coefficients
        self.pde_coeffs = torch.tensor(pde_coeffs, dtype=torch.float32)
        ## Time-Space Domain
        self.T        = torch.tensor(T, dtype=torch.float32)
        self.domain   = torch.tensor(domain, dtype=torch.float32)
        self.dim      = dim
        ## Initial Condition
        self.initial_condition = initial_condition

        # Solver
        ## Neural Network
        self.neural_network = neural_network
        ## Auxiliary Diffusion for FK Stochastic Representation
        self.auxiliary_diffusion = auxiliary_diffusion(
            pde_coeffs, num_diffusions_per_input)

        # Option to specify which loss function to use
        self.use_modified_loss = use_modified_loss

        # Performance trackers
        self.training_losses = []
        self.l2_residual = None

        self.use_mc_reference = use_mc_reference
        self.mc_reference = None
        self.comparison_resolution = comparison_resolution

        if use_mc_reference:
            self.mc_reference = MCReference(dim, domain, T,
                                            pde_coeffs, initial_condition)
            self.l2_residual = []

        self.plot_logger = None

    def evaluate_neural_network(self, x):
        """Assumes that x is (num_inputs, self.dim)
        Breaks it up into batch_sizes groups where
        the final is larger than others (as to not create a lone input)
        """
        return self.neural_network(x)

    def functional_nn(self, x):
        """Reshapes x so that it can take in a 1D vector of inputs"""
        # Assumes 1D vector of inputs - non batched
        return self.evaluate_neural_network(x.reshape(-1, 1)).reshape(*x.shape)

    def feynman_kac_stochastic_representation(self, xT):
        """Return the FK Stochastic Representation given
        the Auxiliary Diffusion."""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # Generate points uniformly on the domain
        # Assumes that were are using UniformDomainDataset
        x0 = batch[0]

        ## 1. Neural Network Approximation
        # Shape is (batch_size, dim)
        output = self.evaluate_neural_network(x0)

        ## 2. Feynman-Kac Stochastic Representation of solution
        # Auxiliary diffusion, Only use one simulation per input point
        # Shape (N, self.batch_size, self.dim)
        diffusion = self.auxiliary_diffusion.simulate(x0=x0, T=self.T)
        target = self.feynman_kac_stochastic_representation(diffusion)

        ## 3. Compute the loss
        loss = self.compute_loss(target, output)

        ## 4. Logging stuff for tracking
        plot_logger = PlotLogger(self.logger.experiment)
        self.log("empirical_loss", loss)
        self.training_losses.append(loss.item())

        domain_mesh = torch.linspace(*self.domain, self.comparison_resolution)

        if self.use_mc_reference:
            # L2 Error
            residual = self.functional_nn(domain_mesh) \
                        - self.mc_reference.exact(domain_mesh)
            l2_residual = torch.mean(residual ** 2)
            self.log("l2_residual", l2_residual)
            self.l2_residual.append(l2_residual.item())

            l1_residual = torch.mean(abs(residual))
            self.log("l1_residual", l1_residual)


        if PLOT_LEVEL >= 2:
            # Plot samples over NN and MC reference
            if self.current_epoch % 10 == 0 and self.use_mc_reference:
                step = self.current_epoch // 10
                nn_value = self.functional_nn(domain_mesh)
                mc_value = self.mc_reference.exact(domain_mesh)
                #
                evaluation_points = torch.flatten(x0)
                target_values = target

                plot_logger.plot_target_values(self.current_epoch, step,
                    domain_mesh.detach(), nn_value.detach(), mc_value.detach(),
                    evaluation_points, output.detach(), target_values
                    )


        # Integration is Expensive, only do every 100 epochs.
        if self.current_epoch % 100 == 0:
            step = self.current_epoch // 100
            # Mass of nn which represents a density
            # Neural Network operates on batches
            nn_value = self.functional_nn(domain_mesh)
            mass = torch.trapezoid(nn_value, domain_mesh)
            self.log("mass", mass)

            if self.use_mc_reference:
                # Mass of mc
                mc_value = self.mc_reference.exact(domain_mesh)
                mc_mass = torch.trapezoid(mc_value, domain_mesh)
                self.log("mc_mass", mc_mass)

                if PLOT_LEVEL >= 1:
                    # Density plots
                    plot_logger.plot_density_against_reference(
                        self.current_epoch, step, domain_mesh.detach(),
                        nn_value.detach(), mc_value.detach()
                    )

                    plot_logger.semilogy_density_against_reference(
                        self.current_epoch, step, domain_mesh.detach(),
                        nn_value.detach(), mc_value.detach()
                    )

        if PLOT_LEVEL >= 1:
            if self.current_epoch == 0:
                ic_value = self.initial_condition(domain_mesh)
                ic_mass = torch.trapezoid(ic_value, domain_mesh)

                plot_logger.plot_initial_condition(
                    domain_mesh.detach(), ic_value.detach(), ic_mass
                    )

        return loss

    def compute_loss(self, target, output):
        if self.use_modified_loss:
            # Shape (self.batch_size, self.dim)
            mean_target = torch.mean(target, dim=0)
            ## 3. Compute the loss
            # Shape is (batch_size, dim) by broadcasting
            residual = mean_target - output
            # Shape is (batch_size) after taking the norm
            l2_residual = torch.square(torch.linalg.vector_norm(residual, dim=1))
            # Average over the inputs
            # New shape ()
            loss = torch.mean(l2_residual, dim=0)
        else:
            # Shape is (N, batch_size, dim) by broadcasting
            residual = target - output
            # Shape is (N, batch_size) after taking the norm
            l2_residual = torch.square(torch.linalg.vector_norm(residual, dim=2))
            # Average over the samples per x0.
            # New shape (batch_size)
            mean_over_samples = torch.mean(l2_residual, dim=0)
            # Average over the inputs
            # New shape ()
            loss = torch.mean(mean_over_samples, dim=0)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        # Use Constant Learning Rate
        # # Decay the learning rate by 0.1 every 200 steps
        # scheduler = StepLR(optimizer, step_size=300, gamma=0.7)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler
        #     }
        # }
        return optimizer

    def __call__(self, x):
        """Want it to be called like a normal vectorised function"""
        return self.functional_nn(x)


class LinearFilterPDESolver(PDESolver):
    def __init__(self, ou_coefs, initial_condition, T, domain, dim, neural_network,
                 use_modified_loss=False, num_diffusions_per_input=1,
                 use_mc_reference=False, comparison_resolution=1000):
        # TODO: Is passing class constructor as argument sensible?
        super().__init__(ou_coefs, initial_condition,
                         T, domain, dim,
                         neural_network, AuxiliaryLinearDiffusion,
                         use_modified_loss=use_modified_loss,
                         num_diffusions_per_input=num_diffusions_per_input,
                         use_mc_reference=use_mc_reference,
                         comparison_resolution=comparison_resolution)


    def feynman_kac_stochastic_representation(self, xT):
        """Given Auxiliary Diffusion at time T, return
        phi(\hat{X}_T) \exp(\int_{0}^{T} r(\hat{X}_T) dt)
        """
        M = self.pde_coeffs[0]
        # Tr M = M for 1D
        return self.initial_condition(xT) * torch.exp(-M*self.T)


class BasicNet(nn.Module):
    def __init__(self, neurons, dim, domain):
        super().__init__()

        self.dim = dim
        # TODO: Maybe multiply final output by something to zero it outside domain?
        self.domain = torch.tensor(domain, dtype=torch.float32)

        self.neurons = neurons
        # TODO: Do for loops next time
        self.layers = nn.Sequential(
            ## Part 1
            nn.Linear(dim, neurons[0], bias=True),
            nn.Tanh(),

            ## Part 2 (repeat part 1)
            nn.Linear(neurons[0], neurons[1], bias=True),
            nn.Tanh(),

            ## Part 3 (get scalar output)
            nn.Linear(neurons[1], 1, bias=True),
            # Ensure positive output
            nn.Softplus(),
        )

    def forward(self, x):
        """Assuming that x has shape (batch_size, self.dim)"""
        assert len(x.shape) >= 2, f"{x.shape} needs to be 2D. E.g., (batch_size, self.dim)"
        return self.layers(x)


class AuxiliaryLinearDiffusion:
    def __init__(self, ou_coefs, num_diffusions_per_input):
        self.ou_coefs = torch.tensor(ou_coefs, dtype=torch.float32)
        self.num_diffusions_per_input = num_diffusions_per_input

    def simulate(self, x0, T):
        """Use closed-form formula to simulate auxiliary diffusion at time T.
        Output dimension is (N, *x_0.shape).
        This is typically (N, self.batch_size, self.dim)
        """
        M     = self.ou_coefs[0]
        mean  = self.ou_coefs[1]
        std   = self.ou_coefs[2]

        drift = torch.exp(-M*T) * (x0 - mean)
        noise_variance = (1. - torch.exp(-2.*M*T)) / (2. * M)
        # std is sigma in the paper
        noise = std * torch.sqrt(noise_variance) \
                    * torch.randn(size=(self.num_diffusions_per_input, *x0.shape))

        diffusion = mean + drift + noise
        return diffusion


class UniformDomainDataset(Dataset):
    """This is a bit of hack. Size 1 dataset which
    returns random inputs each time."""
    def __init__(self, domain, mc_sample_size, dim=1, transform=None, use_quasi_mc=False):
        self.domain = domain
        self.mc_sample_size = mc_sample_size
        self.dim = dim
        self.use_quasi_mc = use_quasi_mc

        if use_quasi_mc:
            # Already scaled to be uniform on [a, b]
            power_of_2 = round(np.log2(self.mc_sample_size))
            self.x = self._get_sobol_points(size=power_of_2)

        self.size = 1
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Use this if you want to use Sobol Points
        if self.use_quasi_mc:
            x = self.x
        # Use this if you want to generate pure mc, instead of quasi mc
        else:
            x = torch.rand(size=(self.mc_sample_size, self.dim))
            a, b = self.domain
            # Transform to domain of
            x = a + x*(b-a)

        if self.transform:
            x = self.transform(x)
        return x

    def _get_sobol_points(self, size):
        distribution = ot.ComposedDistribution(
            [ot.Uniform(*self.domain)]*self.dim)
        sequence = ot.SobolSequence(self.dim)
        samplesize = 2**size # Sobol' sequences are in base 2
        experiment = ot.LowDiscrepancyExperiment(sequence, distribution,
                                                 samplesize, False)
        sample = experiment.generate()
        # They used tf.consant, I don't know if this is important
        samp = torch.tensor(sample, dtype=torch.float32)
        return samp


# TODO: Tailored for LinearFilter, I think you just need to change
# _get_sobol_values to use a custom formula as we have for LinearFilterPDESolver
class MCReference:
    """Contains sobol points and diffusion simulator"""
    def __init__(self, dim, domain, T, ou_coefs, initial_condition,
                 num_diffusions_per_point=2**10):
        self.dim = dim
        # Domain needs to be normal number
        self.domain   = domain
        self.T        = torch.tensor(T, dtype=torch.float32)
        self.ou_coefs = torch.tensor(ou_coefs, dtype=torch.float32)
        self.num_diffusions_per_point = num_diffusions_per_point
        self.auxiliary_diffusion = AuxiliaryLinearDiffusion(
            ou_coefs, self.num_diffusions_per_point)
        self.initial_condition = initial_condition

        # Compute the MC Reference (exact) on Sobol Points
        self.sobol_pts = self._get_sobol_points(size=10)
        self.sobol_vals = self._get_sobol_values(initial_condition)
        # Easy to interpolate because we're working on 1D.
        self.exact_interpolated = interp1d(torch.squeeze(self.sobol_pts),
                                            torch.squeeze(self.sobol_vals),
                                            kind='cubic')

    def exact(self, x):
        """do not have an exact sol here... can use a precomputed MC"""
        ret = torch.zeros_like(x, dtype=torch.float32)
        # self.sobol_pts is a vector, there is no need to reduce
        sobol_min = torch.min(self.sobol_pts)
        sobol_max = torch.max(self.sobol_pts)

        min_mask = (x < sobol_min)
        max_mask = x > sobol_max
        reg_mask = (sobol_min <= x) & (x <= sobol_max)

        ret[min_mask] = torch.tensor(self.exact_interpolated(sobol_min),
                                     dtype=torch.float32)
        ret[max_mask] = torch.tensor(self.exact_interpolated(sobol_max),
                                     dtype=torch.float32)
        ret[reg_mask] = torch.tensor(self.exact_interpolated(x[reg_mask]),
                                     dtype=torch.float32)

        return ret

    def _get_sobol_points(self, size):
        distribution = ot.ComposedDistribution(
            [ot.Uniform(*self.domain)]*self.dim)
        sequence = ot.SobolSequence(self.dim)
        samplesize = 2**size # Sobol' sequences are in base 2
        experiment = ot.LowDiscrepancyExperiment(sequence, distribution,
                                                 samplesize, False)
        sample = experiment.generate()
        # They used tf.consant, I don't know if this is important
        samp = torch.tensor(sample, dtype=torch.float32)
        return samp

    def _get_sobol_values(self, phi):
        """MC Reference evaluated on Sobol points.
        Get Expectation of phi(X_T) exp( - Tr M T)
        """
        M = self.ou_coefs[0]
        T = self.T
        exacts = torch.zeros(len(self.sobol_pts))
        # Do one point at a time
        # Cannot simulate 128 * (2**15) simultaneously
        for i, x in enumerate(self.sobol_pts):
            # Sobol point is (1,)
            # So diffusion will have shape (N, 1)
            diffusion = self.auxiliary_diffusion.simulate(x0=x, T=T)
            vals = phi(diffusion)*torch.exp(-M*T)
            exacts[i] = torch.mean(vals)
        return exacts

    def __call__(self, x):
        """Return self.exact(x)"""
        return self.exact(x)


class PlotLogger:
    """Helper class to generate plots during training"""
    def __init__(self, writer):
        self.writer = writer

    def add_image_to_writer(self, fig, name, step):
        """Turn a plot into an image to be put into Tensorboard"""
        fig.canvas.draw()
        rgba_image = np.asarray(fig.canvas.buffer_rgba())
        # https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data
        # Needs to be Channel x Height x Width
        # Originally Height x Width x Channel
        rgba_image = rgba_image.transpose(2, 0, 1)
        self.writer.add_image(name, rgba_image, step)

    def plot_density_against_reference(self, epoch, step,
                                       domain_mesh, nn_value, mc_value):
        fig, ax = plt.subplots()
        ax.plot(domain_mesh, nn_value, color='grey', label="NNet")
        ax.plot(domain_mesh, mc_value, color='red', label="MC")
        ax.set_ylim(-0.25, torch.max(mc_value) * 1.1)
        ax.legend(loc=1)
        # TODO: Fix learning rates later
        ax.set_title(f'Epoch: {epoch:5d} LR: {1e-2:.5f}')
        ax.set_ylabel("Density")
        ax.set_xlabel("Domain")

        self.add_image_to_writer(fig, "Density Plot", step)
        plt.close()

    def semilogy_density_against_reference(self, epoch, step,
                                       domain_mesh, nn_value, mc_value):
        fig, ax = plt.subplots()
        ax.semilogy(domain_mesh, nn_value, color='grey', label="NNet")
        ax.semilogy(domain_mesh, mc_value, color='red', label="MC")
        ax.legend(loc=1)
        # TODO: Fix learning rates later
        ax.set_title(f'Epoch: {epoch:5d} LR: {1e-2:.5f}')
        ax.set_ylabel("Density")
        ax.set_xlabel("Domain")

        self.add_image_to_writer(fig, "Log Scale Density Plot", step)
        plt.close()

    def plot_initial_condition(self, domain_mesh, ic_value, ic_mass):
        fig, ax = plt.subplots()
        ax.plot(domain_mesh, ic_value, color='grey', label="Initial Condition")
        ax.legend(loc=1)
        # TODO: Fix learning rates later
        ax.set_title(f'Mass {ic_mass:.5f}')
        ax.set_ylabel("Density")
        ax.set_xlabel("Domain")

        self.add_image_to_writer(fig, "Initial Condition", 0)
        plt.close()

    def plot_target_values(self, epoch, step,
            domain_mesh, nn_value, mc_value,
            evaluation_points, values, target_values):
        fig, ax = plt.subplots()
        ax.plot(domain_mesh, nn_value, color='grey', label="NNet")
        ax.plot(domain_mesh, mc_value, color='red', label="MC")
        ax.set_ylim(-0.25, torch.max(mc_value) * 1.5)
        # TODO: Fix learning rates later
        ref_l2 = torch.mean(torch.square(mc_value - nn_value))
        obv_l2 = torch.mean(torch.square(target_values - values))
        mod_l2 = torch.mean(torch.square(torch.mean(target_values - values, dim=0)))
        ax.set_title(f'Epoch: {epoch:5d} RefL: {ref_l2:.3f} ObvL: {obv_l2:.3f} ModL: {mod_l2:.3f}')
        ax.set_ylabel("Density")
        ax.set_xlabel("Domain")

        # Assume dim = 1
        # evaluation_points is going to be (batch_size, dim)
        # value is going to be (batch_size, dim)
        # target_values is going to be (N, batch_size, dim)

        ax.scatter(evaluation_points.flatten(), values.flatten(), label="NNet")
        for i in range(target_values.shape[0]):
            ax.scatter(evaluation_points.flatten(), target_values[i].flatten(),
                       label=f"MC_{i}")

        mc_mean = torch.mean(target_values, dim=0)
        ax.scatter(evaluation_points.flatten(), mc_mean.flatten(),
                    label=f"MC_mean", marker="x")

        ax.legend(loc=1)

        self.add_image_to_writer(fig, "Target Values", step)
        plt.close()