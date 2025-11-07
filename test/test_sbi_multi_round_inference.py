# MULTI-ROUND INFERENCE (INFÉRENCE NON-AMORTIE, CIBLÉE)
# --------------------------------------------------------
# - On souhaite estimer précisément le postérieur p(θ | x₀) pour UNE observation
#   spécifique (x₀), par exemple un patient donné.
# - L'inférence se fait en plusieurs rounds :
#     1. Simulations avec θ tirés du prior (comme d'hab).
#     2. On échantillonne ensuite les θ depuis le postérieur estimé à x₀.
#     3. On simule à nouveau, puis on réentraîne le modèle.
#     4. On répète ce processus pour affiner le postérieur autour de x₀.
# - Avantage : beaucoup plus efficace et précis si on cible un seul x₀.
# - Inconvénient : le modèle n’est pas réutilisable pour d’autres observations.

# Utile si tu veux une estimation très précise des paramètres pour une seule observation.

import torch
import matplotlib.pyplot as plt
from sbi.analysis import pairplot
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

# 2 rounds: first round simulates from the prior (like single round inference), second round simulates parameter set
# that were sampled from the obtained posterior (reduce the prior for this specific observation).
num_rounds = 2
num_dim = 3

# The specific observation we want to focus the inference on.
x_o = torch.zeros(num_dim,)
prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))
simulator = lambda theta: theta + torch.randn_like(theta) * 0.1

# Ensure compliance with sbi's requirements.
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator = process_simulator(simulator, prior, prior_returns_numpy)
check_sbi_inputs(simulator, prior)

inference = NPE(prior)

# start rounds
posteriors = []
proposal = prior

for _ in range(num_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=500)

    # In `SNLE` and `SNRE`, you should not pass the `proposal` to
    # `.append_simulations()`
    density_estimator = inference.append_simulations(
        theta, x, proposal=proposal).train()

    # The proposal is the posterior from the previous round.
    posterior = inference.build_posterior(density_estimator)
    posteriors.append(posterior)
    # nouveau prior à partir de la posterieur conditionné à x_o specifiquement
    proposal = posterior.set_default_x(x_o)


# Plot of the posterior conditional to x_o (no longer amortized)
posterior_samples = posterior.sample((10000,), x=x_o)

# plot posterior samples
fig, ax = pairplot(
    posterior_samples, points = x_o, limits=[[-2, 2], [-2, 2], [-2, 2]], figsize=(5, 5)
)

plt.sca(ax[1, 1])
plt.show()


