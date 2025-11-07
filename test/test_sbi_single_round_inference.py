#  SINGLE-ROUND INFERENCE (INFÉRENCE AMORTIE)
# ------------------------------------------------
# - L'objectif est d'entraîner un modèle général capable d'inférer p(θ | x)
#   pour n'importe quelle observation x (ex: différents patients).
# - On tire les paramètres θ du prior, on simule x, puis on entraîne un
#   réseau de neurones pour approximer le postérieur.
# - Avantage : une fois le modèle entraîné, il est rapide à utiliser
#   pour de nouvelles observations.
# - Inconvénient : nécessite beaucoup de simulations, parfois inutiles,
#   si on s'intéresse seulement à un cas particulier.

#  Utile si tu veux un modèle réutilisable pour plein d'observations différentes.


import torch
import matplotlib.pyplot as plt
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

# espace des dimensions du parametre theta
num_dim = 3

# fonction simulateur de données
def simulator(theta):
    # linear gaussian
    return theta + 1.0 + torch.randn_like(theta) * 0.1

# le prior de theta (de dimension 3 ici du coup)
prior = BoxUniform(low=-2 * torch.ones(num_dim), high=2 * torch.ones(num_dim))

# Check prior, return PyTorch prior. S'assure que les contraintes du modele utilisé sont bien respectés et fait l'echantillonnage
prior, num_parameters, prior_returns_numpy = process_prior(prior)

# Check simulator, returns PyTorch simulator able to simulate batches. Retourne les données simulées à partir des paramètres echantillonnés
simulator = process_simulator(simulator, prior, prior_returns_numpy)

# Consistency check after making ready for sbi. Verifie les dimensions, formats et consistence des données simulées. Souleve une exception en cas de probleme.
check_sbi_inputs(simulator, prior)

# NPE = posterior estimation, ces paramètres sont prior, simulator, number of training steps, batch size, and learning rate
inference = NPE(prior=prior)

# on peut utiliser simulator_for_sbi pour faire en parallele avec joblib
theta, x = simulate_for_sbi(simulator, proposal =prior, num_workers=1, num_simulations = 2000, seed=0) # il existe un argument pour une bar de progression mais ca ne marche que pour les fichieres .py

print("theta.shape", theta.shape)
print("x.shape", x.shape)

# on donne les données a l'inference
inference = inference.append_simulations(theta, x)
density_estimator = inference.train()

# compute the posterior distribution
posterior = inference.build_posterior(density_estimator)


# generate the first observation
theta_1 = prior.sample((1,))
x_obs_1 = simulator(theta_1)
# now generate a second observation
theta_2 = prior.sample((1,))
x_obs_2 = simulator(theta_2)

# visualize the posterior given one observation
posterior_samples_1 = posterior.sample((10000,), x=x_obs_1)

# plot posterior samples
fig, ax = analysis.pairplot(
    posterior_samples_1, limits=[[-2, 2], [-2, 2], [-2, 2]], figsize=(5, 5),
    labels=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"],
    points=theta_1 # add ground truth thetas
)

plt.sca(ax[1, 1])
plt.show()

# assess the quality of the posterior is checking whether parameters sampled from the posterior can reproduce the observation when we simulate data with them.

theta_posterior = posterior.sample((10000,), x=x_obs_1)  # sample from posterior
x_predictive = simulator(theta_posterior)  # simulate data from posterior
fig, ax  = analysis.pairplot(x_predictive,
         points=x_obs_1,  # plot with x_obs as a point
         figsize=(6, 6),
         labels=[r"$x_1$", r"$x_2$", r"$x_3$"])

plt.sca(ax[1, 1])
plt.show()