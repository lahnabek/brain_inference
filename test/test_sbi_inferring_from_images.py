# ---------------------------------------------------------------
# Pourquoi utiliser un réseau d'embedding dans SBI ?
# ---------------------------------------------------------------

# Problème :
# - L'observation x est parfois une donnée de grande dimension ou complexe :
#   image, série temporelle, matrice de connectivité, etc.
# - Les méthodes classiques supposent que x est un vecteur de taille fixe et raisonnable.
# - Si x est trop grand ou structuré (ex : image 2D, signal temporel), la méthode
#   peut devenir inefficace ou impraticable.

# Solution : réseau d'embedding
# - On introduit un réseau d'embedding f(x) qui transforme x en un vecteur de plus
#   petite dimension.
# - Ce vecteur est ensuite utilisé pour apprendre la postérieure.
# - Le réseau est entraîné en même temps que l'inférence, pour capturer les
#   informations les plus utiles de x en vue de prédire θ.

# Avantages :
# - Permet d'utiliser des données complexes comme observations.
# - Réduction de la dimension de x, ce qui améliore l'efficacité de l'entraînement.
# - L'embedding peut apprendre à extraire les caractéristiques pertinentes automatiquement.

# Inconvénients :
# - Complexité supplémentaire : nécessite de définir et entraîner un réseau.
# - Peut introduire un biais si l'embedding est mal adapté à la tâche.
# - Moins interprétable : on perd la structure initiale des données dans f(x).




import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sbi.neural_nets.embedding_nets import CNNEmbedding
from sbi import analysis, utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.neural_nets import posterior_nn


seed = 0
torch.manual_seed(seed)


# simulator 
def simulator_model(parameter, return_points=False):
    """Simulator model with two-dimensional input parameter and 1024-D output

    This simulator serves as a basic example for using a neural net for learning
    summary features. It has only two input parameters but generates
    high-dimensional output vectors. The data is generated as follows:
        (-) Input:  parameter = [r, phi] (1) Generate 100 two-dimensional
        points centered around (r cos(phi),r sin(phi))
            and perturbed by a Gaussian noise with variance 0.01
        (2) Create a grayscale image I of the scattered points with dimensions
            32 by 32
        (3) Perturb I with an uniform noise with values betweeen 0 and 0.2
        (-) Output: I

    Parameters
    ----------
    parameter : array-like, shape (2)
        The two input parameters of the model, ordered as [r, phi]
    return_points : bool (default: False)
        Whether the simulator should return the coordinates of the simulated
        data points as well

    Returns
    -------
    I: torch tensor, shape (1, 1024)
        Output flattened image
    (optional) points: array-like, shape (100, 2)
        Coordinates of the 2D simulated data points

    """
    r = parameter[0]
    phi = parameter[1]

    sigma_points = 0.10
    npoints = 100
    points = []
    for _ in range(npoints):
        x = r * torch.cos(phi) + sigma_points * torch.randn(1)
        y = r * torch.sin(phi) + sigma_points * torch.randn(1)
        points.append([x, y])
    points = torch.as_tensor(points)

    nx = 32
    ny = 32
    sigma_image = 0.20
    im = torch.zeros(nx, ny)
    for point in points:
        pi = int((point[0] - (-1)) / ((+1) - (-1)) * nx)
        pj = int((point[1] - (-1)) / ((+1) - (-1)) * ny)
        if (pi < nx) and (pj < ny):
            im[pi, pj] = 1
    im = im + sigma_image * torch.rand(nx, ny)
    im = im.T
    im = im.reshape(1, -1)

    if return_points:
        return im, points
    else:
        return im


# look at a sample from the simulator of dimension 1024 (32*32)
def simulate_samples_plot():
    # simulate samples
    true_parameter = torch.tensor([0.70, torch.pi / 4])
    x_observed, x_points = simulator_model(true_parameter, return_points=True)

    # plot the observation
    fig, ax = plt.subplots(
        facecolor="white", figsize=(11.15, 5.61), ncols=2, constrained_layout=True
    )
    circle = plt.Circle((0, 0), 1.0, color="k", ls="--", lw=0.8, fill=False)
    ax[0].add_artist(circle)
    ax[0].scatter(x_points[:, 0], x_points[:, 1], s=20)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_xlim(-1, +1)
    ax[0].set_xticks([-1, 0.0, +1.0])
    ax[0].set_ylim(-1, +1)
    ax[0].set_yticks([-1, 0.0, +1.0])
    ax[0].set_title(r"original simulated points with $r = 0.70$ and $\phi = \pi/4$")
    ax[1].imshow(x_observed.view(32, 32), origin="lower", cmap="gray")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("noisy observed data (gray image with 32 x 32 pixels)")
    plt.show()


# choice of an embedding_net : sbi pre-configured MLP CNN and permutation-invariant nn. Here CNN. Ce réseau de convolution est utilisé pour transformer des images en un vecteur de caractéristiques (embedding) plus petit. Cela permet de réduire la complexité tout en extrayant les caractéristiques importantes de l'image pour l'inférence bayésienne.
embedding_net = CNNEmbedding(
    input_shape=(32, 32),
    in_channels=1, # 1 channel for grayscale, 3 for RGB
    out_channels_per_layer=[6], # features de sortie par layers
    num_conv_layers=1, # nombre de layers CNN
    num_linear_layers=1, # nombre de layers linéaire apres le CNN
    output_dim=8, # nombre de features de sortie
    kernel_size=5, # taille des filtres (impair)
    pool_kernel_size=8 # taille de la fenâtre de pooling (réduction de dimension en conservant l'info)
)


# set prior distribution for the parameters
prior = utils.BoxUniform(
    low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 2 * torch.pi])
)

# make a SBI-wrapper on the simulator object for compatibility
prior, num_parameters, prior_returns_numpy = process_prior(prior)
simulator_wrapper = process_simulator(simulator_model, prior, prior_returns_numpy)
check_sbi_inputs(simulator_wrapper, prior)


# instantiate the neural density estimator
neural_posterior = posterior_nn(model="maf", embedding_net=embedding_net) # toujours utilidé mais par default c'est MLP je crois

# setup the inference procedure with NPE
inferer = NPE(prior=prior, density_estimator=neural_posterior)

# run the inference procedure on one round and 10000 simulated data points
theta, x = simulate_for_sbi(simulator_wrapper, prior, num_simulations=10_000)

density_estimator = inferer.append_simulations(theta, x).train(training_batch_size=256)
posterior = inferer.build_posterior(density_estimator)



# visualizing the posterior

# generate posterior samples
true_parameter = torch.tensor([0.50, torch.pi / 4])
x_observed = simulator_model(true_parameter)
samples = posterior.set_default_x(x_observed).sample((50000,))

# create the figure
fig, ax = analysis.pairplot(
    samples,
    points=true_parameter,
    labels=["r", r"$\phi$"],
    limits=[[0, 1], [0, 2 * torch.pi]],
    fig_kwargs=dict(
        points_colors="r",
        points_offdiag={"markersize": 6},
    ),
    figsize=(5, 5),
)

plt.sca(ax[1, 1])
plt.title("Posterior samples")
plt.show()