"""
This module provides inference pipelines using the `sbi` library for simulation-based Bayesian inference.

It contains:
- `make_prior`: Build a BoxUniform prior given bounds and parameter types.
- `single_round_inference`: Perform amortized inference using a single simulation round.
- `multi_round_inference`: Perform targeted inference using sequential rounds (non-amortized).
- `compare_pca`: Compare PCA embeddings between simulated and real data for visualization and evaluation.
"""



from sbi.neural_nets.embedding_nets import (
    FCEmbedding,
    CNNEmbedding, # currently proposed for CNN-based embedding, litterature recommends PCA anyway but sbi does not support PCA embedding
    PermutationInvariantEmbedding
)
import os
import torch

from utils import *
from functools import partial
from params import *
from sbi import *
import torch.nn as nn
from test_pca import load_data_with_labels, split_groups
from embedding_pca import EmbeddingNetWithPCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr


#############" fonction inferences ###############


def make_prior(type=None, borne_min=None, borne_max=None, list_inferer=None):
    """
    Creates a BoxUniform prior distribution over all parameters.

    Parameters
    ----------
    type : list of str
        List specifying the prior type ("uniform" or "gaussian") for each parameter.

    borne_min : torch.Tensor
        Lower bounds for each parameter.

    borne_max : torch.Tensor
        Upper bounds for each parameter.

    list_inferer : list of str
        Names of the parameters to infer.

    Returns
    -------
    prior : BoxUniform
        Uniform (or approximated Gaussian) prior defined over parameter bounds.
    """
    if type is None or borne_min is None or borne_max is None or list_inferer is None:
        raise ValueError("All arguments (type, borne_min, borne_max, list_inferer) are required.")

    if len(type) != len(list_inferer):
        raise ValueError("The 'type' list must be the same length as 'list_inferer'.")

    low_list = []
    high_list = []

    for i, t in enumerate(type):
        if t == 'uniform':
            low_list.append(borne_min[i])
            high_list.append(borne_max[i])
        elif t == 'gaussian':
            mean = (borne_min[i] + borne_max[i]) / 2.0
            std = (borne_max[i] - borne_min[i]) / 4.0  # covers 95% in ±2σ
            low_list.append(mean - 2 * std)
            high_list.append(mean + 2 * std)
        else:
            raise ValueError(f"Unsupported prior type '{t}' for parameter {list_inferer[i]}. Choose 'uniform' or 'gaussian'.")

    low_tensor = torch.tensor(low_list, dtype=torch.float32)
    high_tensor = torch.tensor(high_list, dtype=torch.float32)

    prior = BoxUniform(low=low_tensor, high=high_tensor)

    return prior


def compare_pca(Z_sim, Z_real, pca_sim, pca_real, EXP_NAME, save_dir="results"):
    """
    Compares PCA decompositions between simulated and real data.

    Saves cumulative explained variance, cosine similarity of eigenvectors,
    and Pearson correlation of projected scores.

    Parameters
    ----------
    Z_sim : ndarray
        PCA projections of simulated data.

    Z_real : ndarray
        PCA projections of real data.

    pca_sim : PCA object
        PCA fitted on simulated data.

    pca_real : PCA object
        PCA fitted on real data.

    EXP_NAME : str
        Name of the experiment (for output directory).

    save_dir : str
        Directory where results will be saved.
    """

    output_dir = os.path.join(save_dir, EXP_NAME, "pca_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # (1) Variance expliquée cumulée
    explained_sim = np.cumsum(pca_sim.explained_variance_ratio_)
    explained_real = np.cumsum(pca_real.explained_variance_ratio_)

    plt.figure()
    plt.plot(explained_sim, label="Simulations", color='orange')
    plt.plot(explained_real, label="Données réelles", color='blue')
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("Variance expliquée (cumulée)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "variance_expliquee_cumulee.png"))
    plt.close()

    # (2) Cosine similarity entre vecteurs propres
    angles = []
    for i in range(min(pca_sim.components_.shape[0], pca_real.components_.shape[0])):
        angle = 1 - cosine(pca_sim.components_[i], pca_real.components_[i])
        angles.append(angle)

    plt.figure()
    plt.plot(angles, marker='o')
    plt.xlabel("Composante principale")
    plt.ylabel("Similarité cosinus (1 = alignés)")
    plt.title("Alignement des vecteurs propres")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "cosine_similarity_eigenvectors.png"))
    plt.close()

    # (3) Corrélation de Pearson entre les scores projetés
    pearson_corrs = []
    max_components = min(Z_sim.shape[1], Z_real.shape[1])
    for i in range(max_components):
        r, _ = pearsonr(Z_sim[:, i], Z_real[:, i])
        pearson_corrs.append(r)

    plt.figure()
    plt.bar(range(1, max_components + 1), pearson_corrs)
    plt.xlabel("Composante principale")
    plt.ylabel("Corrélation de Pearson")
    plt.title("Corrélation des scores projetés (PCA)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "correlation_scores_projetes.png"))
    plt.close()

    # (4) Sauvegarde des valeurs dans un fichier texte
    with open(os.path.join(output_dir, "pca_metrics.txt"), "w") as f:
        f.write("Variance expliquée cumulée (sim):\n")
        f.write(np.array2string(explained_sim, precision=4))
        f.write("\n\nVariance expliquée cumulée (réelle):\n")
        f.write(np.array2string(explained_real, precision=4))
        f.write("\n\nCosine similarity (vecteurs propres):\n")
        f.write(np.array2string(np.array(angles), precision=4))
        f.write("\n\nCorrélation de Pearson (scores):\n")
        f.write(np.array2string(np.array(pearson_corrs), precision=4))

    print(f"[✓] Analyse PCA enregistrée dans : {output_dir}")


def single_round_inference(
    simulator_func,
    params_obs,
    list_inferer,
    borne_min,
    borne_max,
    MODEL_NN,
    EXP_NAME,
    NUM_SIMULATION,
    NUM_SAMPLES,
    REAL_OBS,
    PRIOR_TYPE,
    CTRL,
    SUBJ_IDX,
    USE_PATIENT_SC,
    RANDOM_SC,
    USE_QI_CHANGE,
    pca_model=None,
    save_dir="results",
    n_components_pca=10,
    use_pca_on_simulated_data=False,
    use_cnn=False
):
    """
    Runs amortized (single-round) simulation-based inference using the sbi library.

    Parameters
    ----------
    simulator_func : callable
        Function that simulates data given a vector of parameters. It must accept a batch of parameters
        and return simulated features or signals compatible with embedding strategies.

    params_obs : dict
        Parameters or settings used to generate the observation (e.g., subject index, control flag, etc.).
        This is passed to `simulator_func` for the observation generation.

    list_inferer : list of str
        Names of parameters to infer (must match order and names expected in prior and simulator).

    borne_min : torch.Tensor
        Lower bounds for each parameter to infer, used to define the uniform prior.

    borne_max : torch.Tensor
        Upper bounds for each parameter to infer, used to define the uniform prior.

    MODEL_NN : str
        Neural density estimator type, one of {"maf", "nsf", "mdn"} supported by sbi.

    EXP_NAME : str
        Identifier for the experiment (used to name output directories and files).

    NUM_SIMULATION : int
        Number of simulations to perform for training the inference network.

    NUM_SAMPLES : int
        Number of posterior samples to draw once inference is complete.

    REAL_OBS : bool
        If True, the observation is drawn from real data. If False, the observation is simulated using params_obs.

    PRIOR_TYPE : list of str
        Type of prior per parameter: typically "uniform" or "gaussian". Currently only "uniform" is supported.

    CTRL : bool
        Whether to simulate/control for "control" subjects.

    SUBJ_IDX : int
        Subject index to use for generating the observation.

    USE_PATIENT_SC : bool
        Whether to use the patient-specific structural connectivity (SC).

    RANDOM_SC : bool
        If True, use randomized SC in the simulation.

    USE_QI_CHANGE : bool
        If True, apply changes in quality index to the simulation.

    pca_model : sklearn.decomposition.PCA or None
        If not None, apply this PCA model as a feature embedding to reduce dimension.

    save_dir : str, optional
        Root directory where results will be saved. Default is "results".

    n_components_pca : int, optional
        Number of PCA components to retain if PCA is applied to simulated data. Default is 10.

    use_pca_on_simulated_data : bool, optional
        If True, apply PCA on the simulated features before posterior training. Otherwise, use raw features.

    use_cnn : bool, optional
        If True, use a CNN-based embedding of simulated signals instead of PCA or raw features.

    Returns
    -------
    posterior : sbi posterior object
        Trained posterior distribution approximator.

    samples : torch.Tensor
        Samples drawn from the trained posterior given the observation.

    """

    seed = params_obs['seed']

    print("[1] Vérification du simulateur")
    if simulator_func is None:
        raise ValueError("Aucun simulateur spécifié. Veuillez fournir une fonction de simulation.")
    
    print("[2] Définition du prior")
    prior = make_prior(type=PRIOR_TYPE, borne_min=borne_min, borne_max=borne_max, list_inferer=list_inferer)
    prior, _, prior_returns_numpy = process_prior(prior)

    print("[3] Préparation du simulateur SBI")
    simulator_wrapped = partial(
        simulator_func,
        params=default_params,
        list_inferer=list_inferer,
        kernel_hrf=kernel,
        CTRL=CTRL,
        SUBJ_IDX=SUBJ_IDX,
        USE_PATIENT_SC=USE_PATIENT_SC,
        RANDOM_SC=RANDOM_SC,
        REAL_OBS=REAL_OBS,
        USE_QI_CHANGE = USE_QI_CHANGE
    )
    simulator = process_simulator(simulator_wrapped, prior, prior_returns_numpy)

    print(f"[4] Génération des données d'entraînement avec {NJOBS} workers")
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=NUM_SIMULATION, num_workers=NJOBS, seed=seed)
    theta, x = theta.to(torch.float32), x.to(torch.float32)
    print(f"    theta.shape = {theta.shape}  # attendu: (NUM_SIMULATION, dim_param)")
    print(f"    x.shape = {x.shape}          # attendu: (NUM_SIMULATION, dim_observation)")
    print("[✓] Simulations générées")

    

    if use_pca_on_simulated_data:
        print("[NOT embedding] PCA sur les données simulées avant l'embedding identity")
        try:
            print(" PCA sur les données simulées")
            x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
            pca_sim = PCA(n_components=n_components_pca)
            Z_sim = pca_sim.fit_transform(x_np)
            print(f"    Z_sim.shape = {Z_sim.shape}  # Projections PCA des simulations")
            # transforme Z_sim en tensor pour la suite
            x = torch.tensor(Z_sim, dtype=torch.float32)

            # adapte si nécessaire
            X, y = load_data_with_labels(f"generated_data/FC_data_flattened_{PARC_SIZE}.npz")
            X_ctrl, _ = split_groups(X, y)
            pca_real = PCA(n_components=10).fit(X_ctrl)
            Z_real = pca_real.transform(X_ctrl)
            compare_pca(Z_sim, Z_real, pca_sim, pca_real, EXP_NAME)
        except Exception as e:
            print("[!] Erreur dans la comparaison PCA :", e)

    
    # Suppose que pca_real a été entraînée sur les données réelles
    elif pca_model is not None:
        print("[embedding] Utilisation de PCA pré-entraînée sur les données réelles")
        embedding_net = EmbeddingNetWithPCA(pca_model=pca_model, apply_mlp=False)
    

    
    elif use_cnn:
        # Transforme x (tensor de vecteurs) en un tensor 3D pour CNN
        print("[embedding] using CNN embedding")
        cnn_x = []
        for FC_vect in x:
            FC_mtx = vector_to_symmetric_matrix(FC_vect, size=PARC_SIZE)
            cnn_x.append(FC_mtx)
        cnn_x = torch.stack(cnn_x)  # Combine les matrices en un seul tensor 3D
        x = cnn_x

        embedding_net = CNNEmbedding(
            input_shape=(PARC_SIZE, PARC_SIZE),  # Forme de l'entrée pour le CNN
            in_channels=1, # Nombre de canaux d'entrée (1 pour les matrices FC)
            out_channels_per_layer=[6], # Nombre de canaux de sortie pour chaque couche
            num_conv_layers=1, 
            num_linear_layers=1,
            output_dim=10,  # Dimension de sortie de l'embedding
            kernel_size=5,
            pool_kernel_size=8
            )   
    else:
        print("[embedding] Aucune PCA ou CNN spécifié, utilisation d'un réseau d'embedding vide")
        embedding_net = nn.Identity()

    print("[5] Entraînement du modèle de densité")
    density_estimator = posterior_nn(model=MODEL_NN, embedding_net=embedding_net)
    inference = NPE(prior=prior, density_estimator=density_estimator)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(show_train_summary=True)
    posterior = inference.build_posterior(density_estimator)
    print("[✓] Modèle entraîné et postérieur construit")

    print("[6] Génération de l'observation")
    if REAL_OBS:
        theta_obs, FC_obs = generate_observation(
            params=params_obs,
            simulator=simulator,
            list_inferer=list_inferer,
            REAL_OBS=REAL_OBS,
            CTRL=CTRL,
            SUBJ_IDX=SUBJ_IDX,
            RANDOM_SC=RANDOM_SC,
            USE_PATIENT_SC=USE_PATIENT_SC,
            
            data_path=f"generated_data/FC_data_flattened_{PARC_SIZE}.npz")
        # theta_obs numpy array attendu: (len(list_inferer),)
    else:
        theta_dict, FC_obs = generate_observation(
            params=params_obs,
            simulator=simulator,
            list_inferer=list_inferer,
            REAL_OBS=REAL_OBS,
            CTRL=CTRL,
            SUBJ_IDX=SUBJ_IDX,
            RANDOM_SC=RANDOM_SC,
            USE_PATIENT_SC=USE_PATIENT_SC,
            data_path=f"generated_data/FC_data_flattened_{PARC_SIZE}.npz")
        theta_obs = torch.tensor([theta_dict[p] for p in list_inferer], dtype=torch.float32).numpy()
    print(f"    FC_obs.shape = {FC_obs.shape}  # attendu: (dim_observation,) ou (nb_regions, nb_variables)")
    print(f"    theta_obs = {theta_obs}        # attendu: (len(list_inferer),)")
    print("[✓] Observation générée")

    print("[8] Inférence postérieure...")
    posterior, theta_posterior_samples, x_predictive_all = compute_predictions_from_posterior(
        posterior,
        FC_obs,
        simulator=simulator,
        N_samples=NUM_SAMPLES,
    )
    posterior.set_default_x(FC_obs)  # Définit FC_obs comme observation par défaut pour le postérieur
    # theta_posterior_samples.shape attendu : (NUM_SAMPLES, dim_param)
    # x_predictive_all.shape attendu : (NUM_SAMPLES, dim_observation)
    print(f"    theta_posterior_samples.shape = {theta_posterior_samples.shape}")
    print(f"    x_predictive_all.shape = {x_predictive_all.shape}")

    if isinstance(theta_posterior_samples, torch.Tensor):
        posterior_samples_list = theta_posterior_samples.detach().cpu().numpy()
    else:
        posterior_samples_list = theta_posterior_samples

    print(f"    posterior_samples_list.shape = {posterior_samples_list.shape}  # attendu: (NUM_SAMPLES, dim_param)")
    print(f"    Nombre d'inférences : {len(list_inferer)}")
    print(f"    REAL_OBS = {REAL_OBS}")

    print("[9] Évaluation et log de l'expérience")
    os.makedirs(f"{save_dir}/{EXP_NAME}", exist_ok=True)
    evaluate_and_log_experiment(
        posterior,              # postérieur complet
        x_predictive_all,       # simulations prédites (shape: [N_samples, dim])
        posterior_samples_list, # liste d'échantillons postérieurs (numpy array)
        FC_obs,                 # observation
        theta_obs,              # vrai paramètre si connu (sinon None)
        exp_name=EXP_NAME,
        list_inferer=list_inferer,
        num_samples=NUM_SAMPLES,
        num_simulations=NUM_SIMULATION,
        borne_min=borne_min,
        borne_max=borne_max,
        real_obs=REAL_OBS,
    )
    print("[✓] Inférence complète terminée")


def multi_round_inference(
    simulator_func,
    params_obs,
    list_inferer,
    borne_min,
    borne_max,
    MODEL_NN,
    EXP_NAME,
    NUM_SIMULATION,
    NUM_SAMPLES,
    REAL_OBS,
    PRIOR_TYPE,
    CTRL,
    SUBJ_IDX,
    USE_PATIENT_SC,
    RANDOM_SC,
    USE_QI_CHANGE,
    pca_model=None,
    save_dir="results",
    n_components_pca=10,
    use_pca_on_simulated_data=False,
    use_cnn=False,
    NUM_ROUNDS=3
):
    """
    Runs sequential simulation-based inference (multi-round NPE) for a single observation.

    Parameters
    ----------
    simulator_func : callable
        Function that returns simulated data given parameter values.

    params_obs : dict
        Parameters for generating the observation (or seeds).

    list_inferer : list of str
        Names of parameters to infer.

    borne_min : torch.Tensor
        Lower bounds of prior for each parameter.

    borne_max : torch.Tensor
        Upper bounds of prior for each parameter.

    MODEL_NN : str
        Neural density estimator architecture to use ("maf", "mdn", etc.).

    EXP_NAME : str
        Name of the experiment (used for saving results).

    NUM_SIMULATION : int
        Number of simulations per round.

    NUM_SAMPLES : int
        Number of posterior samples to generate.

    REAL_OBS : bool
        Whether the observation comes from real data.

    PRIOR_TYPE : list of str
        List indicating prior type for each parameter ("uniform" or "gaussian").

    CTRL : bool
        Whether to use control subjects in the simulation.

    SUBJ_IDX : int
        Subject index to use for real or synthetic observations.

    USE_PATIENT_SC : bool
        Whether to use patient structural connectivity.

    RANDOM_SC : bool
        Whether to randomize structural connectivity.

    USE_QI_CHANGE : bool
        Whether to apply quality index change in simulation.

    pca_model : sklearn PCA or None
        If provided, applies the PCA model as embedding.

    save_dir : str
        Directory where results are saved.

    n_components_pca : int
        Number of PCA components if PCA is used on simulated data.

    use_pca_on_simulated_data : bool
        Whether to apply PCA directly to simulated data (before embedding).

    use_cnn : bool
        Whether to use a CNN-based embedding.

    NUM_ROUNDS : int
        Number of sequential simulation rounds.
    """


    seed = params_obs['seed']

    print("[1] Vérification du simulateur")
    if simulator_func is None:
        raise ValueError("Aucun simulateur spécifié. Veuillez fournir une fonction de simulation.")
    
    print("[2] Définition du prior")
    prior = make_prior(type=PRIOR_TYPE, borne_min=borne_min, borne_max=borne_max, list_inferer=list_inferer)
    prior, _, prior_returns_numpy = process_prior(prior)

    print("[3] Préparation du simulateur SBI")
    simulator_wrapped = partial(
        simulator_func,
        params=default_params,
        list_inferer=list_inferer,
        kernel_hrf=kernel,
        CTRL=CTRL,
        SUBJ_IDX=SUBJ_IDX,
        USE_PATIENT_SC=USE_PATIENT_SC,
        RANDOM_SC=RANDOM_SC,
        REAL_OBS=REAL_OBS,
        USE_QI_CHANGE = USE_QI_CHANGE
    )
    simulator = process_simulator(simulator_wrapped, prior, prior_returns_numpy)

    
    
    print("[4] Génération de l'observation")
    if REAL_OBS:
        theta_obs, FC_obs = generate_observation(
            params=params_obs,
            simulator=simulator,
            list_inferer=list_inferer,
            REAL_OBS=REAL_OBS,
            CTRL=CTRL,
            SUBJ_IDX=SUBJ_IDX,
            RANDOM_SC=RANDOM_SC,
            USE_PATIENT_SC=USE_PATIENT_SC,
            
            data_path=f"generated_data/FC_data_flattened_{PARC_SIZE}.npz")
        # theta_obs numpy array attendu: (len(list_inferer),)
    else:
        theta_dict, FC_obs = generate_observation(
            params=params_obs,
            simulator=simulator,
            list_inferer=list_inferer,
            REAL_OBS=REAL_OBS,
            CTRL=CTRL,
            SUBJ_IDX=SUBJ_IDX,
            RANDOM_SC=RANDOM_SC,
            USE_PATIENT_SC=USE_PATIENT_SC,
            data_path=f"generated_data/FC_data_flattened_{PARC_SIZE}.npz")
        theta_obs = torch.tensor([theta_dict[p] for p in list_inferer], dtype=torch.float32).numpy()
    print(f"    FC_obs.shape = {FC_obs.shape}  # attendu: (dim_observation,) ou (nb_regions, nb_variables)")
    print(f"    theta_obs = {theta_obs}        # attendu: (len(list_inferer),)")
    print("[✓] Observation générée")


    # prepare the embedding network
    if use_pca_on_simulated_data:
        print("[NOT embedding] PCA sur les données simulées avant l'embedding identity")
        try:
            print(" PCA sur les données simulées")
            x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
            pca_sim = PCA(n_components=n_components_pca)
            Z_sim = pca_sim.fit_transform(x_np)
            print(f"    Z_sim.shape = {Z_sim.shape}  # Projections PCA des simulations")
            # transforme Z_sim en tensor pour la suite
            x = torch.tensor(Z_sim, dtype=torch.float32)

            # adapte si nécessaire
            X, y = load_data_with_labels(f"generated_data/FC_data_flattened_{PARC_SIZE}.npz")
            X_ctrl, _ = split_groups(X, y)
            pca_real = PCA(n_components=10).fit(X_ctrl)
            Z_real = pca_real.transform(X_ctrl)
            compare_pca(Z_sim, Z_real, pca_sim, pca_real, EXP_NAME)
        except Exception as e:
            print("[!] Erreur dans la comparaison PCA :", e)

    
    # Suppose que pca_real a été entraînée sur les données réelles
    elif pca_model is not None:
        print("[embedding] Utilisation de PCA pré-entraînée sur les données réelles")
        embedding_net = EmbeddingNetWithPCA(pca_model=pca_model, apply_mlp=False)
    
    elif use_cnn:
        # Transforme x (tensor de vecteurs) en un tensor 3D pour CNN
        print("[embedding] using CNN embedding")
        cnn_x = []
        for FC_vect in x:
            FC_mtx = vector_to_symmetric_matrix(FC_vect, size=PARC_SIZE)
            cnn_x.append(FC_mtx)
        cnn_x = torch.stack(cnn_x)  # Combine les matrices en un seul tensor 3D
        x = cnn_x

        embedding_net = CNNEmbedding(
            input_shape=(PARC_SIZE, PARC_SIZE),  # Forme de l'entrée pour le CNN
            in_channels=1, # Nombre de canaux d'entrée (1 pour les matrices FC)
            out_channels_per_layer=[6], # Nombre de canaux de sortie pour chaque couche
            num_conv_layers=1, 
            num_linear_layers=1,
            output_dim=10,  # Dimension de sortie de l'embedding
            kernel_size=5,
            pool_kernel_size=8
            )   
    else:
        print("[embedding] Aucune PCA ou CNN spécifié, utilisation d'un réseau d'embedding vide")
        embedding_net = nn.Identity()


    print(f"[5] Génération des données d'entraînement avec {NJOBS} workers et {NUM_ROUNDS} rounds")
    density_estimator = posterior_nn(model=MODEL_NN, embedding_net=embedding_net)
    inference = NPE(prior=prior, density_estimator=density_estimator)
    proposal = prior
    posteriors = []

    for round_idx in range(NUM_ROUNDS):
        print(f"\n🔁 Round {round_idx+1}/{NUM_ROUNDS}")
        theta, x = simulate_for_sbi(simulator, proposal=proposal, num_simulations=NUM_SIMULATION, num_workers=NJOBS, seed=seed)
        theta, x = theta.to(torch.float32), x.to(torch.float32)
        inference = inference.append_simulations(theta, x, proposal=proposal)
        density_estimator = inference.train(show_train_summary=True)
        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(FC_obs)
    print(f"    theta.shape = {theta.shape}  # attendu: (NUM_SIMULATION, dim_param)")
    print(f"    x.shape = {x.shape}          # attendu: (NUM_SIMULATION, dim_observation)")
    print("[✓] Simulations générées")

    # 5. Inférence postérieure finale
    final_posterior = posteriors[-1]

    print("[8] Inférence postérieure...")
    posterior, theta_posterior_samples, x_predictive_all = compute_predictions_from_posterior(
        final_posterior,
        FC_obs,
        simulator=simulator,
        N_samples=NUM_SAMPLES,
    )
    posterior.set_default_x(FC_obs)  # Définit FC_obs comme observation par défaut pour le postérieur
    # theta_posterior_samples.shape attendu : (NUM_SAMPLES, dim_param)
    # x_predictive_all.shape attendu : (NUM_SAMPLES, dim_observation)
    print(f"    theta_posterior_samples.shape = {theta_posterior_samples.shape}")
    print(f"    x_predictive_all.shape = {x_predictive_all.shape}")

    if isinstance(theta_posterior_samples, torch.Tensor):
        posterior_samples_list = theta_posterior_samples.detach().cpu().numpy()
    else:
        posterior_samples_list = theta_posterior_samples

    print(f"    posterior_samples_list.shape = {posterior_samples_list.shape}  # attendu: (NUM_SAMPLES, dim_param)")
    print(f"    Nombre d'inférences : {len(list_inferer)}")
    print(f"    REAL_OBS = {REAL_OBS}")

    print("[9] Évaluation et log de l'expérience")
    os.makedirs(f"{save_dir}/{EXP_NAME}", exist_ok=True)
    evaluate_and_log_experiment(
        posterior,              # postérieur complet
        x_predictive_all,       # simulations prédites (shape: [N_samples, dim])
        posterior_samples_list, # liste d'échantillons postérieurs (numpy array)
        FC_obs,                 # observation
        theta_obs,              # vrai paramètre si connu (sinon None)
        exp_name=EXP_NAME,
        list_inferer=list_inferer,
        num_samples=NUM_SAMPLES,
        num_simulations=NUM_SIMULATION,
        borne_min=borne_min,
        borne_max=borne_max,
        real_obs=REAL_OBS,
    )
    print("[✓] Multi round inférence complète terminée")




