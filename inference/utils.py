
"""
Module for simulation, posterior analysis, and visualization of functional connectivity data. All relevant function to use during the inference process are included here.

This module includes:
- Simulation and observation generation of BOLD and FC data.
- Utility functions for matrix/vector conversion and data extraction.
- Posterior sample analysis: diagnostics, plotting, calibration.
- Evaluation metrics and experiment logging.
- PCA analysis on real FC datasets.

Sections:
1. Imports and global configuration
2. Simulation and observation functions
3. Data transformation utilities
4. Posterior and inference analysis functions
5. Visualization and plotting utilities
6. Evaluation and logging functions
7. PCA and real data analysis

Usage:
Import and call functions as needed for simulation, inference, diagnostics, and visualization
within a neuroimaging / SBI pipeline context.
"""

# ---------------------------
# 1. Imports and Global Config
# ---------------------------

import sys
import os
import warnings
from datetime import datetime
import json
import random
import zipfile

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from scipy.stats import skew, kurtosis, entropy, pearsonr
from scipy.spatial.distance import mahalanobis
from scipy import signal

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.neural_nets import posterior_nn
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from tqdm import tqdm
from tqdm_joblib import tqdm_joblib  # pip install tqdm_joblib
import joblib

from inference.params import kernel, default_params, NJOBS, PARC_SIZE

# Internal imports
from tvb_model_reference.simulation_file.parameter.parameter_M_Berlin_dt01s import Parameter  # type: ignore
import tvb_model_reference.simulation_file.nuu_tools_simulation_human as tools  # type: ignore

# Add project root to PYTHONPATH for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize global parameters object
parameters = Parameter()



# ---------------------------------
# 2. Simulation and Observation Functions
# ---------------------------------




def generate_BOLD_filename(params, USE_QI_CHANGE, folder_root='save_file/'):
    """
    Generate a filename for the BOLD signal file based on model parameters.

    Parameters
    ----------
    params : dict
        Dictionary containing all model parameters required to generate the filename.
        Must include keys like 'Qi', 'ratio_coupling_EE_EI', 'coupling_strength',
        'seed', 'Iext', and if `USE_QI_CHANGE` is True, also 'nodes_Qichange',
        'node_name_Qichange', and each 'Qi_change_curr_i'.
    USE_QI_CHANGE : bool
        Whether to include Qi_change information in the filename.
    folder_root : str, optional
        Base directory where the file will be saved. Default is 'save_file/'.

    Returns
    -------
    str
        Full path to the generated BOLD filename.
    """
    if USE_QI_CHANGE:
        Qi_change_curr = [params[f"Qi_change_curr_{i}"] for i in range(len(params['nodes_Qichange']))]
        filename = (
            f"sig_BOLD_b_0_Qi_{params['Qi']}"
            f"{params['node_name_Qichange']}{Qi_change_curr}"
            f"EtoEIratio{params['ratio_coupling_EE_EI']}_coupling{params['coupling_strength']}"
            f"seed{params['seed']}_noise{params['Iext']}.npy"
        )
    else:
        filename = (
            f"sig_BOLD_b_0_Qi_{params['Qi']}"
            f"EtoEIratio{params['ratio_coupling_EE_EI']}_coupling{params['coupling_strength']}"
            f"seed{params['seed']}_noise{params['Iext']}.npy"
        )

    return os.path.join(folder_root, filename)


def simulator_BOLD(kernel_hrf, params, SUBJ_IDX, CTRL, RANDOM_SC, USE_PATIENT_SC, REAL_OBS, USE_QI_CHANGE):
    """
    Run a brain network simulation and generate corresponding BOLD signals.

    This function simulates neural activity using a connectome-based model configured by the given parameters.
    It applies a hemodynamic response function (HRF) convolution to compute BOLD signals from neural activity
    and returns the downsampled BOLD time series. Optionally, the function saves the resulting BOLD data to disk.

    Parameters
    ----------
    kernel_hrf : np.ndarray
        Hemodynamic response function (HRF) kernel used for convolution with simulated activity.
    params : dict
        Dictionary of simulation parameters, including:
            - 'Qi' : float
                Baseline global adaptation parameter.
            - 'Qi_change_curr_i' : float
                Adaptation parameter for node(s) to modify, for each i.
            - 'node_name_Qichange' : str
                Description of the modified node group.
            - 'nodes_Qichange' : list of int or list of list of int
                Indices of regions where Qi is modified.
            - 'ratio_coupling_EE_EI' : float
                Ratio of E→E to E→I inter-regional coupling.
            - 'coupling_strength' : float
                Global scaling factor for structural connectivity.
            - 'Iext' : float
                External input current (stimulus or noise).
            - 'seed' : int
                Random seed for initialization.
            - 'add_transient' : int
                Extra time (ms) discarded at start to remove transients.
            - 'cut_time' : int
                Duration (ms) of warm-up period to discard.
            - 'run_time' : int
                Total duration (ms) of recorded simulation after warm-up.
            - 'save_BOLD' : bool
                Whether to save the generated BOLD signal.
    SUBJ_IDX : int
        Subject index for selecting patient-specific connectome data.
    CTRL : bool
        If True, use control subject data when `USE_PATIENT_SC` is enabled.
    RANDOM_SC : bool
        If True, generate randomized structural connectivity for control.
    USE_PATIENT_SC : bool
        If True, use patient-specific structural connectivity data.
    REAL_OBS : bool
        Whether the observed patient data is real (not synthetic).
    USE_QI_CHANGE : bool
        Whether to apply a custom Qi value to selected nodes.

    Returns
    -------
    BOLD_subsamp : np.ndarray
        Downsampled BOLD signal of shape [n_regions, n_timepoints].

    Raises
    ------
    ValueError
        If the number of nodes in the connectome is insufficient.
    FileNotFoundError
        If a required result file is not found during post-processing.
    """

    

    # Extract parameters
    #print("[simulator_BOLD] Début de la simulation BOLD", flush=True)

    Qi = params['Qi']
    ratio_coupling_EE_EI = params['ratio_coupling_EE_EI']
    coupling_strength = params['coupling_strength']
    seed = params['seed']
    Iext = params['Iext']
    node_name_Qichange = params['node_name_Qichange']
    nodes_Qichange = params['nodes_Qichange']
    add_transient = params['add_transient']
    cut_time = params['cut_time']
    run_time = params['run_time']
    if USE_QI_CHANGE:
        Qi_change_curr = []
        for i in range(len(params['nodes_Qichange'])):
            Qi_change_curr.append(params[f"Qi_change_curr_{i}"])


    # Seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    

    # time of simulation (we don't really to cut a transcient when using the mode 'valid' of convolution )
    cut_transient = cut_time + add_transient  # ms to discard initially
    run_sim = run_time + cut_transient  # total simulation time in ms

    # Handle patient-specific structural connectivity if enabled
    if USE_PATIENT_SC and REAL_OBS:
        sc_paths = extract_SC_patient(SUBJ_IDX,CTRL,RANDOM_SC)
        parameters.parameter_connection_between_region["path_weight"] = sc_paths["weights"]
        parameters.parameter_connection_between_region["path_length"] = sc_paths["tract_lengths"]
        parameters.parameter_connection_between_region["path_region_labels"] = sc_paths["region_labels"]
        parameters.parameter_connection_between_region["path_centres"] = sc_paths["centres"]

    

    # Initialize simulator (first time to define the simulator, could be done outside the function once and for all or some other way to avoid re-initialization)
    simulator = tools.init(
        parameters.parameter_simulation,
        parameters.parameter_model,
        parameters.parameter_connection_between_region,
        parameters.parameter_coupling,
        parameters.parameter_integrator,
        parameters.parameter_monitor,
        my_seed=int(seed)
    )

    
    # Vérification du nombre de noeuds dans le connectome
    num_nodes = simulator.number_of_nodes
    if num_nodes <= 1:
        raise ValueError(f"Trop peu de noeuds ({num_nodes}) dans la connectome.")

    # Define Qi values for each region
    # If USE_QI_CHANGE is True, Qi will be modified for specific nodes, otherwise same value is applied to all nodes
    Qi_val = Qi.numpy() if isinstance(Qi, torch.Tensor) else Qi
    Qi_allreg = np.full(num_nodes, Qi_val)
    if USE_QI_CHANGE:
        for i, list_node in enumerate(nodes_Qichange):
            if not isinstance(list_node, list):
                list_node = [list_node]
            Qi_allreg[list_node] = Qi_change_curr[i] 


    
    parameters.parameter_model.update({
        'b_e': 0, #bvals
        'Q_i': Qi_allreg.tolist(),
        'ratio_EI_EE': ratio_coupling_EE_EI,
        'K_ext_i': type(parameters.parameter_model['K_ext_e'])(0) ,# E→I coupling disabled, bvals=0
        'external_input_ex_ex': Iext,
        'external_input_in_ex': Iext
    })

    

    # Initialisation aléatoire
    
    initE = np.random.rand() * 0.001 # ratemax_init = 0.001
    initI = initE * 4 # ratio I/E = 4 
    initW = initE**2 * 20. * 0 / (0.001**2) # bvals = 0
    parameters.parameter_model['initial_condition'].update({
        'E': [initE, initE],
        'I': [initI, initI],
        'W_e': [initW, initW]
    })

    parameters.parameter_coupling["parameter"]["a"] = coupling_strength


    # Stimulus parameters (right now there is no stimulation applied, but we can set it up)
    stimval = 0. # Hz, stimulus strength
    stimdur = 50 # ms, duration of the s
    stim_region = np.array([18,19]) # left and right insula
    weight = list(np.zeros(simulator.number_of_nodes))
    for reg in stim_region:
        weight[reg] = stimval # region and stimulation strength of the region 0 
    parameters.parameter_stimulus["tau"]= stimdur # stimulus duration [ms]
    parameters.parameter_stimulus["T"]= 2000.0 # interstimulus interval [ms]
    parameters.parameter_stimulus["weights"]= weight
    parameters.parameter_stimulus["variables"]=[0] #variable to kick
    parameters.parameter_stimulus['onset'] = cut_transient + 0.5*(run_sim-cut_transient)
    parameters.parameter_stimulus['onset'] = cut_transient + parameters.parameter_stimulus["T"]/2
    stim_time = parameters.parameter_stimulus['onset']
    stim_region_name_l = simulator.connectivity.region_labels[stim_region]
    if len(stim_region_name_l) > 1:
        name_large_regions = []
        for reg_name_curr in stim_region_name_l:
            idx_char = 0
            while reg_name_curr[idx_char].islower() and \
                idx_char < len(reg_name_curr):
                    idx_char += 1
            name_large_regions.append(reg_name_curr[:idx_char])
        stim_region_name = '-'.join(np.unique(name_large_regions))
    else:
        stim_region_name = ''.join(stim_region_name_l)
    #print(stim_region_name)

    # Second init to update the simulmator with stimulation, parameters and Qi values (to keep if the simulator is already initialized befor the function call)
    simulator = tools.init(parameters.parameter_simulation,
                                parameters.parameter_model,
                                parameters.parameter_connection_between_region,
                                parameters.parameter_coupling,
                                parameters.parameter_integrator,
                                parameters.parameter_monitor,
                                parameter_stimulation=parameters.parameter_stimulus,
                                my_seed = int(seed))

    # Run simulation
    raw_result = tools.run_simulation(
        simulator,
        run_sim,
        parameters.parameter_simulation,
        parameters.parameter_monitor
    )
    
    # Extract time points for BOLD signal
    tinterstim = parameters.parameter_stimulus["T"]
    time_after_last_stim = (run_sim - cut_transient)//tinterstim*tinterstim + cut_transient
    time_begin_all = np.arange(cut_transient, time_after_last_stim, tinterstim)

    # Collect results for each time point
    Esig_alltime = []
    Isig_alltime = []
    for time_begin in time_begin_all:
        try:
            result = tools.get_result(raw_result, 'results/synch', time_begin, time_begin + tinterstim)
            if result:
                time_s = result[0][0] * 1e-3 - result[0][0][0] * 1e-3  # convert ms to sec
                
                Esig_alltime.append(result[0][1][:, 0, :] * 1e3)  # scale to Hz
                Isig_alltime.append(result[0][1][:, 1, :] * 1e3)
        except FileNotFoundError:
            print(f"might be wrong path to parameters of simulation check params.py")
    
    # Extraction des signaux
    Esig = np.concatenate(Esig_alltime, axis=0)
    Isig = np.concatenate(Isig_alltime, axis=0)
    EIsig = 0.8 * Esig + 0.2 * Isig
    
    # Suppression du transitoire
    dt = time_s[1] - time_s[0] # must be the same as in the kernel file (0.005)
    dt_BOLD = 1.0  # BOLD signal sampling time in seconds
    
    FR_sum = result[0][1][:,0,:]*1e3*0.8 \
            + result[0][1][:,1,:]*1e3*0.2

    # Convolution HRF
    BOLD = np.array([
        signal.fftconvolve(EIsig[:, i], kernel_hrf, mode='valid') # valid to avoid edge effects
        for i in range(len(FR_sum[0]))
    ])

    # Sous-échantillonnage 
    ratio_dt = int(dt_BOLD/ dt)
    trunc_len = (BOLD.shape[1] // ratio_dt) * ratio_dt
    BOLD_subsamp = BOLD[:, :trunc_len].reshape(BOLD.shape[0], -1, ratio_dt).mean(axis=2)

    # Save BOLD signal
    filename = generate_BOLD_filename(params, USE_QI_CHANGE)
    np.save(filename, BOLD_subsamp)

    # if you have issues saving files when partitioning the cluster, uncomment the following lines and remove the previous np.save line
    """tmp_path = f"/tmp/abc_{os.getpid()}.npy"

    np.save(tmp_path, BOLD_subsamp)
    # Petit délai aléatoire de 0 à 3 secondes avant la sauvegarde pour le cluster
    time.sleep(random.uniform(0, 1.5))
    shutil.move(tmp_path, filename)"""

    return BOLD_subsamp



def simulator_FC(theta, params, list_inferer, SUBJ_IDX, CTRL, RANDOM_SC, USE_PATIENT_SC, REAL_OBS, USE_QI_CHANGE, kernel_hrf=kernel):
    """
    Simulate a functional connectivity (FC) matrix from BOLD signals
    using parameter inference and optionally patient-specific connectomes.

    This function overrides specific parameters from the base `params` dictionary
    using values from `theta`, then either loads or simulates BOLD signals and
    computes the functional connectivity (FC) matrix via Pearson correlation.
    The output is the upper triangular (non-redundant) part of the FC matrix.

    Parameters
    ----------
    theta : torch.Tensor or dict
        Parameter values to override in `params`:
            - If `dict`: keys must match those in `list_inferer`.
            - If `Tensor` (1D): values are applied in order of `list_inferer`.

    params : dict
        Base parameter dictionary for the model simulation.

    list_inferer : list of str
        List of parameter keys from `params` to override with `theta`.

    SUBJ_IDX : int
        Subject index for selecting patient-specific connectivity.

    CTRL : bool
        If True, use control SC (connectivity) when `USE_PATIENT_SC` is True.

    RANDOM_SC : bool
        If True, apply randomization to structural connectivity.

    USE_PATIENT_SC : bool
        If True, use subject-specific structural connectivity.

    REAL_OBS : bool
        If True, indicates real patient observation is used.

    USE_QI_CHANGE : bool
        If True, apply region-specific adaptation parameters.

    kernel_hrf : np.ndarray, optional
        Hemodynamic response function (HRF) kernel for BOLD simulation (default: `kernel`).

    Returns
    -------
    torch.Tensor
        1D tensor containing the upper-triangular (excluding diagonal) part of the FC matrix.

    Notes
    -----
    - The FC matrix is computed as the Pearson correlation across brain regions.
    - BOLD signals are either loaded from file (if already saved) or simulated on the fly.
    - The output is suitable for use in optimization/inference pipelines.
    """

    #print("[simulator_FC] Début de la simulation de la matrice FC", flush=True)

    # Copy params to avoid side effects
    if params is None:
        params = default_params
    params_c = params.copy()

    # Override parameters using theta values
    if isinstance(theta, dict):
        for key in list_inferer:
            params_c[key] = float(theta[key])
    else:
        for i, key in enumerate(list_inferer):
            val = theta[i].item() if isinstance(theta[i], torch.Tensor) else float(theta[i])
            params_c[key] = val

    # Generate full path for BOLD filename (inside 'save_file/' folder)
    filename = generate_BOLD_filename(params_c, USE_QI_CHANGE)
    if not filename.startswith('save_file/'):
        filename = os.path.join('save_file', filename)

    # Load or simulate BOLD signals
    if os.path.exists(filename):
        bold = np.load(filename)
        #print("Bold loaded")
    else:
        #print('Simulating Bold signal')
        bold = simulator_BOLD(kernel_hrf = kernel_hrf, params=params_c,  SUBJ_IDX = SUBJ_IDX, CTRL = CTRL, RANDOM_SC = RANDOM_SC, USE_PATIENT_SC = USE_PATIENT_SC, REAL_OBS = REAL_OBS, USE_QI_CHANGE = USE_QI_CHANGE)
        #print("done!")

    # Compute FC matrix as correlation across regions
    FC = np.corrcoef(bold) # pearson correlation matrix

    # Extract upper-triangular part excluding diagonal to transform to 1D vector
    triu_idx = np.triu_indices_from(FC, k=1)
    FC_vector = FC[triu_idx]

    return torch.tensor(FC_vector, dtype=torch.float32)


def generate_observation(
    params,
    simulator,
    list_inferer,
    REAL_OBS,
    CTRL,
    SUBJ_IDX,
    data_path=f"generated_data/FC_data_flattened_{PARC_SIZE}.npz"
):
    """
    Generate a functional connectivity (FC) observation for use in Bayesian inference.

    This function supports two modes:
    - Simulation: Generates FC vector by running a forward model with specified parameters.
    - Real observation: Loads a real FC vector (control or patient) from a `.npz` file.

    Parameters
    ----------
    params : dict
        Dictionary of default model parameters. Must include all keys in `list_inferer`.

    simulator : callable
        Function that takes a batch of parameter tensors and returns corresponding FC vectors.
        Typically a wrapper of `simulator_FC`.

    list_inferer : list of str
        Names of parameters to infer, corresponding to keys in `params`.

    REAL_OBS : bool
        If True, loads FC vector from real data; if False, simulates FC from the model.

    CTRL : bool
        If True and `REAL_OBS` is True, load FC data from control group.
        Otherwise, load data from patient group.

    SUBJ_IDX : int
        Index of the subject to extract from the real FC dataset.

    data_path : str, optional
        Path to the `.npz` file containing precomputed FC vectors for real data
        (default: f"generated_data/FC_data_flattened_{PARC_SIZE}.npz").

    Returns
    -------
    tuple
        If `REAL_OBS` is False:
            (theta_dict, FC_simulated)
                - theta_dict : dict
                    Dictionary of parameter values used for simulation.
                - FC_simulated : torch.Tensor
                    Simulated FC vector (1D).

        If `REAL_OBS` is True:
            (None, FC_real)
                - FC_real : torch.Tensor
                    Real FC vector (1D) loaded from file.

    Raises
    ------
    FileNotFoundError
        If the data file is not found when `REAL_OBS` is True.

    KeyError
        If the expected key ("X_ctrl" or "X_schz") is not found in the loaded data.

    ValueError
        If one or more keys in `list_inferer` are missing from `params`.

    Notes
    -----
    The simulated FC vector is computed using the provided simulator with a batch of one
    parameter vector. This is useful for likelihood-free inference methods like SNPE or ABC.
    """

    if not REAL_OBS:
        # Préparer le dict de paramètres à inférer
        theta_dict = {p: params[p] for p in list_inferer if p in params}
        missing = [p for p in list_inferer if p not in params]
        if missing:
            raise ValueError(f"Parameters missing from `params`: {missing}")

        # Transformer en tensor batché (shape [1, D])
        theta_tensor = torch.tensor([[theta_dict[p] for p in list_inferer]], dtype=torch.float32)
        print(f"[Simulate observation] theta_tensor.shape = {theta_tensor.shape}")

        # Simuler FC
        FC_sim = simulator(theta_tensor)

        return theta_dict, FC_sim


    else:
        # Load real data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = np.load(data_path, allow_pickle=True)

        key = "X_ctrl" if CTRL else "X_schz"
        if key not in data:
            raise KeyError(f"Key '{key}' not found in data file.")

        FC_matrix = data[key][SUBJ_IDX, :]
        FC_tensor = torch.tensor(FC_matrix, dtype=torch.float32)

        return None, FC_tensor


# ---------------------------------
# 3. Data Transformation Utilities
# ---------------------------------


def vector_to_symmetric_matrix(vec, size = PARC_SIZE):
    """
    Reconstruit une matrice symétrique à partir d'un vecteur contenant la partie triangulaire supérieure sans la diagonale.

    Paramètres :
    - vec : np.ndarray de forme (size*(size-1)/2,)
        Vecteur contenant les éléments au-dessus de la diagonale (sans la diagonale).
    - size : int
        Taille de la matrice carrée à reconstruire.

    Retour :
    - mat : np.ndarray de forme (size, size)
        Matrice symétrique reconstruite.
    """
    expected_len = size * (size - 1) // 2
    if len(vec) != expected_len:
        raise ValueError(
            f"Longueur du vecteur ({len(vec)}) ne correspond pas à la taille attendue ({expected_len}) pour une matrice {size}x{size}.")

    mat = np.zeros((size, size))
    
    # Remplir triangle supérieur sans la diagonale
    triu_indices = np.triu_indices(size, k=1)
    mat[triu_indices] = vec

    # Symétriser
    mat += mat.T

    # Ajouter des 1 sur la diagonale
    np.fill_diagonal(mat, 1.0)

    return mat


def extract_SC_patient(
    SUBJ_IDX,
    CTRL,
    RANDOM_SC,
    base_dir="data_tvb/real_connectome_for_tvb"
):
    """
    Extract structural connectivity (SC) data for a subject from a pre-generated TVB-compatible .zip file.

    This function allows access to SC data for either control or schizophrenia subjects.
    The SC data include connectivity weights, tract lengths, region labels, and region centers.

    Parameters
    ----------
    SUBJ_IDX : int or None
        Index of the subject to load (from 0 to 25). If None and `RANDOM_SC` is True,
        a subject is randomly selected from the appropriate group.

    CTRL : bool
        If True, selects the subject from the control group. Otherwise, selects from the schizophrenia group.

    RANDOM_SC : bool
        If True, selects a random subject (overrides `SUBJ_IDX` if provided).

    base_dir : str, optional
        Directory where the SC `.zip` archives are stored (default: "data_tvb/real_connectome_for_tvb").

    Returns
    -------
    dict of str
        Dictionary containing absolute paths to the extracted files:
            - 'weights' : Path to the SC weights matrix (`weights.npy`)
            - 'tract_lengths' : Path to tract length matrix (`tract_lengths.npy`)
            - 'region_labels' : Path to text file of region names (`region_labels.txt`)
            - 'centres' : Path to text file of 3D region coordinates (`centres.txt`)

    Raises
    ------
    FileNotFoundError
        If the corresponding `.zip` file for the subject is not found.

    Notes
    -----
    All files are extracted to a temporary directory: `data_tvb/real_connectome_for_tvb/tmp`.

    Examples
    --------
    >>> extract_SC_patient(SUBJ_IDX=5, CTRL=True, RANDOM_SC=False)
    {
        'weights': 'data_tvb/real_connectome_for_tvb/tmp/weights.npy',
        'tract_lengths': 'data_tvb/real_connectome_for_tvb/tmp/tract_lengths.npy',
        'region_labels': 'data_tvb/real_connectome_for_tvb/tmp/region_labels.txt',
        'centres': 'data_tvb/real_connectome_for_tvb/tmp/centres.txt'
    }
    """
    
    # Create a temporary directory to extract the connectome files into
    tmp_dir = "data_tvb/real_connectome_for_tvb/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    # If no specific subject index is provided or RANDOM_SC is True, select one at random (0–25)
    if RANDOM_SC or SUBJ_IDX is None:
        SUBJ_IDX = np.random.randint(0, 26) # size of our current dataset

    # Determine the subject group: control ("ctrl") or schizophrenia ("schz")
    group = "ctrl" if CTRL else "schz"

    # Construct the filename for the subject's connectome ZIP archive
    zip_name = f"patient_{group}_{SUBJ_IDX}.zip"
    zip_path = os.path.join(base_dir, zip_name)

    # Check that the ZIP file exists; raise an error if not
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"[✗] File not found: {zip_path}")

    # Extract the contents of the ZIP file into the temporary directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    # Return the full paths to the extracted SC data files
    return {
        "weights": os.path.join(tmp_dir, "weights.npy"),               # Structural connectivity weights
        "tract_lengths": os.path.join(tmp_dir, "tract_lengths.npy"),   # Tract distances between regions
        "region_labels": os.path.join(tmp_dir, "region_labels.txt"),   # Brain region labels
        "centres": os.path.join(tmp_dir, "centres.txt")                # Spatial coordinates of regions
    }




# ---------------------------------
# 4. Posterior and Inference Analysis
# ---------------------------------



def compute_predictions_from_posterior(posterior, obs, simulator, N_samples):
    """
    Generate model predictions by sampling from a learned posterior distribution.

    This function performs three main steps:
    1. Samples N parameter sets from the posterior conditioned on an observation.
    2. Simulates predictions (e.g., FC vectors) using the sampled parameters.
    3. Converts the predictions to a NumPy array for further analysis or evaluation.

    Parameters
    ----------
    posterior : sbi.inference.Posteriors
        Trained posterior distribution over model parameters.
    
    obs : torch.Tensor
        Observed data used to condition the posterior (e.g., real or simulated FC vector).

    simulator : callable
        Simulator function that maps parameters to observable predictions.
        Should accept a batched tensor of shape (1, D).

    N_samples : int
        Number of parameter samples to draw from the posterior.

    Returns
    -------
    posterior : sbi.inference.Posteriors
        The input posterior (unchanged).

    theta_posterior_np : np.ndarray
        Array of sampled parameters with shape (N_samples, D).

    x_predictive_all_np : np.ndarray
        Array of model predictions corresponding to each sample.
        Shape: (N_samples, output_dim).
    """
    print("[compute_predictions_from_posterior] Starting...")

    # Step 1: Sample N parameter vectors from the posterior given the observed data
    try:
        theta_posterior = posterior.sample((N_samples,), x=obs, show_progress_bars=True)
    except Exception as e:
        print(f"[Posterior sampling error] {e}")
        raise

    # Step 2: Simulate predictions for each sampled parameter set in parallel
    with tqdm_joblib(tqdm(desc="Simulations", total=N_samples)):
        x_predictive_all = joblib.Parallel(n_jobs=NJOBS)(
            joblib.delayed(simulator)(
                theta=theta_posterior[i].unsqueeze(0)  # Add batch dimension
            )
            for i in range(N_samples)
        )

    # Step 3: Convert list of predictions (tensors or arrays) to a single NumPy array
    try:
        x_predictive_all_np = np.vstack([
            x.detach().cpu().numpy() if hasattr(x, "detach") else x
            for x in x_predictive_all
        ])
    except Exception as e:
        print(f"[NumPy conversion error] {e}")
        raise

    print(f"x_predictive_all_np shape: {x_predictive_all_np.shape}")

    return posterior, theta_posterior.cpu().numpy(), x_predictive_all_np


def save_posterior_plot(samples, true_value, lower, upper, name, save_path):
    """
    Generate and save a posterior distribution plot for a single parameter.

    The plot includes:
    - A histogram and KDE of the posterior samples.
    - A vertical line for the mean of the samples.
    - Optional vertical line for the ground-truth (true_value), if provided.
    - Vertical dashed lines for the 90% credible interval (lower, upper bounds).

    Parameters
    ----------
    samples : array-like
        Posterior samples of the parameter (e.g., from MCMC or SNPE).
    
    true_value : float or None
        Ground-truth value of the parameter, shown as a red dashed line. If None, this line is omitted.
    
    lower : float
        Lower bound of the 90% credible interval.
    
    upper : float
        Upper bound of the 90% credible interval.
    
    name : str
        Name of the parameter (used in the plot title and x-axis label).
    
    save_path : str
        Path to save the resulting plot as an image (e.g., PNG or PDF).

    Returns
    -------
    None
        The plot is saved to disk; nothing is returned.
    """
    plt.figure(figsize=(6, 4))

    # Determine the number of histogram bins based on sample size
    bins = min(50, max(10, len(samples) // 20))

    # Plot histogram and KDE
    sns.histplot(samples, bins=bins, kde=True, color="skyblue", edgecolor="black", stat="density")

    mean = np.mean(samples)

    # Add reference lines
    if true_value is not None:
        plt.axvline(true_value, color="red", linestyle="--", label="True value")
    plt.axvline(mean, color="green", linestyle="-", label=f"Mean = {mean:.3f}")
    plt.axvline(lower, color="black", linestyle="--", alpha=0.5, label="90% CI Lower")
    plt.axvline(upper, color="black", linestyle="--", alpha=0.5, label="90% CI Upper")

    # Finalize plot
    plt.title(f"Posterior of {name}")
    plt.xlabel(name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def quick_stats_dict(x):
    """
    Compute quick summary statistics of an input array or tensor.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Input data, can be a NumPy array or a PyTorch tensor.
        If a tensor, it will be detached and moved to CPU before processing.

    Returns
    -------
    dict
        Dictionary containing:
        - "shape": shape of the input as a list.
        - "global_mean": mean of the feature-wise means.
        - "global_std": mean of the feature-wise standard deviations.
        - "min_abs": minimum value in the entire array.
        - "max_abs": maximum value in the entire array.
        - "std_min": minimum standard deviation among features.
        - "std_max": maximum standard deviation among features.

    Notes
    -----
    Assumes the data is structured with features along axis=1 (columns),
    and observations/samples along axis=0 (rows).
    """
    # Convert torch tensor to numpy array if necessary
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    # Compute mean and std per feature (column-wise)
    mean_per_feature = np.mean(x, axis=0)
    std_per_feature = np.std(x, axis=0)

    return {
        "shape": list(x.shape),
        "global_mean": float(np.mean(mean_per_feature)),
        "global_std": float(np.mean(std_per_feature)),
        "min_abs": float(np.min(x)),
        "max_abs": float(np.max(x)),
        "std_min": float(np.min(std_per_feature)),
        "std_max": float(np.max(std_per_feature)),
    }


def posterior_calibration_and_plot(true_theta, posterior_samples, alpha_list=[0.5, 0.9, 0.95, 0.99], save_path=None):
    """
    Compute calibration of credible intervals for posterior samples of a single test.

    Parameters
    ----------
    true_theta : array-like, shape (n_params,)
        True parameter values for the observation.

    posterior_samples : ndarray, shape (n_samples, n_params)
        Samples drawn from the posterior distribution.

    alpha_list : list of float, optional
        Credibility levels to evaluate, defaults to [0.5, 0.9, 0.95, 0.99].

    save_path : str or None, optional
        If provided, save the calibration plot to this file path.

    Returns
    -------
    coverage_dict : dict
        Dictionary mapping each alpha to a boolean indicating whether 
        the true parameters fall within the corresponding credible interval.

    Notes
    -----
    This function calculates whether the true parameter vector is contained within the 
    credible interval at different credibility levels. It can also plot a calibration curve 
    showing coverage versus credibility level.
    """
    if true_theta is None:
        print("[Warning] true_theta is None.")
        return None

    true_theta = np.asarray(true_theta).squeeze()
    coverage_dict = {}

    for alpha in alpha_list:
        lower = np.percentile(posterior_samples, 100 * (1 - alpha) / 2, axis=0)
        upper = np.percentile(posterior_samples, 100 * (1 + alpha) / 2, axis=0)

        is_in_interval = np.logical_and(true_theta >= lower, true_theta <= upper)
        coverage_dict[alpha] = bool(np.all(is_in_interval))

    if save_path is not None:
        alphas = sorted(alpha_list)
        coverage_vals = [float(coverage_dict[a]) for a in alphas]
        plt.figure(figsize=(6, 5))
        plt.plot(alphas, coverage_vals, marker="o", label="Observed Coverage", color="blue")
        plt.xlabel("Credibility Level (α)")
        plt.ylabel("Observed Coverage")
        plt.title("Posterior Calibration Curve")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ylim(0, 1.05)
        plt.xlim(0, 1.05)
        plt.legend(loc="lower right")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    return coverage_dict


def save_pca_projection(predictions, obs, save_path, theta=None):
    """
    Projette les prédictions et l'observation sur les deux premières composantes principales.
    Colore les prédictions selon le premier paramètre inféré (Qi), avec la colormap 'coolwarm'.
    
    Arguments :
    - predictions : [N, dim_obs] torch.Tensor ou np.ndarray
    - obs         : [dim_obs]    torch.Tensor ou np.ndarray
    - save_path   : chemin pour sauvegarder la figure
    - theta       : [N, dim_param] torch.Tensor ou np.ndarray, contient Qi en première colonne
    """

    # Conversion en numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    if theta is not None and isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy()

    # Vérifications
    if np.isnan(predictions).any() or np.isnan(obs).any():
        raise ValueError("NaN found in predictions or observations before PCA")
    if np.allclose(predictions, predictions[0]):
        raise ValueError("All prediction samples are identical — PCA would be invalid")

    # PCA projection
    pca = PCA(n_components=2)
    x_proj = pca.fit_transform(predictions)
    obs_proj = pca.transform(obs.reshape(1, -1))

    print('x_proj shape:', x_proj.shape)      # (N, 2)
    print('obs_proj shape:', obs_proj.shape)  # (1, 2)

    # Gestion des couleurs : Qi uniquement
    if theta is not None:
        qi_values = theta[:, 0]  # Qi est le premier paramètre
        cmap = plt.cm.coolwarm
    else:
        qi_values = 'blue'
        cmap = None

    # Plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        x_proj[:, 0],
        x_proj[:, 1],
        c=qi_values,
        cmap=cmap,
        alpha=0.8,
        edgecolors='k',
        linewidths=0.3,
        label='Prédictions'
    )
    plt.scatter(
        obs_proj[:, 0],
        obs_proj[:, 1],
        color='black',
        marker='x',
        s=100,
        label='Observation'
    )

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Projection PCA (couleur = Qi)')
    plt.legend()
    
    if theta is not None:
        cbar = plt.colorbar(sc)
        cbar.set_label('Qi (paramètre 0)', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_fc_matrices(fc_obs, fc_pred, save_path):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(fc_obs, cmap="coolwarm", cbar=True)
    plt.title("FC Observation")

    plt.subplot(1, 2, 2)
    sns.heatmap(fc_pred, cmap="coolwarm", cbar=True)
    plt.title("Best FC Prediction")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_metrics(predictions, obs_np):
    """
    Evaluate a list of prediction vectors against a single observation vector using
    a Pearson distance metric (1 - Pearson correlation). If variance is too low,
    fallback to Mean Squared Error (MSE) as the distance.

    Parameters
    ----------
    predictions : list or array-like
        List of predicted vectors to compare against the observation.

    obs_np : array-like
        The observation vector to compare against.

    Returns
    -------
    best_idx : int
        Index of the prediction with the smallest distance to the observation.

    pearson_distances : list of float
        List of distances (1 - Pearson correlation or MSE fallback) for each prediction.

    Raises
    ------
    ValueError
        If all distances are infinite, indicating no valid comparison could be made.
    """
    var_threshold = 1e-8  # Threshold for too low variance
    pearson_distances = []

    obs_vector = np.asarray(obs_np).squeeze()

    for i, pred in enumerate(predictions):
        pred = np.asarray(pred).squeeze()

        # Check shape compatibility
        if pred.shape != obs_vector.shape:
            print(f"[!] Shape mismatch at index {i}: pred {pred.shape}, obs {obs_vector.shape}")
            pearson_distances.append(np.inf)
            continue

        # If variance too low, fallback to MSE
        if np.var(pred) < var_threshold or np.var(obs_vector) < var_threshold:
            print(f"[!] Low variance detected at index {i}, using MSE instead.")
            mse = np.mean((pred - obs_vector) ** 2)
            pearson_distances.append(mse)
            continue

        # Constant vectors cannot have Pearson correlation
        if np.std(pred) == 0 or np.std(obs_vector) == 0:
            print(f"[!] Constant vector detected at index {i}, cannot compute Pearson.")
            pearson_distances.append(np.inf)
            continue

        # Compute Pearson correlation and convert to distance
        try:
            pearson_val, _ = pearsonr(pred, obs_vector)
            if not np.isfinite(pearson_val):
                raise ValueError("Non-finite Pearson correlation")
            distance = 1 - pearson_val
            pearson_distances.append(distance)
        except Exception as e:
            print(f"[!] Pearson calculation failed at index {i}: {e}")
            pearson_distances.append(np.inf)

    # Check at least one valid distance
    if all(np.isinf(pearson_distances)):
        raise ValueError("[ERROR] All Pearson distances are infinite — no valid vector for comparison.")

    best_idx = int(np.argmin(pearson_distances))
    print(f"[INFO] Best index by metric: {best_idx}, distance = {pearson_distances[best_idx]}")

    return best_idx, pearson_distances


def compute_posterior_diagnostics(samples):
    """
    Compute diagnostic statistics for each parameter dimension in posterior samples.

    Parameters
    ----------
    samples : np.ndarray
        Array of posterior samples with shape (n_samples, n_parameters).

    Returns
    -------
    dict
        Dictionary where each key is "param_i" for parameter i, containing:
        - mean: Mean of the samples for this parameter.
        - std: Standard deviation of the samples.
        - skewness: Skewness of the distribution.
        - kurtosis: Kurtosis of the distribution.
        - entropy_kde: Estimated entropy based on Kernel Density Estimation (KDE).
    """
    stats = {}
    for i in range(samples.shape[1]):
        dim = samples[:, i]
        # Fit KDE for entropy estimation
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(dim[:, None])
        log_probs = kde.score_samples(dim[:, None])
        
        stats[f"param_{i}"] = {
            "mean": float(np.mean(dim)),
            "std": float(np.std(dim)),
            "skewness": float(skew(dim)),
            "kurtosis": float(kurtosis(dim)),
            "entropy_kde": float(entropy(np.exp(log_probs))),
        }
    return stats

def plot_marginals(samples, param_names=None, save_path=None):
    """
    Plot marginal distributions (histograms with KDE) for each parameter in posterior samples.

    Parameters
    ----------
    samples : np.ndarray
        Array of posterior samples with shape (n_samples, n_parameters).

    param_names : list of str, optional
        List of parameter names for plot titles. If None, generic names are used.

    save_path : str, optional
        If provided, save the plot to this file path. Otherwise, display is suppressed.
    """
    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(6, 2.5 * n_params), sharex=False)

    for i in range(n_params):
        ax = axes[i] if n_params > 1 else axes
        # Plot histogram with KDE overlay
        sns.histplot(samples[:, i], kde=True, ax=ax, color="skyblue", stat="density", bins=30)
        ax.set_title(param_names[i] if param_names else f"Param {i}")
        ax.set_xlabel("")  # Clean x-labels for clarity

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def analyze_posterior_logprob_map(samples, posterior, x_obs, save_path, show_progress_bars=True):
    """
    Analyze a posterior distribution by computing log-probabilities, estimating MAP,
    and generating diagnostic plots.

    Parameters
    ----------
    samples : torch.Tensor or None
        Posterior samples (if None, samples will be drawn inside the function).

    posterior : object
        SBI posterior object with methods `sample`, `log_prob`, and `map`.

    x_obs : torch.Tensor
        Observation used for conditioning the posterior.

    save_path : str
        Directory path where plots will be saved.

    show_progress_bars : bool, optional (default=True)
        Whether to display progress bars during sampling and optimization.

    Returns
    -------
    dict
        Dictionary containing:
        - "samples": posterior samples tensor
        - "log_probs": log-probabilities of the samples
        - "map_estimate": MAP estimate tensor
    """
    print("[analyze_posterior_logprob_map] Starting posterior analysis...")

    # Ensure samples are drawn conditioned on observation if not provided
    if samples is None:
        samples = posterior.sample((100,), x=x_obs, show_progress_bars=show_progress_bars)

    # Compute log-probabilities of posterior samples
    log_probs = posterior.log_prob(theta=samples)  # shape: (num_samples,)

    # Compute MAP estimate via optimization
    print("Computing MAP estimate...")
    map_estimate = posterior.map(num_init_samples=100, num_to_optimize=10, show_progress_bars=show_progress_bars)

    num_params = samples.shape[1]

    # Plot histogram of log-probabilities
    plt.figure(figsize=(6, 4))
    sns.histplot(log_probs.cpu().numpy(), bins=30, kde=True)
    plt.title("Log-Probability Distribution of Posterior Samples")
    plt.xlabel("log p(θ | x)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "log_probs_histogram.png"))
    plt.close()

    # If dimensionality >= 2, plot 2D scatter of first two params with MAP
    if num_params >= 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha=0.2, label='Samples')
        plt.scatter(map_estimate[0].cpu(), map_estimate[1].cpu(), color='red', marker='*', s=120, label='MAP')
        plt.xlabel("θ₀")
        plt.ylabel("θ₁")
        plt.title("2D Projection of Posterior Samples with MAP")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "2d_projection_with_map.png"))
        plt.close()

    # If dimensionality is small, plot marginals with MAP line
    if num_params <= 5:
        for i in range(num_params):
            plt.figure(figsize=(5, 3))
            sns.histplot(samples[:, i].cpu().numpy(), bins=30, kde=True)
            plt.axvline(map_estimate[i].cpu().item(), color='red', linestyle='--', label='MAP')
            plt.title(f"Marginal Distribution of Parameter θ[{i}]")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"marginal_{i}.png"))
            plt.close()

    return {
        "samples": samples,
        "log_probs": log_probs,
        "map_estimate": map_estimate
    }


# ---------------------------------
# 5. Evaluation and Logging Functions
# ---------------------------------


def evaluate_and_log_experiment(
    posterior,
    predictions,
    posterior_samples,
    obs,
    true_theta=None,
    exp_name=None,
    list_inferer=None,
    num_samples=None,
    num_simulations=None,
    borne_min=None,
    borne_max=None,
    real_obs=None,
    alpha_list=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
):
    """
    Evaluate predictions and posterior samples against observations and log comprehensive metrics and plots.

    Args:
        posterior: The trained posterior distribution object.
        predictions: List or array of predicted vectors.
        posterior_samples: Samples drawn from the posterior distribution.
        obs: Observed data (tensor or ndarray).
        true_theta: True parameter values, if available (tensor or ndarray).
        exp_name: Name of the experiment (used to create directories).
        list_inferer: List of parameter names corresponding to posterior samples.
        num_samples: Number of samples used for inference (optional).
        num_simulations: Number of simulations run (optional).
        borne_min: Tensor or array representing lower bounds (optional).
        borne_max: Tensor or array representing upper bounds (optional).
        real_obs: Real observation data, if different from `obs` (optional).
        alpha_list: List of credible interval levels to evaluate calibration.

    Returns:
        None. Saves metrics and figures to disk.
    """

    # Create directories for storing results and figures
    output_dir = os.path.join("results", exp_name)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Convert observation to numpy array and flatten if tensor
    obs_np = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
    obs_np = obs_np.squeeze()

    # Convert predictions to numpy array
    predictions = np.asarray(predictions)

    # Check shape compatibility between obs and predictions
    if obs_np.ndim != 1 or obs_np.shape[0] != predictions.shape[1]:
        raise ValueError(f"[!] Shape mismatch: obs_np.shape={obs_np.shape}, expected={predictions.shape[1]}")

    # Evaluate metrics (Pearson distance or fallback MSE) and get best prediction index
    best_idx, pearson_distances = evaluate_metrics(predictions, obs_np)

    # --- Global posterior analysis ---
    posterior_samples = np.asarray(posterior_samples)

    # Ensure posterior_samples has two dimensions (samples x params)
    if posterior_samples.ndim == 1:
        posterior_samples = posterior_samples[:, np.newaxis]

    # Compute mean, std, and 90% credible intervals for each parameter
    posterior_mean = np.mean(posterior_samples, axis=0).tolist()
    posterior_std = np.std(posterior_samples, axis=0).tolist()
    lower = np.percentile(posterior_samples, 5, axis=0).tolist()
    upper = np.percentile(posterior_samples, 95, axis=0).tolist()

    # Compute additional statistics: skewness, kurtosis, entropy, and diversity of samples
    try:
        skewness = [float(skew(posterior_samples[:, i])) for i in range(posterior_samples.shape[1])]
        kurt = [float(kurtosis(posterior_samples[:, i])) for i in range(posterior_samples.shape[1])]
        entropies = [
            float(entropy(np.histogram(posterior_samples[:, i], bins=50, density=True)[0] + 1e-12))
            for i in range(posterior_samples.shape[1])
        ]
        diversity = float(np.mean(pairwise_distances(posterior_samples)))
    except Exception as e:
        print(f"[!] Posterior stats failed: {e}")
        skewness, kurt, entropies, diversity = [], [], [], None

    # Plot marginal histograms of posterior samples for each parameter
    try:
        fig, axs = plt.subplots(posterior_samples.shape[1], 1, figsize=(6, 3 * posterior_samples.shape[1]))
        for i in range(posterior_samples.shape[1]):
            ax = axs[i] if posterior_samples.shape[1] > 1 else axs
            samples = posterior_samples[:, i]
            ax.hist(samples, bins=50, alpha=0.7, color="skyblue", density=True)
            title = list_inferer[i] if list_inferer else f'param {i}'
            ax.set_title(f"Posterior Histogram - {title}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            # Display std, skewness, and kurtosis in the corner of each subplot
            ax.text(
                0.95, 0.95,
                f"std={np.std(samples):.3f}\nskew={skewness[i]:.2f}\nkurt={kurt[i]:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10
            )
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "posterior_marginals_overview.png"))
        plt.close()
    except Exception as e:
        print(f"[!] Failed to plot posterior marginal histograms: {e}")

    # Aggregate all calculated metrics into a dictionary
    metrics = {
        "posterior_mean": posterior_mean,
        "posterior_std": posterior_std,
        "credible_interval_90": {"lower": lower, "upper": upper},
        "pearson_stats": {
            "mean_pearson_distance": float(np.mean(pearson_distances)),
            "std_pearson_distance": float(np.std(pearson_distances)),
            "min_pearson_distance": float(np.min(pearson_distances)),
            "max_pearson_distance": float(np.max(pearson_distances)),
        },
        "posterior_distribution_stats": {
            "skewness": skewness,
            "kurtosis": kurt,
            "entropy": entropies,
            "sample_diversity": diversity,
        },
        "best_sample_index": best_idx,
        "num_samples": num_samples,
        "num_simulations": num_simulations,
        "real_obs": real_obs,
        "timestamp": datetime.now().isoformat(),
        "posterior_quick_stats": quick_stats_dict(posterior_samples),
        "prediction_quick_stats": quick_stats_dict(predictions),
        "borne_min": borne_min.detach().cpu().tolist() if borne_min is not None else None,
        "borne_max": borne_max.detach().cpu().tolist() if borne_max is not None else None,
    }

    # If true parameter values are provided, compute additional error metrics and credible interval coverage
    if true_theta is not None:
        true_theta_np = (
            true_theta.detach().cpu().numpy() if isinstance(true_theta, torch.Tensor) else np.asarray(true_theta)
        ).squeeze()

        # Ensure true_theta is 1D array
        if true_theta_np.ndim == 0:
            true_theta_np = np.array([true_theta_np])
        if true_theta_np.ndim > 1:
            true_theta_np = true_theta_np.squeeze()

        # Compute Mean Squared Error (MSE) and Mean Absolute Error (MAE)
        mse = np.mean((posterior_samples - true_theta_np) ** 2, axis=0).tolist()
        mae = np.mean(np.abs(posterior_samples - true_theta_np), axis=0).tolist()

        # Check if true_theta lies inside the 90% credible interval for each parameter
        inside = ((true_theta_np >= lower) & (true_theta_np <= upper)).tolist()

        # Update metrics dict with true parameter info and error stats
        metrics.update({
            "true_theta": true_theta_np.tolist(),
            "mse": mse,
            "mae": mae,
            "credible_interval_90_inside": inside,
        })

        # Generate calibration plot of posterior credible intervals if possible
        try:
            posterior_calibration_and_plot(
                true_theta_np,
                posterior_samples,
                alpha_list=alpha_list,
                save_path=os.path.join(fig_dir, "posterior_calibration.png"),
            )
        except Exception as e:
            print(f"[!] Posterior calibration failed: {e}")

    # Save posterior plots for each parameter with credible interval and true value
    for i, name in enumerate(list_inferer or []):
        try:
            save_posterior_plot(
                samples=posterior_samples[:, i],
                true_value=true_theta[i] if true_theta is not None else None,
                lower=lower[i],
                upper=upper[i],
                name=name,
                save_path=os.path.join(fig_dir, f"posterior_{name}.png"),
            )
        except Exception as e:
            print(f"[!] Posterior plot failed for {name}: {e}")
    
    # Save PCA projection plot comparing predictions and observation with posterior samples as color or labels
    try:
        save_pca_projection(predictions, obs_np, os.path.join(fig_dir, "pca_predictions_vs_obs.png"), theta=posterior_samples)
    except Exception as e:
        print(f"[!] PCA projection failed: {e}")

    # Attempt to reconstruct and save Functional Connectivity (FC) matrices from best prediction and observation
    try:
        best_fc = np.asarray(predictions[best_idx]).flatten()
        obs_vec = np.asarray(obs_np).flatten()

        # Calculate expected matrix dimension for FC based on vector length (triangular matrix vectorization)
        fc_dim = int((1 + np.sqrt(1 + 8 * len(best_fc))) // 2)
        expected_len = fc_dim * (fc_dim - 1) // 2
        if len(best_fc) != expected_len:
            raise ValueError(f"[ERROR] Length {len(best_fc)} does not match expected {expected_len} for {fc_dim}x{fc_dim} matrix.")

        # Convert vectors back to symmetric matrices
        FC_pred = vector_to_symmetric_matrix(best_fc, fc_dim)
        FC_obs = vector_to_symmetric_matrix(obs_vec, fc_dim)

        # Save comparison figure and arrays
        save_fc_matrices(FC_obs, FC_pred, os.path.join(fig_dir, "best_FC_matrix_comparison.png"))
        np.save(os.path.join(fig_dir, "best_predicted_fc.npy"), best_fc)
        np.save(os.path.join(fig_dir, "fc_observation.npy"), obs_vec)
    except Exception as e:
        print(f"[!] FC matrix comparison failed: {e}")

    # Generate and save posterior marginal plots for quick visual diagnostics
    plot_marginals(posterior_samples, list_inferer, save_path=os.path.join(fig_dir, "posterior_marginals.png"))

    # Compute posterior diagnostics (mean, std, skewness, kurtosis, entropy)
    diagnostics = compute_posterior_diagnostics(posterior_samples)
    metrics["posterior_diagnostics"] = diagnostics

    # Save all collected metrics as a JSON file for later analysis
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Analyze posterior log-probabilities over parameter space, save plots/logs
    analyze_posterior_logprob_map(samples=posterior_samples, posterior=posterior, obs=obs, save_path=fig_dir, show_progress_bars=True)

    # Save the posterior model itself for reproducibility or later use
    try:
        posterior_path = os.path.join(output_dir, "density_estimator.pt")
        torch.save(posterior, posterior_path)
        print(f"[✔] Posterior saved at {posterior_path}")
    except Exception as e:
        print(f"[!] Failed to save posterior: {e}")

# ---------------------------------
# 6. PCA and Real Data Analysis
# ---------------------------------


def plot_cumulative_pca_from_real_data(
    data_path=f"generated_data/FC_data_flattened_{PARC_SIZE}.npz",
    save_path="results/cumulative_pca_real_data.png",
    show=True,
    title="Cumulative PCA on Real FC Data"
):
    """
    Load real functional connectivity data (vectorized) and plot cumulative explained variance from PCA.

    Args:
        data_path (str): Path to the .npz file containing vectorized FC data under key 'X'.
        save_path (str): Path where the plot image will be saved.
        show (bool): Whether to display the plot interactively.
        title (str): Title for the plot.

    Raises:
        FileNotFoundError: If the specified data file does not exist.
    """

    # Check if data file exists before loading
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load the .npz file with numpy (allow_pickle=True for safety if needed)
    data = np.load(data_path, allow_pickle=True)

    # Key in the .npz file under which the vectorized data is stored
    key = "X"

    # Extract the vectorized FC data array, shape expected: [n_subjects, n_features]
    X = data[key]  # Vectorized FC matrices
    print(f"[✓] Loaded {X.shape[0]} subjects with {X.shape[1]} features each from key '{key}'.")

    # Perform PCA on the data without dimensionality reduction (all components)
    pca = PCA()
    pca.fit(X)

    # Extract explained variance ratio per component and compute cumulative sum
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker='o',
        linestyle='--',
        color='blue'
    )
    # Horizontal lines marking 90% and 95% explained variance thresholds
    plt.axhline(0.9, color='red', linestyle=':', label='90% variance')
    plt.axhline(0.95, color='green', linestyle=':', label='95% variance')

    # Axis labels and title
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(title)

    # Grid and legend for better readability
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot to disk if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"[✓] Figure saved to: {save_path}")

    # Show plot interactively if requested, otherwise close to free memory
    if show:
        plt.show()
    else:
        plt.close()
