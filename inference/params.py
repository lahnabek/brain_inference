"""
# params.py

This file defines the default configuration and regional mappings for running
simulation-based inference (SBI) experiments using the `sbi` library.

It includes:
- Paths to data and kernel resources
- Brain region indices and groupings
- Default model parameters and inference options
- Utility functions for saving/loading parameters
"""

import json
import os
import torch
import numpy as np

# ------------------------------------------------------------
# Base directory of the project
# ------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ------------------------------------------------------------
# Experiment identifiers and output folder
# ------------------------------------------------------------

EXP_NAME = 'test_code'  # Used to name folders/results
save_dir = f"results/{EXP_NAME}"  # Where to save outputs

# ------------------------------------------------------------
# Paths to required data files
# ------------------------------------------------------------

path_params = {
    "connectivity_dir": os.path.join(BASE_DIR, "data_tvb", "connectome_TVB"),
    "simulation_file_dir": os.path.join(BASE_DIR, "simulation_file"),
    "kernel_dir": os.path.join(BASE_DIR, "data_tvb", "kernel"),
    "parameter_simulation_model_dir": os.path.join(BASE_DIR, "results", "synch"),
}

# Hemodynamic response kernel (HRF)
kernel = np.load(os.path.join(BASE_DIR, "tvb_model_reference", "data_tvb", "kernel", "kernel_hrf_dt0.005.npy"))

# ------------------------------------------------------------
# Brain region dictionary and groups
# ------------------------------------------------------------

# now using 68 regions (34 per hemisphere)
# nodes 20 à 24 ; 54 à 58 ; 66 à 68; 31 à 33 ; 38 à ? => nodes with caracteristics that we cannot replicate for now
# 0-33 : right hemisphere
# 34-67 : left hemisphere
# you can check the name of regions also in generated_data/regions_corticales_{PARC_SIZE}_indices.csv

region_dict = {
    0: 'rh_lateralorbitofrontal',
    1: 'rh_parsorbitalis',
    2: 'rh_frontalpole',
    3: 'rh_medialorbitofrontal',
    4: 'rh_parstriangularis',
    5: 'rh_parsopercularis',
    6: 'rh_rostralmiddlefrontal',
    7: 'rh_superiorfrontal',
    8: 'rh_caudalmiddlefrontal',
    9: 'rh_precentral',
    10: 'rh_paracentral',
    11: 'rh_rostralanteriorcingulate',
    12: 'rh_caudalanteriorcingulate',
    13: 'rh_posteriorcingulate',
    14: 'rh_isthmuscingulate',
    15: 'rh_postcentral',
    16: 'rh_supramarginal',
    17: 'rh_superiorparietal',
    18: 'rh_inferiorparietal',
    19: 'rh_precuneus',
    20: 'rh_cuneus',
    21: 'rh_pericalcarine',
    22: 'rh_lateraloccipital',
    23: 'rh_lingual',
    24: 'rh_fusiform',
    25: 'rh_parahippocampal',
    26: 'rh_entorhinal',
    27: 'rh_temporalpole',
    28: 'rh_inferiortemporal',
    29: 'rh_middletemporal',
    30: 'rh_bankssts',
    31: 'rh_superiortemporal',
    32: 'rh_transversetemporal',
    33: 'rh_insula',
    34: 'lh_lateralorbitofrontal',
    35: 'lh_parsorbitalis',
    36: 'lh_frontalpole',
    37: 'lh_medialorbitofrontal',
    38: 'lh_parstriangularis',
    39: 'lh_parsopercularis',
    40: 'lh_rostralmiddlefrontal',
    41: 'lh_superiorfrontal',
    42: 'lh_caudalmiddlefrontal',
    43: 'lh_precentral',
    44: 'lh_paracentral',
    45: 'lh_rostralanteriorcingulate',
    46: 'lh_caudalanteriorcingulate',
    47: 'lh_posteriorcingulate',
    48: 'lh_isthmuscingulate',
    49: 'lh_postcentral',
    50: 'lh_supramarginal',
    51: 'lh_superiorparietal',
    52: 'lh_inferiorparietal',
    53: 'lh_precuneus',
    54: 'lh_cuneus',
    55: 'lh_pericalcarine',
    56: 'lh_lateraloccipital',
    57: 'lh_lingual',
    58: 'lh_fusiform',
    59: 'lh_parahippocampal',
    60: 'lh_entorhinal',
    61: 'lh_temporalpole',
    62: 'lh_inferiortemporal',
    63: 'lh_middletemporal',
    64: 'lh_bankssts',
    65: 'lh_superiortemporal',
    66: 'lh_transversetemporal',
    67: 'lh_insula'
}


# Region groupings
groupe_cuneus = [20, 21, 22, 23, 54, 55, 56, 57]
groupe_auditif = [29, 30, 31, 32, 63, 64, 65, 66]
groupe_langage = [38, 39, 63, 65]

aires_auditives = groupe_auditif
aires_langage = groupe_langage
insula = [33, 67]

# Network groupings
dmn = [13, 47, 19, 53, 2, 36, 4, 5, 38, 39, 11, 12, 45, 46]
dmn_frontale = [2, 36, 4, 5, 38, 39]
dmn_cingulaire = [13, 47, 11, 45, 12, 46]
dmn_parietale = [19, 53]

reseau_salience = [11, 12, 45, 46, 33, 67]
reseau_exec_control = [6, 7, 8, 17, 18, 40, 41, 42, 51, 52]
hippocampal_like = [25, 26, 27, 59, 60, 61]
temporaux = [24, 25, 26, 27, 28, 29, 59, 60, 61, 62, 63]

# ------------------------------------------------------------
# Default model parameters for simulation
# ------------------------------------------------------------

default_params = {
    'Qi': 5.0,
    'Iext': 0.001,
    'coupling_strength': 0.15,
    'ratio_coupling_EE_EI': 1.4,
    'node_name_Qichange': 'PCC',
    'nodes_Qichange': [dmn_frontale, dmn_cingulaire, dmn_parietale],
    'save_BOLD': True, # not used, always save BOLD
    'add_transient': 0,
    'cut_time': 0,
    'run_time': 100 * 10**3,  # 100 seconds
    'seed': 1,
}

# ------------------------------------------------------------
# Inference parameters and observed data
# ------------------------------------------------------------

# List of parameters to infer
list_inferer = ['Qi']
USE_QI_CHANGE = False # Whether to use Qi change parameters in specific regions
if USE_QI_CHANGE:
    for i in range(len(default_params['nodes_Qichange'])):
        default_params[f"Qi_change_curr_{i}"] = 5.0
        list_inferer.append(f"Qi_change_curr_{i}")

# Observed data parameters if using a simulated observation
params_obs = default_params.copy()
params_obs.update({
    'Qi': 5.0,
})

# Prior bounds (for uniform or truncated normal)
borne_min = torch.tensor([4.5])
borne_max = torch.tensor([5.5])

# ------------------------------------------------------------
# SBI control settings
# ------------------------------------------------------------

MODE = 'single'  # "single" or "multi"
MODEL_NN = 'maf'  # ["mdn", "maf", "nsf"] other options available but maf is great
NUM_SAMPLES = 100
NUM_SIMULATION = 1000 # enough to have a good score when using simulated observations
NUM_ROUNDS = 3
REAL_OBS = True
CTRL = True
SUBJ_IDX = 0
RANDOM_SC = False
USE_PATIENT_SC = True
PRIOR_TYPE = ['gaussian']
NJOBS = -1  # For multiprocessing (if applicable)
PARC_SIZE = 68  # Number of regions in the Desikan atlas


# ------------------------------------------------------------
# Utility functions for saving/loading parameter dicts
# ------------------------------------------------------------

def save_params_to_json(params_dict, save_dir, filename='params.json'):
    """
    Save a parameter dictionary to a JSON file, converting torch.Tensors to lists.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    serializable_params = {}
    for k, v in params_dict.items():
        if isinstance(v, torch.Tensor):
            serializable_params[k] = v.tolist()
        else:
            serializable_params[k] = v
    with open(path, 'w') as f:
        json.dump(serializable_params, f, indent=4)
    print(f"[INFO] Parameters saved to {path}")

def load_params_from_json(path):
    """
    Load parameters from a JSON file. Recasts `borne_min` and `borne_max` to torch.Tensor if found.
    """
    with open(path, 'r') as f:
        params = json.load(f)
    if 'borne_min' in params:
        params['borne_min'] = torch.tensor(params['borne_min'])
    if 'borne_max' in params:
        params['borne_max'] = torch.tensor(params['borne_max'])
    return params