import sys
import os
import warnings
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from functools import partial
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

# Set project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*use_inf_as_na.*")

# Optuna logging
import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

# Local imports
from inference.utils import *
from inference.params import *
from test_pca import *

# Mapping of brain region indices to names (right then left hemisphere)
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



# Load empirical matrix
_, empirical_matrix = generate_observation(
    params=default_params,
    simulator=simulator_FC,
    list_inferer=list_inferer,
    REAL_OBS=True,
    CTRL=True,
    SUBJ_IDX=0,
    RANDOM_SC=False,
    USE_PATIENT_SC=True,
    data_path=f"generated_data/FC_data_flattened_{PARC_SIZE}.npz"
)

# Load and split control and schizophrenia data
X, y = load_data_with_labels(f"generated_data/FC_data_flattened_{PARC_SIZE}.npz")
X_ctrl, X_schz = split_groups(X, y)


def pca_analysis_with_simulations(empirical_data_real, simulated_vecs, empirical_vec, n_components=50, save_path=None, save_all=False):
    """Performs PCA on real data and projects simulated data, then evaluates similarity."""
    print("Starting PCA analysis with simulations...")
    pca = PCA(n_components=n_components)
    Z_real = pca.fit_transform(empirical_data_real)
    Z_sim = pca.transform(simulated_vecs)
    Z_emp = pca.transform(empirical_vec.reshape(1, -1))[0]

    max_components = n_components if save_all else min(n_components, 10)
    n_pages = int(np.ceil(max_components / 10))
    for page in range(n_pages):
        start, end = page * 10, min((page + 1) * 10, max_components)
        n_rows = int(np.ceil((end - start) / 5))
        plt.figure(figsize=(15, 3 * n_rows))
        for i, comp_idx in enumerate(range(start, end)):
            plt.subplot(n_rows, 5, i + 1)
            sns.histplot(Z_sim[:, comp_idx], bins=30, color='orange', alpha=0.6, kde=True, label='Simulations')
            sns.histplot(Z_real[:, comp_idx], bins=30, color='blue', alpha=0.6, kde=True, label='X_ctrl')
            plt.title(f"PC {comp_idx + 1}")
            plt.xlabel("Score")
            plt.ylabel("Count")
            plt.legend()
        plt.tight_layout()
        if save_path:
            filename = f"pca_projection_histograms_{start+1}_to_{end}.png"
            plt.savefig(os.path.join(save_path, filename))
            plt.close()
        else:
            plt.show()

    pearson_global = [
        abs(np.corrcoef(sim_scores[:max_components], Z_emp[:max_components])[0, 1])
        for sim_scores in Z_sim
    ]
    return pearson_global, pca


def compute_FC_matrixes(node_name, list_nodes, list_values, empirical_matrix, empirical_data_real=X_ctrl, save_path="fc_matrix_results"):
    """Runs simulations varying Qi parameters in selected brain regions and compares simulated FC to empirical FC."""
    print(f"Computing FC matrices for region: {node_name}")
    os.makedirs(save_path, exist_ok=True)

    list_inferer = []
    params = default_params.copy()
    params["nodes_Qichange"] = list_nodes
    params["node_name_Qichange"] = node_name

    for i in range(len(params['nodes_Qichange'])):
        params[f"Qi_change_curr_{i}"] = 5.0
        list_inferer.append(f"Qi_change_curr_{i}")

    simulator_wrapped = partial(
        simulator_FC,
        params=params,
        list_inferer=list_inferer,
        kernel_hrf=kernel,
        CTRL=CTRL,
        SUBJ_IDX=SUBJ_IDX,
        USE_PATIENT_SC=USE_PATIENT_SC,
        RANDOM_SC=RANDOM_SC,
        REAL_OBS=REAL_OBS,
        USE_QI_CHANGE=True
    )

    combinations = list(itertools.product(*list_values))
    thetas = [list(comb) for comb in combinations]

    def run_one(theta):
        sim_vec = simulator_wrapped(theta)
        sim_vec_np = sim_vec.detach().cpu().numpy() if hasattr(sim_vec, 'detach') else np.array(sim_vec)
        empirical_np = empirical_matrix.detach().cpu().numpy() if hasattr(empirical_matrix, 'detach') else np.array(empirical_matrix)
        pearson = np.corrcoef(sim_vec_np, empirical_np)[0, 1]
        mae = np.mean(np.abs(sim_vec_np - empirical_np))
        sim_matrix = vector_to_symmetric_matrix(sim_vec_np)
        return {
            "theta": theta,
            "pearson": pearson,
            "mae": mae,
            "sim_vec": sim_vec_np,
            "sim_matrix": sim_matrix
        }

    results = Parallel(n_jobs=-1)(delayed(run_one)(theta) for theta in tqdm(thetas, desc="Simulations"))

    obs_mat = vector_to_symmetric_matrix(empirical_matrix)
    simulated_vecs = np.array([res["sim_vec"] for res in results])
    empirical_vec = empirical_matrix.detach().cpu().numpy() if hasattr(empirical_matrix, 'detach') else np.array(empirical_matrix)

    pearson_global_pca, pca = pca_analysis_with_simulations(empirical_data_real, simulated_vecs, empirical_vec, n_components=10, save_path=save_path)

    for i, res in enumerate(results):
        res["pearson_global_pca"] = pearson_global_pca[i]

    results_sorted = sorted(results, key=lambda r: r["pearson"], reverse=True)
    best_5 = results_sorted[:5]
    worst_5 = results_sorted[-5:]

    error_summary = [{
        "index": int(i),
        "theta": [float(x) for x in res["theta"]],
        "pearson": float(res["pearson"]),
        "mae": float(res["mae"]),
        "pearson_global_pca": float(res["pearson_global_pca"])
    } for i, res in enumerate(results)]

    with open(os.path.join(save_path, "summary_errors.json"), "w") as f:
        json.dump(error_summary, f, indent=2)

    for group_name, group in [("best", best_5), ("worst", worst_5)]:
        for i, res in enumerate(group):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            sns.heatmap(res["sim_matrix"], cmap='coolwarm', ax=axs[0], cbar=True)
            axs[0].set_title(f"Simulated FC ({group_name} #{i+1}) (Theta: {res['theta']})")
            sns.heatmap(obs_mat, cmap='coolwarm', ax=axs[1], cbar=True)
            axs[1].set_title("Empirical FC")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"fc_matrix_{group_name}_{i}.png"))
            plt.close()

    print(f"{len(results)} simulations completed. Results saved to '{save_path}'.")
    return results


def main():
    """
    Run multiple functional connectivity (FC) matrix computations for various brain regions.
    All parameters are pre-defined in a structured list of dictionaries.
    """
    all_runs = [
        {
            "node_name": "cuneus",
            "list_nodes": [[20,54], [21,55], [22,56], [23,57]],
            "list_values": [[4,5,6,10]] * 4,
            "save_path": "results/cuneus_results"
        },
        {
            "node_name": "lingual",
            "list_nodes": [[24], [58]],
            "list_values": [[2,4,5,6,10]] * 2,
            "save_path": "results/g_vs_r_lingual_results"
        },
        {
            "node_name": "salient",
            "list_nodes": [[33, 67], [11,45]],
            "list_values": [[2,4,5,6,10]] * 2,
            "save_path": "results/salient_results"
        },
        {
            "node_name": "hypo",
            "list_nodes": [[25, 59], [26,60], [27,61]],
            "list_values": [[4,5,6,10]] * 3,
            "save_path": "results/hypo_results"
        },
        {
            "node_name": "dmn",
            "list_nodes": [dmn_frontale, dmn_cingulaire, dmn_parietale],
            "list_values": [[4,5,6,10]] * 3,
            "save_path": "results/dmn_results"
        },
        {
            "node_name": "moteur",
            "list_nodes": [[9, 43], [15, 49], [10, 44]],
            "list_values": [[4,5,6,10], [10,4,5,6], [10,4,5,6]],
            "save_path": "results/moteur_results"
        },
        {
            "node_name": "executif",
            "list_nodes": [[6, 40], [7, 41], [8, 42]],
            "list_values": [[4,6,10]] * 3,
            "save_path": "results/executif_results"
        },
        {
            "node_name": "auditif",
            "list_nodes": [[31, 65], [32, 66]],
            "list_values": [[2,4,5,6]] * 2,
            "save_path": "results/auditif_results"
        },
        {
            "node_name": "dmn_fine",
            "list_nodes": [[13, 47], [19, 53], [2, 36]],
            "list_values": [[4,6,10]] * 3,
            "save_path": "results/dmn_fine_results"
        },
        {
            "node_name": "vision_hierarchie",
            "list_nodes": [[21, 55], [22, 56], [24, 58]],
            "list_values": [[4,5,6]] * 3,
            "save_path": "results/vision_hierarchie_results"
        },
        {
            "node_name": "salience_core",
            "list_nodes": [[11, 12, 45, 46, 33, 67]],
            "list_values": [[2,4,6,10]],
            "save_path": "results/salience_core_results"
        }
    ]

    # Global empirical data loaded or passed externally
    empirical_matrix = ...  # e.g. np.load(...)
    empirical_data_real = X_ctrl  # If using PCA-based analysis

    for run in all_runs:
        compute_FC_matrixes(
            node_name=run["node_name"],
            list_nodes=run["list_nodes"],
            list_values=run["list_values"],
            empirical_matrix=empirical_matrix,
            empirical_data_real=empirical_data_real,
            save_path=run["save_path"]
        )



if __name__ == "__main__":
    main()
    print("All computations completed successfully.")