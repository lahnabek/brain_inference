"""
sbi_simulation.py

Main script to run Simulation-Based Inference (SBI) for brain connectivity models,
supporting both single-round and multi-round inference.

OVERVIEW:
---------
This script serves as the entry point to launch SBI inference using
simulation functions and inference routines.

It provides the main execution functions:
- `main_inf()` to run inference (single or multi-round based on a mode flag)
- `test_simulator()` to validate and visualize the BOLD signal simulator outputs
- `main_pca()` to perform PCA on real data and run inference using PCA components

KEY MODULES AND ORGANIZATION:
-----------------------------
- Inference logic is implemented in the `inference_core` module, which
  contains functions like `single_round_inference` and `multi_round_inference`.
- Simulation and utility functions, including `simulator_FC`, data loading,
  and helper functions, reside in the `utils` module.
- Parameters and hyperparameters are centralized in `params.py`.


PARAMETER CONFIGURATION:
------------------------
Modify the `params.py` file to configure:
- Prior bounds (`borne_min`, `borne_max`)
- Neural network models (`MODEL_NN`)
- Number of simulations (`NUM_SIMULATION`)
- Number of samples (`NUM_SAMPLES`)
- Observed data configuration (`REAL_OBS`, `CTRL`, `SUBJ_IDX`, etc.)
"""

import sys
import os

# Détecte automatiquement la racine du projet (où se trouve ce script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from params import *
from inference_core import *
from utils import *
from params import default_params
import numpy as np
import torch
from test_pca import load_pca, load_data_with_labels, split_groups, analyze_component_discriminability


def main_inf():
    """
    Main entry point to run the inference procedure.

    Depending on the global variable MODE, this function runs either a single-round or
    multi-round inference using the specified simulator function and parameters.

    The function handles configuration for the inference, such as the number of samples,
    simulation parameters, model choice, and data settings, and then calls the appropriate
    inference routine.

    Raises:
        ValueError: If MODE is not 'single' or 'multi'.
    """

    # Run single-round inference if MODE is set to "single"
    if MODE == "single":
        print("▶ Starting single-round inference...")
        single_round_inference(
            simulator_func=simulator_FC,      # Function to simulate functional connectivity data
            params_obs=params_obs,             # Observation parameters
            list_inferer=list_inferer,        # List of parameters to infer
            borne_min=borne_min,               # Lower bounds of parameters
            borne_max=borne_max,               # Upper bounds of parameters
            MODEL_NN=MODEL_NN,                 # Neural network model specification
            EXP_NAME=EXP_NAME,                 # Experiment name for saving results
            NUM_SIMULATION=NUM_SIMULATION,    # Number of simulations to run
            NUM_SAMPLES=NUM_SAMPLES,           # Number of posterior samples to generate
            REAL_OBS=REAL_OBS,                 # Real observed data (if any)
            PRIOR_TYPE=PRIOR_TYPE,             # Type of prior distribution used
            CTRL=CTRL,                        # Control flag (e.g., control vs patient)
            SUBJ_IDX=SUBJ_IDX,                 # Subject index to process
            RANDOM_SC=RANDOM_SC,               # Flag to use random structural connectivity
            USE_PATIENT_SC=USE_PATIENT_SC,    # Flag to use patient-specific structural connectivity
            USE_QI_CHANGE=USE_QI_CHANGE,      # Flag to apply quality index changes
            save_dir=save_dir,                # Directory to save inference outputs
            n_components_pca=10               # Number of PCA components to use in analysis
        )

    # Run multi-round inference if MODE is set to "multi"
    elif MODE == "multi":
        print("▶ Starting multi-round inference...")
        multi_round_inference(
            NUM_ROUNDS=3,                     # Number of inference rounds
            simulator_func=simulator_FC,      # Simulation function for FC data
            params_obs=params_obs,             # Observation parameters
            list_inferer=list_inferer,        # Parameters to infer
            borne_min=borne_min,               # Lower bounds
            borne_max=borne_max,               # Upper bounds
            MODEL_NN=MODEL_NN,                 # Neural network model
            EXP_NAME=EXP_NAME,                 # Experiment identifier
            NUM_SIMULATION=NUM_SIMULATION,    # Simulations per round
            NUM_SAMPLES=NUM_SAMPLES,           # Samples per round
            REAL_OBS=REAL_OBS,                 # Real observed data
            PRIOR_TYPE=PRIOR_TYPE,             # Prior distribution type
            CTRL=CTRL,                        # Control flag
            SUBJ_IDX=SUBJ_IDX,                 # Subject index
            RANDOM_SC=RANDOM_SC,               # Use random SC flag
            USE_PATIENT_SC=USE_PATIENT_SC,    # Use patient SC flag
            USE_QI_CHANGE=USE_QI_CHANGE,      # Use QI change flag
            save_dir=save_dir,                # Output directory
            n_components_pca=10               # PCA components for analysis
        )

    # Raise an error if MODE has an unexpected value
    else:
        raise ValueError(f"Invalid MODE '{MODE}'. Use 'single' or 'multi'.")



def test_simulator():
    """
    Test the BOLD signal simulator and visualize its outputs.

    This function runs the BOLD simulator with default parameters and settings,
    saves the generated BOLD time series, and produces various plots to help
    verify the simulation's correctness and characteristics.

    Steps included:
    - Run the simulator with specified parameters.
    - Save the simulated BOLD signals for later inspection.
    - Plot example BOLD time series for selected brain regions.
    - Display a heatmap of the entire BOLD dataset.
    - Print basic statistics (mean, std, min, max) of BOLD signals.
    - Compute and plot the functional connectivity (FC) matrix based on Pearson correlation.
    - Reconstruct and visualize the FC matrix from its vectorized form.
    """
    print("▶ Starting simulation test for the BOLD simulator...")

    # Run the BOLD simulator with the configured parameters and settings
    print("🔧 Simulation parameters:")
    bold = simulator_BOLD(
        kernel_hrf=kernel,
        params=default_params,
        SUBJ_IDX=SUBJ_IDX,
        CTRL=CTRL,
        RANDOM_SC=RANDOM_SC,
        USE_PATIENT_SC=USE_PATIENT_SC,
        REAL_OBS=REAL_OBS
    )

    # Save the simulated BOLD data to a .npy file for future verification
    np.save('save_file/abc.npy', bold)
    print("done saving")

    # Extract number of brain regions and time points from the simulated data shape
    n_regions, n_timepoints = bold.shape

    # Plot BOLD time series for 5 evenly spaced regions
    selected_regions = np.linspace(0, n_regions - 1, 5, dtype=int)
    plt.figure(figsize=(12, 6))
    for region in selected_regions:
        plt.plot(bold[region], label=f"Region {region}")
    plt.xlabel("Time (TRs)")
    plt.ylabel("BOLD Signal")
    plt.title("Example BOLD Signals")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Display a heatmap of the entire BOLD signals matrix
    sns.heatmap(bold, annot=False, cmap="coolwarm", cbar=True)
    plt.title("BOLD Signals Heatmap")
    plt.xlabel("Time (TRs)")
    plt.ylabel("Regions")
    plt.tight_layout()
    plt.show()

    # Calculate and print mean and standard deviation of BOLD signals per region
    bold_means = bold.mean(axis=1)
    bold_stds = bold.std(axis=1)
    print("Mean BOLD signals (per region):")
    print(bold_means)
    print("\nStandard deviation of BOLD signals (per region):")
    print(bold_stds)

    # Print global min, max and mean values of the BOLD signals
    print(f"\nGlobal BOLD min: {bold.min():.4f} / max: {bold.max():.4f} / mean: {bold.mean():.4f}")

    # Compute the functional connectivity (FC) matrix using Pearson correlation
    FC = np.corrcoef(bold)
    plt.figure(figsize=(8, 6))
    sns.heatmap(FC, cmap='coolwarm', annot=False, cbar=True)
    plt.title("Functional Connectivity (Pearson) Matrix")
    plt.xlabel("Regions")
    plt.ylabel("Regions")
    plt.tight_layout()
    plt.show()

    # Extract the upper triangular values (vectorized FC) without the diagonal
    triu_idx = np.triu_indices_from(FC, k=1)
    FC_vector = FC[triu_idx]
    FC_torch = torch.tensor(FC_vector, dtype=torch.float32)

    # Calculate the dimension of the original FC matrix from the vector length
    fc_dim = int((1 + np.sqrt(1 + 8 * len(FC_vector))) // 2)
    print("\nFC vector shape:", FC_torch.shape)
    print("FC dim (reconstructed):", fc_dim)

    # Reconstruct the symmetric FC matrix from the vectorized form
    vect_FC = vector_to_symmetric_matrix(FC_torch.numpy(), fc_dim)
    plt.figure(figsize=(8, 6))
    sns.heatmap(vect_FC, cmap='coolwarm', annot=False, cbar=True)
    plt.title("Reconstructed Functional Connectivity Matrix")
    plt.xlabel("Regions")
    plt.ylabel("Regions")
    plt.tight_layout()
    plt.show()


def main_pca():
    """
    Perform PCA on real data and run a single-round inference using the PCA model.

    This function loads flattened functional connectivity (FC) data with labels,
    splits the data into control and patient groups, computes PCA and evaluates
    component discriminability, then runs the inference pipeline using the PCA
    model to reduce data dimensionality.

    Steps included:
    - Load real FC data and corresponding labels.
    - Split data into control and schizophrenia groups.
    - Analyze discriminability of PCA components (up to 50 components).
    - Run single-round inference with the PCA model included.
    """
    # Load flattened FC data and labels from a .npz file
    X, y = load_data_with_labels(f"generated_data/FC_data_flattened_{PARC_SIZE}.npz")

    # Split data into control and patient groups for potential group-specific analysis
    X_ctrl, X_schz = split_groups(X, y)  # Note: If plotting by group, y must be defined accordingly

    # Compute PCA and analyze how well components discriminate between groups (no plot)
    _, pca_model = analyze_component_discriminability(X, y, n_components=50, PLOT=False)

    print("▶ Starting single-round inference with PCA on real data...")

    # Run the inference pipeline with the PCA model passed as an argument
    single_round_inference(
        simulator_func=simulator_FC,
        params_obs=params_obs,
        list_inferer=list_inferer,
        borne_min=borne_min,
        borne_max=borne_max,
        MODEL_NN=MODEL_NN,
        EXP_NAME=EXP_NAME,
        NUM_SIMULATION=NUM_SIMULATION,
        NUM_SAMPLES=NUM_SAMPLES,
        REAL_OBS=REAL_OBS,
        PRIOR_TYPE=PRIOR_TYPE,
        CTRL=CTRL,
        SUBJ_IDX=SUBJ_IDX,
        RANDOM_SC=RANDOM_SC,
        USE_PATIENT_SC=USE_PATIENT_SC,
        USE_QI_CHANGE=USE_QI_CHANGE,
        save_dir=save_dir,
        pca_model=pca_model,
    )




    
if __name__ == "__main__":
    test_simulator()




    

    

