"""
SC/FC Data Preparation Pipeline for Inference with The Virtual Brain (TVB)

This script performs the following steps:
1. Extract and filter structural connectivity (SC) and functional connectivity (FC) matrices from a .mat dataset,
   using a specified parcellation scale.
2. Save the corresponding cortical region names to a CSV file.
3. Extract FC matrices, flatten their upper triangular parts, and store them in a compressed format for later use.
4. Generate structural connectomes (SC) as .zip files compatible with TVB simulations,
   saving them under `data_tvb/real_connectome_for_tvb`.

Outputs:
- Filtered SC/FC matrices (.h5), cortical regions (.csv), and flattened FC data (.npz) → in ../generated_data/
- TVB-compatible structural connectome .zip files → in ./data_tvb/real_connectome_for_tvb/
"""

import os
import numpy as np
import h5py
import scipy.io as sio
import pandas as pd
import zipfile
import json

# === Global Configuration Parameters ===



# Available parcellation discretisations with their internal indices.
discretisation_possible = {83: 0, 129: 1, 234: 2, 463: 3, 1015: 4}

# Choose one parcellation scale to filter the data. 
# This corresponds to the number of regions; change if needed.
discretisation_chosen = discretisation_possible[83]

# Define key project directories assuming this script is run from within tvb_model_reference.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
original_data_path = os.path.join(project_root, "original_dataset")                     # Raw dataset location
generated_data_path = os.path.join(project_root, "generated_data")                      # Folder for processed outputs
tvb_zip_output_path = os.path.join(project_root, "data_tvb", "real_connectome_for_tvb") # TVB SC .zip output

# Ensure output directories exist
os.makedirs(generated_data_path, exist_ok=True)
os.makedirs(tvb_zip_output_path, exist_ok=True)



# === Step 1: Filter SC/FC Matrices Extraction ===



# Load parcellation labels to identify cortical regions at chosen scale
labels_file = os.path.join(original_data_path, "labels_index_CORTICAL_Laus2008_all_scales.mat")
file_path = os.path.join(original_data_path, "27_SCHZ_CTRL_dataset.mat")

labels_data = sio.loadmat(labels_file)

# Get indices of cortical regions for chosen parcellation scale (convert to zero-based) (reduces number of regions)
ixc = labels_data["ixc"][0, discretisation_chosen][0] - 1

# Output file path for filtered SC/FC data in HDF5 format
output_h5_path = os.path.join(generated_data_path, f"SC_FC_dataset_filtered_{len(ixc)}x{len(ixc)}.h5")

with h5py.File(file_path, "r") as f:
    with h5py.File(output_h5_path, "w") as hf:
        # Iterate over dataset groups, skip demographics metadata
        for key in f["SC_FC_Connectomes"].keys():
            if key == "demographics":
                continue
            group = f["SC_FC_Connectomes"][key]
            hf.create_group(key)
            for sub_key in group.keys():
                # Extract matrix data at chosen discretisation
                data = group[sub_key][discretisation_chosen, 0]
                matrix = f[data][:]
                # Filter matrix rows and columns by cortical indices
                filtered_matrix = matrix[:, ixc, :][:, :, ixc]
                hf[key].create_dataset(sub_key, data=filtered_matrix)



# === Step 2: Save Cortical Region Names ===

# Extract cortical region names corresponding to chosen discretisation
llist = labels_data["llist"]
region_names = [llist[0, discretisation_chosen][i, 0][0] for i in range(llist[0, discretisation_chosen].shape[0])]

# Save cortical region names with index starting at 0 (not 1)
df_regions = pd.DataFrame({
    "Index": range(len(region_names)),  # Start at 0
    "Cortical Region": region_names
})

# Save to CSV
csv_path = os.path.join(generated_data_path, f"regions_corticales_{len(region_names)}_indices.csv")
df_regions.to_csv(csv_path, index=False)



print(f"[✓] Region labels saved to:\n - {csv_path}")



# === Step 3: Flatten FC Matrices for Machine Learning ===

def extract_flattened_fc(h5_file, key="FC_correlation"):
    """
    Extract FC matrices, flatten their upper triangular part, and label them by group.
    
    Args:
        h5_file (str): Path to the HDF5 file with filtered FC data.
        key (str): Dataset key for FC correlation matrices.
        
    Returns:
        X (np.array): Flattened FC data for all subjects.
        y (np.array): Labels (0=control, 1=schizophrenia).
        X_ctrl (np.array): Flattened FC for control group.
        X_schz (np.array): Flattened FC for schizophrenia group.
    """
    X, y, X_ctrl, X_schz = [], [], [], []
    with h5py.File(h5_file, "r") as f:
        for group, label in zip(["ctrl", "schz"], [0, 1]):
            for i in range(f[key][group].shape[0]):
                mat = f[key][group][i]
                # Extract upper triangular part without diagonal to avoid redundancy
                upper_triangle = mat[np.triu_indices_from(mat, k=1)]
                X.append(upper_triangle)
                y.append(label)
                (X_ctrl if label == 0 else X_schz).append(upper_triangle)
    return np.array(X), np.array(y), np.array(X_ctrl), np.array(X_schz)

# Run the flattening and save the results as compressed .npz file
X, y, X_ctrl, X_schz = extract_flattened_fc(output_h5_path)
npz_output = os.path.join(generated_data_path, f"FC_data_flattened_{len(ixc)}.npz")
np.savez(npz_output, X=X, y=y, X_schz=X_schz, X_ctrl=X_ctrl)

# === Step 4: Generate TVB-compatible Structural Connectome .zip Files ===

def generate_all_SC_tvb_zips(h5_path, output_dir, parcellation_size):
    """
    Generate .zip files containing SC connectomes formatted for TVB.

    Args:
        h5_path (str): Path to filtered SC data (.h5).
        output_dir (str): Directory to save generated .zip files.
        parcellation_size (int): Number of regions in parcellation.

    Process:
        For each subject in control and schizophrenia groups,
        save weights, tract lengths, region labels, and dummy centre coordinates,
        then archive them as a .zip file compatible with TVB.
    """
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        for group in ["ctrl", "schz"]:
            sc_data = f["SC_number_of_fibers"][group]
            length_data = f["SC_length_of_fibers"][group]
            for subj_idx in range(sc_data.shape[0]):
                subj_name = f"patient_{group}_{subj_idx}"
                subj_dir = os.path.join(output_dir, subj_name)
                os.makedirs(subj_dir, exist_ok=True)

                # Save structural weights and fiber lengths as numpy arrays
                np.save(os.path.join(subj_dir, "weights.npy"), sc_data[subj_idx])
                np.save(os.path.join(subj_dir, "tract_lengths.npy"), length_data[subj_idx])

                # Write region labels as plain text file
                with open(os.path.join(subj_dir, "region_labels.txt"), "w") as f_labels:
                    for i in range(parcellation_size):
                        f_labels.write(f"Region_{i}\n")

                # Dummy centre coordinates (replace with real if available)
                np.savetxt(os.path.join(subj_dir, "centres.txt"), np.random.rand(parcellation_size, 3))

                # Create .zip archive of all necessary files for TVB
                zip_path = os.path.join(output_dir, f"{subj_name}.zip")
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    for fname in ["weights.npy", "tract_lengths.npy", "region_labels.txt", "centres.txt"]:
                        zipf.write(os.path.join(subj_dir, fname), arcname=fname)

                # Clean up temporary folder
                for f_tmp in os.listdir(subj_dir):
                    os.remove(os.path.join(subj_dir, f_tmp))
                os.rmdir(subj_dir)



# Generate the TVB SC .zip connectomes with chosen parcellation size
generate_all_SC_tvb_zips(output_h5_path, tvb_zip_output_path, parcellation_size=len(ixc))

print("[✓] Data successfully generated.")

