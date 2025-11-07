import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os 
import sys
from scipy import stats
from scipy import signal
sys.path.append(os.path.abspath(".."))



labels = np.load("save_file/stim_region_labels.npy", allow_pickle=True)
bold = np.load("save_file/sig_BOLD__b_0_Qi_PCC5.0_repeatedinsulastim_0.0EtoEIratio1.0_coupling0.15seed0_noise0.000315.npy", allow_pickle=True)
bold_TAN = np.load("save_file/sig_BOLD__b_0_Qi_PCC[5.0]_repeatedinsulastim_0.0EtoEIratio1.0_coupling0.15seed0_noise0.000315_TAN.npy", allow_pickle=True)
print('done loading')
print(labels)


# Créer une figure avec 2 lignes et 2 colonnes
fig, axs = plt.subplots(2, 2, figsize=(16, 10))

# --------- Ligne 1 : BOLD classique ---------
sns.heatmap(bold, annot=False, cmap="coolwarm", cbar=True, ax=axs[0, 0])
axs[0, 0].set_title("Heatmap of BOLD signal")

corr_matrix = np.corrcoef(bold)
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", cbar=True, ax=axs[0, 1])
axs[0, 1].set_title("Functional Connectivity (Pearson)")

# --------- Ligne 2 : BOLD TAN ---------
sns.heatmap(bold_TAN, annot=False, cmap="coolwarm", cbar=True, ax=axs[1, 0])
axs[1, 0].set_title("Heatmap of BOLD signal (TAN)")

corr_matrix_TAN = np.corrcoef(bold_TAN)
sns.heatmap(corr_matrix_TAN, annot=False, cmap="coolwarm", cbar=True, ax=axs[1, 1])
axs[1, 1].set_title("Functional Connectivity (Pearson) - TAN")

# Mise en page
plt.tight_layout()
plt.show()
print("FC (standard) min/max :", corr_matrix.min(), corr_matrix.max())
print("FC (TAN) min/max :", corr_matrix_TAN.min(), corr_matrix_TAN.max())

diff = np.abs(bold - bold_TAN)
print("Erreur absolue moyenne :", np.mean(diff))
print("Erreur max :", np.max(diff))

bold_flat = bold.flatten()
bold_TAN_flat = bold_TAN.flatten()

corr = np.corrcoef(bold_flat, bold_TAN_flat)[0, 1]
print("Corrélation entre les deux BOLD :", corr)

