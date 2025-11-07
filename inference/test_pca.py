"""
This script provides a set of tools for analyzing and visualizing functional connectivity (FC) matrices 
using Principal Component Analysis (PCA). It includes functions for:

- Loading and preprocessing FC data
- Performing PCA and visualizing explained variance
- Assessing the discriminability of PCA components between groups (e.g., CTRL vs SCHZ)
- Simulating new samples based on group means
- Reconstructing FC matrices from PCA-encoded representations
- Visualizing original and reconstructed FC matrices
- Saving and loading trained PCA models

It is designed for exploratory analysis of high-dimensional brain connectivity data in a classification context.
"""



import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import joblib
from scipy.stats import pearsonr
import seaborn as sns
from params import PARC_SIZE

# pca components to try : whiten = True



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




def load_data_with_labels(data_path):
    """
    Charge les données et leurs étiquettes à partir d'un fichier .npz contenant les clés 'X' et 'y'.

    Args:
        data_path (str): Chemin vers le fichier .npz contenant les données.

    Returns:
        tuple: 
            - X (ndarray): Données d'entrée de forme [n_samples, n_features].
            - y (ndarray): Étiquettes associées (valeurs binaires attendues).

    Raises:
        FileNotFoundError: Si le fichier spécifié n'existe pas.
        KeyError: Si les clés 'X' et 'y' ne sont pas présentes dans le fichier.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise KeyError("Les clés 'X' et 'y' sont requises dans le fichier.")
    X = data["X"]
    y = data["y"]
    return X, y


def split_groups(X, y):
    """
    Sépare les données en deux groupes selon les étiquettes : groupe contrôle (y == 0) et groupe patient (y == 1).

    Args:
        X (ndarray): Données d'entrée [n_samples, n_features].
        y (ndarray): Étiquettes binaires (0 = contrôle, 1 = patient).

    Returns:
        tuple:
            - X_ctrl (ndarray): Données du groupe contrôle (y == 0).
            - X_schz (ndarray): Données du groupe patient (y == 1).
    """
    return X[y == 0], X[y == 1]  # X_ctrl, X_schz


def plot_cumulative_pca(X, save_path=None, show=True, title="Cumulative PCA"):
    pca = PCA()
    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--', color='blue')
    plt.axhline(0.9, color='red', linestyle=':', label='90% variance')
    plt.axhline(0.95, color='green', linestyle=':', label='95% variance')
    plt.xlabel("Nombre de composantes principales")
    plt.ylabel("Variance expliquée cumulée")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[✓] Figure sauvegardée dans: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return pca


def analyze_component_discriminability(X, y, n_components=10, per_page=10, PLOT=True):
    """
    Affiche les histogrammes de projection des sujets (CTRL vs SCHZ) pour les composantes principales.

    Args:
        X (ndarray): Données [n_samples, n_features]
        y (ndarray): Labels binaires (0 = CTRL, 1 = SCHZ)
        n_components (int): Nombre de composantes à considérer
        per_page (int): Nombre de composantes par figure
    """
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    if PLOT:
        n_pages = int(np.ceil(n_components / per_page))

        for page in range(n_pages):
            start = page * per_page
            end = min((page + 1) * per_page, n_components)
            n_subplots = end - start
            n_rows = int(np.ceil(n_subplots / 5))

            plt.figure(figsize=(15, 3 * n_rows))

            for i, comp_idx in enumerate(range(start, end)):
                plt.subplot(n_rows, 5, i + 1)
                plt.hist(Z[y == 0, comp_idx], alpha=0.6, label="CTRL", bins=20)
                plt.hist(Z[y == 1, comp_idx], alpha=0.6, label="SCHZ", bins=20)
                plt.title(f"PC {comp_idx + 1}")
                plt.xlabel("Score")
                plt.ylabel("Count")

            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                plt.legend(loc='upper right')

            plt.suptitle(f"Composantes {start + 1} à {end}", fontsize=16)
            plt.tight_layout()
            plt.show()

    return Z, pca



def simulate_data(mean_vector, std=0.1, n_samples=10):
    return mean_vector + np.random.normal(0, std, size=(n_samples, mean_vector.shape[0]))


def pca_encode(X, pca):
    return pca.transform(X)


def pca_decode(Z, pca):
    return pca.inverse_transform(Z)


def save_pca(pca, filepath):
    joblib.dump(pca, filepath)
    print(f"[✓] PCA sauvegardée dans {filepath}")


def load_pca(filepath):
    return joblib.load(filepath)




def test_reconstruction(X, pca, n_components=10):
    # Crée une nouvelle PCA avec les n premières composantes
    pca_tmp = PCA(n_components=n_components)
    pca_tmp.components_ = pca.components_[:n_components]
    pca_tmp.mean_ = pca.mean_
    pca_tmp.explained_variance_ = pca.explained_variance_[:n_components]
    pca_tmp.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_components]

    Z = pca_encode(X, pca_tmp)
    X_reconstructed = pca_decode(Z, pca_tmp)

    mse = np.mean((X - X_reconstructed)**2)
    mae = np.mean(np.abs(X - X_reconstructed))

    # Pearson corr : moyenné sur chaque paire (ligne par ligne)
    corrs = [pearsonr(X[i], X_reconstructed[i])[0] for i in range(X.shape[0])]
    avg_corr = np.mean(corrs)

    print(f"[✓] MSE de reconstruction avec {n_components} composantes: {mse:.4f}")
    print(f"[✓] MAE de reconstruction: {mae:.4f}")
    print(f"[✓] Corrélation de Pearson moyenne: {avg_corr:.4f}")

    return mse, mae, avg_corr, X_reconstructed

def show_random_reconstruction(X, pca, size=PARC_SIZE, title="Reconstruction aléatoire", n_components=None):
    """
    Tire une matrice aléatoire dans X, encode via PCA, la reconstruit et affiche original vs reconstruction.
    
    Args:
        X (ndarray): données réelles [n_samples, n_features]
        pca (PCA): objet PCA déjà entraîné
        size (int): taille de la matrice symétrique
        title (str): titre du plot
        n_components (int or None): nombre de composantes à utiliser (par défaut: toutes)
    """
    idx = np.random.randint(0, X.shape[0])
    x = X[idx]

    # Si on veut réduire le nombre de composantes
    if n_components is not None:
        pca_tmp = PCA(n_components=n_components)
        pca_tmp.components_ = pca.components_[:n_components]
        pca_tmp.mean_ = pca.mean_
        pca_tmp.explained_variance_ = pca.explained_variance_[:n_components]
        pca_tmp.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_components]
    else:
        pca_tmp = pca

    # Encode & decode
    z = pca_encode([x], pca_tmp)
    x_rec = pca_decode(z, pca_tmp)[0]

    # Corrélation de Pearson
    r, _ = pearsonr(x, x_rec)

    # Reconstruire les matrices symétriques
    mat_orig = vector_to_symmetric_matrix(x, size)
    mat_rec = vector_to_symmetric_matrix(x_rec, size)

    # Affichage
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sns.heatmap(mat_orig, ax=axes[0], cmap="coolwarm", square=True, cbar=False)
    axes[0].set_title("Original")

    sns.heatmap(mat_rec, ax=axes[1], cmap="coolwarm", square=True, cbar=False)
    axes[1].set_title(f"Reconstruit\n(r = {r:.4f})")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()



def plot_fc_matrices(matrices, n_cols=5, title="FC matrices", vmin=None, vmax=None, symmetric_input=True, size=PARC_SIZE):
    """
    Affiche une grille de matrices FC avec seaborn.

    Args:
        matrices (ndarray): [n_samples, n_features] ou [n_samples, size, size]
        n_cols (int): nombre de colonnes dans la grille
        title (str): titre global
        vmin, vmax (float): plage des couleurs
        symmetric_input (bool): si True, applique vector_to_symmetric_matrix() élément par élément
        size (int): taille des matrices symétriques à reconstruire si input vectorisé
    """
    n = matrices.shape[0]

    # Si les matrices sont vectorisées : on les reconstruit une par une
    if matrices.ndim == 2 and symmetric_input:
        matrices = np.array([vector_to_symmetric_matrix(vec, size=size) for vec in matrices])

    # Vérification de forme
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Les matrices doivent être de forme [n, size, size] après transformation.")

    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        sns.heatmap(
            matrices[i],
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            square=True,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(f"Sample {i+1}", fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()



def plot_multiple_reconstructions(X, pca, size=PARC_SIZE, components_list=[1, 5, 10, 20, 30, 40, 50], random_idx=None):
    """
    Affiche les reconstructions d'une FC matrix pour différents nombres de composantes PCA,
    comparées à l'originale.

    Args:
        X (ndarray): données [n_samples, n_features]
        pca (PCA): objet PCA préalablement entraîné
        size (int): taille des matrices symétriques à reconstruire
        components_list (list of int): liste des nombres de composantes à tester
        random_idx (int or None): index de l'échantillon à utiliser (aléatoire si None)
    """
    if random_idx is None:
        random_idx = np.random.randint(0, X.shape[0])
    x = X[random_idx]

    # Matrice originale
    mat_orig = vector_to_symmetric_matrix(x, size)

    # Initialisation des figures
    n_recons = len(components_list)
    n_cols = 4
    n_rows = int(np.ceil((n_recons + 1) / n_cols))  # +1 pour l'originale

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Afficher la matrice originale
    sns.heatmap(mat_orig, ax=axes[0], cmap="coolwarm", square=True, cbar=False)
    axes[0].set_title("Original", fontsize=12)

    # Reconstructions
    for i, n_comp in enumerate(components_list, start=1):
        # Création d'une PCA temporaire réduite
        pca_tmp = PCA(n_components=n_comp)
        pca_tmp.components_ = pca.components_[:n_comp]
        pca_tmp.mean_ = pca.mean_
        pca_tmp.explained_variance_ = pca.explained_variance_[:n_comp]
        pca_tmp.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_comp]

        z = pca_encode([x], pca_tmp)
        x_rec = pca_decode(z, pca_tmp)[0]
        mat_rec = vector_to_symmetric_matrix(x_rec, size)

        # Pearson correlation
        r, _ = pearsonr(x, x_rec)

        sns.heatmap(mat_rec, ax=axes[i], cmap="coolwarm", square=True, cbar=False)
        axes[i].set_title(f"{n_comp} comp.\nr = {r:.2f}", fontsize=10)

    # Retirer axes inutilisés
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Reconstructions PCA par nombre de composantes", fontsize=16)
    plt.tight_layout()
    plt.show()



def main_test_pca():
    # Charger les données avec labels
    X, y = load_data_with_labels(f"generated_data/FC_data_flattened_{PARC_SIZE}.npz")

    # Tracer la variance expliquée cumulée
    pca = plot_cumulative_pca(X, save_path="results/cumulative_pca.png", show=False)

    # Séparer les groupes
    X_ctrl, X_schz = split_groups(X, y)

    # Analyse des composantes discriminantes
    Z, pca_reduced = analyze_component_discriminability(X, y, n_components=50)

    # Suppose que X et pca sont déjà chargés
    plot_multiple_reconstructions(X, pca_reduced, size=PARC_SIZE)


    # Simulation autour des moyennes
    mean_ctrl = X_ctrl.mean(axis=0)
    sim_ctrl = simulate_data(mean_ctrl, std=0.05, n_samples=5)
    plot_fc_matrices(sim_ctrl, title="Simulations CTRL")
    # Reconstruction test
    mse, mae, corr, X_rec = test_reconstruction(X_ctrl, pca, n_components=50)

    # Visualisation reconstruction
    plot_fc_matrices(X_rec[:10], title="Reconstructions CTRL (10 premiers)")


    show_random_reconstruction(X_ctrl, pca, size=PARC_SIZE, n_components=50, title="CTRL - Reconstruction aléatoire")
    show_random_reconstruction(X_schz, pca, size=PARC_SIZE, n_components=50, title="SCHZ - Reconstruction aléatoire")

    # Sauvegarder le modèle PCA
    save_pca(pca_reduced, "pca_all_real_data.joblib")



if __name__ == "__main__":
    main_test_pca()