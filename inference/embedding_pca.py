import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np

class EmbeddingNetWithPCA(nn.Module):
    def __init__(self, pca_model: PCA, apply_mlp: bool = False, mlp_hidden: int = 64):
        """
        Embedding net utilisant une PCA pré-entraînée (sur données réelles) compatible avec sbi.
        
        Args:
            pca_model: Instance sklearn.decomposition.PCA entraînée.
            apply_mlp: Si True, ajoute un MLP après la PCA.
            mlp_hidden: Nombre de neurones cachés si MLP activé.
        """
        super().__init__()
        self.pca_model = pca_model
        self.apply_mlp = apply_mlp
        self.pca_dim = pca_model.n_components_

        if apply_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.pca_dim, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, mlp_hidden // 2),
                nn.ReLU(),
                nn.Linear(mlp_hidden // 2, self.pca_dim)
            )

    def forward(self, x):
        """
        x : torch.Tensor de forme (batch_size, dim_observation)
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            raise ValueError("L'entrée x doit être un Tensor PyTorch")

        # Application de la PCA (pas entraînée ici, doit être fournie)
        x_pca = self.pca_model.transform(x_np)
        x_pca = torch.tensor(x_pca, dtype=torch.float32, device=x.device)

        if self.apply_mlp:
            x_pca = self.mlp(x_pca)

        return x_pca
