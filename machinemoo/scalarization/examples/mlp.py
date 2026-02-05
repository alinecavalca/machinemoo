import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.typing as npt
from typing import Any, Tuple, Optional

# Imports do framework MachineMoo
from machinemoo.scalarization.core import BaseScalarizer, LipschitzTorchMixin
from machinemoo.scalarization.lipschitz_estimation import calculate_torch_lipschitz_constant
from machinemoo.utils.typing import MatrixLike
from machinemoo.utils.logging_config import get_logger

logger = get_logger(__name__)

class MLP(nn.Module):
    """
    Rede Neural Simples (Perceptron Multicamadas).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class MLPScalarization(LipschitzTorchMixin, BaseScalarizer):
    """
    Estratégia de Escalarização para MLP com suporte a Fairness.
    Herda de LipschitzTorchMixin para cálculo automático de limites inferiores (objs_lb).
    """
    def __init__(
        self,
        num_objs: int,
        X: MatrixLike,
        y: Any,
        fair_feat: str,
        epochs: int = 200,
        lower_bound_estimate: str | float = "zero", # "zero" ou "lipschitz"
        hidden_dim: int = 64,
        learning_rate: float = 0.001
    ) -> None:
        # Inicializa a classe base (define self._M, self.lower_bound_estimate)
        super().__init__(num_objs=num_objs, lower_bound_estimate=lower_bound_estimate)
        
        self.fair_feat = fair_feat
        # Garante ordenação consistente dos grupos
        self.fair_att = np.unique(X[fair_feat]).tolist()
        
        # Prepara dados (converte para tensor)
        X_np = np.array(X) if hasattr(X, 'to_numpy') else X
        y_np = np.array(y) if hasattr(y, 'to_numpy') else y
        
        self.X_tensor = torch.tensor(X_np, dtype=torch.float32)
        self.y_tensor = torch.tensor(y_np, dtype=torch.float32)
        
        # Referências originais (Pandas) para facilitar manipulação de índices
        self.X_pd = X
        self.y_pd = y

        # Configuração do Modelo
        input_dim = X.shape[1]
        output_dim = 1
        self.num_epochs = epochs
        self.learning_rate = learning_rate
        
        self.model = MLP(input_dim, hidden_dim, output_dim)

        # Pré-cálculo da Constante de Lipschitz (se necessário)
        if self.lower_bound_estimate == "lipschitz":
            self.L = np.zeros(self.M) 
            group_values = np.array(self.X_pd[self.fair_feat])
            
            for i, g in enumerate(self.fair_att):
                if i >= self.M: 
                    break
                
                mask = (group_values == g)
                Xg = self.X_tensor[mask]
                yg = self.y_tensor[mask]

                # Função de perda local para estimativa da curvatura (Hessiana)
                def loss_fn_closure():
                     out = self.model(Xg).squeeze()
                     return nn.BCELoss()(out, yg)

                self.L[i] = calculate_torch_lipschitz_constant(
                        model=self.model,
                        loss_fn=loss_fn_closure,
                        num_iterations=20,
                    )
            logger.info(f"Lipschitz constants calculated: {self.L}")
        else:
            self.L = None

    def training(
        self,
        weight: npt.NDArray[np.float64]
    ) -> Tuple[MLP, npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
        """
        Treina o modelo com um vetor de pesos de escalarização.
        Retorna: (modelo_treinado, objetivos, gradientes)
        """
        feature_vals = np.array(self.X_pd[self.fair_feat])
        unique_groups = np.unique(feature_vals)
        group_weights_lookup = weight
        indices = np.searchsorted(unique_groups, feature_vals)
        sample_weights = group_weights_lookup[indices]
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

        criterion = nn.BCELoss(weight=sample_weights_tensor, reduction="mean")
        optimizer = optim.RAdam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        self.model.train()
        for _ in range(self.num_epochs):
            optimizer.zero_grad()
            y_pred = self.model(self.X_tensor).squeeze()
            loss = criterion(y_pred, self.y_tensor)
            loss.backward()
            optimizer.step()

        # 3. Avaliação (Cálculo de Objetivos e Gradientes)
        self.model.eval()
        
        objs = np.zeros(self.M)
        gradients = []
        
        for i, g_val in enumerate(self.fair_att):
            if i >= self.M: 
                break
            
            mask = (self.X_pd[self.fair_feat] == g_val).to_numpy()
            X_g = self.X_tensor[mask]
            y_g = self.y_tensor[mask]
            
            if len(X_g) == 0:
                objs[i] = 0.0
                total_params = sum(p.numel() for p in self.model.parameters())
                gradients.append(np.zeros(total_params))
                continue

            self.model.zero_grad()
            output = self.model(X_g).squeeze()
            loss = nn.BCELoss(reduction='mean')(output, y_g)
            
            objs[i] = loss.item()

            # Se Lipschitz está ativo, precisamos calcular os gradientes dos objetivos
            if self.lower_bound_estimate == "lipschitz":
                loss.backward()
                grads = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        grads.append(param.grad.view(-1).cpu().numpy())
                    else:
                        grads.append(np.zeros(param.numel()))
                gradients.append(np.concatenate(grads))

        # Retorno: Gradiente é opcional (None se não for Lipschitz)
        if self.lower_bound_estimate == "lipschitz":
            return self.model, objs, np.array(gradients)
        
        return self.model, objs, None