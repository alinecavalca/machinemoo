from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from machinemoo import Scalarization

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        task_names: list[str],
        input_dim: int = 50,
        hidden_dim: int = 32,
        num_labels: int = 2,
        dropout_rate: float = 0.3
    ) -> None:
        super(MultiTaskModel, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.shared_fc = nn.Linear(input_dim, hidden_dim)

        # Task-specific heads stored in a ModuleDict
        self.classifiers = nn.ModuleDict(
            {task: nn.Linear(hidden_dim, num_labels) for task in task_names}
        )

    def forward(self, embeddings):
        x = self.dropout(embeddings)
        features = self.shared_fc(x)

        # Compute logits for each task and return a dict
        return {task: head(features) for task, head in self.classifiers.items()}


class NLPScalarization(Scalarization):
    def __init__(
        self,
        train_dataloader,
        device,
        task_names: list[str],
        gradient: bool = False
    ) -> None:
        super(NLPScalarization, self).__init__(len(task_names))
        self.task_names = task_names
        self.__M = len(task_names)
        self.epochs = 3
        self.device = device
        self.train_dataloader = train_dataloader
        self.use_gradient = gradient
        self.model = MultiTaskModel(task_names=task_names).to(device)

    def get_gradients(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        layer: nn.Module | None = None
    ) -> npt.NDArray[np.float64]:
        params = list(layer.parameters()) if layer else list(model.parameters())
        grads = torch.autograd.grad(loss, params, retain_graph=True)
        return np.concatenate([g.detach().cpu().numpy().flatten() for g in grads])

    def compute_task_gradients(self, batch: dict, criterion: nn.Module) -> npt.NDArray[np.float64]:
        self.model.eval()
        embeddings = batch["embedding"].to(self.device)
        logits_dict = self.model(embeddings)

        grads = []
        for task in self.task_names:
            labels = batch[task].to(self.device)
            loss = criterion(logits_dict[task], labels)
            layer = self.model.classifiers[task]
            grads.append(self.get_gradients(self.model, loss, layer))
        return np.stack(grads)

    def training(
        self,
        weight: npt.NDArray[np.float64]
    ) -> tuple[nn.Module, npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[nn.Module, npt.NDArray[np.float64]]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=2e-5, weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm(
                self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"
            ):
                embeddings = batch["embedding"].to(self.device)
                logits_dict = self.model(embeddings)
                optimizer.zero_grad()

                losses = []
                for i, task in enumerate(self.task_names):
                    labels = batch[task].to(self.device)
                    task_loss = criterion(logits_dict[task], labels)
                    losses.append(weight[i] * task_loss)

                loss = sum(losses) / self.__M
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(
                f"Epoch {epoch+1} - Average loss: {total_loss / len(self.train_dataloader):.4f}"
            )

        # Final evaluation
        total_batches = 0
        eval_losses = np.zeros(self.M)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in self.train_dataloader:
                embeddings = batch["embedding"].to(self.device)
                logits_dict = self.model(embeddings)
                for i, task in enumerate(self.task_names):
                    labels = batch[task].to(self.device)
                    loss = criterion(logits_dict[task], labels)
                    eval_losses[i] += loss.item()
                total_batches += 1
        # objs /= len(self.train_dataloader)
        objs = np.array([loss / total_batches for loss in eval_losses])

        if self.use_gradient:
            batch = next(iter(self.train_dataloader))
            gradient = self.compute_task_gradients(batch, criterion)
            return self.model, objs, gradient
        return self.model, objs
