import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from transformers import AutoModel

from machinemoo import Scalarization
from machinemoo import get_logger
from machinemoo import calculate_torch_lipschitz_constant

# Setting a seed for reproducibility
from machinemoo.utils.seed_config import set_np_torch_seed
set_np_torch_seed(42)

logger = get_logger(f"moo.{__name__}")


class MultiTaskBertFineTuner(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_tasks: int,
        num_classes_per_task: int | list[int],
        hidden_intermediate: int = 16,
        reduced_size: int = 8,
        dropout_prob: float = 0.6,
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, cache_dir=".")

        # Gradient checkpointing
        #self.model.gradient_checkpointing_enable()

        for param in self.model.embeddings.parameters():
            param.requires_grad = False

        # Freeze all layers except the last one
        for layer in self.model.encoder.layer[:-1]:
            for param in layer.parameters():
                param.requires_grad = False

        if isinstance(num_classes_per_task, int):
            num_classes_per_task = [num_classes_per_task] * num_tasks

        hidden_size = self.model.config.hidden_size

        # fully connected before heads
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_intermediate),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_intermediate, reduced_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
        )

        # Heads para cada tarefa
        self.task_heads = nn.ModuleList(
            [
                nn.Linear(reduced_size, num_classes) 
                for num_classes in num_classes_per_task
            ]
        )

    def forward(self, input_ids, attention_mask) -> list[torch.Tensor]:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0]
        reduced = self.mlp(cls_token)
        return [head(reduced) for head in self.task_heads]


class NLPScalarization(Scalarization):
    def __init__(
        self,
        train_dataloader,
        device,
        model_name,
        task_names,
        num_classes_per_task=2,
        lower_bound_estimate: str | float = 0.01, # Options: "zero", "lipschitz", or a float value
    ) -> None:
        super(NLPScalarization, self).__init__(len(task_names))
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.__M = len(task_names)
        self.device = device
        self.train_dataloader = train_dataloader
        self.epochs = 2
        self.lower_bound_estimate = lower_bound_estimate

        self.model = MultiTaskBertFineTuner(
            model_name=model_name,
            num_tasks=self.num_tasks,
            num_classes_per_task=num_classes_per_task,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        
        if self.lower_bound_estimate == "lipschitz":
            batch = next(iter(self.train_dataloader))
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            self.L = np.zeros(self.num_tasks)
            for i, task in enumerate(self.task_names):
                labels = batch[task].to(self.device)
                
                loss_fn = lambda: criterion(self.model(input_ids, attention_mask)[i], labels)
                self.L[i] = calculate_torch_lipschitz_constant(
                        model=self.model,
                        loss_fn=loss_fn,
                        device=self.device,
                        num_iterations=20,
                        layer=self.model.task_heads[i]
                    )
            logger.debug("Lipschitz constants for each group:", self.L)

    def get_gradients(
        self,
        model: torch.nn.Module,
        loss: torch.Tensor,
        layer: torch.nn.Module | None = None
    ) -> npt.NDArray[np.float64]:
        
        params = model.parameters() if layer is None else layer.parameters()
        grads = torch.autograd.grad(loss, list(params), retain_graph=True)
        
        return np.concatenate([g.detach().cpu().numpy().flatten() for g in grads])


    def compute_task_gradients(self, batch, criterion):
        self.model.eval()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        grads = []
        for i, task in enumerate(self.task_names):
            labels = batch[task].to(self.device)
            logits = self.model(input_ids, attention_mask)[i]
            loss = criterion(logits, labels)
            grad = self.get_gradients(self.model, loss, layer=self.model.task_heads[i])
            grads.append(grad)

        torch.cuda.empty_cache()
        return np.stack(grads)

    def _compute_lipschitz(self) -> np.ndarray:
        """Returns a lower estimative of the objective values with lipschitz estamation.

        Returns:
            np.ndarray: Lower estimative for each objective function.
        """
        w_gradient = self.w@np.array([self.gradient[idx] for idx in range(self.M)])
        objs_delta = 1/(2*self.w@self.L)*w_gradient@w_gradient
        self.__objs_lower = self.objs - objs_delta
        return self.__objs_lower

    def training(
        self,
        weights: npt.NDArray[np.float64]
    ) -> tuple[torch.nn.Module, npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[torch.nn.Module, npt.NDArray[np.float64]]:
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=2e-5, weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.model.train()
            for batch in tqdm(
                self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs} - Training", leave=False
            ):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits_list = self.model(input_ids, attention_mask)
                losses = []
                for i, task in enumerate(self.task_names):
                    labels = batch[task].to(self.device)
                    loss = criterion(logits_list[i], labels)
                    losses.append(loss * weights[i])

                loss = sum(losses)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del input_ids, attention_mask, logits_list, loss
                torch.cuda.empty_cache()

        # Evaluation
        self.model.eval()
        eval_losses = np.zeros(self.num_tasks)
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(self.train_dataloader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits_list = self.model(input_ids, attention_mask)
                for i, task in enumerate(self.task_names):
                    labels = batch[task].to(self.device)
                    loss = criterion(logits_list[i], labels)
                    eval_losses[i] += loss.item()

                del input_ids, attention_mask, logits_list
                torch.cuda.empty_cache()

        # objs /= len(self.train_dataloader)
        objs = np.array([loss / len(self.train_dataloader) for loss in eval_losses])

        if self.lower_bound_estimate == "lipschitz":
            batch = next(iter(self.train_dataloader))
            gradient = self.compute_task_gradients(batch, criterion)
            return self.model, objs, gradient
        
        return self.model, objs
