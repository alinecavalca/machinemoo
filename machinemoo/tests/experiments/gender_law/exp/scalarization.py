import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Optional, Union
from transformers import AutoModelForSequenceClassification

from machinemoo import Scalarization

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

class MultiTaskBertFineTuner(nn.Module):
    def __init__(self, model_name: str, num_tasks: int, num_classes_per_task: Union[int, list[int]]):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            cache_dir='.'
        )
        
        # Freeze embeddings
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False

        for layer in self.model.bert.encoder.layer[:-3]:
            for param in layer.parameters():
                param.requires_grad = False

        hidden_size = self.model.config.hidden_size
        reduced_size = 8
        self.projection = nn.Linear(hidden_size, reduced_size)
        self.dropout = nn.Dropout(p=0.3)

        if isinstance(num_classes_per_task, int):
            num_classes_per_task = [num_classes_per_task] * num_tasks

        self.task_heads = nn.ModuleList([
            nn.Linear(reduced_size, num_classes) for num_classes in num_classes_per_task
        ])

    def forward(self, input_ids, attention_mask) -> list[torch.Tensor]:
        output = self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.last_hidden_state[:, 0]
        reduced_output = self.dropout(torch.relu(self.projection(pooled_output)))
        return [head(reduced_output) for head in self.task_heads]

class NLPScalarization(Scalarization):
    def __init__(self, train_dataloader, device, model_name, task_names, num_classes_per_task=2, gradient=False):
        super().__init__(len(task_names))
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.__M = len(task_names)
        self.device = device
        self.train_dataloader = train_dataloader
        self.epochs = 2
        self.use_gradient = gradient

        self.model = MultiTaskBertFineTuner(
            model_name=model_name, 
            num_tasks=self.num_tasks, 
            num_classes_per_task=num_classes_per_task
        ).to(device)

    def get_gradients(self, model: torch.nn.Module, loss: torch.Tensor, layer: torch.nn.Module = None) -> np.ndarray:
        params = model.parameters() if layer is None else layer.parameters()
        grads = torch.autograd.grad(loss, list(params), retain_graph=True)
        return np.concatenate([g.detach().cpu().numpy().flatten() for g in grads])

    def compute_task_gradients(self, batch, criterion):
        self.model.eval()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        logits_list = self.model(input_ids, attention_mask)

        grads = []
        for task in self.task_names:
            labels = batch[task].to(self.device)
            loss = criterion(logits_list[task], labels)
            grad = self.get_gradients(self.model, loss, layer=self.model.task_heads[i])
            grads.append(grad)

        return np.stack(grads)

    def training(self, weights: Union[np.ndarray, list[float]]):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.model.train()
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs} - Training"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits_list = self.model(input_ids, attention_mask)
                losses = []
                for i, task in enumerate(self.task_names):
                    labels = batch[task].to(self.device)
                    loss = criterion(logits_list[i], labels)
                    losses.append(loss*weights[i])

                loss = sum(losses) / self.num_tasks
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluation
        self.model.eval()
        eval_losses = np.zeros(self.num_tasks)
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(self.train_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits_list = self.model(input_ids, attention_mask)
                for i, task in enumerate(self.task_names):
                    labels = batch[task].to(self.device)
                    loss = criterion(logits_list[i], labels)
                    objs[i] += loss.item()

        #objs /= len(self.train_dataloader)
        objs = np.array([loss / len(self.train_dataloader) for loss in eval_losses])
        #objs = total_losses
        
        gradient = None
        if self.use_gradient:
            batch = next(iter(self.train_dataloader))
            gradient = self.compute_task_gradients(batch, criterion)


        return self.model, objs, gradient
