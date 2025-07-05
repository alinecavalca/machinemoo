import random
from collections import Counter
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def make_group_hate_label_func(group_name: str) -> Callable[[dict], int]:
    def label_func(example: dict) -> int:
        # labels = example["annotators"]["label"]
        # majority_label = Counter(labels).most_common(1)[0][0]
        # is_hate = 0 if majority_label == 1 else 1
        # if is_hate:
        for group_list in example["bias_type"]:
            if group_name in group_list:
                return 1
        return 0

    return label_func


def make_hate_speech_label_func() -> Callable[[dict], int]:
    def label_func(example: dict) -> int:
        labels = example["annotators"]["label"]
        majority_label = Counter(labels).most_common(1)[0][0]
        return 0 if majority_label == 1 else 1

    return label_func


def build_label_funcs_from_config(config: dict) -> dict[str, Callable[[dict], int]]:
    funcs = {}
    for task, spec in config.items():
        # if spec["type"] == "hate_speech":
        #    funcs[task] = make_hate_speech_label_func()
        if spec["type"] == "group_bias":
            funcs[task] = make_group_hate_label_func(spec["group"])
        else:
            raise ValueError(f"Unknown task type: {spec['type']}")
    return funcs


def extract_bert_embeddings(
    dataset,
    tokenizer_name="bert-base-uncased",
    max_length=128,
    batch_size=32,
    device="cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(tokenizer_name).to(device)
    model.eval()

    texts = [" ".join(d["sentences"]["sentence"]) for d in dataset]

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoding)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings.cpu())

    return torch.cat(embeddings, dim=0)  # Shape: (N, hidden_size)


def apply_pca_and_save(
    embeddings: torch.Tensor, n_components=50, output_path="embeddings_pca.npy"
):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings.numpy())
    np.save(output_path, reduced)
    return reduced


def get_dataloader(dataset, task_config):
    label_funcs = build_label_funcs_from_config(task_config)
    task_names = list(task_config.keys())

    embeddings = extract_bert_embeddings(dataset)
    pca_embeddings = apply_pca_and_save(
        embeddings, n_components=50, output_path="embeddings_pca.npy"
    )

    dataset_pca = MultiTaskPCADataset(
        pca_embeddings, dataset, task_names=task_names, task_label_funcs=label_funcs
    )
    dataloader = DataLoader(dataset_pca, batch_size=64, shuffle=True)
    return dataloader


class MultiTaskPCADataset(Dataset):
    def __init__(self, pca_embeddings, raw_dataset, task_names, task_label_funcs):
        self.embeddings = torch.tensor(pca_embeddings, dtype=torch.float)
        self.raw_dataset = raw_dataset
        self.task_names = task_names
        self.label_funcs = task_label_funcs

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        data = self.raw_dataset[idx]
        item = {"embedding": self.embeddings[idx]}

        for task in self.task_names:
            item[task] = torch.tensor(self.label_funcs[task](data), dtype=torch.long)

        return item
