"""Small PyTorch classifiers, persisted inference, and evaluation metrics."""

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from torch import nn

ROOT = Path(__file__).resolve().parent


def network(width, hidden=0):
    if hidden == 0:
        return nn.Sequential(nn.Linear(width, 1))
    return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))


def fit_torch(x, y, config):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x = torch.tensor(np.asarray(x, dtype=np.float32))
    y = torch.tensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
    models, histories = [], []
    loss_function = nn.BCEWithLogitsLoss()
    for seed in [42, 43, 44]:
        torch.manual_seed(seed)
        model = network(x.shape[1], config['hidden'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        history = []
        model.train()
        for _ in range(config['epochs']):
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        model.eval()
        models.append(model)
        histories.append(history)
    return models, np.mean(histories, axis=0).tolist()


def probabilities(models, x):
    x = torch.tensor(np.asarray(x, dtype=np.float32))
    with torch.inference_mode():
        seed_probabilities = [model(x).sigmoid().numpy().ravel() for model in models]
    return np.mean(seed_probabilities, axis=0)


def score(y, probability):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    predicted = probability >= 0.5
    accuracy = accuracy_score(y, predicted)
    n, z = len(y), 1.96
    center = (accuracy + z*z/(2*n)) / (1 + z*z/n)
    radius = z*np.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {
        'accuracy': float(accuracy), 'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)), 'f1': float(f1_score(y, predicted, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        'brier': float(brier_score_loss(y, probability)), 'log_loss': float(log_loss(y, probability, labels=[0, 1])),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        'accuracy_ci95': [float(center-radius), float(center+radius)], 'passengers': n,
    }


def load_bundle(directory=ROOT / 'artifacts'):
    directory = Path(directory)
    checkpoint = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
    models = []
    for state in checkpoint['states']:
        model = network(checkpoint['width'], checkpoint['hidden'])
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    preprocessor = joblib.load(directory / 'preprocessor.joblib')
    manifest = json.loads((directory / 'manifest.json').read_text())
    return preprocessor, models, manifest


def predict(frame, bundle):
    preprocessor, models, _ = bundle
    return probabilities(models, preprocessor.transform(frame))
