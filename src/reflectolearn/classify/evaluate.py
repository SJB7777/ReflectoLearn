import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..logger import setup_logger
from .data import NlayerDataset
from .model import NlayerClassifier


def evaluate_checkpoint(checkpoint_path, dataset_path, batch_size=128, device=None, top_n_errors=10):
    """
    Evaluate a classification checkpoint with both quantitative and qualitative metrics.

    Returns:
        metrics (dict): Dictionary containing Accuracy, ±1-layer Accuracy, MAE
    """
    logger = setup_logger()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load model checkpoint ---
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = NlayerClassifier(num_classes=checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Dataset & DataLoader ---
    dataset = NlayerDataset(dataset_path)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Evaluation loop ---
    y_true, y_pred = [], []
    with torch.no_grad():
        for R, n_layer in tqdm(val_loader, desc="Evaluating"):
            R = R.to(device)
            logits = model(R)
            pred = torch.argmax(logits, dim=1) + 1  # 0~num_classes-1 → 1~num_classes
            y_true.extend(n_layer.numpy())
            y_pred.extend(pred.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- Quantitative metrics ---
    accuracy = np.mean(y_true == y_pred) * 100
    close_acc = np.mean(np.abs(y_true - y_pred) <= 1) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    logger.info(f"Accuracy: {accuracy:.2f}%, ±1-layer Accuracy: {close_acc:.2f}%, MAE: {mae:.2f}")

    # --- Per-class accuracy ---
    per_class_acc = {}
    for i in range(1, checkpoint['num_classes'] + 1):
        idx = y_true == i
        acc = np.mean(y_pred[idx] == y_true[idx]) * 100
        per_class_acc[i] = acc
    logger.info(f"Per-class Accuracy: {per_class_acc}")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=list(range(1, checkpoint['num_classes'] + 1)))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.show()

    # --- Error distribution ---
    errors = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.hist(errors, bins=np.arange(errors.min()-0.5, errors.max()+1.5, 1), edgecolor='black')
    plt.xlabel("Prediction Error (Predicted - True)")
    plt.ylabel("Count")
    plt.title("Prediction Error Distribution")
    plt.show()

    # --- Top-N largest errors ---
    df = pd.DataFrame({"True": y_true, "Pred": y_pred})
    df['Error'] = df['Pred'] - df['True']
    top_errors = df.iloc[np.argsort(np.abs(df['Error']))[::-1]].head(top_n_errors)
    logger.info(f"Top {top_n_errors} largest errors:\n{top_errors}")

    # --- Return metrics for further processing ---
    metrics = {
        "accuracy": accuracy,
        "close_accuracy": close_acc,
        "mae": mae,
        "per_class_accuracy": per_class_acc,
        "top_errors": top_errors
    }

    return metrics


if __name__ == "__main__":
    from pathlib import Path


    checkpoint_path = Path("results/xrr_classifier_checkpoint/20251013_131654.pt")
    dataset_path = Path(r"X:\\XRR_AI\\hdf5_XRR\\data\\251013.h5")
    metrics = evaluate_checkpoint(checkpoint_path, dataset_path)
    print(metrics)
