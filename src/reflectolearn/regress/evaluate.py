import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..logger import setup_logger
from .data import ThicknessDataset
from .model import ThicknessRegressor


def evaluate_regressor(checkpoint_path: str, dataset: ThicknessDataset, batch_size: int = 128, device=None):
    logger = setup_logger()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ThicknessRegressor(checkpoint["input_dim"], checkpoint["n_layer"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for R, y in tqdm(loader, desc="Evaluating"):
            R = R.to(device)
            preds = model(R).cpu().numpy()
            y_true.append(y.numpy())
            y_pred.append(preds)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logger.info(f"MAE: {mae:.4f}, R²: {r2:.4f}")

    # Scatter plot: true vs pred
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.4)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Thickness")
    plt.ylabel("Predicted Thickness")
    plt.title(f"Thickness Regression (n={checkpoint['n_layer']})")
    plt.show()

    return mae, r2


if __name__ == "__main__":
    from .data import ThicknessDataset
    checkpoint_path = r"results\regress\thickness_n3.pt"
    dataset_path = r"D:\03_Resources\Data\XRR_AI\data\250929.h5"
    dataset = ThicknessDataset(dataset_path, 3)

    mae, r2 = evaluate_regressor(checkpoint_path, dataset)
    print(mae, r2)
