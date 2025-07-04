from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np


class ModelManager(ABC):
    """
    Abstract Base Class for managing PyTorch model training and prediction workflows.

    Defines a common interface for building, training, evaluating, predicting,
    saving, and loading models.
    """

    class XRR_MLP(nn.Module):
        def __init__(self, input_dim: int = 626, output_dim: int = 20):

            super().__init__()

            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )

        def forward(self, x):
            return self.model(x)

    def __init__(
        self,
        learning_rate: float = 1e-3,
        device: str | torch.device | None = None,
        criterion: nn.Module | None = None,
        optimizer_cls: type[optim.Optimizer] = optim.Adam,
        model_params: dict[str, Any] | None = None,  # To pass params to _build_model
    ):
        """
        Initializes the base manager.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.learning_rate = learning_rate
        self.model_params = model_params if model_params else {}

        # Build the model using the abstract method and provided parameters
        self.model = self.XRR_MLP(**self.model_params).to(self.device)

        # Setup criterion (loss function)
        self.criterion = criterion if criterion is not None else nn.MSELoss()

        # Setup optimizer
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.learning_rate)

        print(f"Initialized {self.__class__.__name__} on device: {self.device}")
        # Optionally print model summary if desired
        # print(f"Model Architecture:\n{self.model}")

    @abstractmethod
    def train(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        model_path: str | Path,
        epochs: int = 50,
        batch_size: int = 64,
        val_ratio: float = 0.1,
    ) -> dict[str, list[float]]:
        """
        Trains the model using provided data.

        Handles data preparation, training loop, validation loop, model saving,
        and returns training history.

        Returns:
            dictionary containing training history (e.g., {'train_loss': [...], 'val_loss': [...]}).
        """

    @abstractmethod
    def predict(
        self, X_input: np.ndarray, model_path: str | Path | None = None
    ) -> np.ndarray:
        pass

    def save_model(self, path: str | Path):
        """
        Saves the current model state dictionary to the specified path.
        Ensures parent directories exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model state saved to {path}")

    def load_model(self, path: str | Path, weights_only: bool = True):
        """
        Loads the model state dictionary from the specified path.

        Args:
            path (str | Path): Path to the saved model state dictionary.
            weights_only (bool): Passed to torch.load for security. Defaults to True.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found at {path}")

        # Use map_location to ensure model loads onto the correct device
        state_dict = torch.load(
            path, map_location=self.device, weights_only=weights_only
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)  # Ensure model is on correct device
        print(f"Model state loaded from {path}")


class MLPManager(ModelManager):
    """XRR_MLP 모델과 학습/예측 기능을 통합 관리하는 클래스"""

    class XRR_MLP(nn.Module):
        def __init__(self, input_dim: int = 626, output_dim: int = 20):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),

                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),

                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),

                nn.Linear(128, 16),  # ✅ 추가된 레이어
                nn.ReLU(),
                nn.BatchNorm1d(16),

                nn.Linear(16, output_dim),
            )

        def forward(self, x):
            return self.model(x)

    def __init__(
        self,
        input_dim: int = 626,
        output_dim: int = 20,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            learning_rate=learning_rate,
            device=device,
            criterion=nn.MSELoss(),
            optimizer_cls=optim.Adam,
            model_params={"input_dim": input_dim, "output_dim": output_dim},
        )
        self.device = device
        self.model = self.XRR_MLP(input_dim=input_dim, output_dim=output_dim).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        model_path: str,
        epochs: int = 50,
        batch_size: int = 64,
        val_ratio: float = 0.1,
    ) -> dict:
        """
        모델 학습 및 저장 함수
        """
        # 데이터 형상 변환
        if len(X_data.shape) == 3:
            X_data = X_data.reshape(X_data.shape[0], -1)

        # 텐서 변환
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)

        # 데이터셋 및 로더 생성
        dataset = TensorDataset(X_tensor, y_tensor)
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        # 학습 히스토리
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * x_batch.size(0)

            avg_loss = total_loss / train_size
            history["train_loss"].append(avg_loss)

            # 검증
            val_loss = self._validate(val_loader, val_size)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}"
            )

        # 모델 저장
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        return history

    def _validate(self, val_loader: DataLoader, val_size: int) -> float:
        """검증 데이터에 대한 손실 계산"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                preds = self.model(x_val)
                loss = self.criterion(preds, y_val)
                val_loss += loss.item() * x_val.size(0)

        return val_loss / val_size

    def predict(self, X_input: np.ndarray, model_path: str | None = None) -> np.ndarray:
        """
        모델 예측 함수

        Args:
            X_input: 입력 데이터 (shape: (N, 313, 2) or (N, 626))
            model_path: 저장된 모델 경로 (선택, 제공 시 로드)

        Returns:
            예측 결과 (shape: (N, output_dim))
        """
        # 모델 로드 (필요 시)
        if model_path:
            self.load_model(model_path)

        # 데이터 형상 변환
        if len(X_input.shape) == 3:
            X_input = X_input.reshape(X_input.shape[0], -1)

        # 텐서 변환 및 예측
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(self.device)
        self.model.eval()

        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()

        return preds


class SimpleCNN(nn.Module):
    """A basic CNN architecture example, adaptable via parameters."""

    def __init__(self, input_channels: int, height: int, width: int, output_dim: int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (16, H/2, W/2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # Calculate flattened size dynamically after conv blocks
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, height, width)
            dummy_output = self.conv_block2(self.conv_block1(dummy_input))
            # Check resulting shape (should be e.g., [1, 32, 78, 1])
            # print(f"Shape after conv blocks: {dummy_output.shape}")
            self._flattened_size = dummy_output.numel()  # Total number of elements

        if self._flattened_size <= 0:
            raise ValueError(
                f"Calculated flattened size is non-positive ({self._flattened_size}). "
                f"Check CNN architecture and input shape ({input_channels, height, width})."
            )

        # --- Fully Connected Block ---
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_block(x)
        return x


class CNNModelManager(ModelManager):
    """Manages a SimpleCNN model for image-like data."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],  # (channels, height, width)
        output_dim: int,
        learning_rate: float = 1e-3,
        device: str | torch.device | None = None,
        criterion: nn.Module | None = None,
        optimizer_cls: type[optim.Optimizer] = optim.Adam,
    ):
        """
        Initializes the CNNModelManager.
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        # Pass model-specific parameters needed by _build_model to base class
        model_params = {
            "input_channels": input_shape[0],
            "height": input_shape[1],
            "width": input_shape[2],
            "output_dim": output_dim,
        }
        super().__init__(learning_rate, device, criterion, optimizer_cls, model_params)

    def _build_model(self, **kwargs) -> nn.Module:
        """Builds the SimpleCNN model using parameters passed during init."""
        # Expects input_channels, height, width, output_dim from kwargs (via self.model_params)
        required_params = ["input_channels", "height", "width", "output_dim"]
        if not all(p in kwargs for p in required_params):
            raise ValueError(f"_build_model requires parameters: {required_params}")
        return SimpleCNN(**kwargs)

    def _prepare_data(
        self, X_data: np.ndarray, y_data: np.ndarray | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Prepares image data for CNN: ensures shape (N, C, H, W) and converts to tensor.
        """
        n_samples = X_data.shape[0]
        expected_shape_no_batch = self.input_shape  # (C, H, W)
        current_shape_no_batch = X_data.shape[1:]

        # 1. Check if current shape matches expected (ignoring batch dim)
        if current_shape_no_batch != expected_shape_no_batch:
            # Attempt common fixes: e.g., add channel dim if input is (N, H, W) and C=1
            if len(current_shape_no_batch) == 2 and expected_shape_no_batch[0] == 1:
                # Input is (N, H, W), expected is (1, H, W)
                if current_shape_no_batch == expected_shape_no_batch[1:]:
                    X_data = X_data[
                        :, np.newaxis, :, :
                    ]  # Add channel dim: (N, 1, H, W)
                    print(f"Data Reshaped: Added channel dimension -> {X_data.shape}")
                else:
                    raise ValueError(
                        f"Input shape {X_data.shape} height/width mismatch expected {expected_shape_no_batch}"
                    )
            # Attempt reshape if total elements match (e.g., flatten -> C,H,W)
            elif np.prod(current_shape_no_batch) == np.prod(expected_shape_no_batch):
                X_data = X_data.reshape((n_samples,) + expected_shape_no_batch)
                print(f"Data Reshaped: Automatically reshaped to -> {X_data.shape}")

            else:
                raise ValueError(
                    f"Input data shape {X_data.shape} is incompatible with expected model input shape {(n_samples,) + expected_shape_no_batch}"
                )

        # 2. Convert to Tensor
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        y_tensor = (
            torch.tensor(y_data, dtype=torch.float32) if y_data is not None else None
        )

        return X_tensor, y_tensor

    def train(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        model_path: str | Path,
        epochs: int = 50,
        batch_size: int = 64,
        val_ratio: float = 0.1,
    ) -> dict[str, list[float]]:
        """Trains the CNN model."""
        save_best_only = True  # Default behavior to save best model
        model_path = Path(model_path)  # Ensure Path object
        X_tensor, y_tensor = self._prepare_data(X_data, y_data)

        if y_tensor is None:
            raise ValueError("Target data (y_data) cannot be None for training.")

        # Create datasets and dataloaders
        dataset = TensorDataset(X_tensor, y_tensor)
        if not (0 <= val_ratio < 1):
            print("Warning: Invalid val_ratio. Setting to 0 (no validation).")
            val_ratio = 0

        if val_ratio == 0:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            val_loader = None
            train_size = len(dataset)
            val_size = 0
            save_best_only = False  # Cannot save best without validation
            print("Training without validation split.")
        else:
            val_size = int(len(dataset) * val_ratio)
            train_size = len(dataset) - val_size
            if train_size <= 0 or val_size <= 0:
                raise ValueError(
                    f"Dataset size ({len(dataset)}) too small for val_ratio ({val_ratio})."
                )
            train_set, val_set = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(
                val_set, batch_size=batch_size
            )  # No shuffle for validation

        # Training loop
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            epoch_train_loss = 0.0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                # Forward pass
                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item() * x_batch.size(0)

            avg_epoch_train_loss = epoch_train_loss / train_size
            history["train_loss"].append(avg_epoch_train_loss)

            # Validation step
            avg_epoch_val_loss = float("nan")  # Default if no validation
            if val_loader:
                avg_epoch_val_loss = self._run_validation(val_loader)
                history["val_loss"].append(avg_epoch_val_loss)

                # Save best model logic
                if save_best_only and avg_epoch_val_loss < best_val_loss:
                    best_val_loss = avg_epoch_val_loss
                    self.save_model(model_path)  # Use the inherited save method

                    print(
                        f"Epoch {epoch+1}: Val loss improved to {avg_epoch_val_loss:.4f}. Model saved to {model_path}"
                    )
            # Print progress
            val_loss_str = f"{avg_epoch_val_loss:.4f}" if val_loader else "N/A"
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_epoch_train_loss:.4f} - "
                f"Val Loss: {val_loss_str}"
            )

        # Save the last model if not saving best only
        if not save_best_only:
            self.save_model(model_path)
            print(f"Saved final model state after {epochs} epochs to {model_path}")

        return history

    def predict(
        self, X_input: np.ndarray, model_path: str | Path | None = None
    ) -> np.ndarray:
        """Makes predictions using the trained CNN model."""
        if model_path:
            self.load_model(model_path)  # Use inherited load method

        # Prepare input data
        try:
            X_tensor, _ = self._prepare_data(X_input, None)
        except ValueError as e:
            print(f"Error during prediction data preparation: {e}")
            raise

        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            preds_tensor = self.model(X_tensor.to(self.device))

        return preds_tensor.cpu().numpy()

    def _run_validation(self, val_loader: DataLoader) -> float:
        """Runs the validation loop and returns the average loss."""
        self.model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0
        num_samples = 0
        with torch.no_grad():  # Disable gradient calculations
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)
                total_val_loss += loss.item() * x_batch.size(
                    0
                )  # Accumulate weighted loss
                num_samples += x_batch.size(0)

        if num_samples == 0:
            print("Warning: Validation set is empty.")
            return float("inf")  # Or 0.0 or NaN, depending on desired behavior
        return total_val_loss / num_samples  # Return average loss


if __name__ == "__main__":
    # 가상 데이터 생성
    N, input_dim, output_dim = 1000, 626, 20
    X_data = np.random.randn(N, 313, 2)  # (N, 313, 2)
    y_data = np.random.randn(N, output_dim)  # (N, 20)

    # 모델 관리자 초기화
    manager = MLPManager(input_dim=input_dim, output_dim=output_dim, learning_rate=1e-3)

    # 학습
    history = manager.train(
        X_data=X_data,
        y_data=y_data,
        model_path="model.pth",
        epochs=5,
        batch_size=64,
        val_ratio=0.1,
    )

    # 예측
    X_test = np.random.randn(100, 313, 2)
    predictions = manager.predict(X_test, model_path="model.pth")
    print(f"Predictions shape: {predictions}")
