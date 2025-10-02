from pathlib import Path

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


# 1. Dataset: 입력(R)과 정답(n_layer)만 반환하도록 수정 (동일)
class XRRDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.len = len(f["R"])  # 길이만 저장

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as f:
            R = f["R"][idx].astype("float32")
            n_layer = f["n_layer"][idx].astype("int64")
        return torch.tensor(R), torch.tensor(n_layer)


# 2. Model: 분류(Classification) 기능만 남기고 단순화 (동일)
class XRRClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_feat = nn.Linear(32, 128)
        self.classifier = nn.Linear(128, num_classes)
        # Regressor 헤드는 완전히 제거

    def forward(self, R):
        x = R.unsqueeze(1)
        feat = self.encoder(x).squeeze(-1)
        feat = F.relu(self.fc_feat(feat))
        n_layer_logits = self.classifier(feat)
        return n_layer_logits


# 3. Training Function: 분류 작업에 맞게 재구성 및 결과 저장 로직 추가
def train_classifier(dataset, max_epoch=10, batch_size=128):
    # Train/Validation 데이터 분리
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 모델 초기화
    num_classes = 6
    model = XRRClassifier(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 결과를 저장할 딕셔너리 초기화
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
    }

    # Training loop
    for epoch in range(1, max_epoch + 1):
        # --- Training ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for R, n_layer in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            R, n_layer = R.to(device), n_layer.to(device)

            optimizer.zero_grad()
            n_layer_logits = model(R)

            # 층 개수는 1~6 -> loss 계산을 위해 0~5 인덱스로 변환
            loss = loss_fn(n_layer_logits, n_layer - 1)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 정확도 계산
            _, predicted = torch.max(n_layer_logits.data, 1)
            train_total += n_layer.size(0)
            train_correct += (predicted == (n_layer - 1)).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for R, n_layer in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                R, n_layer = R.to(device), n_layer.to(device)
                n_layer_logits = model(R)
                loss = loss_fn(n_layer_logits, n_layer - 1)
                val_loss += loss.item()

                _, predicted = torch.max(n_layer_logits.data, 1)
                val_total += n_layer.size(0)
                val_correct += (predicted == (n_layer - 1)).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # 에폭별 결과를 history에 저장
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(
            f"Epoch {epoch:02d}/{max_epoch}: "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

    # 모델의 가중치와 학습 히스토리를 딕셔너리로 묶어서 반환
    return {
        'model_state_dict': model.state_dict(),
        'history': history,
        'num_classes': num_classes # 모델 재로딩에 필요한 정보
    }


if __name__ == "__main__":
    file = r"D:\03_Resources\Data\XRR_AI\data\250929.h5"
    save_path = "./results/xrr_classifier_checkpoint.pt" # checkpoint라는 이름으로 변경

    # print("--- 1. 모델 학습 시작 ---")
    # # 데이터셋 생성 및 모델 학습
    # dataset = XRRDataset(file)
    # # 학습 결과에는 model_state_dict와 history(loss, acc)가 포함됨
    # checkpoint = train_classifier(dataset, max_epoch=10)

    # # 학습된 모델 (가중치 + 학습 결과) 저장
    # torch.save(checkpoint, save_path)
    # print(f"\nModel checkpoint saved to {save_path}")

    # # --- 2. 저장된 결과 불러오기 및 확인 ---
    # print("\n--- 저장된 학습 결과 불러오기 및 확인 ---")
    # loaded_checkpoint = torch.load(save_path)

    # # 학습 히스토리 정보 확인
    # loaded_history = loaded_checkpoint['history']
    # print(f"Total Epochs: {len(loaded_history['train_loss'])}")
    # print(f"Final Train Loss: {loaded_history['train_loss'][-1]:.4f}")
    # print(f"Final Val Accuracy: {loaded_history['val_accuracy'][-1]:.2f}%")

    # # 모델 가중치 불러와서 사용 예시
    # # 새로운 모델 인스턴스 생성
    # loaded_model = XRRClassifier(num_classes=loaded_checkpoint['num_classes'])
    # # 저장된 가중치를 로드
    # loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    # loaded_model.eval()
    # print("Model state dict successfully loaded.")


    import matplotlib as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from torch.utils.data import DataLoader, Dataset, random_split


    # --- Load checkpoint ---
    checkpoint_path = "./results/xrr_classifier_checkpoint.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint['num_classes']
    model = XRRClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded. Using device: {device}")

    # --- Dataset & DataLoader (validation set) ---
    file_path = r"D:\03_Resources\Data\XRR_AI\data\250929.h5"
    dataset = XRRDataset(file_path)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    # --- Evaluation ---
    y_true = []
    y_pred = []

    with torch.no_grad():
        for R, n_layer in val_loader:
            R = R.to(device)
            logits = model(R)
            pred = torch.argmax(logits, dim=1) + 1  # 0~5 → 1~6
            y_true.extend(n_layer.numpy())
            y_pred.extend(pred.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Accuracy
    accuracy = np.mean(y_true == y_pred) * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # ±1층 정확도
    close_acc = np.mean(np.abs(y_true - y_pred) <= 1) * 100
    print(f"Validation ±1 Layer Accuracy: {close_acc:.2f}%")

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"Mean Absolute Error (layers): {mae:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5,6])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
