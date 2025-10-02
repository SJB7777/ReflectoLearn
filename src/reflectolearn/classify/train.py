import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..logger import setup_logger
from .train import NlayerClassifier


def train_classifier(dataset, max_epoch=10, batch_size=128):
    logger = setup_logger()

    # Train/Validation 데이터 분리
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 모델 초기화
    num_classes = 6
    model = NlayerClassifier(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

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

        logger.info(
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
