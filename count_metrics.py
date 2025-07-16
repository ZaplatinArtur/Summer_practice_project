import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

def evaluate_model(model_path: str, test_dir: str, batch_size: int = 32, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Вычисляет метрики F1-score, Precision, Recall на тестовой выборке.
    
    Args:
        model_path (str): Путь к сохраненной модели (.pth)
        test_dir (str): Путь к тестовой директории
        batch_size (int): Размер батча для тестового загрузчика
        device (str): Устройство для инференса ('cuda' или 'cpu')
    
    Returns:
        dict: Словарь с метриками (accuracy, f1_score, precision, recall)
    """
    # Инициализация тестового датасета и загрузчика данных
    test_dataset = CustomImageDataset(
        root_dir=test_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        target_size=(512, 512)  # Соответствует входу модели
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Загрузка модели
    model = ModelNatoTanks().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Инициализация списков для сбора предсказаний и меток
    all_preds = []
    all_labels = []
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    # Функция потерь
    criterion = nn.CrossEntropyLoss()
    
    # Прогресс-бар для тестового набора
    progress_bar = tqdm(test_loader, desc="Оценка на тестовом наборе")
    
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Прямой проход
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Статистика
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Сбор предсказаний и меток
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Обновление прогресс-бара
            progress_bar.set_postfix({
                'test_loss': test_loss / test_total,
                'test_accuracy': 100. * test_correct / test_total
            })
    
    # Вычисление средних значений
    test_loss = test_loss / len(test_dataset)
    test_acc = 100. * test_correct / test_total
    
    # Вычисление метрик
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    test_precision = precision_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='weighted')
    
    # Формирование результата
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'f1_score': test_f1,
        'precision': test_precision,
        'recall': test_recall
    }
    
    # Вывод результатов
    print(f"Результаты на тестовом наборе:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"F1-score: {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    
    return metrics

if __name__ == "__main__":
    # Параметры
    model_path = ...  # Путь к сохраненной модели
    test_dir = ...  # Путь к тестовой директории
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Вычисление метрик
    metrics = evaluate_model(model_path, test_dir, batch_size, device)