from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from model_architecture import ModelNatoTanks
from torchvision import transforms
import torch
from src/dataset import CustomImageDataset


BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (512,512)
def train_model(root_dir: str, val_dir: str, model_save_path: str = "model.pth"):
    # Инициализация тренировочного датасета и загрузчика данных
    train_dataset = CustomImageDataset(
        root_dir=root_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        target_size=TARGET_SIZE
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Инициализация валидационного датасета и загрузчика данных
    val_dataset = CustomImageDataset(
        root_dir=val_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        target_size=TARGET_SIZE
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Инициализация модели
    model = ModelNatoTanks().to(DEVICE)
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Планировщик скорости обучения
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Переменные для отслеживания лучшей модели
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    epochs_accuracy_arr = []
    epochs_loss_arr = []
    # Цикл обучения
    for epoch in range(NUM_EPOCHS):
        # Тренировочный режим
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Прогресс-бар для тренировочной эпохи
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Обнуление градиентов
            optimizer.zero_grad()
            
            # Прямой проход
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Обратное распространение и оптимизация
            loss.backward()
            optimizer.step()
            
            # Статистика
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Обновление прогресс-бара
            progress_bar.set_postfix({
                'train_loss': running_loss / total,
                'train_accuracy': 100. * correct / total
            })
        
        # Средние значения за эпоху (тренировка)
        train_loss = running_loss / len(train_dataset)
        train_acc = 100. * correct / total
        
        # Валидационный режим
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Прогресс-бар для валидации
        val_progress_bar = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{NUM_EPOCHS} [Val]")
        
        with torch.no_grad():  # Отключаем вычисление градиентов для валидации
            for images, labels in val_progress_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                # Прямой проход
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Статистика
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Обновление прогресс-бара
                val_progress_bar.set_postfix({
                    'val_loss': val_loss / val_total,
                    'val_accuracy': 100. * val_correct / val_total
                })
        
        # Средние значения за эпоху (валидация)
        val_loss = val_loss / len(val_dataset)
        val_acc = 100. * val_correct / val_total
        epochs_accuracy_arr.append(val_acc)
        epochs_loss_arr.append(val_loss)
        # Обновление планировщика на основе валидационной потери
        scheduler.step(val_loss)
        
        # Сохранение лучшей модели на основе валидационной потери
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            model_save_path = f"model_{epoch}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Модель сохранена: {model_save_path} (val_loss: {val_loss:.4f})")
        
        # Логирование
        print(f"Эпоха {epoch+1}/{NUM_EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
    
    print("Обучение завершено!")
    return epochs_accuracy_arr,epochs_loss_arr


if __name__ == "__main__":
    root_train = ...#папка ст тренировачными данными
    root_val = ...#папка с валидационными данными
    # Обучаем модель
    epochs_accuracy_arr,epochs_loss_arr = train_model(root_train,root_val)