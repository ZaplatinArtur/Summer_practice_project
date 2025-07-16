import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

def load_model(model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Загружает сохраненную модель ModelNatoTanks.
    
    Args:
        model_path (str): Путь к файлу модели (.pth)
        device (str): Устройство для загрузки модели ('cuda' или 'cpu')
    
    Returns:
        nn.Module: Загруженная модель
    """
    model = ModelNatoTanks().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Перевод модели в режим инференса
    return model

def preprocess_image(image_path: str, target_size: tuple = (512, 512)):
    """
    Предобрабатывает изображение для инференса.
    
    Args:
        image_path (str): Путь к изображению
        target_size (tuple): Целевой размер изображения (height, width)
    
    Returns:
        torch.Tensor: Предобработанный тензор изображения
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Стандартные значения для ImageNet
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Добавляем размерность батча
    return image

def predict_image(model: nn.Module, image_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Выполняет инференс на одном изображении.
    
    Args:
        model (nn.Module): Загруженная модель
        image_path (str): Путь к изображению
        device (str): Устройство для инференса ('cuda' или 'cpu')
    
    Returns:
        tuple: Предсказанный класс (int) и вероятности (list)
    """
    # Предобработка изображения
    image = preprocess_image(image_path).to(device)
    
    # Инференс
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, dim=1)
    
    return predicted_class.item(), probabilities.squeeze().cpu().numpy().tolist()

def main():
    # Параметры
    model_path = "/kaggle/working/model_20.pth"
    image_path = "/kaggle/input/dataset-nato-final/dataset_augmentation/test/leopard/leopard_20.jpg"  # Укажите путь к изображению
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Загрузка модели
    model = load_model(model_path, device)
    
    # Инференс
    predicted_class, probabilities = predict_image(model, image_path, device)
    
    # Вывод результатов
    print(f"Предсказанный класс: {predicted_class}")
    print(f"Вероятности: {probabilities}")

if __name__ == "__main__":
    main()