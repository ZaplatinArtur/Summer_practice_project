import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def _ensure_tensor(img):
    """Конвертирует PIL.Image в тензор при необходимости"""
    if isinstance(img, Image.Image):
        return transforms.ToTensor()(img)
    return img

def show_images(images, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    # Конвертируем все изображения в тензоры
    images = [_ensure_tensor(img) for img in images]
    images = images[:nrow]
    
    # Ресайз изображений
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    
    # Визуализация
    fig, axes = plt.subplots(1, nrow, figsize=(nrow*2, 2))
    if nrow == 1:
        axes = [axes]
    
    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def show_single_augmentation(original_img, augmented_img, title="Аугментация"):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    # Конвертируем в тензоры
    original_img = _ensure_tensor(original_img)
    augmented_img = _ensure_tensor(augmented_img)
    
    # Ресайз
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    aug_resized = resize_transform(augmented_img)
    
    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')
    
    aug_np = aug_resized.numpy().transpose(1, 2, 0)
    aug_np = np.clip(aug_np, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


def show_multiple_augmentations(original_img, augmented_imgs, titles, max_per_row=4):
    """
    Визуализирует оригинальное изображение и несколько аугментаций
    с автоматическим разбиением на строки (максимум max_per_row в строке)
    """
    # Конвертируем в тензоры
    original_img = _ensure_tensor(original_img)
    augmented_imgs = [_ensure_tensor(img) for img in augmented_imgs]
    
    # Собираем все изображения
    all_imgs = [original_img] + augmented_imgs
    all_titles = ["Оригинал"] + titles
    
    # Рассчитываем количество строк и столбцов
    n_total = len(all_imgs)
    n_rows = (n_total + max_per_row - 1) // max_per_row
    n_cols = min(n_total, max_per_row)
    
    # Создаем фигуру с сеткой подграфиков
    fig, axs = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(n_cols * 3, n_rows * 3)
    )
    
    # Если только одна строка, делаем axs итерируемым
    if n_rows == 1:
        axs = [axs]
    
    # Ресайз изображений
    resize_transform = transforms.Resize((192, 192), antialias=True)
    
    # Визуализируем изображения
    for i in range(n_rows):
        row_start = i * max_per_row
        row_end = min((i + 1) * max_per_row, n_total)
        row_items = row_end - row_start
        
        for j in range(row_items):
            idx = row_start + j
            img = all_imgs[idx]
            title = all_titles[idx]
            
            # Получаем ось для текущего изображения
            if n_rows > 1:
                ax = axs[i][j] if row_items > 1 else axs[i]
            else:
                ax = axs[j] if row_items > 1 else axs
            
            # Ресайзим изображение
            img_resized = resize_transform(img)
            img_np = img_resized.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            ax.imshow(img_np)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
    
    # Скрываем пустые подграфики
    if n_total % max_per_row != 0:
        empty_cells = max_per_row - (n_total % max_per_row)
        for j in range(row_items, row_items + empty_cells):
            if n_rows > 1:
                axs[-1][j].axis('off')
            else:
                axs[j].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Сравнение аугментаций ({n_total} изображений)", fontsize=14)
    plt.show()