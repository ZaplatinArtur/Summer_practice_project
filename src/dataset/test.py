import os
import torch
from torchvision import transforms
from PIL import Image
from customImageDataset import CustomImageDataset
from extra_augs import (AddGaussianNoise, RandomErasingCustom, CutOut, 
                       Solarize, Posterize, AutoContrast, ElasticTransform)


current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, '..', '..', 'data', 'train')
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))

original_img, label = dataset[0]
class_names = dataset.get_class_names()
print(f"Оригинальное изображение, класс: {class_names[label]}")

class_names = dataset.get_class_names()
print(f"Оригинальное изображение, класс: {class_names[label]}")

show_images(original_img)