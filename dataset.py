"""
Модуль для загрузки и обработки данных для модели AST.

Содержит классы Dataset и DataLoader для работы с изображениями,
разметкой ограждений, теней и метаданными съёмки.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ASTDataset(Dataset):
    """
    Датасет для обучения модели AST.
    
    Поддерживает загрузку изображений, разметки ориентированных bounding box (OBB),
    типов ограждений, метаданных о положении солнца и информации о связности теней.
    
    Ожидаемая структура директории с данными:
        data/
        ├── images/
        │   ├── img_001.png
        │   └── ...
        ├── annotations/
        │   ├── img_001.json
        │   └── ...
        └── splits/
            ├── train.txt
            └── val.txt
    
    Формат JSON аннотации:
    {
        "image_path": "images/img_001.png",
        "gsd": 0.1,  # м/пиксель
        "sun_azimuth": 135.5,  # радианы
        "sun_elevation": 0.78,  # радианы
        "objects": [
            {
                "id": 1,
                "type": "fence",  # или "shadow"
                "fence_type": 2,  # класс типа ограждения (0-3)
                "bbox": [x, y, w, h, theta],  # OBB формат
                "shadow_group_id": null  # для группировки фрагментов теней
            },
            ...
        ]
    }
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (1024, 1024),
        augment: bool = True,
        min_object_area: int = 100
    ):
        """
        Инициализация датасета.
        
        Args:
            data_root: Корневая директория с данными.
            split: Название сплита ('train', 'val', 'test').
            image_size: Размер изображений для ресайза (height, width).
            augment: Применять ли аугментации.
            min_object_area: Минимальная площадь объекта для фильтрации.
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.min_object_area = min_object_area
        
        # Загрузка списка файлов
        self.image_paths = self._load_image_list()
        
        # Создание трансформаций
        self.transforms = self._get_transforms()
        
    def _load_image_list(self) -> List[str]:
        """Загрузка списка изображений для текущего сплита."""
        split_file = os.path.join(self.data_root, "splits", f"{self.split}.txt")
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Файл сплита не найден: {split_file}")
        
        with open(split_file, 'r', encoding='utf-8') as f:
            image_names = [line.strip() for line in f if line.strip()]
        
        return image_names
    
    def _get_transforms(self) -> A.Compose:
        """Создание пайплайна трансформаций."""
        if self.augment and self.split == "train":
            return A.Compose([
                A.RandomResizedCrop(
                    height=self.image_size[0],
                    width=self.image_size[1],
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='xywh',
                label_fields=['labels', 'fence_types', 'object_ids', 'shadow_group_ids'],
                min_area=self.min_object_area,
                min_visibility=0.3
            ))
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='xywh',
                label_fields=['labels', 'fence_types', 'object_ids', 'shadow_group_ids'],
                min_area=self.min_object_area,
                min_visibility=0.3
            ))
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Получение элемента датасета.
        
        Returns:
            Словарь с изображением, таргетами и метаданными.
        """
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.data_root, "images", f"{image_name}.png")
        annot_path = os.path.join(self.data_root, "annotations", f"{image_name}.json")
        
        # Загрузка изображения
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # Загрузка аннотаций
        with open(annot_path, 'r', encoding='utf-8') as f:
            annot = json.load(f)
        
        # Извлечение метаданных
        gsd = annot.get("gsd", 0.1)
        sun_azimuth = annot.get("sun_azimuth", 0.0)
        sun_elevation = annot.get("sun_elevation", 0.0)
        
        # Подготовка bounding box и меток
        bboxes = []
        labels = []  # 0 - background, 1 - fence, 2 - shadow
        fence_types = []
        object_ids = []
        shadow_group_ids = []
        
        for obj in annot.get("objects", []):
            obj_type = obj["type"]
            if obj_type == "fence":
                label = 1
            elif obj_type == "shadow":
                label = 2
            else:
                continue
            
            bbox = obj["bbox"]  # [x, y, w, h, theta]
            
            # Проверка минимальной площади
            area = bbox[2] * bbox[3]
            if area < self.min_object_area:
                continue
            
            bboxes.append(bbox[:4])  # xywh для albumentations
            labels.append(label)
            fence_types.append(obj.get("fence_type", 0))
            object_ids.append(obj.get("id", len(labels)))
            shadow_group_ids.append(obj.get("shadow_group_id", -1))
        
        # Применение трансформаций
        if len(bboxes) > 0:
            transformed = self.transforms(
                image=image,
                bboxes=bboxes,
                labels=labels,
                fence_types=fence_types,
                object_ids=object_ids,
                shadow_group_ids=shadow_group_ids
            )
            
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]
            fence_types = transformed["fence_types"]
            object_ids = transformed["object_ids"]
            shadow_group_ids = transformed["shadow_group_ids"]
        else:
            # Если объектов нет, применяем только трансформации изображения
            transformed = self.transforms(image=image)
            image = transformed["image"]
        
        # Восстановление угла поворота (theta) после трансформаций
        # В реальном сценарии нужно корректировать угол при поворотах
        full_bboxes = []
        for i, bbox in enumerate(bboxes):
            # Для упрощения оставляем угол без изменений
            # В production нужно учитывать применённые вращения
            theta = annot["objects"][i]["bbox"][4] if i < len(annot["objects"]) else 0.0
            full_bboxes.append(list(bbox) + [theta])
        
        return {
            "image": image,
            "image_name": image_name,
            "gsd": gsd,
            "sun_azimuth": sun_azimuth,
            "sun_elevation": sun_elevation,
            "bboxes": torch.as_tensor(full_bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "fence_types": torch.as_tensor(fence_types, dtype=torch.long),
            "object_ids": torch.as_tensor(object_ids, dtype=torch.long),
            "shadow_group_ids": torch.as_tensor(shadow_group_ids, dtype=torch.long),
        }


def create_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (1024, 1024),
    augment: bool = True,
    shuffle: bool = True
) -> DataLoader:
    """
    Создание DataLoader для заданного сплита.
    
    Args:
        data_root: Корневая директория с данными.
        split: Название сплита.
        batch_size: Размер батча.
        num_workers: Количество воркеров для загрузки данных.
        image_size: Размер изображений.
        augment: Применять ли аугментации.
        shuffle: Перемешивать ли данные.
    
    Returns:
        DataLoader для итерации по данным.
    """
    dataset = ASTDataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        augment=augment and (split == "train")
    )
    
    # Для валидации и теста не перемешиваем
    if split != "train":
        shuffle = False
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train")
    )
    
    return dataloader


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Функция для объединения элементов батча.
    
    Обрабатывает случаи, когда количество объектов в разных изображениях отличается.
    
    Args:
        batch: Список элементов датасета.
    
    Returns:
        Словарь с объединёнными тензорами и метаданными.
    """
    images = torch.stack([item["image"] for item in batch])
    image_names = [item["image_name"] for item in batch]
    gsds = torch.as_tensor([item["gsd"] for item in batch], dtype=torch.float32)
    sun_azimuths = torch.as_tensor([item["sun_azimuth"] for item in batch], dtype=torch.float32)
    sun_elevations = torch.as_tensor([item["sun_elevation"] for item in batch], dtype=torch.float32)
    
    # Bounding box и метки могут иметь разную длину, поэтому используем списки
    bboxes = [item["bboxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    fence_types = [item["fence_types"] for item in batch]
    object_ids = [item["object_ids"] for item in batch]
    shadow_group_ids = [item["shadow_group_ids"] for item in batch]
    
    return {
        "images": images,
        "image_names": image_names,
        "gsds": gsds,
        "sun_azimuths": sun_azimuths,
        "sun_elevations": sun_elevations,
        "bboxes": bboxes,
        "labels": labels,
        "fence_types": fence_types,
        "object_ids": object_ids,
        "shadow_group_ids": shadow_group_ids,
    }
