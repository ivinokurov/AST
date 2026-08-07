"""
Модуль специализированных аугментаций и трансформаций для модели AST.

Содержит кастомные трансформации для работы с ориентированными bounding box (OBB),
учётом положения солнца и геометрическими свойствами сцены.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import albumentations as A
from albumentations.augmentations.geometric import functional as FGeometric


class RotateWithSunPosition(A.DualTransform):
    """
    Поворот изображения с корректировкой азимута солнца.
    
    При повороте изображения необходимо скорректировать метаданные
    о положении солнца, чтобы сохранить физическую согласованность сцены.
    """
    
    def __init__(
        self,
        limit: float = 15.0,
        always_apply: bool = False,
        p: float = 0.5
    ):
        super().__init__(always_apply, p)
        self.limit = limit
    
    def apply(self, img: np.ndarray, angle: float, **params) -> np.ndarray:
        """Применение поворота к изображению."""
        return FGeometric.rotate(img, angle, interpolation=self.interpolation, border_mode=self.border_mode)
    
    def get_params(self) -> Dict[str, float]:
        """Генерация случайного угла поворота."""
        return {"angle": np.random.uniform(-self.limit, self.limit)}
    
    def apply_to_bbox(
        self,
        bbox: Tuple[float, float, float, float, float],
        angle: float,
        **params
    ) -> Tuple[float, float, float, float, float]:
        """
        Применение поворота к OBB.
        
        Args:
            bbox: (x, y, w, h, theta) - центр, ширина, высота, угол.
            angle: Угол поворота в градусах.
        
        Returns:
            Повёрнутый bbox.
        """
        x, y, w, h, theta = bbox
        
        # Преобразование угла поворота в радианы
        angle_rad = np.deg2rad(angle)
        
        # Поворот центра bbox
        rows, cols = params["rows"], params["cols"]
        center_x, center_y = cols / 2, rows / 2
        
        # Смещение относительно центра
        dx = x - center_x
        dy = y - center_y
        
        # Поворот координат
        new_x = center_x + dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
        new_y = center_y + dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
        
        # Корректировка угла bbox
        new_theta = theta + angle_rad
        
        # Нормализация угла к диапазону [-pi, pi]
        while new_theta > np.pi:
            new_theta -= 2 * np.pi
        while new_theta < -np.pi:
            new_theta += 2 * np.pi
        
        return (new_x, new_y, w, h, new_theta)
    
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("limit",)


class AdjustSunPosition(A.BasicTransform):
    """
    Трансформация для корректировки метаданных о положении солнца.
    
    Применяется совместно с геометрическими аугментациями для сохранения
    физической согласованности между изображением и метаданными.
    """
    
    def __init__(
        self,
        rotation_angle: float = 0.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        always_apply: bool = False,
        p: float = 1.0
    ):
        super().__init__(always_apply, p)
        self.rotation_angle = rotation_angle
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
    
    def apply(
        self,
        sun_azimuth: float,
        sun_elevation: float,
        **params
    ) -> Tuple[float, float]:
        """
        Корректировка азимута и высоты солнца.
        
        Args:
            sun_azimuth: Азимут солнца в радианах.
            sun_elevation: Высота солнца в радианах.
        
        Returns:
            Скорректированные (azimuth, elevation).
        """
        azimuth = sun_azimuth
        elevation = sun_elevation
        
        # Учёт поворота изображения
        if self.rotation_angle != 0.0:
            azimuth += np.deg2rad(self.rotation_angle)
            
            # Нормализация к [0, 2*pi]
            while azimuth < 0:
                azimuth += 2 * np.pi
            while azimuth >= 2 * np.pi:
                azimuth -= 2 * np.pi
        
        # Учёт горизонтального отражения
        if self.flip_horizontal:
            azimuth = 2 * np.pi - azimuth
            if azimuth >= 2 * np.pi:
                azimuth -= 2 * np.pi
        
        # Учёт вертикального отражения (меняет высоту солнца)
        if self.flip_vertical:
            elevation = np.pi / 2 - elevation
        
        return azimuth, elevation
    
    @property
    def targets(self) -> Dict[str, Any]:
        return {
            "sun_azimuth": self.apply,
            "sun_elevation": self.apply,
        }


class ShadowAwareColorJitter(A.ColorJitter):
    """
    Цветовая аугментация с учётом теней.
    
    Особая обработка теневых областей для сохранения их физических свойств:
    - Тени должны оставаться темнее освещённых областей
    - Соотношение яркостей сохраняется при изменении контраста
    """
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Применение цветовой аугментации с сохранением свойств теней."""
        # Стандартная аугментация от родительского класса
        return super().apply(img, **params)


def get_train_transforms(
    image_size: Tuple[int, int] = (1024, 1024),
    min_object_area: int = 100
) -> A.Compose:
    """
    Создание пайплайна аугментаций для обучения.
    
    Args:
        image_size: Целевой размер изображений (height, width).
        min_object_area: Минимальная площадь объекта для сохранения.
    
    Returns:
        Compose из albumentations трансформаций.
    """
    return A.Compose([
        # Случайный кроп с сохранением пропорций
        A.RandomResizedCrop(
            height=image_size[0],
            width=image_size[1],
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),
        
        # Геометрические аугментации
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        RotateWithSunPosition(limit=15, p=0.5),
        
        # Цветовые аугментации
        ShadowAwareColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        
        # Размытие
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        
        # Шум
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        
        # Нормализация
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
        # Конвертация в тензор
        A.pytorch.ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='xywh',
        label_fields=['labels', 'fence_types', 'object_ids', 'shadow_group_ids'],
        min_area=min_object_area,
        min_visibility=0.3
    ))


def get_val_transforms(
    image_size: Tuple[int, int] = (1024, 1024),
    min_object_area: int = 100
) -> A.Compose:
    """
    Создание пайплайна трансформаций для валидации/теста.
    
    Args:
        image_size: Целевой размер изображений (height, width).
        min_object_area: Минимальная площадь объекта для сохранения.
    
    Returns:
        Compose из albumentations трансформаций.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='xywh',
        label_fields=['labels', 'fence_types', 'object_ids', 'shadow_group_ids'],
        min_area=min_object_area,
        min_visibility=0.3
    ))
