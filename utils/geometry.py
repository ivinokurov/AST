"""
Модуль геометрических операций для работы с OBB и другими примитивами.

Содержит функции для преобразования между форматами bounding box,
вычисления IoU для ориентированных прямоугольников и другие геометрические утилиты.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional, Union
import cv2


def obb_to_polygon(obb: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Преобразование OBB в полигон (4 вершины).
    
    Args:
        obb: Массив формата [x, y, w, h, theta] или батч [N, 5].
             (x, y) - центр, w - ширина, h - высота, theta - угол в радианах.
    
    Returns:
        Массив вершин формы [4, 2] или [N, 4, 2].
    """
    if isinstance(obb, torch.Tensor):
        obb = obb.cpu().numpy()
    
    single = obb.ndim == 1
    if single:
        obb = obb[np.newaxis, :]
    
    x, y, w, h, theta = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3], obb[:, 4]
    
    # Половины размеров
    half_w = w / 2
    half_h = h / 2
    
    # Вершины в локальной системе координат (до поворота)
    corners_local = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ]).transpose(2, 0, 1)  # [4, N, 2]
    
    # Матрица поворота
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    R = np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])  # [N, 2, 2]
    
    # Поворот вершин
    corners_rotated = np.einsum('nij,jni->nij', R, corners_local)  # [4, N, 2]
    
    # Смещение к центру
    center = np.stack([x, y], axis=1)  # [N, 2]
    corners_global = corners_rotated.transpose(1, 0, 2) + center[:, np.newaxis, :]  # [N, 4, 2]
    
    if single:
        return corners_global[0]
    
    return corners_global


def polygon_to_obb(polygon: np.ndarray) -> np.ndarray:
    """
    Преобразование полигона (4 вершины) в OBB формат.
    
    Args:
        polygon: Массив вершин формы [4, 2] или [N, 4, 2].
    
    Returns:
        Массив OBB формата [5] или [N, 5]: (x, y, w, h, theta).
    """
    single = polygon.ndim == 2
    if single:
        polygon = polygon[np.newaxis, :, :]
    
    n_boxes = polygon.shape[0]
    obbs = []
    
    for i in range(n_boxes):
        pts = polygon[i]  # [4, 2]
        
        # Центр - среднее вершин
        center = pts.mean(axis=0)
        
        # Вычисление ширины, высоты и угла через minAreaRect
        rect = cv2.minAreaRect(pts.astype(np.float32))
        (cx, cy), (w, h), angle = rect
        
        # Конвертация угла в радианы
        theta = np.deg2rad(angle)
        
        obbs.append([center[0], center[1], w, h, theta])
    
    result = np.array(obbs)
    
    if single:
        return result[0]
    
    return result


def rotated_iou(
    obbs1: Union[np.ndarray, torch.Tensor],
    obbs2: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Вычисление IoU для ориентированных bounding box.
    
    Args:
        obbs1: OBB формата [N, 5] или [5].
        obbs2: OBB формата [M, 5] или [5].
    
    Returns:
        Матрица IoU формы [N, M] или скаляр.
    """
    is_tensor = isinstance(obbs1, torch.Tensor) or isinstance(obbs2, torch.Tensor)
    
    if isinstance(obbs1, torch.Tensor):
        obbs1 = obbs1.cpu().numpy()
    if isinstance(obbs2, torch.Tensor):
        obbs2 = obbs2.cpu().numpy()
    
    single1 = obbs1.ndim == 1
    single2 = obbs2.ndim == 1
    
    if single1:
        obbs1 = obbs1[np.newaxis, :]
    if single2:
        obbs2 = obbs2[np.newaxis, :]
    
    n1, n2 = len(obbs1), len(obbs2)
    iou_matrix = np.zeros((n1, n2), dtype=np.float32)
    
    for i in range(n1):
        poly1 = obb_to_polygon(obbs1[i]).astype(np.int32)
        area1 = cv2.contourArea(poly1)
        
        for j in range(n2):
            poly2 = obb_to_polygon(obbs2[j]).astype(np.int32)
            area2 = cv2.contourArea(poly2)
            
            # Пересечение полигонов
            intersect_pts = cv2.intersectConvexConvex(
                poly1.reshape(-1, 1, 2),
                poly2.reshape(-1, 1, 2)
            )[0]
            
            if intersect_pts is None or len(intersect_pts) == 0:
                iou = 0.0
            else:
                intersect_area = cv2.contourArea(intersect_pts)
                union_area = area1 + area2 - intersect_area
                iou = intersect_area / max(union_area, 1e-6)
            
            iou_matrix[i, j] = iou
    
    if single1 and single2:
        result = iou_matrix[0, 0]
    elif single1:
        result = iou_matrix[0, :]
    elif single2:
        result = iou_matrix[:, 0]
    else:
        result = iou_matrix
    
    if is_tensor:
        return torch.from_numpy(result)
    
    return result


def bbox_distance(
    obb1: Union[np.ndarray, torch.Tensor],
    obb2: Union[np.ndarray, torch.Tensor],
    metric: str = "center"
) -> float:
    """
    Вычисление расстояния между двумя OBB.
    
    Args:
        obb1: Первый OBB [5].
        obb2: Второй OBB [5].
        metric: Тип метрики ('center', 'edge').
    
    Returns:
        Расстояние в пикселях.
    """
    if isinstance(obb1, torch.Tensor):
        obb1 = obb1.cpu().numpy()
    if isinstance(obb2, torch.Tensor):
        obb2 = obb2.cpu().numpy()
    
    if metric == "center":
        # Расстояние между центрами
        center1 = obb1[:2]
        center2 = obb2[:2]
        return np.linalg.norm(center1 - center2)
    
    elif metric == "edge":
        # Минимальное расстояние между границами
        poly1 = obb_to_polygon(obb1)
        poly2 = obb_to_polygon(obb2)
        
        min_dist = float('inf')
        for p1 in poly1:
            for p2 in poly2:
                dist = np.linalg.norm(p1 - p2)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    else:
        raise ValueError(f"Неизвестная метрика: {metric}")


def clip_bb_to_image(
    obb: Union[np.ndarray, torch.Tensor],
    img_height: int,
    img_width: int
) -> np.ndarray:
    """
    Обрезка OBB по границам изображения.
    
    Args:
        obb: OBB формата [5].
        img_height: Высота изображения.
        img_width: Ширина изображения.
    
    Returns:
        Обрезанный OBB.
    """
    if isinstance(obb, torch.Tensor):
        obb = obb.cpu().numpy()
    
    x, y, w, h, theta = obb
    
    # Получение вершин
    polygon = obb_to_polygon(obb)
    
    # Клиппинг вершин по границам изображения
    polygon[:, 0] = np.clip(polygon[:, 0], 0, img_width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, img_height - 1)
    
    # Обратное преобразование в OBB
    return polygon_to_obb(polygon)


def scale_obb(
    obb: Union[np.ndarray, torch.Tensor],
    scale_x: float,
    scale_y: Optional[float] = None
) -> np.ndarray:
    """
    Масштабирование OBB относительно центра.
    
    Args:
        obb: OBB формата [5].
        scale_x: Коэффициент масштабирования по X.
        scale_y: Коэффициент масштабирования по Y (если None, равен scale_x).
    
    Returns:
        Масштабированный OBB.
    """
    if isinstance(obb, torch.Tensor):
        obb = obb.cpu().numpy()
    
    if scale_y is None:
        scale_y = scale_x
    
    x, y, w, h, theta = obb
    
    return np.array([x, y, w * scale_x, h * scale_y, theta])


def rotate_point(
    point: Tuple[float, float],
    angle: float,
    center: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
    """
    Поворот точки вокруг центра.
    
    Args:
        point: Координаты точки (x, y).
        angle: Угол поворота в радианах.
        center: Центр поворота (по умолчанию - начало координат).
    
    Returns:
        Повёрнутая точка (x, y).
    """
    x, y = point
    
    if center is None:
        cx, cy = 0.0, 0.0
    else:
        cx, cy = center
    
    # Смещение относительно центра
    dx = x - cx
    dy = y - cy
    
    # Поворот
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    new_x = cx + dx * cos_a - dy * sin_a
    new_y = cy + dx * sin_a + dy * cos_a
    
    return (new_x, new_y)
