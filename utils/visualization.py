"""
Модуль визуализации результатов модели AST.

Содержит функции для отрисовки ориентированных bounding box,
теней, полей смещения и других элементов сцены.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from PIL import Image, ImageDraw, ImageFont


# Цвета для различных классов объектов
CLASS_COLORS = {
    0: (128, 128, 128),   # background - серый
    1: (0, 128, 0),       # fence - зелёный
    2: (0, 0, 255),       # shadow - красный
}

# Цвета для типов ограждений
FENCE_TYPE_COLORS = {
    0: (0, 128, 0),     # тип 0 - тёмно-зелёный
    1: (0, 255, 0),     # тип 1 - светло-зелёный
    2: (128, 255, 0),   # тип 2 - жёлто-зелёный
    3: (255, 255, 0),   # тип 3 - жёлтый
}


def obb_to_polygon(obb: np.ndarray) -> np.ndarray:
    """
    Преобразование OBB в полигон (4 вершины).
    
    Args:
        obb: Массив [x, y, w, h, theta].
    
    Returns:
        Массив вершин [4, 2].
    """
    x, y, w, h, theta = obb
    
    half_w = w / 2
    half_h = h / 2
    
    # Вершины в локальной системе
    corners_local = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])
    
    # Матрица поворота
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    R = np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])
    
    # Поворот и смещение
    corners_global = corners_local @ R.T + np.array([x, y])
    
    return corners_global.astype(np.int32)


def draw_obb(
    image: np.ndarray,
    obb: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
    label: Optional[str] = None
) -> np.ndarray:
    """
    Отрисовка ориентированного bounding box на изображении.
    
    Args:
        image: Изображение в формате BGR или RGB.
        obb: OBB формата [x, y, w, h, theta].
        color: Цвет линии (R, G, B).
        thickness: Толщина линии.
        label: Текстовая метка (опционально).
    
    Returns:
        Изображение с нарисованным OBB.
    """
    result = image.copy()
    
    # Получение вершин
    polygon = obb_to_polygon(obb)
    
    # Рисование полигона
    cv2.polylines(result, [polygon], True, color, thickness)
    
    # Добавление метки
    if label is not None:
        # Позиция метки - верхняя левая вершина
        top_left = polygon[0]
        
        # Фон для текста
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        pt1 = (top_left[0], top_left[1] - text_size[1] - 5)
        pt2 = (top_left[0] + text_size[0], top_left[1])
        
        cv2.rectangle(result, pt1, pt2, color, -1)
        
        # Текст
        text_pt = (top_left[0], top_left[1] - 3)
        cv2.putText(result, label, text_pt, font, font_scale, (255, 255, 255), thickness)
    
    return result


def draw_detections(
    image: np.ndarray,
    detections: Dict[str, Any],
    score_threshold: float = 0.5,
    show_scores: bool = True,
    show_types: bool = False
) -> np.ndarray:
    """
    Отрисовка всех детекций на изображении.
    
    Args:
        image: Изображение в формате RGB.
        detections: Словарь с предсказаниями модели.
        score_threshold: Порог уверенности для отображения.
        show_scores: Показывать ли значения уверенности.
        show_types: Показывать ли типы ограждений.
    
    Returns:
        Изображение с нарисованными детекциями.
    """
    result = image.copy()
    
    bboxes = detections.get("bboxes", [])
    scores = detections.get("scores", [])
    labels = detections.get("labels", [])
    fence_types = detections.get("fence_types", [])
    
    for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
        if score < score_threshold:
            continue
        
        # Выбор цвета
        if label == 1 and show_types and len(fence_types) > i:
            # Для ограждений используем цвет типа
            ftype = fence_types[i]
            color = FENCE_TYPE_COLORS.get(ftype, (0, 128, 0))
        else:
            color = CLASS_COLORS.get(label, (128, 128, 128))
        
        # Формирование метки
        label_text = ""
        if label == 1:
            label_text = "fence"
            if show_types and len(fence_types) > i:
                label_text += f" T{fence_types[i]}"
        elif label == 2:
            label_text = "shadow"
        
        if show_scores:
            label_text += f" {score:.2f}"
        
        # Отрисовка
        result = draw_obb(result, bbox, color, thickness=2, label=label_text)
    
    return result


def draw_shadow_groups(
    image: np.ndarray,
    detections: Dict[str, Any],
    groups: Dict[int, List[int]],
    score_threshold: float = 0.5
) -> np.ndarray:
    """
    Отрисовка группировки фрагментов теней.
    
    Args:
        image: Изображение в формате RGB.
        detections: Словарь с предсказаниями.
        groups: Словарь {group_id: [object_indices]}.
        score_threshold: Порог уверенности.
    
    Returns:
        Изображение с нарисованными группами.
    """
    result = draw_detections(image, detections, score_threshold)
    
    bboxes = detections.get("bboxes", [])
    labels = detections.get("labels", [])
    
    # Назначение цветов группам
    group_colors = {}
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]
    
    for group_idx, (group_id, indices) in enumerate(groups.items()):
        color = palette[group_idx % len(palette)]
        
        # Отрисовка линий между центрами группы
        centers = []
        for idx in indices:
            if idx < len(bboxes) and labels[idx] == 2:  # Только тени
                bbox = bboxes[idx]
                center = (int(bbox[0]), int(bbox[1]))
                centers.append(center)
        
        # Соединение центров линиями
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                cv2.line(result, centers[i], centers[j], color, 1)
    
    return result


def draw_offset_fields(
    image: np.ndarray,
    offset_field: np.ndarray,
    scale: float = 10.0,
    step: int = 20
) -> np.ndarray:
    """
    Визуализация поля смещений.
    
    Args:
        image: Изображение-подложка.
        offset_field: Поле смещений формы [H, W, 2].
        scale: Коэффициент масштабирования векторов.
        step: Шаг сетки для отрисовки.
    
    Returns:
        Изображение с нарисованным полем смещений.
    """
    result = image.copy()
    h, w = offset_field.shape[:2]
    
    # Прореживание для наглядности
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = offset_field[y, x] * scale
            
            start_pt = (x, y)
            end_pt = (int(x + dx), int(y + dy))
            
            # Цвет зависит от величины смещения
            magnitude = np.sqrt(dx**2 + dy**2)
            color_intensity = min(255, int(magnitude * 50))
            color = (color_intensity, 0, 255 - color_intensity)
            
            cv2.arrowedLine(result, start_pt, end_pt, color, 1, tipLength=0.3)
    
    return result


def create_comparison_grid(
    images: List[np.ndarray],
    titles: List[str],
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Создание сетки изображений для сравнения.
    
    Args:
        images: Список изображений.
        titles: Заголовки для каждого изображения.
        output_size: Размер выходного изображения (ширина, высота).
    
    Returns:
        Сетка изображений с заголовками.
    """
    n_images = len(images)
    
    # Определение размеров сетки
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images / n_cols))
    
    # Приведение изображений к одному размеру
    if output_size is not None:
        resized_images = []
        for img in images:
            img_resized = cv2.resize(img, output_size)
            resized_images.append(img_resized)
        images = resized_images
    else:
        # Использование максимального размера
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        
        resized_images = []
        for img in images:
            img_resized = cv2.resize(img, (max_w, max_h))
            resized_images.append(img_resized)
        images = resized_images
        output_size = (max_w, max_h)
    
    # Высота заголовка
    title_height = 30
    
    # Создание пустой сетки
    grid_h = n_rows * (output_size[1] + title_height)
    grid_w = n_cols * output_size[0]
    
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    # Заполнение сетки
    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // n_cols
        col = idx % n_cols
        
        y_start = row * (output_size[1] + title_height) + title_height
        y_end = y_start + output_size[1]
        x_start = col * output_size[0]
        x_end = x_start + output_size[0]
        
        grid[y_start:y_end, x_start:x_end] = img
        
        # Добавление заголовка
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        text_x = x_start + (output_size[0] - text_size[0]) // 2
        text_y = y_start - 8
        
        cv2.putText(grid, title, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    return grid


def save_visualization(
    image: np.ndarray,
    detections: Dict[str, Any],
    output_path: str,
    **kwargs
):
    """
    Сохранение визуализации в файл.
    
    Args:
        image: Изображение в формате RGB.
        detections: Предсказания модели.
        output_path: Путь для сохранения.
        **kwargs: Дополнительные параметры для draw_detections.
    """
    result = draw_detections(image, detections, **kwargs)
    
    # Конвертация в BGR для OpenCV
    if len(result.shape) == 3 and result.shape[2] == 3:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    else:
        result_bgr = result
    
    cv2.imwrite(output_path, result_bgr)
