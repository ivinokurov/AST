"""
Модуль пост-обработки результатов детекции.

Содержит функции для NMS, группировки фрагментов теней
и фильтрации предсказаний модели AST.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


def nms_obb(
    bboxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.4
) -> List[int]:
    """
    Non-Maximum Suppression для ориентированных bounding box.
    
    Args:
        bboxes: Массив OBB формата [N, 5] (x, y, w, h, theta).
        scores: Уверенности детекций [N].
        iou_threshold: Порог IoU для подавления.
    
    Returns:
        Индексы оставшихся детекций.
    """
    if len(bboxes) == 0:
        return []
    
    from utils.geometry import rotated_iou
    
    # Сортировка по убыванию уверенности
    order = np.argsort(scores)[::-1]
    keep = []
    
    while len(order) > 0:
        # Добавление лучшей детекции
        idx = order[0]
        keep.append(idx)
        
        if len(order) == 1:
            break
        
        # Вычисление IoU с остальными
        remaining = order[1:]
        ious = []
        
        for r_idx in remaining:
            iou = rotated_iou(
                torch.as_tensor(bboxes[idx]),
                torch.as_tensor(bboxes[r_idx])
            ).item()
            ious.append(iou)
        
        # Фильтрация по порогу
        ious = np.array(ious)
        mask = ious <= iou_threshold
        order = order[1:][mask]
    
    return keep


def group_shadow_fragments(
    bboxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    connectivity_probs: Optional[np.ndarray] = None,
    conn_threshold: float = 0.5,
    max_radius_meters: float = 3.5,
    gsd: float = 0.1
) -> Dict[int, List[int]]:
    """
    Группировка фрагментов теней в единые объекты.
    
    Алгоритм:
    1. Построение графа связности между фрагментами теней
    2. Удаление рёбер с низкой вероятностью связности
    3. Поиск связных компонент
    
    Args:
        bboxes: OBB формата [N, 5].
        labels: Метки классов [N].
        scores: Уверенности детекций [N].
        connectivity_probs: Матрица вероятностей связности [N, N].
        conn_threshold: Порог вероятности для объединения.
        max_radius_meters: Максимальный радиус связности в метрах.
        gsd: GSD снимка (м/пиксель).
    
    Returns:
        Словарь {group_id: [indices_in_bboxes]}.
    """
    # Фильтрация только теней
    shadow_mask = labels == 2
    shadow_indices = np.where(shadow_mask)[0]
    
    if len(shadow_indices) < 2:
        # 0 или 1 тень - группировка не нужна
        return {i: [idx] for i, idx in enumerate(shadow_indices)}
    
    # Перевод максимального радиуса в пиксели
    max_radius_pixels = max_radius_meters / gsd
    
    # Построение графа связности
    n_shadows = len(shadow_indices)
    adjacency = defaultdict(list)
    
    for i in range(n_shadows):
        idx_i = shadow_indices[i]
        center_i = bboxes[idx_i][:2]
        
        for j in range(i + 1, n_shadows):
            idx_j = shadow_indices[j]
            center_j = bboxes[idx_j][:2]
            
            # Расстояние между центрами
            distance = np.linalg.norm(center_i - center_j)
            
            if distance > max_radius_pixels:
                continue
            
            # Проверка вероятности связности
            if connectivity_probs is not None:
                prob = connectivity_probs[idx_i, idx_j]
                if prob < conn_threshold:
                    continue
            
            # Добавление ребра
            adjacency[idx_i].append(idx_j)
            adjacency[idx_j].append(idx_i)
    
    # Поиск связных компонент (DFS)
    visited = set()
    groups = {}
    group_id = 0
    
    for start_idx in shadow_indices:
        if start_idx in visited:
            continue
        
        # Новый компонент связности
        component = []
        stack = [start_idx]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            
            visited.add(current)
            component.append(current)
            
            # Добавление соседей
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        groups[group_id] = component
        group_id += 1
    
    # Добавление одиночных теней как отдельных групп
    for idx in shadow_indices:
        if idx not in visited:
            groups[group_id] = [idx]
            group_id += 1
    
    return groups


def filter_detections(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    score_threshold: float = 0.5,
    per_class_nms: bool = True,
    nms_threshold: float = 0.4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Фильтрация детекций по порогу уверенности и NMS.
    
    Args:
        bboxes: OBB формата [N, 5].
        scores: Уверенности [N].
        labels: Метки классов [N].
        score_threshold: Порог уверенности.
        per_class_nms: Применять NMS раздельно для каждого класса.
        nms_threshold: Порог IoU для NMS.
    
    Returns:
        Отфильтрованные (bboxes, scores, labels).
    """
    # Порог уверенности
    mask = scores >= score_threshold
    bboxes = bboxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    if len(bboxes) == 0:
        return bboxes, scores, labels
    
    # NMS
    if per_class_nms:
        keep_indices = []
        
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_bboxes = bboxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            cls_keep = nms_obb(cls_bboxes, cls_scores, nms_threshold)
            
            # Преобразование индексов к глобальным
            for idx in cls_keep:
                keep_indices.append(cls_indices[idx])
        
        keep_indices = sorted(keep_indices)
    else:
        keep_indices = nms_obb(bboxes, scores, nms_threshold)
    
    # Применение маски
    bboxes = bboxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]
    
    return bboxes, scores, labels


def compute_connectivity_matrix(
    shadow_features: np.ndarray,
    shadow_centers: np.ndarray,
    shadow_offsets: np.ndarray,
    mlp_weights: np.ndarray,
    mlp_bias: np.ndarray
) -> np.ndarray:
    """
    Вычисление матрицы вероятностей связности для теней.
    
    Формула:
    p_ij = sigmoid(MLP([q_i; q_j; p_i - p_j; Δ_i; Δ_j]))
    
    Args:
        shadow_features: Признаки запросов теней [N, D].
        shadow_centers: Центры OBB теней [N, 2].
        shadow_offsets: Поля смещений теней [N, 2].
        mlp_weights: Веса MLP.
        mlp_bias: Смещение MLP.
    
    Returns:
        Матрица вероятностей [N, N].
    """
    n_shadows = len(shadow_features)
    probs = np.zeros((n_shadows, n_shadows), dtype=np.float32)
    
    for i in range(n_shadows):
        for j in range(i + 1, n_shadows):
            # Конкатенация признаков
            feature_vector = np.concatenate([
                shadow_features[i],
                shadow_features[j],
                shadow_centers[i] - shadow_centers[j],
                shadow_offsets[i],
                shadow_offsets[j]
            ])
            
            # Применение MLP
            logit = np.dot(feature_vector, mlp_weights) + mlp_bias
            prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
            
            probs[i, j] = prob
            probs[j, i] = prob  # Симметричность
    
    return probs


def merge_shadow_groups(
    detections: Dict[str, Any],
    groups: Dict[int, List[int]],
    method: str = "union"
) -> Dict[str, Any]:
    """
    Объединение фрагментов теней в группы.
    
    Args:
        detections: Словарь с детекциями.
        groups: Словарь группировки.
        method: Метод объединения ('union', 'minAreaRect').
    
    Returns:
        Обновлённые детекции с информацией о группах.
    """
    bboxes = detections["bboxes"]
    labels = detections["labels"]
    
    # Назначение group_id каждому объекту
    shadow_group_ids = [-1] * len(bboxes)
    
    for group_id, indices in groups.items():
        for idx in indices:
            if idx < len(shadow_group_ids):
                shadow_group_ids[idx] = group_id
    
    detections["shadow_group_ids"] = shadow_group_ids
    
    # При необходимости можно добавить объединённые OBB для групп
    if method == "union":
        merged_bboxes = []
        merged_labels = []
        
        for group_id, indices in groups.items():
            if len(indices) == 1:
                continue
            
            # Получение всех bbox группы
            group_bboxes = [bboxes[i] for i in indices if i < len(bboxes)]
            
            # Вычисление объединяющего OBB через minAreaRect
            from utils.geometry import obb_to_polygon, polygon_to_obb
            
            all_corners = []
            for bbox in group_bboxes:
                corners = obb_to_polygon(bbox)
                all_corners.append(corners)
            
            all_corners = np.vstack(all_corners)
            merged_bbox = polygon_to_obb(all_corners)
            
            merged_bboxes.append(merged_bbox)
            merged_labels.append(2)  # shadow
        
        if len(merged_bboxes) > 0:
            detections["merged_shadow_bboxes"] = np.array(merged_bboxes)
            detections["merged_shadow_labels"] = np.array(merged_labels)
    
    return detections
