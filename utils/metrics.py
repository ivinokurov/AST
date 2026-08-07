"""
Модуль метрик для оценки качества модели AST.

Содержит реализации mAP для ориентированных bounding box,
метрики связности теней и другие показатели качества.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class OBBMetrics:
    """
    Метрики для оценки детекции ориентированных bounding box.
    
    Поддерживает вычисление mAP (mean Average Precision) с учётом поворота,
    а также отдельные метрики для ограждений и теней.
    """
    
    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        max_detections: int = 300
    ):
        """
        Инициализация метрик.
        
        Args:
            iou_thresholds: Пороги IoU для вычисления mAP.
            max_detections: Максимальное количество детекций на изображение.
        """
        if iou_thresholds is None:
            self.iou_thresholds = [0.5, 0.75]  # Стандартные пороги
        else:
            self.iou_thresholds = iou_thresholds
        
        self.max_detections = max_detections
        
        # Накопление результатов
        self.reset()
    
    def reset(self):
        """Сброс накопленных результатов."""
        self.all_predictions = []
        self.all_ground_truths = []
        self.image_ids = []
    
    def update(
        self,
        predictions: Dict[str, Any],
        ground_truths: Dict[str, Any],
        image_id: str
    ):
        """
        Добавление результатов для одного изображения.
        
        Args:
            predictions: Словарь с предсказаниями модели.
            ground_truths: Словарь с ground truth разметкой.
            image_id: Идентификатор изображения.
        """
        self.all_predictions.append(predictions)
        self.all_ground_truths.append(ground_truths)
        self.image_ids.append(image_id)
    
    def compute_map(
        self,
        class_agnostic: bool = False
    ) -> Dict[str, float]:
        """
        Вычисление mAP по всем накопленным данным.
        
        Args:
            class_agnostic: Игнорировать ли классы объектов.
        
        Returns:
            Словарь с метриками mAP для разных порогов IoU.
        """
        from utils.geometry import rotated_iou
        
        results = {}
        
        for iou_thresh in self.iou_thresholds:
            aps = []
            
            # Группировка по классам
            if class_agnostic:
                classes = [None]
            else:
                classes = list(set(
                    gt.get("labels", [])
                    for gts in self.all_ground_truths
                    for gt in gts
                ))
            
            for cls in classes:
                # Сбор всех детекций и ground truth для класса
                all_scores = []
                all_is_tp = []  # True positive
                
                for preds, gts in zip(self.all_predictions, self.all_ground_truths):
                    pred_boxes = preds.get("bboxes", [])
                    pred_scores = preds.get("scores", [])
                    pred_labels = preds.get("labels", [])
                    
                    gt_boxes = gts.get("bboxes", [])
                    gt_labels = gts.get("labels", [])
                    
                    # Фильтрация по классу
                    if cls is not None:
                        pred_mask = [l == cls for l in pred_labels]
                        gt_mask = [l == cls for l in gt_labels]
                        
                        pred_boxes = [b for b, m in zip(pred_boxes, pred_mask) if m]
                        pred_scores = [s for s, m in zip(pred_scores, pred_mask) if m]
                        gt_boxes = [b for b, m in zip(gt_boxes, gt_mask) if m]
                    
                    if len(gt_boxes) == 0:
                        continue
                    
                    if len(pred_boxes) == 0:
                        continue
                    
                    # Сортировка предсказаний по уверенности
                    sorted_indices = np.argsort(pred_scores)[::-1][:self.max_detections]
                    sorted_boxes = [pred_boxes[i] for i in sorted_indices]
                    
                    # Вычисление TP/FP
                    gt_matched = [False] * len(gt_boxes)
                    
                    for box in sorted_boxes:
                        # Вычисление IoU со всеми GT
                        ious = []
                        for gt_box in gt_boxes:
                            iou = rotated_iou(
                                torch.as_tensor(box),
                                torch.as_tensor(gt_box)
                            ).item()
                            ious.append(iou)
                        
                        # Поиск лучшего совпадения
                        if len(ious) > 0:
                            best_idx = np.argmax(ious)
                            if ious[best_idx] >= iou_thresh and not gt_matched[best_idx]:
                                all_scores.append(pred_scores[sorted_indices[len(all_scores)]])
                                all_is_tp.append(1)
                                gt_matched[best_idx] = True
                            else:
                                all_scores.append(pred_scores[sorted_indices[len(all_scores)]])
                                all_is_tp.append(0)
                
                # Вычисление AP
                if len(all_scores) > 0:
                    ap = self._compute_ap(all_scores, all_is_tp)
                    aps.append(ap)
            
            # Средний AP по классам
            map_value = np.mean(aps) if len(aps) > 0 else 0.0
            results[f"mAP@{iou_thresh:.2f}"] = map_value
        
        return results
    
    def _compute_ap(
        self,
        scores: List[float],
        is_tp: List[int]
    ) -> float:
        """
        Вычисление Average Precision.
        
        Args:
            scores: Уверенности детекций.
            is_tp: Индикаторы true positive (1 или 0).
        
        Returns:
            Значение AP.
        """
        if len(scores) == 0:
            return 0.0
        
        # Сортировка по убыванию уверенности
        sorted_indices = np.argsort(scores)[::-1]
        is_tp = np.array(is_tp)[sorted_indices]
        
        # Накопленная сумма TP
        cum_tp = np.cumsum(is_tp)
        cum_fp = np.cumsum(1 - is_tp)
        
        # Precision и Recall
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)
        recall = cum_tp / (np.sum(is_tp) + 1e-6)
        
        # Interpolated AP (11-point interpolation)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.any(recall >= t):
                p = np.max(precision[recall >= t])
                ap += p / 11
        
        return ap


class ConnectivityMetrics:
    """
    Метрики для оценки качества группировки фрагментов теней.
    
    Оценивает правильность объединения разорванных фрагментов
    в единые тени.
    """
    
    def __init__(self):
        """Инициализация метрик связности."""
        self.reset()
    
    def reset(self):
        """Сброс накопленных результатов."""
        self.tp = 0  # Правильно объединённые пары
        self.fp = 0  # Неправильно объединённые пары
        self.fn = 0  # Пропущенные объединения
        self.total_pairs = 0
    
    def update(
        self,
        predicted_groups: Dict[int, List[int]],
        ground_truth_groups: Dict[int, List[int]]
    ):
        """
        Обновление метрик для одного изображения.
        
        Args:
            predicted_groups: Словарь {group_id: [object_ids]}.
            ground_truth_groups: Ground truth группировка.
        """
        # Построение множества пар для GT
        gt_pairs = set()
        for group in ground_truth_groups.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pair = tuple(sorted([group[i], group[j]]))
                    gt_pairs.add(pair)
        
        # Построение множества пар для предсказаний
        pred_pairs = set()
        for group in predicted_groups.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pair = tuple(sorted([group[i], group[j]]))
                    pred_pairs.add(pair)
        
        # Подсчёт TP, FP, FN
        self.tp += len(gt_pairs & pred_pairs)
        self.fp += len(pred_pairs - gt_pairs)
        self.fn += len(gt_pairs - pred_pairs)
        self.total_pairs += len(gt_pairs)
    
    def compute(self) -> Dict[str, float]:
        """
        Вычисление итоговых метрик связности.
        
        Returns:
            Словарь с precision, recall и F1-score.
        """
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        return {
            "connectivity_precision": precision,
            "connectivity_recall": recall,
            "connectivity_f1": f1,
            "total_gt_pairs": self.total_pairs
        }


class SunPositionMetrics:
    """
    Метрики для оценки точности предсказания положения солнца.
    
    Вычисляет ошибку предсказания азимута и высоты солнца.
    """
    
    def __init__(self):
        """Инициализация метрик положения солнца."""
        self.reset()
    
    def reset(self):
        """Сброс накопленных результатов."""
        self.azimuth_errors = []
        self.elevation_errors = []
    
    def update(
        self,
        predicted_azimuth: float,
        predicted_elevation: float,
        true_azimuth: float,
        true_elevation: float
    ):
        """
        Обновление метрик для одного изображения.
        
        Args:
            predicted_azimuth: Предсказанный азимут (радианы).
            predicted_elevation: Предсказанная высота (радианы).
            true_azimuth: Истинный азимут (радианы).
            true_elevation: Истинная высота (радианы).
        """
        # Ошибка азимута с учётом периодичности
        az_diff = abs(predicted_azimuth - true_azimuth)
        az_diff = min(az_diff, 2 * np.pi - az_diff)
        self.azimuth_errors.append(np.rad2deg(az_diff))
        
        # Ошибка высоты
        el_diff = abs(predicted_elevation - true_elevation)
        self.elevation_errors.append(np.rad2deg(el_diff))
    
    def compute(self) -> Dict[str, float]:
        """
        Вычисление итоговых метрик.
        
        Returns:
            Словарь с MAE для азимута и высоты.
        """
        return {
            "azimuth_mae_deg": np.mean(self.azimuth_errors),
            "elevation_mae_deg": np.mean(self.elevation_errors),
            "sun_position_mae_deg": np.mean(self.azimuth_errors + self.elevation_errors)
        }


def compute_all_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    image_ids: List[str]
) -> Dict[str, float]:
    """
    Вычисление всех метрик для набора данных.
    
    Args:
        predictions: Список предсказаний для каждого изображения.
        ground_truths: Список ground truth для каждого изображения.
        image_ids: Идентификаторы изображений.
    
    Returns:
        Словарь со всеми вычисленными метриками.
    """
    # Детекция OBB
    obb_metrics = OBBMetrics()
    for pred, gt, img_id in zip(predictions, ground_truths, image_ids):
        obb_metrics.update(pred, gt, img_id)
    
    map_results = obb_metrics.compute_map()
    
    # Итоговый словарь
    results = {**map_results}
    
    return results
