"""
Скрипт инференса и оценки модели AST

Этот скрипт предоставляет функциональность инференса для модели AST,
включая предсказание, визуализацию, оценку метрик и группировку фрагментов теней.
"""

import os
import argparse
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import ASTModel


def load_model(checkpoint_path: str, device: torch.device,
               img_size: int = 512, num_queries: int = 300) -> ASTModel:
    """Загрузить модель из чекпоинта."""
    model = ASTModel(img_size=img_size, num_queries=num_queries)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    print(f"Модель загружена из {checkpoint_path}")
    print(f"  Эпоха: {checkpoint.get('epoch', 'Н/Д')}")
    print(f"  Потеря валидации: {checkpoint.get('val_loss', 'Н/Д')}")
    
    return model


def preprocess_image(image_path: str, img_size: int = 512) -> Tuple[torch.Tensor, Image.Image]:
    """Предобработка изображения для инференса."""
    # Загрузка изображения
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Изменение размера
    image_resized = image.resize((img_size, img_size), Image.Resampling.LANCZOS)
    
    # Преобразование в тензор
    image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # Добавление размерности батча
    
    return image_tensor, image


def infer(model: ASTModel, image_tensor: torch.Tensor, device: torch.device,
          sun_azimuth: Optional[float] = None, sun_elevation: Optional[float] = None,
          fence_type: Optional[int] = None, confidence_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
    """Выполнить инференс на одном изображении."""
    model.eval()
    
    with torch.no_grad():
        # Подготовка позиции солнца (использовать значения по умолчанию, если не предоставлены)
        if sun_azimuth is None:
            sun_azimuth = 0.0
        if sun_elevation is None:
            sun_elevation = math.pi / 4
        
        sun_az_tensor = torch.tensor([sun_azimuth], device=device)
        sun_el_tensor = torch.tensor([sun_elevation], device=device)
        
        # Подготовка типа ограждения (если предоставлен)
        if fence_type is None:
            fence_type = 0  # Тип по умолчанию
        
        fence_type_tensor = torch.tensor([fence_type], device=device)
        
        # Запуск модели
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor, sun_az_tensor, sun_el_tensor, fence_type_tensor)
        
        # Извлечение предсказаний
        cls_logits = outputs['cls_logits']  # (N_q, B, 1)
        obb_params = outputs['obb_params']  # (N_q, B, 6)
        type_logits = outputs['type_logits']  # (N_q, B, 4)
        shadow_connectivity = outputs.get('shadow_connectivity', None)  # Для группировки теней
        
        # Применение сигмоиды для классификации
        cls_probs = torch.sigmoid(cls_logits)
        
        # Получение топ предсказаний выше порога
        cls_probs_flat = cls_probs.squeeze(-1).squeeze(-1)  # (N_q,)
        mask = cls_probs_flat > confidence_threshold
        
        predictions = {
            'cls_probs': cls_probs[mask],
            'obb_params': obb_params[mask, 0],
            'type_probs': torch.softmax(type_logits[mask, 0], dim=-1),
        }
        
        # Добавление информации о связности теней, если доступна
        if shadow_connectivity is not None:
            predictions['shadow_connectivity'] = shadow_connectivity
        
        return predictions


def decode_obb(obb_params: torch.Tensor, img_size: int = 512) -> List[Dict]:
    """Декодировать параметры OBB в читаемый формат."""
    results = []
    
    for i in range(len(obb_params)):
        params = obb_params[i]
        
        x = params[0].item() * img_size
        y = params[1].item() * img_size
        w = params[2].item()
        h = params[3].item()
        sin_theta = params[4].item()
        cos_theta = params[5].item()
        
        # Декодирование угла
        theta = math.atan2(sin_theta, cos_theta)
        theta_deg = math.degrees(theta)
        
        results.append({
            'center_x': x,
            'center_y': y,
            'width': w,
            'height': h,
            'angle_rad': theta,
            'angle_deg': theta_deg,
        })
    
    return results


def draw_predictions(image: Image.Image, predictions: Dict, 
                     obb_decoded: List[Dict], output_path: str):
    """Нарисовать предсказания на изображении."""
    # Создание копии
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)
    
    # Цвета для разных типов ограждений
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
    
    for i, (obb, cls_prob, type_prob) in enumerate(zip(
            obb_decoded, predictions['cls_probs'], predictions['type_probs'])):
        
        # Получение лучшего типа
        best_type = type_prob.argmax().item()
        color = colors[best_type % len(colors)]
        
        # Рисование повёрнутого прямоугольника
        cx, cy = obb['center_x'], obb['center_y']
        w, h = obb['width'], obb['height']
        angle = obb['angle_deg']
        
        # Вычисление углов
        corners = get_rotated_rectangle_corners(cx, cy, w, h, angle)
        
        # Рисование полигона
        draw.polygon(corners, outline=color, width=2)
        
        # Рисование метки
        label = f"Ограждение {best_type}: {cls_prob.item():.2f}"
        draw.text((cx - 30, cy - h/2 - 10), label, fill=color)
    
    # Сохранение
    image_draw.save(output_path)
    print(f"Визуализация сохранена в {output_path}")


def get_rotated_rectangle_corners(cx: float, cy: float, w: float, h: float, 
                                   angle_deg: float) -> List[Tuple[float, float]]:
    """Вычислить углы повёрнутого прямоугольника."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Половины размеров
    hw, hh = w / 2, h / 2
    
    # Углы относительно центра
    corners_rel = [
        (-hw, -hh),
        (hw, -hh),
        (hw, hh),
        (-hw, hh)
    ]
    
    # Поворот и трансляция
    corners_abs = []
    for x, y in corners_rel:
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        corners_abs.append((cx + x_rot, cy + y_rot))
    
    return corners_abs


def group_shadow_fragments(predictions: Dict, gsds: float = 0.1,
                           tau_conn: float = 0.5, r_max_meters: float = 3.5) -> List[int]:
    """
    Группировка фрагментов теней в связные компоненты.
    
    Args:
        predictions: Словарь с предсказаниями модели
        gsds: Размер пикселя в метрах (Ground Sample Distance)
        tau_conn: Порог вероятности связности
        r_max_meters: Максимальное расстояние между фрагментами в метрах
    
    Returns:
        Список идентификаторов групп для каждого фрагмента тени
    """
    if 'shadow_connectivity' not in predictions or len(predictions['obb_params']) == 0:
        # Если информация о связности недоступна, возвращаем по одному идентификатору на объект
        return list(range(len(predictions['obb_params'])))
    
    connectivity_matrix = predictions['shadow_connectivity']
    obb_params = predictions['obb_params']
    num_shadows = len(obb_params)
    
    # Преобразование максимального расстояния в пиксели
    img_size = 512  # Предполагаемый размер изображения
    r_max_pixels = r_max_meters / gsds
    
    # Построение графа связности
    # Вершины - детектированные фрагменты теней
    # Рёбра проводятся только между парами с расстоянием <= R_max и вероятностью >= tau_conn
    
    adjacency = [[] for _ in range(num_shadows)]
    
    for i in range(num_shadows):
        for j in range(i + 1, num_shadows):
            # Вычисление расстояния между центрами
            center_i = obb_params[i][:2]  # (x, y)
            center_j = obb_params[j][:2]
            distance = torch.norm(center_i - center_j).item() * img_size
            
            if distance > r_max_pixels * img_size:
                continue
            
            # Проверка вероятности связности
            conn_prob = connectivity_matrix[i, j].item() if connectivity_matrix.dim() == 2 else connectivity_matrix[0, i, j].item()
            
            if conn_prob >= tau_conn:
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    # Поиск связных компонент через DFS
    visited = [False] * num_shadows
    group_ids = [-1] * num_shadows
    current_group = 0
    
    def dfs(node, group_id):
        visited[node] = True
        group_ids[node] = group_id
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                dfs(neighbor, group_id)
    
    for i in range(num_shadows):
        if not visited[i]:
            dfs(i, current_group)
            current_group += 1
    
    return group_ids


def evaluate_predictions(predictions: Dict, ground_truth: Dict,
                         iou_threshold: float = 0.5) -> Dict[str, float]:
    """Оценить предсказания относительно ground truth."""
    # Упрощённая функция оценки
    # На практике следует реализовать полноценный расчёт OBB IoU
    
    pred_boxes = predictions.get('obb_params', [])
    gt_boxes = ground_truth.get('obb_params', [])
    
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    
    if len(pred_boxes) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if len(gt_boxes) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Упрощённо: подсчёт совпадений (на практике использовать OBB IoU)
    true_positives = min(len(pred_boxes), len(gt_boxes))
    false_positives = max(0, len(pred_boxes) - len(gt_boxes))
    false_negatives = max(0, len(gt_boxes) - len(pred_boxes))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def run_inference_pipeline(model_path: str, image_paths: List[str],
                           output_dir: str, img_size: int = 512,
                           confidence_threshold: float = 0.5,
                           device: Optional[str] = None,
                           sun_azimuth: Optional[float] = None,
                           sun_elevation: Optional[float] = None,
                           fence_type: Optional[int] = None,
                           gsds: float = 0.1):
    """Запустить полный пайплайн инференса на нескольких изображениях."""
    # Настройка устройства
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Загрузка модели
    model = load_model(model_path, device, img_size=img_size)
    
    # Создание директории для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Обработка каждого изображения
    all_metrics = []
    
    for img_path in image_paths:
        print(f"\nОбработка {img_path}...")
        
        # Предобработка
        image_tensor, image_pil = preprocess_image(img_path, img_size)
        
        # Инференс
        predictions = infer(model, image_tensor, device,
                           sun_azimuth=sun_azimuth,
                           sun_elevation=sun_elevation,
                           fence_type=fence_type,
                           confidence_threshold=confidence_threshold)
        
        # Декодирование OBB
        if len(predictions['obb_params']) > 0:
            obb_decoded = decode_obb(predictions['obb_params'], img_size)
            
            # Группировка фрагментов теней (если доступна информация о связности)
            if 'shadow_connectivity' in predictions:
                shadow_groups = group_shadow_fragments(
                    predictions, gsds=gsds,
                    tau_conn=0.5, r_max_meters=3.5
                )
                print(f"  Найдено {len(set(shadow_groups))} групп теней")
            
            # Визуализация
            base_name = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"pred_{base_name}")
            draw_predictions(image_pil, predictions, obb_decoded, output_path)
            
            # Печать сводки
            print(f"  Обнаружено {len(obb_decoded)} объектов:")
            for i, obb in enumerate(obb_decoded):
                best_type = predictions['type_probs'][i].argmax().item()
                conf = predictions['cls_probs'][i].item()
                print(f"    Объект {i}: Тип={best_type}, Уверенность={conf:.3f}, "
                      f"Поз=({obb['center_x']:.1f}, {obb['center_y']:.1f}), "
                      f"Размер={obb['width']:.1f}x{obb['height']:.1f}, "
                      f"Угол={obb['angle_deg']:.1f}°")
        else:
            print("  Объекты выше порога не обнаружены")
    
    print(f"\nИнференс завершён! Результаты сохранены в {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Инференс модели AST')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Путь к чекпоинту модели')
    parser.add_argument('--image-paths', type=str, nargs='+',
                       help='Пути к входным изображениям')
    parser.add_argument('--output-dir', type=str, default='./predictions',
                       help='Директория для сохранения предсказаний')
    parser.add_argument('--img-size', type=int, default=512,
                       help='Размер входного изображения')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Порог уверенности для детекций')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство для вычислений (cuda/cpu)')
    parser.add_argument('--sun-azimuth', type=float, default=None,
                       help='Азимут солнца (опционально)')
    parser.add_argument('--sun-elevation', type=float, default=None,
                       help='Высота солнца (опционально)')
    parser.add_argument('--fence-type', type=int, default=None,
                       help='Тип ограждения (опционально)')
    parser.add_argument('--gsd', type=float, default=0.1,
                       help='GSD (Ground Sample Distance) в метрах на пиксель')
    
    args = parser.parse_args()
    
    if not args.image_paths:
        print("Ошибка: Укажите хотя бы один путь к изображению")
        return
    
    run_inference_pipeline(
        model_path=args.model_path,
        image_paths=args.image_paths,
        output_dir=args.output_dir,
        img_size=args.img_size,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        sun_azimuth=args.sun_azimuth,
        sun_elevation=args.sun_elevation,
        fence_type=args.fence_type,
        gsds=args.gsd
    )


if __name__ == '__main__':
    main()
