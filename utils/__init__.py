"""
Инициализация пакета utils.

Модули утилит для работы с геометрией, метриками и визуализацией.
"""

from .geometry import (
    obb_to_polygon,
    polygon_to_obb,
    rotated_iou,
    bbox_distance,
    clip_bb_to_image,
    scale_obb,
    rotate_point
)

from .metrics import (
    OBBMetrics,
    ConnectivityMetrics,
    SunPositionMetrics,
    compute_all_metrics
)

from .visualization import (
    draw_obb,
    draw_detections,
    draw_shadow_groups,
    draw_offset_fields,
    create_comparison_grid,
    save_visualization,
    CLASS_COLORS,
    FENCE_TYPE_COLORS
)

__all__ = [
    # Геометрия
    "obb_to_polygon",
    "polygon_to_obb",
    "rotated_iou",
    "bbox_distance",
    "clip_bb_to_image",
    "scale_obb",
    "rotate_point",
    
    # Метрики
    "OBBMetrics",
    "ConnectivityMetrics",
    "SunPositionMetrics",
    "compute_all_metrics",
    
    # Визуализация
    "draw_obb",
    "draw_detections",
    "draw_shadow_groups",
    "draw_offset_fields",
    "create_comparison_grid",
    "save_visualization",
    "CLASS_COLORS",
    "FENCE_TYPE_COLORS",
]
