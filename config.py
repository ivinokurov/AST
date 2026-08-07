"""
Конфигурация модели AST и параметры обучения.

Этот модуль содержит все гиперпараметры, настройки архитектуры,
коэффициенты функции потерь и параметры обучения для воспроизводимости экспериментов.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BackboneConfig:
    """Конфигурация экстрактора признаков (Swin Transformer с деформируемым вниманием)."""
    # Параметры стекового слоя (Patch Embedding)
    patch_embed_kernel: int = 4
    patch_embed_stride: int = 4
    embed_dim: int = 96
    
    # Количество блоков на каждом этапе
    depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    
    # Размеры окон для оконного внимания (этап 1)
    window_size: int = 7
    
    # Параметры деформируемого внимания (этапы 2-4)
    num_deformable_levels: int = 4  # L: количество масштабов пирамиды
    num_deformable_points: int = 4  # K: количество точек семплирования
    num_heads_deformable: int = 8
    head_dim: int = 32
    
    # Drop path (стохастическая глубина)
    drop_path_rate: float = 0.2  # Максимальная вероятность на последнем этапе
    
    # Dropout в блоках внимания и MLP
    dropout_rate: float = 0.1


@dataclass
class DecoderConfig:
    """Конфигурация трансформерного декодера."""
    # Общая размерность каналов после выравнивания
    hidden_dim: int = 256
    
    # Количество слоёв декодера
    num_layers: int = 6
    
    # Количество объектных запросов
    num_queries: int = 300
    
    # Размерность content и positional query
    query_dim: int = 256
    
    # Количество голов внимания
    num_heads: int = 8
    
    # Расширение MLP в FFN
    mlp_ratio: int = 4
    
    # Dropout
    dropout_rate: float = 0.1


@dataclass
class OffsetFieldConfig:
    """Конфигурация раздельных полей смещения."""
    # Масштабные коэффициенты для tanh
    scale_fence: float = 0.05   # Ограждения: плавные деформации
    scale_shadow: float = 0.35  # Тени: резкие изломы
    
    # Количество уровней и точек (должно совпадать с backbone)
    num_levels: int = 4
    num_points: int = 4


@dataclass
class GeometricPriorConfig:
    """Конфигурация геометрического приора для перекрёстного внимания."""
    # Количество частот для синусоидального кодирования
    num_freqs_azimuth: int = 4
    num_freqs_elevation: int = 4
    
    # Количество классов типов ограждений
    num_fence_types: int = 4
    
    # Размерность эмбеддинга типа ограждения
    fence_type_embed_dim: int = 64


@dataclass
class HeadConfig:
    """Конфигурация головок предсказания."""
    # Скрытая размерность MLP в головках
    hidden_dim: int = 256
    
    # Количество слоёв в MLP
    num_layers: int = 3
    
    # Активация
    activation: str = "relu"
    
    # Инициализация классификационной головки (focal loss)
    focal_pi: float = 0.01
    
    # Количество классов для детекции (фон + объекты)
    num_classes: int = 2  # background, object
    
    # Количество классов типа ограждения
    num_fence_types: int = 4


@dataclass
class LossConfig:
    """Конфигурация функции потерь."""
    # Коэффициенты взвешенной суммы
    lambda_det: float = 1.0   # L_det: потери детекции
    lambda_def: float = 0.1   # L_def: потери деформируемого внимания
    lambda_cross: float = 0.5 # L_cross: потери перекрёстного внимания
    lambda_conn: float = 0.3  # L_conn: потери связности теней
    
    # Параметры Focal Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Порог для группировки фрагментов теней
    conn_threshold: float = 0.5
    
    # Максимальный радиус для связности (в метрах)
    max_conn_radius_meters: float = 3.5


@dataclass
class TrainConfig:
    """Параметры обучения."""
    # Оптимизатор
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))
    
    # Scheduler
    scheduler_type: str = "cosine"  # cosine, step, linear
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Обучение
    batch_size: int = 8
    num_epochs: int = 300
    num_workers: int = 4
    
    # Gradient clipping
    grad_clip_norm: float = 0.1
    
    # Смешанная точность (AMP)
    use_amp: bool = True
    
    # Логирование
    log_interval: int = 10
    eval_interval: int = 10
    
    # Seed для воспроизводимости
    seed: int = 42


@dataclass
class InferenceConfig:
    """Параметры инференса."""
    # Порог уверенности для детекции
    score_threshold: float = 0.5
    
    # Порог NMS (Non-Maximum Suppression)
    nms_threshold: float = 0.4
    
    # Порог связности для группировки теней
    conn_threshold: float = 0.5
    
    # Максимальный радиус связности (в метрах)
    max_conn_radius_meters: float = 3.5
    
    # GSD (Ground Sampling Distance) по умолчанию, м/пиксель
    default_gsd: float = 0.1


@dataclass
class ASTConfig:
    """Полная конфигурация модели AST."""
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    offset_fields: OffsetFieldConfig = field(default_factory=OffsetFieldConfig)
    geometric_prior: GeometricPriorConfig = field(default_factory=GeometricPriorConfig)
    heads: HeadConfig = field(default_factory=HeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    def to_dict(self) -> dict:
        """Преобразует конфигурацию в словарь для сохранения."""
        import dataclasses
        return {
            'backbone': dataclasses.asdict(self.backbone),
            'decoder': dataclasses.asdict(self.decoder),
            'offset_fields': dataclasses.asdict(self.offset_fields),
            'geometric_prior': dataclasses.asdict(self.geometric_prior),
            'heads': dataclasses.asdict(self.heads),
            'loss': dataclasses.asdict(self.loss),
            'train': dataclasses.asdict(self.train),
            'inference': dataclasses.asdict(self.inference),
        }


def get_default_config() -> ASTConfig:
    """Возвращает конфигурацию по умолчанию."""
    return ASTConfig()
