"""
Модуль логирования для модели AST.

Поддерживает логирование в TensorBoard, WandB и текстовые файлы.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union
import torch
import numpy as np


class Logger:
    """
    Базовый класс для логирования экспериментов.
    
    Поддерживает множественные бэкенды:
    - TensorBoard
    - Weights & Biases (WandB)
    - Текстовые файлы
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        project_name: str = "AST",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        use_file: bool = True
    ):
        """
        Инициализация логгера.
        
        Args:
            log_dir: Директория для сохранения логов.
            project_name: Название проекта.
            use_tensorboard: Использовать ли TensorBoard.
            use_wandb: Использовать ли WandB.
            use_file: Писать ли в текстовые файлы.
        """
        self.log_dir = log_dir
        self.project_name = project_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.use_file = use_file
        
        # Создание директории
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"{project_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Инициализация TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=self.run_dir)
                print(f"TensorBoard логи записываются в: {self.run_dir}")
            except ImportError:
                print("Warning: TensorBoard не установлен. Логи будут сохраняться только в файлы.")
                self.tb_writer = None
                self.use_tensorboard = False
        else:
            self.tb_writer = None
        
        # Инициализация WandB
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project_name, dir=self.run_dir)
                self.wandb = wandb
                print("WandB инициализирован")
            except ImportError:
                print("Warning: WandB не установлен.")
                self.wandb = None
                self.use_wandb = False
        else:
            self.wandb = None
        
        # Файловый логгер
        if use_file:
            self.log_file = os.path.join(self.run_dir, "training_log.json")
            self.file_history = []
        else:
            self.log_file = None
            self.file_history = None
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int
    ):
        """
        Логирование скалярного значения.
        
        Args:
            tag: Название метрики.
            value: Значение.
            step: Номер шага/эпохи.
        """
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)
        
        if self.use_wandb and self.wandb is not None:
            self.wandb.log({tag: value}, step=step)
        
        if self.use_file and self.file_history is not None:
            self.file_history.append({
                "tag": tag,
                "value": value,
                "step": step,
                "timestamp": datetime.now().isoformat()
            })
    
    def log_scalars(
        self,
        scalars: Dict[str, float],
        step: int
    ):
        """
        Логирование нескольких скаляров одновременно.
        
        Args:
            scalars: Словарь {tag: value}.
            step: Номер шага/эпохи.
        """
        for tag, value in scalars.items():
            self.log_scalar(tag, value, step)
    
    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        step: int
    ):
        """
        Логирование изображения.
        
        Args:
            tag: Название изображения.
            image: Изображение в формате HWC (RGB).
            step: Номер шага/эпохи.
        """
        if self.use_tensorboard and self.tb_writer is not None:
            # Конвертация в CHW формат для TensorBoard
            if len(image.shape) == 3:
                image = image.transpose(2, 0, 1)
            self.tb_writer.add_image(tag, image, step)
        
        if self.use_wandb and self.wandb is not None:
            self.wandb.log({tag: self.wandb.Image(image)}, step=step)
    
    def log_histogram(
        self,
        tag: str,
        values: np.ndarray,
        step: int
    ):
        """
        Логирование гистограммы значений.
        
        Args:
            tag: Название гистограммы.
            values: Массив значений.
            step: Номер шага/эпохи.
        """
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_histogram(tag, values, step)
    
    def log_model_weights(
        self,
        model: torch.nn.Module,
        step: int
    ):
        """
        Логирование гистограмм весов модели.
        
        Args:
            model: Модель PyTorch.
            step: Номер шага/эпохи.
        """
        if self.use_tensorboard and self.tb_writer is not None:
            for name, param in model.named_parameters():
                self.tb_writer.add_histogram(f"weights/{name}", param.detach().cpu().numpy(), step)
    
    def save_config(self, config: Union[Dict, Any]):
        """
        Сохранение конфигурации эксперимента.
        
        Args:
            config: Конфигурация (словарь или dataclass).
        """
        config_path = os.path.join(self.run_dir, "config.json")
        
        # Преобразование dataclass в словарь
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif hasattr(config, "__dict__"):
            config_dict = config.__dict__
        else:
            config_dict = config
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Конфигурация сохранена: {config_path}")
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """
        Сохранение итоговых метрик.
        
        Args:
            metrics: Словарь с метриками.
        """
        metrics_path = os.path.join(self.run_dir, "final_metrics.json")
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Метрики сохранены: {metrics_path}")
    
    def close(self):
        """Закрытие всех писателей и сохранение файловых логов."""
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()
        
        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()
        
        if self.use_file and self.file_history is not None:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.file_history, f, indent=2, ensure_ascii=False)
            print(f"Файловый лог сохранён: {self.log_file}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_logger(
    log_dir: str = "logs",
    project_name: str = "AST",
    backend: str = "tensorboard"
) -> Logger:
    """
    Фабричная функция для создания логгера.
    
    Args:
        log_dir: Директория для логов.
        project_name: Название проекта.
        backend: Бэкенд ('tensorboard', 'wandb', 'file', 'all').
    
    Returns:
        Экземпляр Logger.
    """
    if backend == "tensorboard":
        return Logger(log_dir=log_dir, project_name=project_name, use_tensorboard=True)
    elif backend == "wandb":
        return Logger(log_dir=log_dir, project_name=project_name, use_wandb=True)
    elif backend == "file":
        return Logger(log_dir=log_dir, project_name=project_name, use_file=True)
    elif backend == "all":
        return Logger(
            log_dir=log_dir,
            project_name=project_name,
            use_tensorboard=True,
            use_wandb=True,
            use_file=True
        )
    else:
        return Logger(log_dir=log_dir, project_name=project_name)
