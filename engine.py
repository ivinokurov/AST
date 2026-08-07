"""
Движок обучения и валидации модели AST.

Содержит классы Trainer и Validator для организации цикла обучения,
вычисления потерь и метрик, а также логирования результатов.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
import time

from config import TrainConfig, LossConfig
from model import ASTModel, ASTLoss
from utils.metrics import OBBMetrics, ConnectivityMetrics, SunPositionMetrics


class Trainer:
    """
    Класс для организации процесса обучения модели AST.
    
    Поддерживает:
    - Смешанную точность (AMP)
    - Gradient clipping
    - Warmup learning rate scheduler
    - Логирование метрик и потерь
    """
    
    def __init__(
        self,
        model: ASTModel,
        config: TrainConfig,
        loss_config: LossConfig,
        device: torch.device,
        checkpoint_path: Optional[str] = None
    ):
        """
        Инициализация тренера.
        
        Args:
            model: Модель AST для обучения.
            config: Конфигурация обучения.
            loss_config: Конфигурация функции потерь.
            device: Устройство для вычислений (cuda/cpu).
            checkpoint_path: Путь к чекпоинту для продолжения обучения.
        """
        self.model = model.to(device)
        self.config = config
        self.loss_config = loss_config
        self.device = device
        
        # Функция потерь
        self.criterion = ASTLoss(loss_config).to(device)
        
        # Оптимизатор
        if config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas
            )
        else:
            raise ValueError(f"Неизвестный оптимизатор: {config.optimizer}")
        
        # Scheduler с warmup
        self.scheduler = None  # Будет создан после создания dataloader
        
        # AMP scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Метрики
        self.train_metrics = {
            "obb": OBBMetrics(),
            "connectivity": ConnectivityMetrics(),
            "sun_position": SunPositionMetrics()
        }
        
        self.val_metrics = {
            "obb": OBBMetrics(),
            "connectivity": ConnectivityMetrics(),
            "sun_position": SunPositionMetrics()
        }
        
        # Состояние
        self.current_epoch = 0
        self.best_map = 0.0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "map": [],
            "learning_rate": []
        }
        
        # Загрузка чекпоинта
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
    
    def create_scheduler(self, num_batches: int):
        """
        Создание scheduler с учётом количества батчей.
        
        Args:
            num_batches: Количество батчей в эпохе.
        """
        total_steps = num_batches * self.config.num_epochs
        warmup_steps = num_batches * self.config.warmup_epochs
        
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
        
        # Linear warmup + cosine annealing
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return max(0.0, 1.0 - (1.0 - self.config.min_lr / self.config.learning_rate) * progress)
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Обучение за одну эпоху.
        
        Args:
            dataloader: DataLoader для training данных.
            epoch: Номер текущей эпохи.
        
        Returns:
            Словарь со средними потерями и метриками.
        """
        self.model.train()
        self.train_metrics["obb"].reset()
        self.train_metrics["connectivity"].reset()
        self.train_metrics["sun_position"].reset()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Эпоха {epoch+1}/{self.config.num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Перемещение данных на устройство
            images = batch["images"].to(self.device)
            sun_azimuths = batch["sun_azimuths"].to(self.device)
            sun_elevations = batch["sun_elevations"].to(self.device)
            
            # Таргеты (списки для каждого изображения в батче)
            targets = {
                "bboxes": batch["bboxes"],
                "labels": batch["labels"],
                "fence_types": batch["fence_types"],
                "shadow_group_ids": batch["shadow_group_ids"],
                "gsds": batch["gsds"].tolist() if isinstance(batch["gsds"], torch.Tensor) else batch["gsds"]
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(
                    images,
                    sun_azimuths=sun_azimuths,
                    sun_elevations=sun_elevations
                )
                
                # Вычисление потерь
                losses = self.criterion(outputs, targets)
                total_batch_loss = sum(losses.values())
            
            # Backward pass
            if self.config.use_amp and self.scaler is not None:
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
                self.optimizer.step()
            
            # Обновление scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Статистика
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            # Обновление прогресс-бара
            avg_loss = total_loss / num_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Логирование каждые N батчей
            if (batch_idx + 1) % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.history["learning_rate"].append(current_lr)
        
        # Средние потери за эпоху
        avg_loss = total_loss / max(num_batches, 1)
        self.history["train_loss"].append(avg_loss)
        
        return {"loss": avg_loss}
    
    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Валидация модели.
        
        Args:
            dataloader: DataLoader для валидационных данных.
            epoch: Номер текущей эпохи.
        
        Returns:
            Словарь с метриками валидации.
        """
        self.model.eval()
        self.val_metrics["obb"].reset()
        self.val_metrics["connectivity"].reset()
        self.val_metrics["sun_position"].reset()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Эпоха {epoch+1} [Val]")
        
        for batch in pbar:
            images = batch["images"].to(self.device)
            sun_azimuths = batch["sun_azimuths"].to(self.device)
            sun_elevations = batch["sun_elevations"].to(self.device)
            
            targets = {
                "bboxes": batch["bboxes"],
                "labels": batch["labels"],
                "fence_types": batch["fence_types"],
                "shadow_group_ids": batch["shadow_group_ids"],
                "gsds": batch["gsds"].tolist() if isinstance(batch["gsds"], torch.Tensor) else batch["gsds"]
            }
            
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(
                    images,
                    sun_azimuths=sun_azimuths,
                    sun_elevations=sun_elevations
                )
                
                losses = self.criterion(outputs, targets)
                total_batch_loss = sum(losses.values())
            
            total_loss += total_batch_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.history["val_loss"].append(avg_loss)
        
        # Вычисление mAP
        map_results = self.val_metrics["obb"].compute_map()
        
        # Сохранение лучшего результата
        current_map = map_results.get("mAP@0.50", 0.0)
        if current_map > self.best_map:
            self.best_map = current_map
            self.save_checkpoint("best_model.pth")
        
        self.history["map"].append(current_map)
        
        return {"loss": avg_loss, **map_results}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Полный цикл обучения.
        
        Args:
            train_loader: DataLoader для обучения.
            val_loader: DataLoader для валидации (опционально).
        """
        # Создание scheduler
        self.create_scheduler(len(train_loader))
        
        print(f"Начало обучения модели AST")
        print(f"Устройство: {self.device}")
        print(f"Количество эпох: {self.config.num_epochs}")
        print(f"Размер батча: {self.config.batch_size}")
        print(f"Начальный LR: {self.config.learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            # Обучение
            train_stats = self.train_epoch(train_loader, epoch)
            
            # Валидация
            if val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_stats = self.validate(val_loader, epoch)
                
                print(f"\nЭпоха {epoch+1}:")
                print(f"  Train Loss: {train_stats['loss']:.4f}")
                print(f"  Val Loss: {val_stats['loss']:.4f}")
                print(f"  mAP@0.50: {val_stats.get('mAP@0.50', 0.0):.4f}")
                print(f"  Best mAP: {self.best_map:.4f}\n")
            else:
                print(f"\nЭпоха {epoch+1}: Train Loss = {train_stats['loss']:.4f}\n")
            
            # Сохранение чекпоинта каждые N эпох
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
        
        total_time = time.time() - start_time
        print(f"\nОбучение завершено за {total_time/3600:.2f} часов")
        print(f"Лучший mAP: {self.best_map:.4f}")
    
    def save_checkpoint(self, path: str):
        """
        Сохранение чекпоинта модели.
        
        Args:
            path: Путь для сохранения.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_map": self.best_map,
            "history": self.history,
            "config": self.config.__dict__ if hasattr(self.config, "__dict__") else self.config
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Чекпоинт сохранён: {path}")
    
    def load_checkpoint(self, path: str):
        """
        Загрузка чекпоинта модели.
        
        Args:
            path: Путь к чекпоинту.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.best_map = checkpoint.get("best_map", 0.0)
        self.history = checkpoint.get("history", {})
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Чекпоинт загружен: {path} (эпоха {self.current_epoch})")


class Validator:
    """
    Класс для валидации и тестирования модели AST.
    
    Используется для финальной оценки качества на тестовом наборе данных.
    """
    
    def __init__(
        self,
        model: ASTModel,
        device: torch.device,
        score_threshold: float = 0.5
    ):
        """
        Инициализация валидатора.
        
        Args:
            model: Модель для валидации.
            device: Устройство для вычислений.
            score_threshold: Порог уверенности для детекций.
        """
        self.model = model.to(device).eval()
        self.device = device
        self.score_threshold = score_threshold
        
        self.metrics = {
            "obb": OBBMetrics(),
            "connectivity": ConnectivityMetrics(),
            "sun_position": SunPositionMetrics()
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Полная оценка модели на датасете.
        
        Args:
            dataloader: DataLoader с данными.
        
        Returns:
            Словарь со всеми метриками.
        """
        self.metrics["obb"].reset()
        self.metrics["connectivity"].reset()
        self.metrics["sun_position"].reset()
        
        all_predictions = []
        all_ground_truths = []
        all_image_ids = []
        
        for batch in tqdm(dataloader, desc="Evaluation"):
            images = batch["images"].to(self.device)
            sun_azimuths = batch["sun_azimuths"].to(self.device)
            sun_elevations = batch["sun_elevations"].to(self.device)
            image_names = batch["image_names"]
            
            # Предсказание модели
            outputs = self.model(
                images,
                sun_azimuths=sun_azimuths,
                sun_elevations=sun_elevations
            )
            
            # Пост-обработка предсказаний
            predictions = self._process_outputs(outputs, self.score_threshold)
            
            # Ground truth
            ground_truths = []
            for i in range(len(image_names)):
                gt = {
                    "bboxes": batch["bboxes"][i].cpu().numpy().tolist(),
                    "labels": batch["labels"][i].cpu().numpy().tolist(),
                    "fence_types": batch["fence_types"][i].cpu().numpy().tolist()
                }
                ground_truths.append(gt)
            
            # Обновление метрик
            for pred, gt, img_id in zip(predictions, ground_truths, image_names):
                self.metrics["obb"].update(pred, gt, img_id)
                
                all_predictions.append(pred)
                all_ground_truths.append(gt)
                all_image_ids.append(img_id)
        
        # Вычисление итоговых метрик
        results = self.metrics["obb"].compute_map()
        
        return results
    
    def _process_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Пост-обработка выходов модели.
        
        Args:
            outputs: Сырые выходы модели.
            threshold: Порог уверенности.
        
        Returns:
            Список предсказаний для каждого изображения.
        """
        # Извлечение предсказаний из выходов модели
        # Детальная реализация зависит от формата выходов модели
        predictions = []
        
        batch_size = outputs["cls_logits"].shape[1] if len(outputs["cls_logits"].shape) > 1 else 1
        
        for i in range(batch_size):
            pred = {
                "bboxes": [],
                "scores": [],
                "labels": [],
                "fence_types": []
            }
            predictions.append(pred)
        
        return predictions
