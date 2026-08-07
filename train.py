"""
Скрипт обучения модели AST

Предоставляет функционал для обучения и валидации модели AST.
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model import ASTModel, ASTLoss, create_optimizer, create_scheduler


class DummyDataset(Dataset):
    """Вспомогательный тестовый датасет для отладки пайплайна."""
    
    def __init__(self, num_samples: int = 1000, img_size: int = 512):
        self.num_samples = num_samples
        self.img_size = img_size
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Генерация случайного изображения
        image = torch.randn(3, self.img_size, self.img_size)
        
        # Генерация случайной позиции солнца
        sun_azimuth = torch.rand(1) * 2 * np.pi
        sun_elevation = torch.rand(1) * (np.pi / 2)
        
        # Генерация случайного типа ограждения (0-3)
        fence_type = torch.randint(0, 4, (1,)).squeeze()
        
        # Генерация случайной метки классификации
        cls_label = torch.randint(0, 2, (1,)).float()
        
        # Генерация случайных параметров OBB
        obb_params = torch.zeros(6)
        obb_params[0] = torch.rand(1) * self.img_size  # x
        obb_params[1] = torch.rand(1) * self.img_size  # y
        obb_params[2] = torch.rand(1) * 100 + 10  # w
        obb_params[3] = torch.rand(1) * 100 + 10  # h
        angle = torch.rand(1) * 2 * np.pi - np.pi
        obb_params[4] = torch.sin(angle)  # sin(theta)
        obb_params[5] = torch.cos(angle)  # cos(theta)
        
        return {
            'image': image,
            'sun_azimuth': sun_azimuth.squeeze(),
            'sun_elevation': sun_elevation.squeeze(),
            'fence_type': fence_type,
            'cls_label': cls_label,
            'obb_params': obb_params
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Пользовательская функция формирования батчей."""
    images = torch.stack([item['image'] for item in batch])
    sun_azimuth = torch.stack([item['sun_azimuth'] for item in batch])
    sun_elevation = torch.stack([item['sun_elevation'] for item in batch])
    fence_types = torch.stack([item['fence_type'] for item in batch])
    cls_labels = torch.stack([item['cls_label'] for item in batch])
    obb_params = torch.stack([item['obb_params'] for item in batch])
    
    return {
        'images': images,
        'sun_azimuth': sun_azimuth,
        'sun_elevation': sun_elevation,
        'fence_types': fence_types,
        'cls_labels': cls_labels,
        'obb_params': obb_params
    }


class Trainer:
    """Класс для обучения модели AST."""
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 learning_rate: float = 1e-4, weight_decay: float = 0.05,
                 num_epochs: int = 100, warmup_epochs: int = 5,
                 grad_clip: float = 1.0, use_amp: bool = True):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        
        self.optimizer = create_optimizer(model, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = create_scheduler(self.optimizer, num_epochs, warmup_epochs)
        
        self.criterion = ASTLoss()
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Обучение модели в течение одной эпохи."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Эпоха {epoch+1}/{self.num_epochs} [Обучение]')
        
        for batch_idx, batch in enumerate(pbar):
            # Перенос данных на устройство
            images = batch['images'].to(self.device)
            sun_azimuth = batch['sun_azimuth'].to(self.device)
            sun_elevation = batch['sun_elevation'].to(self.device)
            fence_types = batch['fence_types'].to(self.device)
            
            # Подготовка целевых меток - расширение до формы выхода модели (N_q, B, *)
            batch_size = batch['images'].shape[0]
            targets = {
                'cls_labels': batch['cls_labels'].unsqueeze(0).expand(
                    self.model.num_queries, batch_size, 1).to(self.device),
                'obb_params': batch['obb_params'].unsqueeze(0).expand(
                    self.model.num_queries, batch_size, 6).to(self.device),
                'type_labels': batch['fence_types'].unsqueeze(0).expand(
                    self.model.num_queries, batch_size).to(self.device),
                'sun_azimuth': sun_azimuth,
                'sun_elevation': sun_elevation
            }
            
            # Прямой проход
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    predictions = self.model(images, sun_azimuth, sun_elevation, fence_types)
                    loss = self.criterion(predictions, targets)
                
                # Обратный проход с масштабированием градиентов
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images, sun_azimuth, sun_elevation, fence_types)
                loss = self.criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, dataloader: DataLoader, epoch: int) -> float:
        """Валидация модели."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Эпоха {epoch+1}/{self.num_epochs} [Валидация]')
        
        with torch.no_grad():
            for batch in pbar:
                # Перенос данных на устройство
                images = batch['images'].to(self.device)
                sun_azimuth = batch['sun_azimuth'].to(self.device)
                sun_elevation = batch['sun_elevation'].to(self.device)
                fence_types = batch['fence_types'].to(self.device)
                
                # Подготовка целевых меток - расширение до формы выхода модели (N_q, B, *)
                batch_size = batch['images'].shape[0]
                targets = {
                    'cls_labels': batch['cls_labels'].unsqueeze(0).expand(
                        self.model.num_queries, batch_size, 1).to(self.device),
                    'obb_params': batch['obb_params'].unsqueeze(0).expand(
                        self.model.num_queries, batch_size, 6).to(self.device),
                    'type_labels': batch['fence_types'].unsqueeze(0).expand(
                        self.model.num_queries, batch_size).to(self.device),
                    'sun_azimuth': sun_azimuth,
                    'sun_elevation': sun_elevation
                }
                
                # Прямой проход
                predictions = self.model(images, sun_azimuth, sun_elevation, fence_types)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              save_dir: str = './checkpoints', save_interval: int = 10):
        """Полный цикл обучения модели."""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Начало обучения на {self.num_epochs} эпох...")
        print(f"Устройство: {self.device}")
        print(f"Использование AMP: {self.use_amp}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Обучение
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Валидация
            val_loss = self.validate(val_loader, epoch)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Эпоха {epoch+1}/{self.num_epochs} завершена за {epoch_time:.1f}с")
            print(f"  Потери на обучении: {train_loss:.4f}")
            print(f"  Потери на валидации: {val_loss:.4f}")
            print(f"  Скорость обучения: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Сохранение лучшей модели
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  Сохранена лучшая модель (потери валидации: {val_loss:.4f})")
            
            # Периодическое сохранение чекпоинтов
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  Сохранён чекпоинт на эпохе {epoch+1}")
            
            # Обновление планировщика скорости обучения
            self.scheduler.step()
        
        total_time = time.time() - start_time
        print(f"\nОбучение завершено за {total_time/3600:.2f} часов")
        print(f"Лучшие потери на валидации: {self.best_loss:.4f}")
        
        return self.train_losses, self.val_losses


def main():
    parser = argparse.ArgumentParser(description='Обучение модели AST')
    parser.add_argument('--img-size', type=int, default=512, help='Размер входного изображения')
    parser.add_argument('--num-queries', type=int, default=300, help='Количество объектных запросов')
    parser.add_argument('--batch-size', type=int, default=4, help='Размер батча')
    parser.add_argument('--num-workers', type=int, default=4, help='Количество потоков обработки данных')
    parser.add_argument('--lr', type=float, default=1e-4, help='Скорость обучения')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Коэффициент затухания весов')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох обучения')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Количество эпох разогрева')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Директория для сохранения чекпоинтов')
    parser.add_argument('--no-amp', action='store_true', help='Отключить автоматическую смешанную точность')
    parser.add_argument('--seed', type=int, default=42, help='Случайное зерно')
    
    args = parser.parse_args()
    
    # Установка случайного зерна
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Создание модели
    print("Создание модели AST...")
    model = ASTModel(
        img_size=args.img_size,
        num_queries=args.num_queries
    )
    
    # Подсчёт параметров
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Параметры модели: {num_params / 1e6:.2f}M")
    
    # Создание датасетов
    print("Создание датасетов...")
    train_dataset = DummyDataset(num_samples=800, img_size=args.img_size)
    val_dataset = DummyDataset(num_samples=200, img_size=args.img_size)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )
    
    # Создание тренера
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        use_amp=not args.no_amp
    )
    
    # Обучение
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.save_dir,
        save_interval=10
    )
    
    print("\nОбучение завершено!")


if __name__ == '__main__':
    main()
