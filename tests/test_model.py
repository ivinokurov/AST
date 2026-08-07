"""
Юнит-тесты для модели AST.

Запуск тестов: pytest tests/ -v
"""

import pytest
import torch
import numpy as np
import sys
import os

# Добавление корня проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ASTModel, SeparateOffsetFields, GeometricCrossAttention
from config import get_default_config
from utils.geometry import obb_to_polygon, rotated_iou, bbox_distance


class TestASTModel:
    """Тесты архитектуры модели."""
    
    def test_model_creation(self):
        """Проверка создания модели."""
        config = get_default_config()
        model = ASTModel(config)
        
        assert model is not None
        assert isinstance(model, ASTModel)
    
    def test_model_forward(self):
        """Проверка прямого прохода модели."""
        config = get_default_config()
        model = ASTModel(config)
        model.eval()
        
        # Создание фиктивного входного изображения
        batch_size = 2
        image = torch.randn(batch_size, 3, 1024, 1024)
        sun_azimuth = torch.zeros(batch_size)
        sun_elevation = torch.zeros(batch_size)
        
        # Прямой проход
        with torch.no_grad():
            outputs = model(image, sun_azimuth, sun_elevation)
        
        # Проверка выходов
        assert "cls_logits" in outputs
        assert "obb_params" in outputs
        assert "type_logits" in outputs
    
    def test_model_parameter_count(self):
        """Проверка количества параметров модели."""
        config = get_default_config()
        model = ASTModel(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Модель должна иметь разумное количество параметров (> 1M)
        assert total_params > 1_000_000


class TestSeparateOffsetFields:
    """Тесты раздельных полей смещения."""
    
    def test_offset_fields_creation(self):
        """Проверка создания полей смещения."""
        config = get_default_config()
        fields = SeparateOffsetFields(
            query_dim=config.decoder.hidden_dim,
            num_levels=config.offset_fields.num_levels,
            num_points=config.offset_fields.num_points,
            scale_fence=config.offset_fields.scale_fence,
            scale_shadow=config.offset_fields.scale_shadow
        )
        
        assert fields is not None
    
    def test_offset_fields_forward(self):
        """Проверка прямого прохода полей смещения."""
        config = get_default_config()
        fields = SeparateOffsetFields(
            query_dim=config.decoder.hidden_dim,
            num_levels=config.offset_fields.num_levels,
            num_points=config.offset_fields.num_points,
            scale_fence=config.offset_fields.scale_fence,
            scale_shadow=config.offset_fields.scale_shadow
        )
        
        batch_size = 4
        num_queries = 10
        query = torch.randn(batch_size, num_queries, config.decoder.hidden_dim)
        
        with torch.no_grad():
            fence_offsets, shadow_offsets = fields(query)
        
        # Проверка размеров выходов
        expected_shape = (batch_size, num_queries, config.offset_fields.num_levels * config.offset_fields.num_points * 2)
        assert fence_offsets.shape == expected_shape
        assert shadow_offsets.shape == expected_shape


class TestGeometricCrossAttention:
    """Тесты перекрёстного внимания с геометрическим приором."""
    
    def test_cross_attention_creation(self):
        """Проверка создания модуля внимания."""
        config = get_default_config()
        attn = GeometricCrossAttention(
            embed_dim=config.decoder.hidden_dim,
            num_heads=config.decoder.num_heads,
            num_freqs_azimuth=config.geometric_prior.num_freqs_azimuth,
            num_freqs_elevation=config.geometric_prior.num_freqs_elevation,
            num_fence_types=config.geometric_prior.num_fence_types
        )
        
        assert attn is not None
    
    def test_cross_attention_forward(self):
        """Проверка прямого прохода внимания."""
        config = get_default_config()
        attn = GeometricCrossAttention(
            embed_dim=config.decoder.hidden_dim,
            num_heads=config.decoder.num_heads,
            num_freqs_azimuth=config.geometric_prior.num_freqs_azimuth,
            num_freqs_elevation=config.geometric_prior.num_freqs_elevation,
            num_fence_types=config.geometric_prior.num_fence_types
        )
        
        batch_size = 2
        num_fences = 5
        num_shadows = 8
        
        fence_query = torch.randn(batch_size, num_fences, config.decoder.hidden_dim)
        shadow_key = torch.randn(batch_size, num_shadows, config.decoder.hidden_dim)
        fence_centers = torch.randn(batch_size, num_fences, 2)
        shadow_centers = torch.randn(batch_size, num_shadows, 2)
        sun_azimuth = torch.zeros(batch_size)
        sun_elevation = torch.zeros(batch_size)
        fence_type = torch.zeros(batch_size, dtype=torch.long)
        
        with torch.no_grad():
            output = attn(
                fence_query, shadow_key, shadow_key,
                fence_centers=fence_centers,
                shadow_centers=shadow_centers,
                sun_azimuth=sun_azimuth,
                sun_elevation=sun_elevation,
                fence_type=fence_type
            )
        
        # Проверка размера выхода
        assert output.shape == (batch_size, num_fences, num_shadows)


class TestGeometry:
    """Тесты геометрических утилит."""
    
    def test_obb_to_polygon(self):
        """Проверка преобразования OBB в полигон."""
        obb = np.array([0.0, 0.0, 10.0, 5.0, 0.0])  # x, y, w, h, theta
        polygon = obb_to_polygon(obb)
        
        assert polygon.shape == (4, 2)
    
    def test_rotated_iou_self(self):
        """Проверка IoU объекта с самим собой."""
        obb = torch.tensor([[0.0, 0.0, 10.0, 5.0, 0.0]])
        iou = rotated_iou(obb, obb)
        
        # IoU объекта с самим собой должен быть близок к 1
        assert iou.item() > 0.99
    
    def test_bbox_distance_center(self):
        """Проверка вычисления расстояния между центрами."""
        obb1 = np.array([0.0, 0.0, 10.0, 5.0, 0.0])
        obb2 = np.array([3.0, 4.0, 10.0, 5.0, 0.0])
        
        distance = bbox_distance(obb1, obb2, metric="center")
        
        # Расстояние между (0,0) и (3,4) = 5
        assert abs(distance - 5.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
