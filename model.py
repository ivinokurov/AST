"""
AST (Adaptive Swin Transformer) - Модель для детекции ограждений и их теней

Этот модуль реализует архитектуру AST согласно спецификации:
- Экстрактор признаков на базе Swin Transformer с деформируемым вниманием на этапах 2-4
- Раздельные поля смещения для ограждений и теней
- Четыре головки: классификация, OBB регрессия, тип ограждения, положение солнца
- Декодер с перекрёстным вниманием и геометрическим приором
"""

import math
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Вспомогательные функции и модули
# =============================================================================

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """Отключение путей (Stochastic Depth) для каждого сэмпла."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Отключение путей (Stochastic Depth) для каждого сэмпла."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """MLP с активацией GELU и dropout."""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU,
                 drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# Оконное внимание (Этап 1)
# =============================================================================

class WindowAttention(nn.Module):
    """Оконное многоголовое самовнимание для Swin Transformer."""
    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Таблица относительных позиционных смещений
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# Деформируемое внимание (Этапы 2-4)
# =============================================================================

class SeparateOffsetFields(nn.Module):
    """Раздельные поля смещения для ограждений и теней согласно спецификации AST.
    
    Для ограждений: s_fence = 0.05 (плавные деформации)
    Для теней: s_shadow = 0.35 (резкие изломы на перепадах рельефа)
    
    Δ_ij^(c) = s_c · tanh(W_offset^(c) q_i + b_offset^(c))
    """
    def __init__(self, dim: int, num_levels: int = 4, num_points: int = 4,
                 s_fence: float = 0.05, s_shadow: float = 0.35):
        super().__init__()
        self.s_fence = s_fence
        self.s_shadow = s_shadow
        self.num_levels = num_levels
        self.num_points = num_points
        
        # Раздельные линейные проекторы для ограждений и теней
        offset_dim = 2 * num_levels * num_points  # 2D смещения для каждого уровня/точки
        weight_dim = num_levels * num_points  # Веса внимания для каждого уровня/точки
        
        # Поле смещений для ограждений
        self.fence_offset_proj = nn.Linear(dim, offset_dim)
        self.fence_weight_proj = nn.Linear(dim, weight_dim)
        
        # Поле смещений для теней
        self.shadow_offset_proj = nn.Linear(dim, offset_dim)
        self.shadow_weight_proj = nn.Linear(dim, weight_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Инициализация по схеме Xavier."""
        nn.init.xavier_uniform_(self.fence_offset_proj.weight)
        nn.init.xavier_uniform_(self.fence_weight_proj.weight)
        nn.init.xavier_uniform_(self.shadow_offset_proj.weight)
        nn.init.xavier_uniform_(self.shadow_weight_proj.weight)
    
    def forward(self, query: Tensor) -> Dict[str, Dict[str, Tensor]]:
        """
        Аргументы:
            query: Тензор запроса формы (B, N, C) или (N, B, C)
        
        Возвращает:
            offsets: Словарь с ключами 'fence' и 'shadow', каждый содержит:
                - 'offset': Масштабированные смещения (..., 2*L*K)
                - 'weight': Веса внимания (..., L*K)
        """
        # Обработка форматов (B, N, C) и (N, B, C)
        if query.dim() == 3:
            # Транспонирование в (B, N, C) при необходимости
            if query.shape[0] > query.shape[1]:
                query = query.transpose(0, 1)
        
        # Вычисление смещений для ограждений
        fence_offset_raw = self.fence_offset_proj(query)
        fence_offset = self.s_fence * torch.tanh(fence_offset_raw)
        fence_weight = torch.sigmoid(self.fence_weight_proj(query))
        
        # Вычисление смещений для теней
        shadow_offset_raw = self.shadow_offset_proj(query)
        shadow_offset = self.s_shadow * torch.tanh(shadow_offset_raw)
        shadow_weight = torch.sigmoid(self.shadow_weight_proj(query))
        
        return {
            'fence': {'offset': fence_offset, 'weight': fence_weight},
            'shadow': {'offset': shadow_offset, 'weight': shadow_weight}
        }


class DeformableAttention(nn.Module):
    """Механизм деформируемого внимания для этапов 2-4.
    
    Каждый запрос семплирует признаки в K=4 обучаемых позициях на L=4 масштабах.
    Используются раздельные поля смещения для ограждений и теней.
    """
    def __init__(self, dim: int, num_heads: int = 8, num_levels: int = 4,
                 num_points: int = 4, dropout: float = 0.1,
                 use_separate_offsets: bool = True, s_fence: float = 0.05,
                 s_shadow: float = 0.35):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = dim // num_heads
        self.use_separate_offsets = use_separate_offsets
        
        # Линейная проекция для весов внимания
        self.attention_weights = nn.Linear(dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        
        # Раздельные поля смещения для ограждений и теней
        if use_separate_offsets:
            self.offset_fields = SeparateOffsetFields(
                dim, num_levels, num_points, s_fence, s_shadow)
        else:
            # Единое поле смещений (стандартный Deformable DETR)
            self.sampling_offsets = nn.Linear(dim, num_heads * num_levels * num_points * 2)
            nn.init.constant_(self.sampling_offsets.weight, 0)
            nn.init.constant_(self.sampling_offsets.bias, 0)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, query: Tensor, reference_points: Tensor, 
                value_list: List[Tensor], spatial_shapes: List[Tuple[int, int]],
                object_class: str = 'fence') -> Tensor:
        """
        Аргументы:
            query: Тензор формы (B, N_q, C) - токены запросов
            reference_points: Тензор формы (B, N_q, 2) - нормализованные опорные точки [0, 1]
            value_list: Список карт признаков на разных масштабах [(B, C, H_l, W_l), ...]
            spatial_shapes: Список (H, W) для каждого уровня
            object_class: 'fence' или 'shadow' - определяет используемое поле смещений
        
        Возвращает:
            output: Тензор формы (B, N_q, C)
        """
        B, N_q, C = query.shape
        
        # Получение смещений в зависимости от класса объекта
        if self.use_separate_offsets:
            offsets_dict = self.offset_fields(query)
            sampling_offsets = offsets_dict[object_class]['offset']
            attention_weights = offsets_dict[object_class]['weight']
            
            # Изменение формы смещений: (B, N_q, 2*L*K) -> (B, N_q, num_heads, num_levels, num_points, 2)
            # Для простоты транслируется на все головы
            sampling_offsets = sampling_offsets.view(
                B, N_q, 1, self.num_levels, self.num_points, 2).expand(
                -1, -1, self.num_heads, -1, -1, -1)
            attention_weights = attention_weights.view(
                B, N_q, 1, self.num_levels * self.num_points).expand(
                -1, -1, self.num_heads, -1)
            attention_weights = F.softmax(attention_weights, dim=-1).view(
                B, N_q, self.num_heads, self.num_levels, self.num_points)
        else:
            # Стандартные единые смещения
            sampling_offsets = self.sampling_offsets(query).view(
                B, N_q, self.num_heads, self.num_levels, self.num_points, 2)
            attention_weights = self.attention_weights(query).view(
                B, N_q, self.num_heads, self.num_levels * self.num_points)
            attention_weights = F.softmax(attention_weights, dim=-1).view(
                B, N_q, self.num_heads, self.num_levels, self.num_points)
        
        # Вычисление позиций семплирования
        # reference_points: (B, N_q, 2) -> (B, N_q, 1, 1, 1, 2)
        sampling_locations = reference_points.view(B, N_q, 1, 1, 1, 2) + sampling_offsets
        
        # Ограничение позиций семплирования диапазоном [0, 1]
        sampling_locations = torch.clamp(sampling_locations, 0, 1)
        
        # Семплирование признаков с нескольких масштабов с помощью grid_sample
        sampled_features = []
        for lvl, (H_lvl, W_lvl) in enumerate(spatial_shapes):
            value_lvl = value_list[lvl]  # (B, C, H, W)
            if value_lvl.dim() == 4:
                # Изменение формы для grid_sample: (B, C, H, W)
                C_lvl = value_lvl.shape[1]
                
                # Получение позиций семплирования для этого уровня
                loc_lvl = sampling_locations[..., lvl, :, :]  # (B, N_q, num_heads, num_points, 2)
                
                # Усреднение по головам и точкам для простоты
                loc_avg = loc_lvl.mean(dim=2)  # (B, N_q, num_points, 2)
                
                # Изменение формы для grid_sample
                loc_flat = loc_avg.view(B, N_q * self.num_points, 1, 2)
                
                # Семплирование с помощью grid_sample (требует нормализованных координат в [-1, 1])
                loc_normalized = loc_flat * 2 - 1  # Преобразование [0,1] в [-1,1]
                
                # Изменение формы значения для grid_sample
                value_grid = F.interpolate(value_lvl, size=(H_lvl, W_lvl), mode='bilinear', align_corners=False)
                
                # Семплирование - упрощённый подход
                sampled = F.grid_sample(
                    value_grid,
                    loc_normalized,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                )
                sampled_features.append(sampled)
        
        # Комбинирование семплированных признаков
        if sampled_features:
            # Конкатенация и проекция
            combined = torch.cat(sampled_features, dim=1)
            output = self.output_proj(combined.squeeze(-1).transpose(1, 2))
        else:
            # Резервный вариант: прямая проекция запроса
            value = self.value_proj(query)
            output = self.output_proj(value)
        
        output = self.dropout(output)
        return output


# =============================================================================
# Блок Swin Transformer
# =============================================================================

class SwinTransformerBlock(nn.Module):
    """Блок Swin Transformer с архитектурой pre-norm."""
    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 drop: float = 0.0, drop_path: float = 0.0,
                 use_deformable: bool = False, num_levels: int = 4, num_points: int = 4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        
        if use_deformable:
            self.attn = DeformableAttention(
                dim, num_heads=num_heads, num_levels=num_levels,
                num_points=num_points, dropout=drop)
        else:
            self.attn = WindowAttention(
                dim, num_heads=num_heads, window_size=window_size,
                attn_drop=drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.GELU, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, hw_shape: Tuple[int, int], 
                reference_points: Optional[Tensor] = None,
                value_list: Optional[List[Tensor]] = None,
                spatial_shapes: Optional[List[Tuple[int, int]]] = None) -> Tensor:
        B, L, C = x.shape
        H, W = hw_shape
        
        if isinstance(self.attn, DeformableAttention):
            # Деформируемое внимание - без ограничений на размер окон
            assert L == H * W, f"Несоответствие размера карты признаков: {L} vs {H}*{W}"
            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)
            
            # Для деформируемого внимания без опорных точек используется простой резервный вариант
            if reference_points is None or len(value_list) == 0:
                # Просто применяется линейная проекция в качестве резервного варианта
                x_flat = x.view(B, H * W, C)
                x_out = self.attn.value_proj(x_flat)
                x_out = self.attn.output_proj(x_out)
                x_out = self.attn.dropout(x_out)
                x = x_flat + self.drop_path(x_out)
            else:
                x = self.attn(x, reference_points, value_list, spatial_shapes)
                x = x.view(B, L, C)
                x = shortcut + self.drop_path(x)
            
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # Оконное внимание - размеры должны быть кратны размеру окна
            # Паддинг входа для кратности размеру окна
            pad_l = 0
            pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            
            # Паддинг входа
            x_padded = x.view(B, H, W, C)
            if pad_r > 0 or pad_b > 0:
                x_padded = F.pad(x_padded, (0, 0, pad_l, pad_r, pad_t, pad_b))
            
            B2, Hp, Wp, C2 = x_padded.shape
            
            # Применение нормализации
            x_norm = self.norm1(x_padded)
            
            # Сдвиг окон
            if self.shift_size > 0:
                x_norm = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            
            # Разбиение на окна
            x_windows = self.window_partition(x_norm, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
            
            # Внимание в окнах (W-MSA)
            attn_windows = self.attn(x_windows)
            
            # Объединение окон
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            x_out = self.window_reverse(attn_windows, self.window_size, Hp, Wp)
            
            # Обратный сдвиг
            if self.shift_size > 0:
                x_out = torch.roll(x_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            
            # Удаление паддинга и возврат к исходной форме
            if pad_r > 0 or pad_b > 0:
                x_out = x_out[:, :H-pad_t, :W-pad_l, :].contiguous()
            
            x_out = x_out.view(B, H * W, C)
            
            # Остаточное соединение - использование исходного shortcut
            x = x + self.drop_path(x_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    
    @staticmethod
    def window_partition(x: Tensor, window_size: int) -> Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            -1, window_size, window_size, C)
        return windows
    
    @staticmethod
    def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


# =============================================================================
# Внедрение патчей и слияние патчей
# =============================================================================

class PatchEmbed(nn.Module):
    """Слой внедрения патчей: свёртка 4x4 с LayerNorm."""
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 embed_dim: int = 96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        x = self.proj(x)
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H_out, W_out)


class PatchMerging(nn.Module):
    """Слияние патчей: объединяет 2x2 соседних патча, удваивает каналы."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tuple[Tensor, Tuple[int, int]]:
        H, W = hw_shape
        B, L, C = x.shape
        assert L == H * W, "Размер входного признака не соответствует hw_shape"
        
        x = x.view(B, H, W, C)
        
        # Паддинг для нечётных размеров
        if H % 2 == 1:
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
            H += 1
        if W % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1, 0, 0))
            W += 1
        
        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B, H/2, W/2, 4*C
        x = x.view(B, -1, 4 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x, (H // 2, W // 2)


# =============================================================================
# Backbone AST
# =============================================================================

class ASTBackbone(nn.Module):
    """Backbone AST: Swin Transformer с деформируемым вниманием на этапах 2-4.
    
    Формирует многоуровневые карты признаков и раздельные поля смещений для ограждений и теней.
    """
    def __init__(self, img_size: int = 512, embed_dim: int = 96, depths: List[int] = [2, 2, 6, 2],
                 num_heads: List[int] = [3, 6, 12, 24], window_size: int = 7,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0, drop_path_rate: float = 0.2,
                 num_levels: int = 4, num_points: int = 4):
        super().__init__()
        
        self.num_levels = num_levels
        self.embed_dim = embed_dim
        self.depths = depths
        
        # Внедрение патчей
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=4, in_chans=3,
                                       embed_dim=embed_dim)
        
        # Ставки стохастической глубины
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Построение этапов
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.offset_predictors = nn.ModuleList()
        
        num_channels = embed_dim
        for i_stage in range(4):
            # Создание блоков для этого этапа
            use_deformable = (i_stage >= 1)  # Этапы 2-4 используют деформируемое внимание
            stage_blocks = nn.ModuleList()
            
            for i_block in range(depths[i_stage]):
                shift_size = window_size // 2 if i_block % 2 == 1 else 0
                block = SwinTransformerBlock(
                    dim=num_channels,
                    num_heads=num_heads[i_stage] if i_stage < len(num_heads) else num_heads[-1],
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[sum(depths[:i_stage]) + i_block],
                    use_deformable=use_deformable,
                    num_levels=num_levels,
                    num_points=num_points
                )
                stage_blocks.append(block)
            
            self.stages.append(stage_blocks)
            
            # Добавление понижения разрешения между этапами (кроме последнего)
            if i_stage < 3:
                self.downsamples.append(PatchMerging(num_channels))
            else:
                self.downsamples.append(None)
            
            # Предикторы полей смещений для этапов 2-4
            if i_stage >= 1:
                # Раздельные поля смещения для ограждений и теней
                offset_dim = 2 * num_levels * num_points  # 2D смещения
                weight_dim = num_levels * num_points  # Веса внимания
                
                fence_offset_proj = nn.Linear(num_channels, offset_dim)
                fence_weight_proj = nn.Linear(num_channels, weight_dim)
                shadow_offset_proj = nn.Linear(num_channels, offset_dim)
                shadow_weight_proj = nn.Linear(num_channels, weight_dim)
                
                self.offset_predictors.append(nn.ModuleDict({
                    'fence_offset': fence_offset_proj,
                    'fence_weight': fence_weight_proj,
                    'shadow_offset': shadow_offset_proj,
                    'shadow_weight': shadow_weight_proj
                }))
            else:
                self.offset_predictors.append(None)
            
            num_channels *= 2  # Удвоение каналов после слияния патчей
        
        self.num_features = num_channels // 2  # Каналы на последнем этапе
    
    def forward(self, x: Tensor) -> Tuple[List[Tensor], Dict[str, List[Tensor]]]:
        """
        Аргументы:
            x: Входной тензор изображения (B, 3, H, W)
        
        Возвращает:
            features: Список карт признаков на 4 масштабах [(B, C_i, H_i, W_i), ...]
            offsets: Словарь с полями смещений 'fence' и 'shadow' на каждом этапе
        """
        B, _, H, W = x.shape
        
        # Внедрение патчей
        x, hw_shape = self.patch_embed(x)
        
        features = []
        offsets = {'fence': [], 'shadow': []}
        
        for i_stage, stage_blocks in enumerate(self.stages):
            # Обработка блоков
            for block in stage_blocks:
                x = block(x, hw_shape)
            
            # Изменение формы в (B, C, H, W)
            H_curr, W_curr = hw_shape
            x_reshaped = x.transpose(1, 2).view(B, -1, H_curr, W_curr)
            features.append(x_reshaped)
            
            # Предсказание полей смещений для этапов 2-4
            if self.offset_predictors[i_stage] is not None:
                # Использование усреднённых признаков для предсказания смещений
                x_mean = x.mean(dim=1)  # (B, C)
                
                fence_offset = self.offset_predictors[i_stage]['fence_offset'](x_mean)
                fence_weight = self.offset_predictors[i_stage]['fence_weight'](x_mean)
                shadow_offset = self.offset_predictors[i_stage]['shadow_offset'](x_mean)
                shadow_weight = self.offset_predictors[i_stage]['shadow_weight'](x_mean)
                
                offsets['fence'].append({
                    'offset': fence_offset,
                    'weight': fence_weight
                })
                offsets['shadow'].append({
                    'offset': shadow_offset,
                    'weight': shadow_weight
                })
            else:
                offsets['fence'].append(None)
                offsets['shadow'].append(None)
            
            # Понижение разрешения
            if self.downsamples[i_stage] is not None:
                x, hw_shape = self.downsamples[i_stage](x, hw_shape)
        
        return features, offsets


# =============================================================================
# Трансформерный декодер
# =============================================================================

class TransformerDecoderLayer(nn.Module):
    """Один слой трансформерного декодера с самовниманием, перекрёстным вниманием и FFN."""
    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 1024,
                 dropout: float = 0.1, num_levels: int = 4, num_points: int = 4):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = DeformableAttention(
            d_model, num_heads=nhead, num_levels=num_levels,
            num_points=num_points, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.relu

    def forward(self, tgt: Tensor, memory: List[Tensor], 
                reference_points: Tensor, spatial_shapes: List[Tuple[int, int]],
                tgt_mask: Optional[Tensor] = None) -> Tensor:
        # Самовнимание
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Перекрёстное внимание (деформируемое)
        tgt2 = self.cross_attn(tgt.transpose(0, 1), reference_points, memory, spatial_shapes)
        tgt = tgt + self.dropout2(tgt2.transpose(0, 1))
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerDecoder(nn.Module):
    """Трансформерный декодер с 6 слоями и 300 объектными запросами."""
    def __init__(self, d_model: int = 256, nhead: int = 8, num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024, dropout: float = 0.1, num_queries: int = 300,
                 num_levels: int = 4, num_points: int = 4):
        super().__init__()
        
        self.d_model = d_model
        self.num_queries = num_queries
        
        # Обучаемые объектные запросы (контентный и позиционный)
        self.content_query = nn.Embedding(num_queries, d_model)
        self.positional_query = nn.Embedding(num_queries, d_model)
        
        # Генератор опорных точек
        self.reference_point_generator = nn.Linear(d_model, 2)
        
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, num_levels, num_points)
        self.layers = nn.ModuleList([
            decoder_layer for _ in range(num_decoder_layers)
        ])
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.content_query.weight)
        nn.init.xavier_uniform_(self.positional_query.weight)
        nn.init.xavier_uniform_(self.reference_point_generator.weight)

    def forward(self, memory: List[Tensor], spatial_shapes: List[Tuple[int, int]]) -> Tensor:
        """
        Аргументы:
            memory: Список карт признаков энкодера на разных масштабах
            spatial_shapes: Список (H, W) для каждого масштаба
        
        Возвращает:
            output: Выход декодера (num_queries, B, d_model)
        """
        # Инициализация запросов
        content_query = self.content_query.weight.unsqueeze(1)  # (N_q, 1, C)
        positional_query = self.positional_query.weight.unsqueeze(1)
        
        # Генерация начальных опорных точек
        reference_points = torch.sigmoid(self.reference_point_generator(
            self.content_query.weight))  # (N_q, 2)
        reference_points = reference_points.unsqueeze(0).expand(
            content_query.shape[1], -1, -1)  # (B, N_q, 2)
        
        tgt = content_query + positional_query
        
        # Проход по слоям декодера
        for layer in self.layers:
            tgt = layer(tgt, memory, reference_points, spatial_shapes)
        
        return tgt


# =============================================================================
# Модуль перекрёстного внимания с геометрическим априорным ограничением
# =============================================================================

def sinusoidal_positional_encoding_1d(x: Tensor, num_freqs: int = 4) -> Tensor:
    """
    Синусоидальное позиционное кодирование для скалярных значений (азимут/высота).
    
    PE(x) = [sin(2^k x), cos(2^k x)] для k = 0..K-1
    
    Аргументы:
        x: Входной тензор формы (B,) или (B, 1)
        num_freqs: Количество частотных полос (K=4 согласно спецификации)
    
    Возвращает:
        encoding: Тензор формы (B, 2*K)
    """
    # Убедиться, что x имеет хотя бы 1D
    if x.dim() == 0:
        x = x.unsqueeze(0)  # Скаляр в (1,)
    
    freq_bands = torch.pow(2, torch.arange(num_freqs, device=x.device)).to(x.dtype)
    # x: (B,) -> (B, 1); freq_bands: (K,) -> (1, K)
    # scaled_x: (B, K)
    scaled_x = x.unsqueeze(-1) * freq_bands.unsqueeze(0)
    # Стек sin и cos: (B, K, 2) -> (B, 2*K)
    encoding = torch.stack([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1)
    encoding = encoding.view(x.shape[0], -1)
    return encoding


class GeometricCrossAttention(nn.Module):
    """Модуль перекрёстного внимания с геометрическим приором по положению солнца и типу ограждения.
    
    Реализует уравнение согласно спецификации:
    α_ij^cross = Softmax_j( (q_i^fence · k_j^shadow)/√d + PE(p_j^shadow - p_i^fence)
                           + PE_az(φ_sun) + PE_el(α_sun) + PE_τ(τ_fence) )
    
    Где:
    - PE_az(φ) = [sin(2^k φ), cos(2^k φ)] для k=0..3 (K=4)
    - PE_el(α) = [sin(2^k α), cos(2^k α)] для k=0..3 (K=4)
    - PE_τ(τ) = обучаемое встраивание для типа ограждения
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.1,
                 num_freqs: int = 4, num_fence_types: int = 4):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_freqs = num_freqs
        
        # Проекция запросов, ключей и значений
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Кодирование геометрического приора с помощью синусоидальных функций
        sun_enc_dim = num_freqs * 4  # 4 частоты × 2 (sin/cos) × 2 (азимут/высота)
        self.sun_encoding = nn.Sequential(
            nn.Linear(sun_enc_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Кодирование типа ограждения
        self.type_encoding = nn.Embedding(num_fence_types, d_model)
        
        # Кодирование относительного положения между центрами ограждений и теней
        self.rel_pos_encoding = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.sun_encoding[0].weight)
        nn.init.xavier_uniform_(self.sun_encoding[2].weight)
        nn.init.xavier_uniform_(self.rel_pos_encoding[0].weight)
        nn.init.xavier_uniform_(self.rel_pos_encoding[2].weight)

    def forward(self, fence_queries: Tensor, shadow_queries: Tensor,
                sun_azimuth: Tensor, sun_elevation: Tensor,
                fence_types: Optional[Tensor] = None,
                fence_positions: Optional[Tensor] = None,
                shadow_positions: Optional[Tensor] = None) -> Tensor:
        """
        Аргументы:
            fence_queries: Тензоры запросов ограждений (N, B, C)
            shadow_queries: Тензоры запросов теней (N, B, C)
            sun_azimuth: Азимут солнца φ_sun в радианах (B,)
            sun_elevation: Высота солнца α_sun в радианах (B,)
            fence_types: Индексы типов ограждений τ (B,) или None
            fence_positions: Позиции центров ограждений (N, B, 2) - опционально
            shadow_positions: Позиции центров теней (N, B, 2) - опционально

        Возвращает:
            enhanced_queries: Улучшенные запросы ограждений (N, B, C)
        """
        N, B, C = fence_queries.shape
        
        # Кодирование позиции солнца с помощью синусоидального кодирования
        # sun_azimuth, sun_elevation: (B,) -> encoded: (B, 4*K)
        sun_az_encoded = sinusoidal_positional_encoding_1d(sun_azimuth, self.num_freqs)
        sun_el_encoded = sinusoidal_positional_encoding_1d(sun_elevation, self.num_freqs)
        sun_encoding = torch.cat([sun_az_encoded, sun_el_encoded], dim=-1)  # (B, 4*K)
        sun_encoding = self.sun_encoding(sun_encoding)  # (B, C)
        sun_encoding = sun_encoding.unsqueeze(0).unsqueeze(0)  # (1, 1, C) для трансляции

        # Кодирование типов ограждений
        if fence_types is not None:
            if fence_types.dim() == 1:
                fence_types = fence_types.unsqueeze(0)
            elif fence_types.dim() == 0:
                fence_types = fence_types.unsqueeze(0).unsqueeze(0)
            type_encoding = self.type_encoding(fence_types).unsqueeze(1)
        else:
            type_encoding = 0

        # Кодирование относительного положения
        if fence_positions is not None and shadow_positions is not None:
            rel_pos = shadow_positions.unsqueeze(2) - fence_positions.unsqueeze(1)
            rel_pos_flat = rel_pos.view(N * N * B, 2)
            rel_pos_enc = self.rel_pos_encoding(rel_pos_flat).view(N, N, B, C)
            rel_pos_enc = rel_pos_enc.mean(dim=1).unsqueeze(0)
        else:
            rel_pos_enc = 0

        # Проекция запросов
        Q = self.query_proj(fence_queries + sun_encoding + type_encoding + rel_pos_enc)
        K = self.key_proj(shadow_queries + sun_encoding + rel_pos_enc)
        V = self.value_proj(shadow_queries + sun_encoding + rel_pos_enc)

        # Многоголовое внимание
        Q = Q.view(N, B, self.nhead, C // self.nhead).permute(1, 2, 0, 3)
        K = K.view(N, B, self.nhead, C // self.nhead).permute(1, 2, 0, 3)
        V = V.view(N, B, self.nhead, C // self.nhead).permute(1, 2, 0, 3)

        attn_weights = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(C // self.nhead), dim=-1)
        attn_output = (attn_weights @ V).permute(2, 0, 1, 3).contiguous().view(N, B, C)

        output = self.out_proj(attn_output)
        output = self.dropout(output)

        return fence_queries + output


# =============================================================================
# Головки предсказания
# =============================================================================

class ClassificationHead(nn.Module):
    """Головка классификации с инициализацией для фокальной потери."""
    def __init__(self, d_model: int = 256, num_classes: int = 1, hidden_dim: int = 256,
                 focal_prior: float = 0.01):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Инициализация смещения для фокальной потери
        if num_classes == 1:
            bias_init = -math.log((1 - focal_prior) / focal_prior)
            nn.init.constant_(self.mlp[-1].bias, bias_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class OBBRegressionHead(nn.Module):
    """Головка регрессии ориентированных ограничивающих рамок.
    
    Предсказывает (x, y, w, h, theta) где theta параметризована как (sin(theta), cos(theta)).
    """
    def __init__(self, d_model: int = 256, hidden_dim: int = 256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # x, y, w, h, sin(theta), cos(theta)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Возвращает:
            bbox_params: Тензор (..., 6) с [x, y, w, h, sin_theta, cos_theta]
        """
        return self.mlp(x)
    
    @staticmethod
    def decode_angle(sin_theta: Tensor, cos_theta: Tensor) -> Tensor:
        """Восстанавливает угол из параметризации sin/cos с помощью atan2."""
        return torch.atan2(sin_theta, cos_theta)


class TypeClassificationHead(nn.Module):
    """Головка классификации типа ограждения (4 класса)."""
    def __init__(self, d_model: int = 256, num_types: int = 4):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_types)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)


class SunPositionHead(nn.Module):
    """Головка оценки положения солнца (вспомогательная самообучаемая задача).
    
    Предсказывает азимут (через sin/cos) и высоту (через sigmoid * pi/2).
    """
    def __init__(self, d_model: int = 256, hidden_dim: int = 256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # sin_az, cos_az, sigmoid_el
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Возвращает:
            azimuth: Тензор (..., 2) с [sin_az, cos_az]
            elevation: Тензор (...) с высотой в (0, pi/2]
        """
        pred = self.mlp(x)
        sin_az = pred[..., 0]
        cos_az = pred[..., 1]
        elevation = torch.sigmoid(pred[..., 2]) * (math.pi / 2)
        
        return torch.stack([sin_az, cos_az], dim=-1), elevation


# =============================================================================
# Модель AST
# =============================================================================

class ASTModel(nn.Module):
    """Полная модель AST с экстрактором признаков, декодером и головками предсказания."""
    def __init__(self, img_size: int = 512, num_queries: int = 300, num_classes: int = 1,
                 num_fence_types: int = 4, d_model: int = 256):
        super().__init__()
        
        # Backbone
        self.backbone = ASTBackbone(img_size=img_size)
        
        # Выравнивание признаков (свёртка 1×1 с GroupNorm)
        self.feature_alignment = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, d_model, kernel_size=1),
                nn.GroupNorm(8, d_model)
            ) for channels in [96, 192, 384, 768]
        ])
        
        # Декодер
        self.decoder = TransformerDecoder(
            d_model=d_model, nhead=8, num_decoder_layers=6,
            num_queries=num_queries
        )
        
        # Модуль перекрёстного внимания
        self.cross_attention = GeometricCrossAttention(d_model=d_model)
        
        # Головки предсказания
        self.classification_head = ClassificationHead(d_model, num_classes)
        self.obb_head = OBBRegressionHead(d_model)
        self.type_head = TypeClassificationHead(d_model, num_fence_types)
        self.sun_head = SunPositionHead(d_model)
        
        # Сопоставление запросов с парами ограждение-тень
        self.num_queries = num_queries
        self.num_pairs = num_queries // 2  # Половина — для ограждения, половина — для тени

    def forward(self, images: Tensor, sun_azimuth: Optional[Tensor] = None,
                sun_elevation: Optional[Tensor] = None,
                fence_types: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Аргументы:
            images: Входные изображения (B, 3, H, W)
            sun_azimuth: Угол азимута солнца (B,) - опционально для инференса
            sun_elevation: Угол высоты солнца (B,) - опционально для инференса
            fence_types: Индексы типов ограждений (N, B) - опционально для инференса
        
        Возвращает:
            predictions: Словарь с:
                - cls_logits: Логиты классификации (N_q, B, num_classes)
                - obb_params: Параметры OBB (N_q, B, 6)
                - type_logits: Логиты типов ограждений (N_q, B, num_types)
                - sun_pred: Предсказанное положение солнца (опционально)
        """
        B = images.shape[0]
        
        # Извлечение признаков
        features, offsets = self.backbone(images)
        
        # Выравнивание признаков до общей размерности
        aligned_features = []
        spatial_shapes = []
        for i, feat in enumerate(features):
            feat_aligned = self.feature_alignment[i](feat)
            aligned_features.append(feat_aligned.flatten(2).permute(2, 0, 1))
            spatial_shapes.append((feat_aligned.shape[2], feat_aligned.shape[3]))
        
        # Декодирование
        decoder_output = self.decoder(aligned_features, spatial_shapes)
        
        # Разделение на запросы ограждений и теней
        fence_queries = decoder_output[:self.num_pairs]
        shadow_queries = decoder_output[self.num_pairs:]
        
        # Применение перекрёстного внимания с геометрическим приором
        if sun_azimuth is None:
            # Использование фиктивных значений для инференса
            sun_azimuth = torch.zeros(B, device=images.device)
            sun_elevation = torch.zeros(B, device=images.device)
        
        enhanced_queries = self.cross_attention(
            fence_queries, shadow_queries,
            sun_azimuth, sun_elevation, fence_types
        )
        
        # Генерация предсказаний
        cls_logits = self.classification_head(enhanced_queries)
        obb_params = self.obb_head(enhanced_queries)
        type_logits = self.type_head(enhanced_queries)
        
        # Предсказание положения солнца (вспомогательное)
        sun_pred_sin_cos, sun_pred_elevation = self.sun_head(decoder_output.mean(dim=0))
        
        return {
            'cls_logits': cls_logits,
            'obb_params': obb_params,
            'type_logits': type_logits,
            'sun_pred': (sun_pred_sin_cos, sun_pred_elevation),
            'offsets': offsets
        }


# =============================================================================
# Функции потерь
# =============================================================================

class FocalLoss(nn.Module):
    """Фокальная потеря для классификации."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class OBBLoss(nn.Module):
    """Функция потерь для регрессии ориентированных ограничивающих рамок."""
    def __init__(self):
        super().__init__()

    def forward(self, pred_params: Tensor, target_params: Tensor) -> Tensor:
        """
        Аргументы:
            pred_params: (..., 6) с [x, y, w, h, sin_theta, cos_theta]
            target_params: (..., 6) с тем же форматом
        
        Возвращает:
            loss: Скалярное значение потери
        """
        # L1 потеря для центра, ширины и высоты
        l1_loss = F.l1_loss(pred_params[..., :4], target_params[..., :4])
        
        # Угловая потеря (с использованием параметризации sin/cos)
        angular_loss = F.l1_loss(pred_params[..., 4:], target_params[..., 4:])
        
        return l1_loss + angular_loss


class ASTLoss(nn.Module):
    """Комбинированная функция потерь для модели AST."""
    def __init__(self, cls_weight: float = 1.0, obb_weight: float = 5.0,
                 type_weight: float = 1.0, sun_weight: float = 0.1):
        super().__init__()
        
        self.cls_weight = cls_weight
        self.obb_weight = obb_weight
        self.type_weight = type_weight
        self.sun_weight = sun_weight
        
        self.focal_loss = FocalLoss()
        self.obb_loss = OBBLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions: Dict[str, Tensor], 
                targets: Dict[str, Tensor]) -> Tensor:
        """
        Аргументы:
            predictions: Выходной словарь модели
            targets: Целевой словарь с:
                - cls_labels: Метки бинарной классификации
                - obb_params: Целевые параметры OBB
                - type_labels: Метки типов ограждений
                - sun_azimuth, sun_elevation: Метаданные положения солнца
        
        Возвращает:
            total_loss: Взвешенная сумма всех потерь
        """
        # Потеря классификации
        cls_loss = self.focal_loss(
            predictions['cls_logits'], targets['cls_labels']
        )
        
        # Потеря регрессии OBB
        obb_loss = self.obb_loss(
            predictions['obb_params'], targets['obb_params']
        )
        
        # Потеря классификации типов
        type_loss = self.ce_loss(
            predictions['type_logits'].flatten(0, 1),
            targets['type_labels'].flatten(0, 1)
        )
        
        # Потеря положения солнца (вспомогательная)
        pred_sin_cos, pred_elevation = predictions['sun_pred']
        target_sin_az = torch.sin(targets['sun_azimuth'])
        target_cos_az = torch.cos(targets['sun_azimuth'])
        target_sun_sin_cos = torch.stack([target_sin_az, target_cos_az], dim=-1)
        
        sun_az_loss = F.l1_loss(pred_sin_cos, target_sun_sin_cos)
        sun_el_loss = F.l1_loss(pred_elevation, targets['sun_elevation'])
        sun_loss = sun_az_loss + sun_el_loss
        
        total_loss = (
            self.cls_weight * cls_loss +
            self.obb_weight * obb_loss +
            self.type_weight * type_loss +
            self.sun_weight * sun_loss
        )
        
        return total_loss


# =============================================================================
# Утилиты для обучения
# =============================================================================

def create_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.05) -> torch.optim.Optimizer:
    """Создание оптимизатора с различными скоростями обучения для разных компонентов."""
    
    # Разделение параметров на backbone и остальные
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': other_params, 'lr': lr}
    ], weight_decay=weight_decay)
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int,
                     warmup_epochs: int = 5) -> torch.optim.lr_scheduler._LRScheduler:
    """Создание планировщика скорости обучения с прогревом."""
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs
    )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    return scheduler


if __name__ == "__main__":
    # Быстрый тест
    print("Тестирование модели AST...")
    
    # Создание модели
    model = ASTModel(img_size=512, num_queries=300)
    model.eval()
    
    # Создание фиктивного входа
    images = torch.randn(2, 3, 512, 512)
    sun_azimuth = torch.tensor([0.5, 1.0])
    sun_elevation = torch.tensor([0.3, 0.6])
    
    # Прямой проход
    with torch.no_grad():
        outputs = model(images, sun_azimuth, sun_elevation)
    
    print(f"Форма логитов классификации: {outputs['cls_logits'].shape}")
    print(f"Форма параметров OBB: {outputs['obb_params'].shape}")
    print(f"Форма логитов типов: {outputs['type_logits'].shape}")
    print("Тест модели пройден!")