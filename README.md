# AST (Adaptive Swin Transformer) — Детекция ограждений и теней

Модель AST предназначена для детекции ограждений и их теней на аэроснимках с учётом положения солнца и типа ограждения. 

[Геометрия пары «ограждение–тень»](images/fence_shadow.pdf)

Архитектура основана на Swin Transformer с деформируемым вниманием, раздельными полями смещения для ограждений и теней, а также механизмом перекрёстного внимания с геометрическим приором.

## Структура проекта

```
ast_model/
├── config.py            # Конфигурация модели и параметры обучения
├── model.py             # Архитектура модели AST
├── train.py             # Скрипт обучения модели
├── inference.py         # Скрипт инференса и визуализации
├── dataset.py           # Загрузчик данных и аугментации
├── transforms.py        # Специализированные трансформации
├── engine.py            # Движок обучения и валидации
├── logger.py            # Логирование (TensorBoard, WandB)
├── post_processing.py   # Пост-обработка (NMS, группировка теней)
├── requirements.txt     # Зависимости Python
├── README.md            # Документация
├── images/              # Изображения (pdf-файлы) 
├── utils/               # Утилиты
│   ├── __init__.py      # Инициализация пакета utils
│   ├── geometry.py      # Геометрические операции с OBB
│   ├── metrics.py       # Метрики качества (mAP, связность)
│   └── visualization.py # Визуализация результатов
└── tests/               # Юнит-тесты
    └── test_model.py    # Тесты для модели AST
```

## Установка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Для разработки (опционально)
pip install pytest wandb
```

## Подготовка данных

Ожидаемая структура датасета:
```
data/
├── images/          # Изображения (.png, .jpg)
├── annotations/     # Аннотации в формате JSON
└── splits/          # Списки файлов для train/val/test
```

Формат аннотации (JSON):
```json
{
  "gsd": 0.1,
  "sun_azimuth": 2.356,
  "sun_elevation": 0.785,
  "objects": [
    {
      "id": 1,
      "type": "fence",
      "fence_type": 2,
      "bbox": [x, y, w, h, theta],
      "shadow_group_id": null
    }
  ]
}
```

## Обучение

```bash
python train.py \
  --data-root ./data \
  --output-dir ./outputs \
  --epochs 300 \
  --batch-size 8 \
  --lr 2e-4
```

Параметры:
- `--data-root`: Путь к данным
- `--output-dir`: Директория для чекпоинтов и логов
- `--epochs`: Количество эпох обучения
- `--batch-size`: Размер батча
- `--lr`: Начальный learning rate
- `--resume`: Путь к чекпоинту для продолжения обучения

## Инференс

```bash
python inference.py \
  --checkpoint ./outputs/best_model.pth \
  --image ./test_image.png \
  --output-dir ./results \
  --sun-azimuth 2.356 \
  --sun-elevation 0.785 \
  --fence-type 2 \
  --gsd 0.1
```

Параметры:
- `--checkpoint`: Путь к чекпоинту модели
- `--image`: Путь к изображению или директории
- `--output-dir`: Директория для результатов
- `--sun-azimuth`: Азимут солнца (радианы)
- `--sun-elevation`: Высота солнца (радианы)
- `--fence-type`: Тип ограждения (0-3)
- `--gsd`: GSD снимка (м/пиксель)

## Запуск тестов

```bash
pytest tests/ -v
```

## Ключевые компоненты

- **SeparateOffsetFields**: Раздельные поля смещения (s_fence=0.05, s_shadow=0.35)
- **GeometricCrossAttention**: Перекрёстное внимание с кодированием азимута, высоты солнца и типа ограждения
- **ASTLoss**: Комбинированная функция потерь (λ_det=1.0, λ_def=0.1, λ_cross=0.5, λ_conn=0.3)
- **ConnectivityHead**: Головкa связности для группировки фрагментов теней

## Лицензия

Проект распространяется под лицензией MIT.
