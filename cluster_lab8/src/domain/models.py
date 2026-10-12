from dataclasses import dataclass
from typing import List

@dataclass
class DataObject:
    """Модель объекта для кластеризации."""
    id: str
    features: List[float] # Массив из 3 признаков (Частота, ОЗУ, Вес)
    name: str = ""        # Название (например, модель ноутбука)